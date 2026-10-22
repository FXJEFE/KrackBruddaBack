"""3-class BTC models (-1 SELL / 0 HOLD / 1 BUY) on ORIGINAL MT5 features + dpo_causal + TA-Lib extras (features3.py).
Models: XGBoost multi:softprob (class-balanced sample weights) + Keras MLP (class weights), ensemble = mean of probabilities.
Walk-forward: FOLDS expanding folds, purge = H bars. Fold FOLDS = HOLDOUT (never used for calibrator/margin selection).
Decision rule: d = argmax(P(BUY), P(SELL)); signal = d if P(d) - P(HOLD) >= margin else 0 (margin picked on pre-holdout OOS by macro-F1).
p_hit = isotonic(P(d)) -> P(direction d wins the cost-aware barrier race), fitted on pre-holdout purged OOS; EV = p_hit*TP - (1-p_hit)*SL - cost.
Execution gate (trader, unchanged): p_hit >= 0.77 and EV > 0.
DEPLOYED model = ALL features INCLUDING the original strict centred dpo (REQUIRED, hardcoded; user choice for demo paper trading).
DISCLOSURE: dpo (ta.dpo length 21, centred) uses 11 FUTURE bars -> holdout metrics WITH it are inflated vs live, where dpo is NaN on the
newest 11 bars (xgb routes NaN natively; the MLP output is skipped when its input has NaN). An ablation xgb WITHOUT dpo is evaluated on
the same holdout to show the size of the look-ahead effect.
Usage (.myenv311): python models/train_btc3.py [4h 1h 15m 5m 1m]"""
import os, sys, json, time, sqlite3, pathlib, joblib, numpy as np, pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
M = pathlib.Path(__file__).resolve().parent; sys.path[:0] = [str(M), str(M.parent)]; from lkpaths import HOME as ROOT
import features_mt5 as F, features3 as F3
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
A = pathlib.Path(os.environ.get("ARTIFACTS3_DIR") or (ROOT / "models" / "artifacts_btc3")); A.mkdir(parents=True, exist_ok=True)
BAR_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
MAX_DAYS = {"1m": 180, "5m": 540, "15m": 1460}
SERIES = [("XBTUSD", "spot"), ("PI_XBTUSD", "futures")]; FOLDS = 5; GATE_P = 0.77; NJ = int(os.environ.get("NJOBS", 3))

def load(tf):
    c = sqlite3.connect(f"file:{ROOT / 'data' / 'btc_market.db'}?mode=ro", uri=True); fr = []; info = {}; gp = None
    for sym, venue in SERIES:
        d = pd.read_sql("SELECT ts, o open, h high, l low, c close, v volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts", c, params=(sym, tf))
        d = d.drop_duplicates("ts", keep="last").reset_index(drop=True)
        if tf in MAX_DAYS: d = d[d.ts >= d.ts.max() - MAX_DAYS[tf] * 86400000].reset_index(drop=True)
        if len(d) < 1000: continue
        g = F.garch_params(d.close.iloc[: int(len(d) * 0.4)]); gp = gp or g
        f = F3.features(d, g); lb = F3.labels(d, tf, venue, BAR_MS[tf]); f = pd.concat([f, lb], axis=1); f["sym"] = sym; f["venue"] = venue
        f = f.iloc[150:]; f = f[(f.y != -9)].dropna(subset=F3.COLS); fr.append(f)
        info[sym] = {"rows": len(f), "from": str(pd.to_datetime(f.ts.min(), unit="ms")), "to": str(pd.to_datetime(f.ts.max(), unit="ms")),
                     "label_dist": {int(k): int(v) for k, v in f.y.value_counts().sort_index().items()}, "garch": g}
    return pd.concat(fr).sort_values(["ts", "sym"]).reset_index(drop=True), info

def xgb(): return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                                 num_class=3, n_jobs=NJ, eval_metric="mlogloss", tree_method="hist")
def fit_xgb(X, y): return xgb().fit(X, y + 1, sample_weight=compute_sample_weight("balanced", y))
def fit_mlp(X, y):
    import keras
    nrm = keras.layers.Normalization(); nrm.adapt(np.nan_to_num(X.astype("float32")))
    m = keras.Sequential([keras.Input((X.shape[1],)), nrm, keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.2),
                          keras.layers.Dense(32, activation="relu"), keras.layers.Dense(3, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy")
    cw = {i: w for i, w in enumerate(len(y) / (3 * np.maximum(1, np.bincount(y + 1, minlength=3))))}
    m.fit(X.astype("float32"), y + 1, epochs=6, batch_size=1024, verbose=0, class_weight=cw); return m
def pred_mlp(m, X): return np.asarray(m.predict(X.astype("float32"), batch_size=8192, verbose=0))

def decide(P, margin):
    d = np.where(P[:, 2] >= P[:, 0], 1, -1); pd_ = np.where(d == 1, P[:, 2], P[:, 0])
    return np.where(pd_ - P[:, 1] >= margin, d, 0), d, pd_

def evaluate(y, P, sig, d, ph, dt):
    out = {"rows": int(len(y)), "macro_f1": round(float(f1_score(y, sig, average="macro", labels=[-1, 0, 1])), 4)}
    pr, rc, f1, sup = precision_recall_fscore_support(y, sig, labels=[-1, 0, 1], zero_division=0)
    out["per_class"] = {str(c): {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4), "support": int(sup[i])} for i, c in enumerate((-1, 0, 1))}
    try: out["auc_ovr_macro"] = round(float(roc_auc_score(y + 1, P, multi_class="ovr", average="macro")), 4)
    except Exception as e: out["auc_ovr_macro"] = None
    out["label_dist"] = {str(c): int((y == c).sum()) for c in (-1, 0, 1)}; out["signal_dist"] = {str(c): int((sig == c).sum()) for c in (-1, 0, 1)}
    out["confusion(rows=true -1,0,1)"] = confusion_matrix(y, sig, labels=[-1, 0, 1]).tolist()
    hit = (y == d).astype(int); bins = np.array([0, .3, .4, .5, .6, .7, .77, .85, 1.0001]); rel = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (ph >= a) & (ph < b)
        if m.sum(): rel.append({"p_hit_bin": f"{a:.2f}-{min(b,1):.2f}", "n": int(m.sum()), "mean_p_hit": round(float(ph[m].mean()), 4), "realized_hit": round(float(hit[m].mean()), 4)})
    out["p_hit_reliability"] = rel; out["brier_p_hit"] = round(float(np.mean((ph - hit) ** 2)), 4)
    ret = np.where(d == 1, dt.ret_long.values, dt.ret_short.values); ev = ph * dt.tp_pct.values - (1 - ph) * dt.sl_pct.values - dt.cost.values
    for nm, m in (("all_nonhold_signals", sig != 0), ("gate_p>=0.77_ev>0", (sig != 0) & (ph >= GATE_P) & (ev > 0))):
        out[nm] = {"trades": int(m.sum()), "win_rate": round(float(hit[m].mean()), 4) if m.any() else None,
                   "avg_net_ret_pct": round(float(ret[m].mean() * 100), 4) if m.any() else None, "sum_net_ret_pct": round(float(ret[m].sum() * 100), 2) if m.any() else None,
                   "mean_pred_ev_pct": round(float(ev[m].mean() * 100), 4) if m.any() else None,
                   "by_venue": {v: {"trades": int((m & (dt.venue.values == v)).sum()), "avg_net_ret_pct": round(float(ret[m & (dt.venue.values == v)].mean() * 100), 4) if (m & (dt.venue.values == v)).any() else None} for v in ("futures", "spot")}}
    return out

def run(tf):
    t0 = time.time(); d, info = load(tf); bm = BAR_MS[tf]; Hb = F3.H[tf]; X = d[F3.COLS].values; y = d.y.values
    print(tf, json.dumps(info, default=str), f"load {time.time()-t0:.0f}s", flush=True)
    uts = np.sort(d.ts.unique()); e = np.linspace(len(uts) * 0.4, len(uts), FOLDS + 1).astype(int)
    P = np.full((len(d), 3), np.nan); Pab = np.full((len(d), 3), np.nan); fold = np.full(len(d), -1)
    assert "dpo" in F3.COLS, "REQUIRED feature 'dpo' (original centred MT5 dpo) missing from the pipeline"
    LIVE = list(range(len(F3.COLS))); XL = X; NOD = [i for i, c in enumerate(F3.COLS) if c != "dpo"]
    for k in range(FOLDS):
        t_0 = uts[e[k]]; t_1 = uts[min(e[k + 1], len(uts) - 1)]
        tr = (d.ts < t_0 - Hb * bm).values; te = ((d.ts >= t_0) & ((d.ts < t_1) if k < FOLDS - 1 else True)).values
        px = fit_xgb(XL[tr], y[tr]).predict_proba(XL[te]); pm = pred_mlp(fit_mlp(XL[tr], y[tr]), XL[te]); P[te] = (px + pm) / 2; fold[te] = k
        if k == FOLDS - 1: rx = fit_xgb(X[tr][:, NOD], y[tr]); Pab[te] = rx.predict_proba(X[te][:, NOD])   # ablation: WITHOUT dpo
        print(f"  fold {k+1} test_rows {int(te.sum())} {time.time()-t0:.0f}s", flush=True)
    pre = (fold >= 0) & (fold < FOLDS - 1); ho = fold == FOLDS - 1
    # margin + calibrator on pre-holdout OOS only
    best = (-1, 0.0)
    for mg in np.round(np.arange(-0.3, 0.31, 0.02), 2):
        s, _, _ = decide(P[pre], mg); f = f1_score(y[pre], s, average="macro", labels=[-1, 0, 1]); best = max(best, (f, mg))
    margin = float(best[1]); _, dpre, pdpre = decide(P[pre], margin)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pdpre, (y[pre] == dpre).astype(int))
    sig, dh, pdh = decide(P[ho], margin); ph = cal.predict(pdh)
    res = {"tf": tf, "horizon_bars": Hb, "barriers": {"tp_atr": F3.TP_ATR, "sl_atr": F3.SL_ATR}, "data": info, "margin": margin, "margin_prefit_macro_f1": round(float(best[0]), 4),
           "holdout_from": str(pd.to_datetime(d.ts[ho].min(), unit="ms")), "holdout_ensemble": evaluate(y[ho], P[ho], sig, dh, ph, d[ho]),
           "holdout_argmax3_raw": {str(c): int(v) for c, v in zip((-1, 0, 1), np.bincount(P[ho].argmax(1), minlength=3))}}
    s2, d2, p2 = decide(Pab[ho], margin); res["holdout_ABLATION_xgb_without_dpo_liveLike"] = evaluate(y[ho], Pab[ho], s2, d2, p2, d[ho])
    print("  holdout", json.dumps({k: res["holdout_ensemble"][k] for k in ("macro_f1", "auc_ovr_macro", "signal_dist", "gate_p>=0.77_ev>0")}), flush=True)
    print("  ABLATION without dpo", json.dumps({k: res["holdout_ABLATION_xgb_without_dpo_liveLike"][k] for k in ("macro_f1", "auc_ovr_macro", "gate_p>=0.77_ev>0")}), flush=True)
    # final fit on all rows; calibrator refit on ALL purged OOS
    _, dall, pall = decide(P[fold >= 0], margin); calF = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pall, (y[fold >= 0] == dall).astype(int))
    mx = fit_xgb(XL, y); mm = fit_mlp(XL, y); LC = [F3.COLS[i] for i in LIVE]
    imp = dict(zip(LC, mx.feature_importances_.round(5).tolist())); res["live_xgb_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    res["importance_dpo_vs_causal"] = {k: imp[k] for k in ("dpo", "dpo_causal", "dpo_causal_z")}; res["importance_rank"] = {k: sorted(imp, key=lambda q: -imp[q]).index(k) + 1 for k in ("dpo", "dpo_causal", "dpo_causal_z")}
    joblib.dump(mx, A / f"btc3_xgb_{tf}.joblib"); mm.save(A / f"btc3_mlp_{tf}.keras"); joblib.dump({"isotonic": calF, "margin": margin}, A / f"btc3_cal_{tf}.joblib")
    try:
        from onnxmltools import convert_xgboost; from onnxmltools.convert.common.data_types import FloatTensorType
        (A / f"btc3_xgb_{tf}.onnx").write_bytes(convert_xgboost(mx, initial_types=[("input", FloatTensorType([None, len(LC)]))]).SerializeToString()); res["onnx"] = "ok"
    except Exception as ex: res["onnx"] = f"failed: {ex!r}"[:200]
    meta = {"name": f"btc3_{tf}", "timeframe": tf, "classes": [-1, 0, 1], "class_meaning": {"-1": "SELL", "0": "HOLD", "1": "BUY"}, "internal_index": "label + 1",
            "features_in_order": LC, "required_features": ["dpo"], "dpo_lookahead_bars": 11, "feature_source": "mt5_utf8/mt5_signal_script.py ORIGINAL (1:1, strict centred dpo) + dpo_causal/dpo_causal_z + TA-Lib extras",
            "garch_params": next(iter(info.values()))["garch"], "horizon_bars": Hb, "margin": margin, "gate": {"p_hit_min": GATE_P, "ev_after_costs_gt": 0},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "report": f"report3_{tf}.json"}
    (A / f"btc3_{tf}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    o = d[["ts", "sym", "venue", "y"]].copy(); o[["p_sell", "p_hold", "p_buy"]] = P; o["fold"] = fold; o[o.fold >= 0].to_csv(A / f"oos3_{tf}.csv.gz", index=False)
    (A / f"report3_{tf}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8"); print(tf, "done", f"{time.time()-t0:.0f}s", flush=True); return res

if __name__ == "__main__":
    for tf in (sys.argv[1:] or ["4h", "1h", "15m", "5m", "1m"]): run(tf)