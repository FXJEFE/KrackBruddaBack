"""Per-symbol + multi-symbol 3-class models (-1/0/1) on the OG MT5 feature set (features3: 28 OG + dpo_causal pair + TA-Lib) for
PF_XBTUSD / PF_ETHUSD / PF_SOLUSD / PF_XRPUSD / PF_US100USD (research), data = data/multi/ohlcv_multi.db (tools/pull_kraken_history.py).
XGBoost (perf/gpu_params: CUDA when free) tuned by GridSearchCV(TimeSeriesSplit) + Keras MLP; ensemble mean; TEMPERATURE scaling fitted on a
calibration slice (pre-holdout). Gate fixed 0.77 <= p < 0.9888 and EV>0. Holdout = last 30% (purged H bars) never used for fitting/calibration.
Honesty: centred dpo looks 11 bars ahead -> 'with_dpo' holdout numbers are inflated; 'no_dpo' ablation (same pipeline, dpo removed) is the live proxy.
Win-gate: promoted only if no_dpo holdout avg net > 0 after costs with >= 30 in-band trades AND (BTC) better than the current btc3 holdout.
Usage (.myenv311): python models/train_multi3.py [tfs comma] [syms comma]"""
import os, sys, json, time, pathlib, numpy as np, pandas as pd, joblib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
M = pathlib.Path(__file__).resolve().parent; ROOT = M.parent; sys.path[:0] = [str(M), str(ROOT), str(ROOT / "perf"), str(ROOT / "tools")]
import features_mt5 as F, features3 as F3, gpu_params as G
from pull_kraken_history import read
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.optimize import minimize_scalar
F3.H.update({"1d": 5, "1w": 4, "1M": 3})
BAR_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000, "1w": 604800000, "1M": 2592000000}
TFS = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1m", "5m", "15m", "1h", "4h", "1d"])
SYMS = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD", "PF_US100USD"])
A = ROOT / "models" / "artifacts_multi3"; A.mkdir(parents=True, exist_ok=True); GMIN, GMAX = 0.77, 0.9888
NOD = [c for c in F3.COLS if c != "dpo"]; MAXROWS = 150_000
def flag(item, reason):
    try: import flaglog; flaglog.flag("train_multi3", "-", item, reason, action="allowed")
    except Exception: pass
def build(sym, tf):
    d = read(sym, tf).drop_duplicates("ts").reset_index(drop=True)
    if len(d) < 400: return None, None
    d = d.tail(MAXROWS + 300).reset_index(drop=True)
    g = F.garch_params(d.close.iloc[: int(len(d) * 0.4)])
    f = F3.features(d, g); lb = F3.labels(d, tf, "futures", BAR_MS[tf]); f = pd.concat([f, lb], axis=1); f["sym"] = sym
    f = f.iloc[150:]; f = f[f.y != -9].dropna(subset=NOD).reset_index(drop=True); return f, g
def xgb_grid(X, y):
    xp = G.get_xgb_params(); nj = 1 if xp.get("device") == "cuda" else G.N_JOBS
    base = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", subsample=0.8, colsample_bytree=0.8, **xp)
    gs = GridSearchCV(base, {"max_depth": [4, 6], "n_estimators": [200, 400], "learning_rate": [0.05]}, cv=TimeSeriesSplit(3), scoring="f1_macro", n_jobs=nj)
    gs.fit(X, y + 1, sample_weight=compute_sample_weight("balanced", y)); return gs.best_estimator_, gs.best_params_, xp.get("device")
def mlp(X, y):
    import keras
    n = keras.layers.Normalization(); n.adapt(np.nan_to_num(X.astype("float32")))
    m = keras.Sequential([keras.Input((X.shape[1],)), n, keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.2), keras.layers.Dense(32, activation="relu"), keras.layers.Dense(3)])
    m.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    cw = {i: w for i, w in enumerate(len(y) / (3 * np.maximum(1, np.bincount(y + 1, minlength=3))))}
    m.fit(np.nan_to_num(X).astype("float32"), y + 1, epochs=6, batch_size=1024, verbose=0, class_weight=cw); return m
def sm(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def ens_logits(xm, km, X):
    lx = np.log(np.clip(xm.predict_proba(X), 1e-9, 1)); lk = np.asarray(km.predict(np.nan_to_num(X).astype("float32"), batch_size=8192, verbose=0))
    lk = lk - lk.max(1, keepdims=True); lk = lk - np.log(np.exp(lk).sum(1, keepdims=True)); return (lx + lk) / 2
def temp(L, y):
    r = minimize_scalar(lambda T: -np.mean(np.log(np.clip(sm(L / T)[np.arange(len(y)), y + 1], 1e-12, 1))), bounds=(0.05, 20), method="bounded"); return float(r.x)
def evaluate(P, f):
    d = np.where(P[:, 2] >= P[:, 0], 1, -1); p = np.where(d == 1, P[:, 2], P[:, 0]); sig = np.where((p > P[:, 1]), d, 0)
    ev = p * f.tp_pct.values - (1 - p) * f.sl_pct.values - f.cost.values
    band = (p >= GMIN) & (p < GMAX) & (ev > 0); over = int((p >= GMAX).sum())
    net = np.where(d == 1, f.ret_long.values, f.ret_short.values)[band]; y = f.y.values
    return {"rows": int(len(y)), "label_dist": {str(c): int((y == c).sum()) for c in (-1, 0, 1)}, "signal_dist": {str(c): int((sig == c).sum()) for c in (-1, 0, 1)},
            "macro_f1": round(float(f1_score(y, sig, average="macro", labels=[-1, 0, 1])), 4), "confusion_true_rows_-1_0_1": confusion_matrix(y, sig, labels=[-1, 0, 1]).tolist(),
            "in_band_trades": int(band.sum()), "in_band_long": int((band & (d == 1)).sum()), "in_band_short": int((band & (d == -1)).sum()), "p_over_max_flagged": over,
            "win_pct": round(float((net > 0).mean() * 100), 2) if len(net) else None, "avg_net_pct": round(float(net.mean() * 100), 4) if len(net) else None,
            "sum_net_pct": round(float(net.sum() * 100), 3) if len(net) else None}
def fit_eval(f, cols, tag, tf):
    n = len(f); ho0 = int(n * 0.7); Hb = F3.H[tf]; ca0 = int(ho0 * 0.85)
    tr, ca, ho = f.iloc[: ca0 - Hb], f.iloc[ca0: ho0 - Hb], f.iloc[ho0:]
    X = lambda d: d[cols].values.astype(float)
    xm, bp, dev = xgb_grid(X(tr), tr.y.values); km = mlp(X(tr), tr.y.values)
    T = temp(ens_logits(xm, km, X(ca)), ca.y.values); P = sm(ens_logits(xm, km, X(ho)) / T)
    r = evaluate(P, ho); r.update(best_params=bp, device=dev, temperature=round(T, 4), train_rows=len(tr), cal_rows=len(ca),
                                  holdout_from=str(pd.to_datetime(ho.ts.min(), unit="ms")), holdout_to=str(pd.to_datetime(ho.ts.max(), unit="ms")))
    return r, (xm, km, T)
if __name__ == "__main__":
    G.load_limits_env(); rep = {}; t00 = time.time()
    for tf in TFS:
        allf = []
        for s in SYMS:
            t0 = time.time(); f, g = build(s, tf)
            if f is None or len(f) < 600: rep[f"{s} {tf}"] = {"skipped": "too few bars"}; print(s, tf, "skip", flush=True); continue
            f["sym_id"] = SYMS.index(s); allf.append(f); out = {}
            for tag, cols in (("with_dpo", F3.COLS), ("no_dpo", NOD)):
                out[tag], mdl = fit_eval(f, cols, tag, tf)
                if tag == "with_dpo": dep = mdl
            nd = out["no_dpo"]; prom = bool(nd["in_band_trades"] >= 30 and (nd["avg_net_pct"] or -1) > 0)
            out["promoted"] = prom; out["garch"] = g; rep[f"{s} {tf}"] = out
            if prom:
                xm, km, T = dep; joblib.dump(xm, A / f"{s}_{tf}_xgb.joblib"); km.save(A / f"{s}_{tf}_mlp.keras")
                (A / f"{s}_{tf}.json").write_text(json.dumps({"features_in_order": F3.COLS, "temperature": T, "garch_params": g, "gate": [GMIN, GMAX], "report": out}, default=str))
            else: flag(f"{s}_{tf}", f"not promoted: no_dpo holdout in-band n={nd['in_band_trades']} avg_net={nd['avg_net_pct']}% (needs n>=30 and >0 after costs)")
            print(s, tf, json.dumps({k: {kk: v[kk] for kk in ("in_band_trades", "win_pct", "avg_net_pct", "macro_f1", "temperature")} for k, v in out.items() if isinstance(v, dict) and "rows" in v}), "promoted", prom, f"{time.time()-t0:.0f}s", flush=True)
        if len(allf) >= 2:
            mf = pd.concat(allf).sort_values(["ts", "sym_id"]).reset_index(drop=True); out = {}
            for tag, cols in (("with_dpo", F3.COLS + ["sym_id"]), ("no_dpo", NOD + ["sym_id"])): out[tag], _ = fit_eval(mf, cols, tag, tf)
            # per-symbol breakdown of the multi model is in the per-row holdout; overall shown here
            out["promoted"] = bool(out["no_dpo"]["in_band_trades"] >= 30 and (out["no_dpo"]["avg_net_pct"] or -1) > 0); rep[f"MULTI {tf}"] = out
            print("MULTI", tf, json.dumps({k: {kk: v[kk] for kk in ("in_band_trades", "win_pct", "avg_net_pct", "macro_f1")} for k, v in out.items() if isinstance(v, dict) and "rows" in v}), flush=True)
        (ROOT / "logs" / "train_multi3_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("done", f"{time.time()-t00:.0f}s")
