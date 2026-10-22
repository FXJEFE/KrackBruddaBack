"""BTC-only walk-forward training on data/btc_market.db with the exact MT5 29-feature pipeline.
Groups: xgb6 (XGBoost), avg9 (soft-vote LR+GNB+GradientBoosting; HistGB replaced because skl2onnx 1.20/numpy 2.4 cannot convert it), rf28 (RandomForest), + tf_mlp (Keras, 4-bar window of the 28 feats).
Exports: joblib, .onnx (all), xgboost .json, .keras + .h5, per-model metadata .json, HF folders.
Usage: python models/train_btc.py [15m 1h]"""
import os, sys, json, time, sqlite3, pathlib, joblib, numpy as np, pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
M = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(M)); sys.path.insert(0, str(M.parent)); from lkpaths import HOME as ROOT
import features_mt5 as F
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
A = pathlib.Path(os.environ.get("ARTIFACTS_DIR") or (ROOT / "models" / "artifacts_btc")); A.mkdir(exist_ok=True, parents=True)
H = 4; FOLDS = 5; CONS = 0.77; WIN = 4
FEES = {"spot_taker_0.40%": 0.008, "futures_taker_0.05%": 0.001}           # round trip
WEIGHTS = {"xgb6": 0.35, "avg9": 0.40, "rf28": 0.25, "tf_mlp": 0.25}       # golden_weights from fxjefe config.json (+tf)
BAR_MS = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
MAX_DAYS = {"1m": 540, "5m": 1460}  # entry TFs: recent window keeps CPU time sane; HTFs use full history since 2013
ROLE = {"1m": "entry", "5m": "entry", "15m": "bias", "1h": "bias", "4h": "bias"}
SERIES = [("XBTUSD", "spot"), ("PI_XBTUSD", "futures")]

def mk(g):
    if g == "xgb6": return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=5, eval_metric="logloss")
    if g == "avg9": return VotingClassifier([("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))), ("nb", make_pipeline(StandardScaler(), GaussianNB())),
                                             ("gb", GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.5, random_state=0))], voting="soft", flatten_transform=False)
    if g == "rf28": return RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=100, max_features="sqrt", n_jobs=5, random_state=0)

def load(tf, garch_prm=None):
    c = sqlite3.connect(ROOT / "data" / "btc_market.db"); fr = []; info = {}
    for sym, venue in SERIES:
        d = pd.read_sql("SELECT ts, o open, h high, l low, c close, v volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts", c, params=(sym, tf))
        if tf in MAX_DAYS: d = d[d.ts >= d.ts.max() - MAX_DAYS[tf] * 86400000].reset_index(drop=True)
        if len(d) < 500: continue
        if garch_prm is None: garch_prm = F.garch_params(d.close.iloc[: int(len(d) * 0.4)])  # fitted on min-train window only
        f = F.compute(d, garch_prm); f["ts"] = d.ts.values; f["sym"] = sym
        f["fwd"] = d.close.shift(-H).values / d.close.values - 1
        # forward return must span exactly H bars of wall-clock time (no silent gaps)
        f.loc[(d.ts.shift(-H) - d.ts).values != H * BAR_MS[tf], "fwd"] = np.nan
        # LABEL FIX: flat bars (fwd == 0) were labelled 0 = "down", which skewed P(up) low -> structural SELL bias (1m: 14.7% SELL vs 0.9% BUY
        # at conf>=0.55, all 100 signals >=0.77 were SELL). Ties are now excluded: y = 1 up / 0 down, conditioned on the price moving.
        # (rows kept so lag windows stay time-aligned; ties are masked out of training + metrics in run())
        f["y"] = (f.fwd > 0).astype(int)
        f = f.iloc[60:].dropna(subset=F.FEATURES_29[:28] + F.CAUSAL + ["fwd"]); fr.append(f)
        info[sym] = {"rows": len(f), "from": str(pd.to_datetime(f.ts.min(), unit="ms")), "to": str(pd.to_datetime(f.ts.max(), unit="ms"))}
    return pd.concat(fr).sort_values(["ts", "sym"]).reset_index(drop=True), garch_prm, info

def windows(X, sym):
    """Stack last WIN bars per row (same symbol); rows lacking history get NaN."""
    out = [X]
    for k in range(1, WIN):
        s = pd.DataFrame(X).groupby(sym.values).shift(k).values; out.append(s)
    return np.hstack(out)

def tf_model(n_in, Xtr):
    import keras
    norm = keras.layers.Normalization(); norm.adapt(Xtr)
    m = keras.Sequential([keras.Input((n_in,)), norm, keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.2),
                          keras.layers.Dense(32, activation="relu"), keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy"); return m

def fit_tf(Xtr, ytr):
    ok = ~np.isnan(Xtr).any(1); Xtr, ytr = Xtr[ok].astype("float32"), ytr[ok]
    m = tf_model(Xtr.shape[1], Xtr); n = int(len(Xtr) * 0.9)
    import keras
    m.fit(Xtr[:n], ytr[:n], validation_data=(Xtr[n:], ytr[n:]), epochs=8, batch_size=2048, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]); return m

def pred_tf(m, X):
    p = np.full(len(X), np.nan); ok = ~np.isnan(X).any(1)
    if ok.any(): p[ok] = m.predict(X[ok].astype("float32"), batch_size=8192, verbose=0).ravel()
    return p

def backtest(p, d, thr, fee, bar_ms):
    out = []; nxt = {}
    for pi, ts, sym, fwd in zip(p, d.ts.values, d.sym.values, d.fwd.values):
        if np.isnan(pi) or ts < nxt.get(sym, -1): continue
        if pi >= thr: r = fwd
        elif pi <= 1 - thr: r = -fwd
        else: continue
        out.append(r - fee); nxt[sym] = ts + H * bar_ms
    t = np.array(out)
    if not len(t): return {"trades": 0}
    return {"trades": int(len(t)), "win_rate": round(float((t > 0).mean()), 4), "avg_net_pct": round(float(t.mean() * 100), 4), "sum_net_pct": round(float(t.sum() * 100), 1)}

def consensus(P, names):
    w = np.array([WEIGHTS[n] for n in names]); w = w / w.sum(); up = (P > .5).all(1); dn = (P < .5).all(1)
    conf = (np.where(P > .5, P, 1 - P) * w).sum(1); return np.where(up, conf, np.where(dn, 1 - conf, 0.5))

def metrics(y, p, dt, bar_ms):
    ok = ~np.isnan(p); y, p, dd = y[ok], p[ok], dt[ok]; pr = (p >= .5).astype(int)
    r = {"oos_rows": int(ok.sum()), "accuracy": round(accuracy_score(y, pr), 4), "majority_baseline": round(max(y.mean(), 1 - y.mean()), 4),
         "precision_up": round(precision_score(y, pr, zero_division=0), 4), "precision_down": round(precision_score(1 - y, 1 - pr, zero_division=0), 4),
         "auc": round(roc_auc_score(y, p), 4)}
    for fn, fee in FEES.items():
        r[f"bt_{fn}"] = {f"thr_{t}": backtest(p, dd, t, fee, bar_ms) for t in (0.55, 0.60)}
    return r

def run(tf):
    t0 = time.time(); d, gp, info = load(tf); bm = BAR_MS[tf]; G = F.GROUPS
    print(tf, info, f"load {time.time()-t0:.0f}s", flush=True)
    Xw = windows(d[G["rf28"]].values, d.sym)
    uts = np.sort(d.ts.unique()); e = np.linspace(len(uts) * 0.4, len(uts), FOLDS + 1).astype(int)
    names = list(G) + ["tf_mlp"]; oos = {g: np.full(len(d), np.nan) for g in names}; folds = []
    for k in range(FOLDS):
        t_0 = uts[e[k]]; t_1 = uts[min(e[k + 1], len(uts) - 1)]
        tr = ((d.ts < t_0 - H * bm) & (d.fwd != 0)).values; te = ((d.ts >= t_0) & ((d.ts < t_1) if k < FOLDS - 1 else True)).values
        for g, cols in G.items(): oos[g][te] = mk(g).fit(d.loc[tr, cols].values, d.y.values[tr]).predict_proba(d.loc[te, cols].values)[:, 1]
        oos["tf_mlp"][te] = pred_tf(fit_tf(Xw[tr], d.y.values[tr]), Xw[te])
        folds.append({"fold": k + 1, "train_until": str(pd.to_datetime(t_0, unit="ms")), "test_rows": int(te.sum())}); print(" ", folds[-1], f"{time.time()-t0:.0f}s", flush=True)
    o = d[["ts", "sym", "fwd", "y"]].copy()
    for g in names: o[g] = oos[g]
    o[~np.isnan(oos["xgb6"])].to_csv(A / f"oos_{tf}.csv.gz", index=False)
    m = ~np.isnan(oos["xgb6"]) & (d.fwd != 0).values; dt = d[m]; y = dt.y.values; res = {"data": info, "garch_params": gp, "folds": folds}
    for g in names: res[g] = metrics(y, oos[g][m], dt, bm); print(" ", g, json.dumps(res[g]), flush=True)
    tfm = res["tf_mlp"]; tf_ok = tfm["auc"] > 0.51 and tfm["accuracy"] > tfm["majority_baseline"]
    res["tf_mlp_in_consensus"] = bool(tf_ok)
    for label, nm in (("consensus_3", list(G)), ("consensus_4", names)):
        P = np.column_stack([oos[g][m] for g in nm]); okr = ~np.isnan(P).any(1); cp = np.where(okr, consensus(np.nan_to_num(P, nan=.5), nm), np.nan)
        res[label] = {fn: {f"thr_{t}": backtest(cp, dt, t, fee, bm) for t in (0.55, 0.60, 0.65, CONS)} for fn, fee in FEES.items()}
        res[label]["bars_conf>=0.77"] = int(np.nansum((cp >= CONS) | (cp <= 1 - CONS)))
    print("  consensus", json.dumps({k: res[k] for k in ("consensus_3", "consensus_4", "tf_mlp_in_consensus")}), flush=True)
    # ---------- final fit on all rows + exports ----------
    import export_models as X
    sample = {}
    for g, cols in G.items():
        nz = (d.fwd != 0).values; mdl = mk(g).fit(d.loc[nz, cols].values, d.y.values[nz]); name = f"btc_{g}_{tf}"; joblib.dump(mdl, A / f"{name}.joblib")
        sample[g] = d[cols].values[-2000:]; X.export_sklearn(mdl, g, name, len(cols), sample[g], A)
        write_meta(name, g, tf, cols, res, gp)
    nz = (d.fwd != 0).values; km = fit_tf(Xw[nz], d.y.values[nz]); name = f"btc_tf_mlp_{tf}"; km.save(A / f"{name}.keras"); km.save(A / f"{name}.h5")
    X.export_keras(km, name, Xw[-2000:][~np.isnan(Xw[-2000:]).any(1)], A)
    write_meta(name, "tf_mlp", tf, [f"{c}_lag{k}" for k in range(WIN) for c in G["rf28"]], res, gp)
    print(tf, "done", f"{time.time()-t0:.0f}s", flush=True); return res

def write_meta(name, g, tf, cols, res, gp):
    p = A / f"{name}.json"; exp = json.loads(p.read_text()).get("export", {}) if p.exists() else {}
    meta = {"name": name, "group": g, "timeframe": tf, "role": ROLE[tf], "features_in_order": cols, "n_features": len(cols), "horizon_bars": H,
            "target": f"1 if close[t+{H}] > close[t], 0 if lower (flat bars excluded from training)", "output": "P(up) = probability of class 1",
            "data": res["data"], "source_db": "data/btc_market.db (Kraken OHLCVT zips + Trades API, PI_XBTUSD futures charts)",
            "feature_pipeline": "features_mt5.py -> mt5_utf8/mt5_signal_script.py ORIGINAL compute_technical_indicators via mt5_original_adapter (ALIGNMENT_V2 29-feature order)",
            "garch_params": gp, "consensus": {"min_confidence": CONS, "weights": WEIGHTS, "tf_mlp_included": res["tf_mlp_in_consensus"]},
            "thresholds": {"consensus_min_conf": CONS, "single_model_examples": [0.55, 0.60]},
            "walk_forward": {"folds": FOLDS, "embargo_bars": H, "min_train_frac": 0.4, "fold_detail": res["folds"]},
            "oos_metrics": res[g], "consensus_oos": {"3_groups": res["consensus_3"], "4_with_tf": res["consensus_4"]},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "export": exp}
    if g == "tf_mlp": meta["scaler"] = "keras Normalization layer inside the model (mean/var adapted on training rows)"; meta["window"] = WIN
    elif g in ("avg9",): meta["scaler"] = "StandardScaler inside LR and GNB pipelines (params in joblib/ONNX)"
    else: meta["scaler"] = None
    p.write_text(json.dumps(meta, indent=2, default=str))

if __name__ == "__main__":
    allr = {}
    for tf in (sys.argv[1:] or ["4h", "1h", "15m", "5m", "1m"]):
        allr[tf] = run(tf)
        (A / f"report_{tf}.json").write_text(json.dumps(allr[tf], indent=2, default=str))
    # ---- v3 hook (val3 + engine artifacts + Obsidian vault), see INTERFACE_features.md. Detached, never blocks or fails training.
    try:
        import subprocess
        _py = pathlib.Path(os.environ.get("V3_PYTHON", r"C:\Users\localonlylarp\.myenv311\Scripts\python.exe"))
        if os.environ.get("V3_STAGE", "1") != "0" and _py.exists():
            subprocess.Popen([str(_py), str(M / "val3_stage.py")], cwd=str(ROOT), creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as _e: print("v3 hook skipped:", _e)
