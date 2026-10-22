"""TP-readiness exit model (user 10:44: no preset TP; the model decides when to take profit).
Per symbol x TF: entries every STEP bars, side = sign(TA-Lib LINEARREG_SLOPE14) (a neutral proxy, identical for both exit policies).
State rows (only bars where the open trade is in profit, r>0): OG MT5 features (CAUSAL only - centred dpo excluded, it looks ahead) + TA-Lib +
position state: unrealised R, bars held, MFE R, giveback from MFE, rsi change 3, slope*side (HTF/momentum fade proxy), dpo_causal*side.
Label ready=1 if, within the next H bars, price does NOT extend +0.5 ATR further before giving back 1 ATR (taking profit now was right).
XGBoost (gpu_params) GridSearchCV(TimeSeriesSplit) + temperature scaling on a calibration slice; holdout = last 30% by time.
OOS comparison on the SAME holdout entries: FIXED exits (TP1 1.5ATR 30% / TP2 3ATR 30% / swing 4.5ATR 40%, BE after TP1, lock TP1 after TP2)
vs MODEL exits (p_take_profit >= TP_READY takes 30/30/40 legs; BE at +1R; same SL 1.5ATR; same time stop). Costs: maker 0.02% TP/entry, taker 0.05% SL/time.
Usage (.myenv311): python models/train_tp_exit.py [tfs] [syms]"""
import os, sys, json, time, pathlib, numpy as np, pandas as pd, joblib
M = pathlib.Path(__file__).resolve().parent; ROOT = M.parent; sys.path[:0] = [str(M), str(ROOT), str(ROOT / "perf"), str(ROOT / "tools")]
import features_mt5 as F, features3 as F3, gpu_params as G
from pull_kraken_history import read
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.optimize import minimize_scalar
TFS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["5m", "15m", "1h"]
SYMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD", "PF_XRPUSD"]
H = {"1m": 30, "5m": 24, "15m": 16, "1h": 12, "4h": 8}; STEP = {"1m": 15, "5m": 6, "15m": 4, "1h": 2, "4h": 1}
SL, TP1, TP2, SWG, BER, READY = 1.5, 1.5, 3.0, 4.5, 1.0, float(os.environ.get("TP_READY", 0.60))
FR = (0.30, 0.30, 0.40); MAKER, TAKER = 0.0002, 0.0005
A = ROOT / "models" / "artifacts_tp_exit"; A.mkdir(parents=True, exist_ok=True)
BASE = [c for c in F3.COLS if c != "dpo"]
STATE = ["r", "held", "mfe_r", "giveback_r", "rsi_d3", "slope_side", "dpoc_side"]
def ready_labels(h, l, c, atr, Hb):
    n = len(c); out = {}
    for side in (1, -1):
        t_ext = np.full(n, np.inf); t_gb = np.full(n, np.inf)
        for k in range(Hb, 0, -1):
            hk = np.r_[h[k:], np.full(k, np.nan)]; lk = np.r_[l[k:], np.full(k, np.nan)]
            if side == 1: t_ext[hk - c >= 0.5 * atr] = k; t_gb[c - lk >= 1.0 * atr] = k
            else: t_ext[c - lk >= 0.5 * atr] = k; t_gb[hk - c >= 1.0 * atr] = k
        out[side] = (~(t_ext < t_gb)).astype(int)
    return out
def trades(f, lo, hi, tf):
    """Simulate entries in [lo,hi); returns state rows (for training) and per-entry path info."""
    c, h, l, atr = f.close.values, f.high.values, f.low.values, f.tl_atr10.values; Hm = 3 * H[tf]; rows = []; ents = []
    for t in range(lo, min(hi, len(c) - Hm - 1), STEP[tf]):
        if not np.isfinite(atr[t]) or atr[t] <= 0: continue
        s = 1 if f.tl_linreg_slope14.values[t] >= 0 else -1; e = c[t]; R = SL * atr[t]; mfe = 0.0
        for j in range(1, Hm + 1):
            k = t + j; adv = (l[k] - e) * s if s == 1 else (e - h[k])
            if adv <= -R: break
            r = (c[k] - e) * s / R; mfe = max(mfe, ((h[k] if s == 1 else l[k]) - e) * s / R)
            if r > 0: rows.append((k, t, s, r, j, mfe, mfe - r))
        ents.append((t, s))
    return rows, ents
def state_frame(f, rows):
    k = np.array([x[0] for x in rows]); s = np.array([x[2] for x in rows]); X = f[BASE].values[k]
    rsi = f.tl_rsi14.values; rsid = (rsi[k] - rsi[np.maximum(k - 3, 0)]) * s
    st = np.c_[[x[3] for x in rows], [x[4] for x in rows], [x[5] for x in rows], [x[6] for x in rows], rsid, f.tl_linreg_slope14.values[k] * s, f.dpo_causal.values[k] * s]
    return np.c_[X, st], k, s
LEVEL = ["price", "vwap", "supertrend", "hma", "ichimoku_tenkan", "sar"]; DOLLAR = ["atr", "ema_diff", "macd_diff", "price_vwap_diff", "momentum", "chaikin_vol", "volume_delta", "dpo_causal", "spread"]
def stationary(f):
    """exit model only: level/$ features -> ATR units (non-stationary price levels made the first fit useless OOS, AUC 0.50)"""
    a = f.tl_atr10.replace(0, np.nan)
    for c in LEVEL: f[c] = (f[c] - f.close) / a
    for c in DOLLAR: f[c] = f[c] / a
    for c in ("obv", "ad_line"): f[c] = f[c].diff(10) / (f[c].abs().rolling(100).mean() + 1e-9)
    return f
def sim(f, ents, tf, policy, model=None, T=1.0, P=None):
    c, h, l, atr = f.close.values, f.high.values, f.low.values, f.tl_atr10.values; Hm = 3 * H[tf]; res = []
    rsi = f.tl_rsi14.values
    for t, s in ents:
        e = c[t]; R = SL * atr[t]; sl = e - s * R; left = 1.0; pnl = -MAKER; legs = 0; mfe = 0.0; be = False
        tps = [e + s * TP1 * atr[t], e + s * TP2 * atr[t], e + s * SWG * atr[t]]
        for j in range(1, Hm + 1):
            k = t + j
            if (l[k] <= sl if s == 1 else h[k] >= sl): pnl += left * ((sl - e) * s / e - TAKER); left = 0; break
            if policy == "fixed":
                while legs < 3 and (h[k] >= tps[legs] if s == 1 else l[k] <= tps[legs]):
                    q = FR[legs] if legs < 2 else left; pnl += q * ((tps[legs] - e) * s / e - MAKER); left -= q; legs += 1
                    sl = e if legs == 1 else (tps[0] if legs == 2 else sl)
                if left <= 1e-9: break
            else:
                r = (c[k] - e) * s / R; mfe = max(mfe, ((h[k] if s == 1 else l[k]) - e) * s / R)
                if not be and r >= BER: sl = e; be = True
                if r > 0:
                    p = P.get((t, k), 0.0)
                    if p >= READY:
                        q = FR[legs] if legs < 2 else left; pnl += q * ((c[k] - e) * s / e - MAKER); left -= q; legs += 1
                        if legs == 1: sl = e; be = True
                        if left <= 1e-9: break
        if left > 1e-9: pnl += left * ((c[min(t + Hm, len(c) - 1)] - e) * s / e - TAKER)
        res.append(pnl)
    r = np.array(res); return {"trades": len(r), "win_pct": round(float((r > 0).mean() * 100), 2), "avg_net_pct": round(float(r.mean() * 100), 4), "sum_net_pct": round(float(r.sum() * 100), 2)}
if __name__ == "__main__":
    G.load_limits_env(); rep = {}
    for tf in TFS:
        for sym in SYMS:
            t0 = time.time(); d = read(sym, tf).drop_duplicates("ts").reset_index(drop=True).tail(60000).reset_index(drop=True)
            if len(d) < 2000: rep[f"{sym} {tf}"] = {"skipped": "too few bars"}; continue
            g = F.garch_params(d.close.iloc[: int(len(d) * 0.4)]); f = F3.features(d, g); f["high"] = d.high.values; f["low"] = d.low.values
            f = stationary(f.iloc[200:].reset_index(drop=True)); f[BASE] = f[BASE].astype(float)
            lab = ready_labels(f.high.values, f.low.values, f.close.values, f.tl_atr10.values, H[tf])
            n = len(f); ho0 = int(n * 0.7); ca0 = int(ho0 * 0.85)
            rtr, _ = trades(f, 0, ca0 - H[tf] * 3, tf); rca, _ = trades(f, ca0, ho0 - H[tf] * 3, tf); _, eho = trades(f, ho0, n, tf)
            Xtr, ktr, s_tr = state_frame(f, rtr); ytr = np.where(s_tr == 1, lab[1][ktr], lab[-1][ktr])
            Xca, kca, s_ca = state_frame(f, rca); yca = np.where(s_ca == 1, lab[1][kca], lab[-1][kca])
            xp = G.get_xgb_params(); nj = 1 if xp.get("device") == "cuda" else G.N_JOBS
            gs = GridSearchCV(XGBClassifier(objective="binary:logistic", eval_metric="logloss", subsample=0.8, colsample_bytree=0.8, **xp),
                              {"max_depth": [3, 5], "n_estimators": [200, 400], "learning_rate": [0.05]}, cv=TimeSeriesSplit(3), scoring="roc_auc", n_jobs=nj)
            gs.fit(Xtr, ytr); m = gs.best_estimator_; m.set_params(device="cpu")
            pc = np.clip(m.predict_proba(Xca)[:, 1], 1e-9, 1 - 1e-9); lg = np.log(pc / (1 - pc))
            T = float(minimize_scalar(lambda T: -np.mean(yca * np.log(1 / (1 + np.exp(-lg / T))) + (1 - yca) * np.log(1 - 1 / (1 + np.exp(-lg / T)) + 1e-12)), bounds=(0.05, 20), method="bounded").x)
            rho, _ = trades(f, ho0, n, tf); Xho, kho, s_ho = state_frame(f, rho); yho = np.where(s_ho == 1, lab[1][kho], lab[-1][kho])
            pho = 1 / (1 + np.exp(-np.log(np.clip(m.predict_proba(Xho)[:, 1], 1e-9, 1 - 1e-9) / np.clip(1 - m.predict_proba(Xho)[:, 1], 1e-9, 1)) / T))
            fx = sim(f, eho, tf, "fixed"); md = sim(f, eho, tf, "model", P={(x[1], x[0]): float(v) for x, v in zip(rho, pho)})
            out = {"state_rows_train": len(ytr), "ready_rate_train": round(float(ytr.mean()), 3), "holdout_auc": round(float(roc_auc_score(yho, pho)), 4),
                   "holdout_brier": round(float(brier_score_loss(yho, pho)), 4), "temperature": round(T, 3), "best_params": gs.best_params_, "device": xp.get("device"),
                   "holdout_from": str(pd.to_datetime(f.ts.iloc[ho0], unit="ms")), "fixed_tp": fx, "model_tp": md,
                   "model_better": md["avg_net_pct"] > fx["avg_net_pct"]}
            rep[f"{sym} {tf}"] = out; print(sym, tf, json.dumps(out, default=str), f"{time.time()-t0:.0f}s", flush=True)
            joblib.dump(m, A / f"{sym}_{tf}_tp.joblib"); (A / f"{sym}_{tf}_tp.json").write_text(json.dumps({"features": BASE + STATE, "temperature": T, "ready": READY, "garch_params": g, "report": out}, default=str))
            (ROOT / "logs" / "train_tp_exit_report.json").write_text(json.dumps(rep, indent=1, default=str))
