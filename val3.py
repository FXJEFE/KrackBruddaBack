"""val3 - third, stricter validation stage (after walk-forward OOS + calibration in train_btc/select_consensus):
 1. Engine features (feature_engine, the SAME code the live feature_service uses) on unified-DB bars.
 2. Regime HMM + forecasters fitted on the first FIT_FRAC of bars only; everything after is out-of-sample.
 3. Purged + embargoed K-fold CV (de Prado) on the OOS segment, H-bar purge, EMBARGO_FRAC embargo.
 4. Rolling 'live' holdout = last LIVE_DAYS days + all bars appended after the last training run (artifacts_btc meta 'to').
 5. GridSearchCV(cv=PurgedKFold) over XGB hyperparams on hist + live rows (OOS segment excl. holdout).
 6. Classifier A = MT5 28 features (current production inputs), B = A + engine (TA-Lib, GARCH, regime, forecasts),
    B+filter = B with the regime entry filter. Metrics are cost-adjusted (fees+spread+slippage+rollover/funding, per venue).
 7. Promotion: write artifacts_v3/promotion.json {promote: bool}. promote only if B (or B+filter) beats A on the live
    holdout avg net return per trade with >= 20 trades AND purged-CV AUC is not worse. Production artifacts are untouched.
Also: GARCH/GJR/EGARCH/GARCH-X out-of-sample QLIKE, forecast MAE/dir-acc vs naive. Run with .myenv311:
  python models/val3.py [--tfs 1h,4h] [--years 3]"""
import os, sys, json, time, pathlib, argparse, joblib, numpy as np, pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ.setdefault("OMP_NUM_THREADS", "2")   # box shares 6 cores with the learn loop
M = pathlib.Path(__file__).resolve().parent; ROOT = M.parent; sys.path[:0] = [str(M), str(ROOT)]
import features_mt5 as F, features_mt5_ext as E, feature_engine as FE, garch_suite as G, regime as R, forecast as FC
from datahub import unified as U
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
OUT = M / "artifacts_v3"; OUT.mkdir(exist_ok=True)
BAR_MS = {"1h": 3_600_000, "4h": 14_400_000, "15m": 900_000}
SERIES = [("BTCUSD_PERP", "PI_XBTUSD", "futures"), ("BTCUSD", "XBTUSD", "spot")]
H = 4; FIT_FRAC = 0.4; K = 5; EMBARGO_FRAC = 0.01; LIVE_DAYS = 45; THR = 0.60
GRID = {"max_depth": [3, 5], "learning_rate": [0.03, 0.08], "min_child_weight": [1, 20]}

class PurgedKFold:
    """Contiguous K folds; training rows within `purge` bars before a test fold (label overlap) and `embargo` rows after it are dropped."""
    def __init__(self, n_splits=K, purge=H, embargo_frac=EMBARGO_FRAC): self.k, self.purge, self.ef = n_splits, purge, embargo_frac
    def get_n_splits(self, *a, **k): return self.k
    def split(self, X, y=None, groups=None):
        n = len(X); emb = int(n * self.ef); edges = np.linspace(0, n, self.k + 1).astype(int)
        for i in range(self.k):
            a, b = edges[i], edges[i + 1]; tr = np.r_[np.arange(0, max(0, a - self.purge)), np.arange(min(n, b + emb), n)]
            yield tr, np.arange(a, b)

def xgb(**p): return XGBClassifier(**{**dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, eval_metric="logloss", random_state=0), **p})

def bt(p, d, allowed=None):
    """non-overlapping trades, cost-adjusted net return per trade (fraction), plus leverage-10x liquidation count."""
    out, liq, nxt = [], 0, -1
    for i, (pi, ts) in enumerate(zip(p, d.ts.values)):
        if not np.isfinite(pi) or ts < nxt: continue
        lg = pi >= THR; sh = pi <= 1 - THR
        if not (lg or sh): continue
        if allowed is not None and not allowed[i, 0 if lg else 1]: continue
        r = d.net_long.values[i] if lg else d.net_short.values[i]; adv = -d.mae_long.values[i] if lg else d.mae_short.values[i]
        liq += int(adv >= 0.1 - E.MAINT_MARGIN[d.venue.values[i]]); out.append(r); nxt = ts + H * d.bar_ms.values[i]
    t = np.array(out)
    return {"trades": int(len(t)), "win_rate": round(float((t > 0).mean()), 4) if len(t) else None, "avg_net_bp": round(float(t.mean() * 1e4), 2) if len(t) else None,
            "sum_net_pct": round(float(t.sum() * 100), 2) if len(t) else 0.0, "liq_10x": liq}

def score(y, p, d, allowed=None):
    ok = np.isfinite(p) & d.trade.values
    return {"auc": round(float(roc_auc_score(y[ok], p[ok])), 4) if len(set(y[ok])) > 1 else None, "acc": round(float(accuracy_score(y[ok], p[ok] >= .5)), 4), **bt(p, d, allowed)}

def build(canon, raw, venue, tf, years):
    bm = BAR_MS[tf]; b = U.read_bars("kraken", canon, tf); b = b[b.ts >= b.ts.max() - years * 365 * 86_400_000].reset_index(drop=True)
    fund = U.read_funding("kraken", canon) if venue == "futures" else None
    fr = E.funding_block(b.ts.values, None if fund is None or not len(fund) else fund.assign(rate=fund.relative_rate.fillna(0))[["ts", "rate"]]).funding_rate.values
    ems = (float(np.nanmean(fr)), float(np.nanstd(fr) + 1e-12)) if venue == "futures" else None
    b["exog"] = (fr - ems[0]) / ems[1] if ems else 0.0   # GARCH-X exog: funding z (text-sentiment slot, see garch_suite)
    t0 = time.time(); S = FE.compute_stateless(b, bm); print(f"  {raw} {tf}: {len(b)} bars, stateless features {time.time()-t0:.0f}s", flush=True)
    nfit = int(len(b) * FIT_FRAC); art = {"regime": R.Regime().fit(b.close.values[:nfit]), "exog_ms": ems}
    rg = art["regime"].features(b.close.values); SF = pd.concat([S, rg], axis=1)
    art["forecaster"] = FC.Forecaster().fit(SF.iloc[:nfit - 10], b.close.values[:nfit - 10], FE.STATELESS + R.COLS)  # -10: targets of fit rows never reach OOS
    fc = art["forecaster"].features(SF); allf = pd.concat([SF, fc], axis=1)
    m5 = F.compute(b[["ts", "open", "high", "low", "close", "volume"]], F.garch_params(b.close.iloc[:nfit]))
    d = pd.concat([m5, allf], axis=1); d["ts"] = b.ts.values; d["venue"] = venue; d["bar_ms"] = bm
    d["fwd"] = b.close.shift(-H).values / b.close.values - 1; d.loc[(b.ts.shift(-H) - b.ts).values != H * bm, "fwd"] = np.nan
    d["mae_long"], d["mae_short"] = E.path_excursions(b.high.values, b.low.values, b.close.values, H)
    cost = E.round_trip_cost(venue, H * bm)
    fh = np.nan_to_num(fr) * (H * bm / 3_600_000) if venue == "futures" else 0.0
    d["net_long"] = d.fwd - cost - fh; d["net_short"] = -d.fwd - cost + fh; d["y"] = (d.net_long > 0).astype(int); d["trade"] = (d.net_long > 0) | (d.net_short > 0)
    return b, d, art, nfit

def run(canon, raw, venue, tf, years, last_train_to):
    b, d, art, nfit = build(canon, raw, venue, tf, years); bm = BAR_MS[tf]; res = {"bars": len(b), "fit_until": str(pd.to_datetime(b.ts[nfit], unit="ms")), "regime_model": art["regime"].kind}
    oos = slice(nfit, None); close = b.close.values; print("   engine+fitted features built", flush=True)
    # ---- GARCH comparison (OOS QLIKE) ----
    res["garch_qlike"] = {k: dict(zip(("qlike", "n"), G.qlike(np.r_[np.full(nfit, np.nan), d[f"gv_{k}"].values[nfit:]], close))) for k in G.KINDS + ["garchx"]}
    res["garch_best"] = min(res["garch_qlike"], key=lambda k: res["garch_qlike"][k]["qlike"])
    # ---- forecasts (OOS) ----
    res["forecast"] = FC.evaluate(d.iloc[nfit:].reset_index(drop=True)[FC.COLS], close[nfit:], None)
    # ---- regime summary (OOS) ----
    rp = d.iloc[nfit:][[c for c in R.COLS if c.startswith("reg_p_")]]; lab = rp.values.argmax(1)
    res["regime_share"] = {R.REGIMES[i]: round(float((lab == i).mean()), 3) for i in range(4)}
    # ---- classifier val3 ----
    A = F.FEATURES_29[:28]; B = A + E.TALIB_FEATURES[:0] + FE.STATELESS + R.COLS + FC.COLS
    o = d.iloc[nfit + 60:].dropna(subset=B + ["fwd", "mae_long", "mae_short"]).reset_index(drop=True)
    live_from = o.ts.max() - LIVE_DAYS * 86_400_000          # rolling live window ...
    if last_train_to and o.ts.max() - 120 * 86_400_000 < last_train_to < live_from: live_from = last_train_to   # ... extended to all bars since last training
    hold = (o.ts >= live_from).values; dev = ~hold & (o.ts < live_from - H * bm).values
    res["live_holdout"] = {"from": str(pd.to_datetime(live_from, unit="ms")), "rows": int(hold.sum()), "bars_after_last_training": int((o.ts > (last_train_to or 0)).sum())}
    od = o[dev].reset_index(drop=True); tr = od.trade.values
    def allowed(x): return np.column_stack([R.regime_filter(np.ones(len(x), bool), x), R.regime_filter(np.zeros(len(x), bool), x)])
    cv = {}
    for name, cols in (("A_mt5_28", A), ("B_engine", B)):
        p = np.full(len(od), np.nan)
        for trn, tst in PurgedKFold().split(od):
            trn = trn[tr[trn]]; p[tst] = xgb().fit(od.loc[trn, cols].values, od.y.values[trn]).predict_proba(od.loc[tst, cols].values)[:, 1]
        cv[name] = score(od.y.values, p, od)
        if name == "B_engine": cv["B_engine+regime_filter"] = score(od.y.values, p, od, allowed(od))
    res["purged_cv"] = cv; print("   purged CV", json.dumps(cv), flush=True)
    # GridSearchCV over hist + live-appended rows (dev segment), purged CV
    gs = GridSearchCV(xgb(), GRID, cv=PurgedKFold(), scoring="roc_auc", n_jobs=1, refit=False)
    idx = np.nonzero(tr)[0]; gs.fit(od.loc[idx, B].values, od.y.values[idx]); res["grid"] = {"best_params": gs.best_params_, "cv_auc": round(float(gs.best_score_), 4)}
    # live holdout: train on dev (trade rows), test on holdout
    ho = o[hold].reset_index(drop=True); lh = {}
    for name, cols, prm in (("A_mt5_28", A, {}), ("B_engine", B, {}), ("B_engine_tuned", B, gs.best_params_)):
        mdl = xgb(**prm).fit(od.loc[tr, cols].values, od.y.values[tr]); p = mdl.predict_proba(ho[cols].values)[:, 1]; lh[name] = score(ho.y.values, p, ho)
        if name == "B_engine_tuned":
            lh["B_engine_tuned+regime_filter"] = score(ho.y.values, p, ho, allowed(ho)); final = (mdl, cols)
    res["live"] = lh
    base = lh["A_mt5_28"]; cand = max(("B_engine_tuned", "B_engine_tuned+regime_filter"), key=lambda k: (lh[k]["avg_net_bp"] if lh[k]["avg_net_bp"] is not None else -1e9))
    c = lh[cand]; promote = bool(c["trades"] >= 20 and c["avg_net_bp"] is not None and (base["avg_net_bp"] is None or c["avg_net_bp"] > base["avg_net_bp"])
                                 and (cv["B_engine"]["auc"] or 0) >= (cv["A_mt5_28"]["auc"] or 0) - 0.002)
    res["promotion"] = {"candidate": cand, "promote": promote, "rule": "live-holdout avg_net_bp(B) > A, trades>=20, purged-CV AUC(B) >= AUC(A)-0.002"}
    tag = f"{raw}_{tf}"; joblib.dump({"regime": art["regime"], "forecaster_lgb": art["forecaster"].models, "feat_cols": art["forecaster"].feat_cols,
                                      "s_mu": art["forecaster"].s_mu, "s_sd": art["forecaster"].s_sd, "y_sd": art["forecaster"].y_sd,
                                      "clf": final[0], "clf_cols": final[1], "engine_version": FE.ENGINE_VERSION, "bar_ms": bm, "venue": venue,
                                      "regime_filter_in_use": cand.endswith("filter"), "exog_ms": art["exog_ms"],
                                      "mt5_garch_params": F.garch_params(b.close.iloc[:nfit])}, OUT / f"engine_{tag}.joblib")
    art["forecaster"].lstm.save(OUT / f"lstm_{tag}.keras")
    return res

def load_art(tag):
    """Rebuild engine artifacts (used by feature_service and the live/batch parity test)."""
    from tensorflow import keras
    j = joblib.load(OUT / f"engine_{tag}.joblib"); f = FC.Forecaster(); f.models, f.feat_cols, f.s_mu, f.s_sd, f.y_sd = j["forecaster_lgb"], j["feat_cols"], j["s_mu"], j["s_sd"], j["y_sd"]
    f.lstm = keras.models.load_model(OUT / f"lstm_{tag}.keras"); j["art"] = {"regime": j["regime"], "forecaster": f}; return j

def last_training_to():
    try:
        j = json.loads((M / "artifacts_btc" / "btc_xgb6_1h.json").read_text()); return int(pd.Timestamp(j["data"]["PI_XBTUSD"]["to"]).value // 1_000_000)
    except Exception: return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tfs", default="1h,4h"); ap.add_argument("--years", type=float, default=3.0); a = ap.parse_args()
    ltt = last_training_to(); rep = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "engine_version": FE.ENGINE_VERSION, "last_training_to": ltt, "results": {}}
    for tf in a.tfs.split(","):
        for canon, raw, venue in SERIES:
            t0 = time.time(); print(raw, tf, flush=True)
            try: r = run(canon, raw, venue, tf, a.years if tf != "4h" else max(a.years, 6), ltt)
            except Exception as e: import traceback; traceback.print_exc(); r = {"error": repr(e)}
            r["seconds"] = round(time.time() - t0); rep["results"][f"{raw}_{tf}"] = r; print(json.dumps(r, default=str)[:3000], flush=True)
            (OUT / "val3_report.json").write_text(json.dumps(rep, indent=2, default=str))
    rep["promote_any"] = any(v.get("promotion", {}).get("promote") for v in rep["results"].values())
    (OUT / "val3_report.json").write_text(json.dumps(rep, indent=2, default=str))
    (OUT / "promotion.json").write_text(json.dumps({k: v.get("promotion") for k, v in rep["results"].items()} | {"created": rep["created"]}, indent=2))

if __name__ == "__main__": main()
