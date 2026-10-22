"""Cross-source comparison from the unified DB: per canonical symbol x timeframe x source pair ->
coverage, missing bars, close basis (mean/max, bp), spread, volume correlation, and MT5-feature differences
(features_mt5.compute on both sources over the common window). Output: data/unified/comparison/*.csv + SUMMARY.md
(MT5 pairs also copied to data/mt5/comparison). Usage: python -m datahub.compare_sources [--feature-bars 5000]"""
import sys, argparse, itertools, sqlite3, numpy as np, pandas as pd
from . import unified as U
sys.path.insert(0, str(U.ROOT / "models"))
OUT = U.DATA / "unified" / "comparison"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feature-bars", type=int, default=5000); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); c = U.connect(); cov = U.refresh_coverage(); cov.to_csv(OUT / "coverage.csv", index=False)
    import features_mt5 as F
    rows, frows = [], []
    for (sym, tf), g in cov.groupby(["canonical_symbol", "timeframe"]):
        srcs = sorted(g.source)
        for s1, s2 in itertools.combinations(srcs, 2):
            A = U.read_bars(s1, sym, tf).set_index("ts"); B = U.read_bars(s2, sym, tf).set_index("ts")
            lo, hi = max(A.index.min(), B.index.min()), min(A.index.max(), B.index.max())
            if lo >= hi: rows.append({"symbol": sym, "tf": tf, "a": s1, "b": s2, "overlap": 0}); continue
            A, B = A.loc[lo:hi], B.loc[lo:hi]; j = A.join(B, how="inner", lsuffix="_a", rsuffix="_b"); bm = U.TIMEFRAMES[tf]
            exp = (hi - lo) // bm + 1; basis = (j.close_b / j.close_a - 1) * 1e4
            rows.append({"symbol": sym, "tf": tf, "a": s1, "b": s2, "overlap_from": pd.to_datetime(lo, unit="ms"), "overlap_to": pd.to_datetime(hi, unit="ms"),
                         "expected_bars": int(exp), "bars_a": len(A), "bars_b": len(B), "common": len(j), "missing_in_a": int(exp - len(A)), "missing_in_b": int(exp - len(B)),
                         "basis_mean_bp": basis.mean(), "basis_absmean_bp": basis.abs().mean(), "basis_max_abs_bp": basis.abs().max(),
                         "spread_a": j.spread_a.mean(), "spread_b": j.spread_b.mean(), "vol_corr": j.volume_a.corr(j.volume_b),
                         "tickvol_corr": j.tick_volume_a.corr(j.tick_volume_b), "ret_corr": j.close_a.pct_change().corr(j.close_b.pct_change())})
            jj = j.tail(a.feature_bars)
            if len(jj) > 300:
                def fe(sfx):
                    x = jj[[f"{k}_{sfx}" for k in ("open", "high", "low", "close", "volume")]].copy(); x.columns = ["open", "high", "low", "close", "volume"]
                    x["ts"] = jj.index.values; x = x.reset_index(drop=True); return F.compute(x, F.garch_params(x.close))
                fa, fb = fe("a"), fe("b")
                for col in F.FEATURES_29:
                    u, v = fa[col], fb[col]; sd = u.std()
                    frows.append({"symbol": sym, "tf": tf, "a": s1, "b": s2, "feature": col, "mean_abs_diff": (u - v).abs().mean(),
                                  "mean_abs_diff_in_sd": (u - v).abs().mean() / sd if sd else np.nan, "corr": u.corr(v)})
    r = pd.DataFrame(rows); fr = pd.DataFrame(frows)
    r.to_csv(OUT / "price_comparison.csv", index=False); fr.to_csv(OUT / "feature_comparison.csv", index=False)
    rej = pd.read_sql("SELECT source, canonical_symbol, timeframe, reason, COUNT(*) n FROM rejects GROUP BY 1,2,3,4", c); rej.to_csv(OUT / "rejects_summary.csv", index=False)
    md = ["# Multi-source market data comparison", "", "## Coverage (validated bars in unified DB)", cov.to_markdown(index=False), "",
          "## Rejects by reason", rej.to_markdown(index=False) if len(rej) else "none", "", "## Pairwise price comparison",
          r.to_markdown(index=False) if len(r) else "Only one source present - nothing to compare yet.", ""]
    if len(fr): md += ["## Feature differences (MT5 29-feature pipeline on both)", fr.pivot_table(index="feature", columns=["symbol", "tf", "a", "b"], values="mean_abs_diff_in_sd").to_markdown(), ""]
    (OUT / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    if len(r) and "mt5" in set(r.a) | set(r.b):
        m = U.DATA / "mt5" / "comparison"; m.mkdir(parents=True, exist_ok=True)
        for n in ("price_comparison.csv", "feature_comparison.csv", "SUMMARY.md"): (m / n).write_bytes((OUT / n).read_bytes())
    print((OUT / "SUMMARY.md").read_text(encoding="utf-8"))

if __name__ == "__main__": main()
