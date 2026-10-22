"""Kraken raw DB (data/btc_market.db, written by data/btc_market_db.py) -> unified validated DB.
The raw DB stays where the running builder writes it; data/kraken/kraken_market.db is a pointer note only (no copy).
Read-only on the raw DB. Usage: python -m datahub.import_kraken"""
import sqlite3, sys, pandas as pd
from . import unified as U
RAW = U.DATA / "btc_market.db"
TFS = ["1m", "5m", "15m", "1h", "4h", "1d"]

def main():
    (U.DATA / "kraken").mkdir(parents=True, exist_ok=True)
    (U.DATA / "kraken" / "README.txt").write_text("Kraken raw source DB = ..\\btc_market.db (built by data\\btc_market_db.py).\n", encoding="utf-8")
    r = sqlite3.connect(f"file:{RAW}?mode=ro", uri=True)
    for sym in [x[0] for x in r.execute("SELECT DISTINCT symbol FROM ohlcv")]:
        for tf in TFS:
            d = pd.read_sql("SELECT ts, o open, h high, l low, c close, v volume, trades FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts", r, params=(sym, tf))
            if d.empty: continue
            d["ts_utc"] = U.to_utc_ms(d.ts.values)              # fixes rows stored in seconds instead of ms
            d["raw_symbol"] = sym; d["tick_volume"] = d.pop("trades"); d["spread"] = float("nan")
            g, b, gaps = U.validate(d, "kraken", tf); U.upsert(g, b, gaps).close()
            print(sym, tf, "good", len(g), "rejected", len(b), "gaps", len(gaps), flush=True)
    f = pd.read_sql("SELECT symbol, ts, rate, relative_rate FROM funding", r)
    if len(f):
        c = U.connect(); f["ts"] = U.to_utc_ms(f.ts.values)
        c.executemany("INSERT OR REPLACE INTO funding VALUES(?,?,?,?,?,?)", [("kraken", U.canonical_symbol(s), s, int(t), a, b) for s, t, a, b in f.itertuples(index=False)]); c.commit()
        print("funding rows", len(f))
    print(U.refresh_coverage().to_string())

if __name__ == "__main__": main()
