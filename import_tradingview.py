"""TradingView -> data/tradingview/tradingview_market.db -> unified DB.
TradingView has NO official data API. Two paths:
 1) tvdatafeed (unofficial, nologin): max ~5000 bars per request, no pagination -> e.g. M1 covers ~3.5 days, H1 ~7 months,
    D1 ~13.7 years. Each run appends, so running it regularly accumulates history incrementally.
 2) CSV exports from TradingView charts dropped into data/tradingview/inbox/*.csv (columns time,open,high,low,close[,Volume]);
    file name convention EXCHANGE_SYMBOL, TF.csv (TradingView's default export name, e.g. 'KRAKEN_XBTUSD, 60.csv').
    Processed files move to inbox/done. Times in TV exports are epoch seconds (UTC) or ISO with offset.
Usage: python -m datahub.import_tradingview [--symbols KRAKEN:XBTUSD,BITSTAMP:BTCUSD,COINBASE:BTCUSD] [--csv-only]"""
import re, sys, time, sqlite3, argparse, shutil
import pandas as pd
from . import unified as U
DB = U.DATA / "tradingview" / "tradingview_market.db"; INBOX = U.DATA / "tradingview" / "inbox"
TV_LIMIT = 5000
DEFAULT = "KRAKEN:XBTUSD,BITSTAMP:BTCUSD,COINBASE:BTCUSD,BINANCE:BTCUSDT,BITFINEX:BTCUSD"

def raw_db():
    DB.parent.mkdir(parents=True, exist_ok=True); INBOX.mkdir(parents=True, exist_ok=True); c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS bars(raw_symbol TEXT, timeframe TEXT, ts_utc INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
                 origin TEXT, PRIMARY KEY(raw_symbol, timeframe, ts_utc)) WITHOUT ROWID"""); return c

def store(c, d, sym, tf, origin):
    ok = d.ts_utc.notna()   # NaT = ambiguous/nonexistent local time at a DST switch -> quarantined by validate()
    c.executemany("INSERT OR REPLACE INTO bars VALUES(?,?,?,?,?,?,?,?,?)", [(sym, tf, int(t), o, h, l, cl, v, origin) for t, o, h, l, cl, v in
                  d.loc[ok, ["ts_utc", "open", "high", "low", "close", "volume"]].itertuples(index=False)]); c.commit()
    g, b, gaps = U.validate(d.assign(raw_symbol=sym), "tradingview:" + sym.split(":")[0], tf); U.upsert(g, b, gaps).close()   # one source per exchange feed
    print(sym, tf, origin, "bars", len(d), "good", len(g), "rejected", len(b), "first", pd.to_datetime(d.ts_utc.min(), unit="ms"), flush=True)

def from_tvdatafeed(c, symbols):
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed(); I = {"M1": Interval.in_1_minute, "M5": Interval.in_5_minute, "M15": Interval.in_15_minute, "H1": Interval.in_1_hour,
                            "H4": Interval.in_4_hour, "D1": Interval.in_daily}
    for full in symbols:
        ex, sym = full.split(":")
        for tf, iv in I.items():
            try: d = tv.get_hist(symbol=sym, exchange=ex, interval=iv, n_bars=TV_LIMIT)
            except Exception as e: print(full, tf, "failed", e); continue
            if d is None or d.empty: print(full, tf, "no data"); continue
            # tvdatafeed returns naive datetimes in the machine's local timezone
            import tzlocal; d = d.reset_index(); d["ts_utc"] = U.to_utc_ms(d["datetime"].values, tz=tzlocal.get_localzone_name()).values
            store(c, d, full, tf, "tvdatafeed"); time.sleep(1.5)

def from_inbox(c):
    for f in sorted(INBOX.glob("*.csv")):
        m = re.match(r"(?P<ex>[A-Z0-9]+)_(?P<sym>[^,]+),\s*(?P<tf>\w+)", f.stem)
        if not m: print("skip (name must be EXCHANGE_SYMBOL, TF.csv):", f.name); continue
        d = pd.read_csv(f, encoding="utf-8-sig"); d.columns = [x.strip().lower() for x in d.columns]
        d["ts_utc"] = U.to_utc_ms(d["time"].values); d["volume"] = d.get("volume", pd.Series(float("nan"), index=d.index))
        store(c, d, f"{m['ex']}:{m['sym']}", U.canonical_tf(m["tf"]), f"csv:{f.name}")
        (INBOX / "done").mkdir(exist_ok=True); shutil.move(str(f), INBOX / "done" / f.name)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", default=DEFAULT); ap.add_argument("--csv-only", action="store_true"); a = ap.parse_args()
    c = raw_db(); from_inbox(c)
    if not a.csv_only: from_tvdatafeed(c, a.symbols.split(","))
    print(U.refresh_coverage().query("source.str.startswith('tradingview')", engine="python").to_string())

if __name__ == "__main__": main()
