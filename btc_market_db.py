"""BTC market DB builder/updater -> data/btc_market.db (SQLite WAL).
Sources: Kraken official OHLCVT zips (full history + quarterly), Kraken REST Trades (gap fill to now, aggregated to 1m),
Kraken REST OHLC (recent overlap), Kraken Futures charts API PI_XBTUSD (paginated) + historical funding rates.
Usage: python data/btc_market_db.py build | update | report"""
import sys, io, json, time, sqlite3, pathlib, urllib.request, urllib.parse
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import remote_zip
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent)); from lkpaths import HOME; DB = HOME / "data" / "btc_market.db"
B = "https://assets.kraken.com/marketing/institutions/"
FULL = [B + f"Kraken_OHLCVT_Full.zip.part0{i}" for i in range(5)]; QUARTERS = ["2026Q3"]  # Full covers inception..2026-06-30
NATIVE = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h", 1440: "1d"}
RESAMPLE = {"30m": "30min", "1w": "W-MON"}  # built from 1m/1d
TF_MS = {"1m": 60e3, "5m": 300e3, "15m": 900e3, "30m": 1800e3, "1h": 3600e3, "4h": 14400e3, "1d": 86400e3, "1w": 604800e3}
UA = remote_zip.UA

def con():
    c = sqlite3.connect(DB, timeout=60); c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=NORMAL")
    c.execute("""CREATE TABLE IF NOT EXISTS ohlcv(symbol TEXT, timeframe TEXT, ts INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL,
                 trades INTEGER, source TEXT, PRIMARY KEY(symbol, timeframe, ts)) WITHOUT ROWID""")
    c.execute("CREATE TABLE IF NOT EXISTS funding(symbol TEXT, ts INTEGER, rate REAL, relative_rate REAL, PRIMARY KEY(symbol, ts)) WITHOUT ROWID")
    c.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)"); return c

def put(c, sym, tf, df, src, replace=True):
    rows = [(sym, tf, int(r.ts), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v), int(r.trades) if pd.notna(r.trades) else None, src)
            for r in df.itertuples(index=False)]
    c.executemany(f"INSERT OR {'REPLACE' if replace else 'IGNORE'} INTO ohlcv VALUES(?,?,?,?,?,?,?,?,?,?)", rows); c.commit(); return len(rows)

def load_zip(c, urls, tag):
    z = remote_zip.open_zip(urls)
    for info in z.infolist():
        name = info.filename.split("/")[-1]
        if not name.startswith("XBTUSD_"): continue
        iv = int(name[7:-4]);
        if iv not in NATIVE: continue
        df = pd.read_csv(io.BytesIO(z.read(info)), header=None, names=["ts", "o", "h", "l", "c", "v", "trades"])
        df["ts"] = df.ts.astype("int64") * 1000
        print(tag, name, put(c, "XBTUSD", NATIVE[iv], df, f"kraken_ohlcvt_{tag}"), flush=True)

def get(url):
    for k in range(6):
        try: return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
        except Exception as e:
            if k == 5: raise
            time.sleep(2 + 3 * k)

def last_ts(c, sym, tf):
    r = c.execute("SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?", (sym, tf)).fetchone()[0]; return r or 0

def fill_trades(c):
    """Kraken public Trades -> 1m candles from the last stored 1m candle to now (only completed minutes are written)."""
    start = last_ts(c, "XBTUSD", "1m"); since = (start + 60000) * 1_000_000; buf = []; n = 0; t0 = time.time()
    def flush(final):
        nonlocal buf, n
        if not buf: return
        df = pd.DataFrame(buf, columns=["ts", "p", "q"]); df["m"] = (df.ts // 60000 * 60000).astype("int64")
        cut = int(time.time() // 60 * 60000) if final else int(df.m.max())   # keep the newest (maybe partial) minute in buffer
        done = df[df.m < cut]; buf = [x for x in buf if x[0] // 60000 * 60000 >= cut] if not final else []
        if len(done):
            g = done.groupby("m").agg(o=("p", "first"), h=("p", "max"), l=("p", "min"), c=("p", "last"), v=("q", "sum"), trades=("p", "size")).reset_index().rename(columns={"m": "ts"})
            n += put(c, "XBTUSD", "1m", g, "kraken_trades_api"); print("  trades->1m to", pd.to_datetime(g.ts.max(), unit="ms"), n, f"{time.time()-t0:.0f}s", flush=True)
    while True:
        j = get(f"https://api.kraken.com/0/public/Trades?pair=XBTUSD&since={since}&count=1000")
        if j.get("error"):
            if any("Too many" in e for e in j["error"]): time.sleep(6); continue
            raise RuntimeError(j["error"])
        res = j["result"]; tr = next(v for k, v in res.items() if k != "last"); since = int(res["last"])
        buf += [(float(t[2]) * 1000, float(t[0]), float(t[1])) for t in tr]
        if len(tr) < 1000: flush(True); break
        if len(buf) > 100000: flush(False)
        time.sleep(1.0)
    return n

def resample_from_1m(c, since_ts=0):
    """Rebuild 5m/15m/30m/1h/4h/1d from 1m for ts >= since_ts (used for the API-filled tail), and 1w from 1d (all)."""
    df = pd.read_sql("SELECT ts,o,h,l,c,v,trades FROM ohlcv WHERE symbol='XBTUSD' AND timeframe='1m' AND ts>=? ORDER BY ts", c, params=(int(since_ts),))
    if len(df):
        s = df.set_index(pd.to_datetime(df.ts, unit="ms"))
        for tf, rule in (("5m", "5min"), ("15m", "15min"), ("30m", "30min"), ("1h", "1h"), ("4h", "4h"), ("1d", "1D")):
            if tf != "30m" and since_ts == 0: continue  # native files exist for these
            r = s.resample(rule, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum", "trades": "sum"}).dropna(subset=["o"])
            r = r[r.index + pd.Timedelta(rule) <= pd.Timestamp.now("UTC").tz_localize(None)]  # complete buckets only
            r["ts"] = ((r.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("int64"); put(c, "XBTUSD", tf, r.reset_index(drop=True), "resampled_1m")
    d = pd.read_sql("SELECT ts,o,h,l,c,v,trades FROM ohlcv WHERE symbol='XBTUSD' AND timeframe='1d' ORDER BY ts", c)
    s = d.set_index(pd.to_datetime(d.ts, unit="ms"))
    w = s.resample("W-MON", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum", "trades": "sum"}).dropna(subset=["o"])
    w["ts"] = ((w.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("int64"); put(c, "XBTUSD", "1w", w.reset_index(drop=True), "resampled_1d")

def futures(c):
    """PI_XBTUSD trade charts, paginated backwards from now (API returns <=2000 candles per call)."""
    for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"):
        step = int(TF_MS[tf] / 1000); end = int(time.time()); stop = last_ts(c, "PI_XBTUSD", tf) // 1000; tot = 0
        while True:
            j = get(f"https://futures.kraken.com/api/charts/v1/trade/PI_XBTUSD/{tf}?from={max(stop, end - 2000*step)}&to={end}")
            cs = j.get("candles", [])
            if not cs: break
            df = pd.DataFrame([{"ts": int(x["time"]), "o": x["open"], "h": x["high"], "l": x["low"], "c": x["close"], "v": x["volume"], "trades": None} for x in cs])
            df = df[df.ts + TF_MS[tf] <= time.time() * 1000]; tot += put(c, "PI_XBTUSD", tf, df, "kraken_futures_charts")
            first = int(cs[0]["time"]) // 1000
            if first <= stop or first >= end or not j.get("more_candles", True) and len(cs) < 2: break
            end = first - 1; time.sleep(0.3)
        print("PI_XBTUSD", tf, tot, flush=True)
    j = get("https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol=PI_XBTUSD")
    rows = [("PI_XBTUSD", int(pd.Timestamp(r["timestamp"]).value // 10**6), r.get("fundingRate"), r.get("relativeFundingRate")) for r in j.get("rates", [])]
    c.executemany("INSERT OR REPLACE INTO funding VALUES(?,?,?,?)", rows); c.commit(); print("funding", len(rows))

def fix_resampled(c):
    c.execute("DELETE FROM ohlcv WHERE ts < 1000000000000"); c.commit(); resample_from_1m(c, 0)

def update(c=None):
    """Incremental: trades->1m gap, rebuild tail timeframes, futures tail, funding. Called by trader.py."""
    c = c or con(); t = last_ts(c, "XBTUSD", "1m"); n = fill_trades(c)
    resample_from_1m(c, since_ts=(t // 604800000) * 604800000 - 86400000 if t else 0)
    futures(c); c.execute("INSERT OR REPLACE INTO meta VALUES('last_update', ?)", (time.strftime("%Y-%m-%dT%H:%M:%S"),)); c.commit(); return n

def report(c=None):
    c = c or con(); out = {"db_mb": round(DB.stat().st_size / 1e6, 1) if DB.exists() else 0, "series": []}
    for sym, tf, n, a, b in c.execute("SELECT symbol,timeframe,COUNT(*),MIN(ts),MAX(ts) FROM ohlcv GROUP BY symbol,timeframe"):
        exp = (b - a) / TF_MS[tf] + 1
        out["series"].append({"symbol": sym, "tf": tf, "rows": n, "from": str(pd.to_datetime(a, unit="ms")), "to": str(pd.to_datetime(b, unit="ms")),
                              "missing_bars": int(exp - n), "missing_pct": round(100 * (1 - n / exp), 2)})
    big = {}
    for tf in ("1m", "15m", "1h"):
        ts = pd.read_sql("SELECT ts FROM ohlcv WHERE symbol='XBTUSD' AND timeframe=? ORDER BY ts", c, params=(tf,)).ts
        d = ts.diff() / TF_MS[tf]; i = d.nlargest(3).index
        big[tf] = [{"after": str(pd.to_datetime(ts[k - 1], unit="ms")), "bars": int(d[k])} for k in i]
    out["largest_gaps"] = big
    out["funding_rows"] = c.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM funding").fetchone()
    return out

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"; c = con()
    if cmd == "build":
        load_zip(c, FULL, "full")
        for q in QUARTERS: load_zip(c, [B + f"Kraken_OHLCVT_{q}.zip"], q)
        resample_from_1m(c, 0); update(c)
    elif cmd == "update": update(c)
    elif cmd == "fix": fix_resampled(c)
    print(json.dumps(report(c), indent=1, default=str))
