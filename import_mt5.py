"""MT5 -> data/mt5/mt5_market.db (raw, broker server time kept + UTC) -> unified DB.
Uses ONLY an already running/logged-in terminal (mt5.initialize() with no credentials). Imports all Market Watch
symbols (symbols_get() with .select True) for M1..D1 from 2008-01-01, chunked copy_rates_range (monthly for M1/M5,
yearly otherwise), incremental resume per symbol/timeframe, plus symbol specs. Records actual first bar.
Server time -> UTC: MT5 returns bar times as server-local epoch. Offset is measured now (last tick time - UTC);
brokers on GMT+2/+3 follow EU DST (New York close), so with MT5_SERVER_TZ unset and offset 2/3h we use 'Europe/Athens'
(EET/EEST) which reproduces that DST schedule; otherwise a fixed offset. Override: MT5_SERVER_TZ=Etc/GMT-2 etc.
Usage: python -m datahub.import_mt5 [--symbols BTCUSD,ETHUSD] [--tfs M1,M5] [--ticks-days 7]"""
import os, sys, time, sqlite3, argparse, datetime as dt
import numpy as np, pandas as pd
from . import unified as U
DB = U.DATA / "mt5" / "mt5_market.db"
START = dt.datetime(2008, 1, 1)

def tfmap(mt5):
    return {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}

def raw_db():
    DB.parent.mkdir(parents=True, exist_ok=True); c = sqlite3.connect(DB, timeout=60); c.execute("PRAGMA journal_mode=WAL")
    c.executescript("""CREATE TABLE IF NOT EXISTS rates(symbol TEXT, timeframe TEXT, time_server INTEGER, ts_utc INTEGER, open REAL, high REAL, low REAL,
        close REAL, tick_volume REAL, spread_points INTEGER, real_volume REAL, PRIMARY KEY(symbol, timeframe, time_server)) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS ticks(symbol TEXT, time_msc_server INTEGER, ts_utc INTEGER, bid REAL, ask REAL, last REAL, volume REAL, flags INTEGER,
        PRIMARY KEY(symbol, time_msc_server, flags, bid, ask)) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS symbol_specs(symbol TEXT PRIMARY KEY, canonical_symbol TEXT, description TEXT, path TEXT, digits INTEGER, point REAL,
        trade_contract_size REAL, volume_min REAL, volume_step REAL, volume_max REAL, spread INTEGER, spread_float INTEGER, swap_long REAL, swap_short REAL,
        swap_mode INTEGER, margin_initial REAL, margin_maintenance REAL, margin_hedged REAL, currency_base TEXT, currency_profit TEXT, currency_margin TEXT,
        trade_calc_mode INTEGER, leverage_account INTEGER, margin_1lot REAL, updated TEXT);
      CREATE TABLE IF NOT EXISTS coverage(symbol TEXT, timeframe TEXT, first_bar_utc INTEGER, last_bar_utc INTEGER, rows INTEGER, capped INTEGER, note TEXT,
        PRIMARY KEY(symbol, timeframe));
      CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);""")
    return c

def server_tz(mt5, sym):
    if os.environ.get("MT5_SERVER_TZ"): return os.environ["MT5_SERVER_TZ"]
    t = mt5.symbol_info_tick(sym); off = round((t.time - time.time()) / 3600) if t else 0
    return "Europe/Athens" if off in (2, 3) else f"Etc/GMT{-off:+d}" if off else "UTC"

def chunks(tf, start, end):
    step = pd.DateOffset(months=1) if tf in ("M1", "M5") else pd.DateOffset(years=1)
    a = pd.Timestamp(start)
    while a < end:
        b = min(a + step, pd.Timestamp(end)); yield a.to_pydatetime(), b.to_pydatetime(); a = b

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols"); ap.add_argument("--tfs", default="M1,M5,M15,H1,H4,D1"); ap.add_argument("--ticks-days", type=int, default=7)
    a = ap.parse_args()
    import MetaTrader5 as mt5
    if not mt5.initialize():  # attaches to the running, logged-in terminal; never passes credentials
        print("MT5 initialize failed:", mt5.last_error(), "- is terminal64.exe installed, running and logged in?"); sys.exit(2)
    ti = mt5.terminal_info(); ai = mt5.account_info(); c = raw_db()
    c.execute("INSERT OR REPLACE INTO meta VALUES('maxbars', ?)", (str(ti.maxbars),)); c.commit()
    print("terminal", ti.name, "build", mt5.version(), "maxbars", ti.maxbars, "connected", ti.connected)
    syms = a.symbols.split(",") if a.symbols else [s.name for s in mt5.symbols_get() if s.select]
    tz = server_tz(mt5, syms[0]); c.execute("INSERT OR REPLACE INTO meta VALUES('server_tz', ?)", (tz,)); print("server tz ->", tz)
    end = dt.datetime.now() + dt.timedelta(days=1)
    for s in syms:
        mt5.symbol_select(s, True); i = mt5.symbol_info(s)
        if i is None: print("skip", s); continue
        m1 = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, s, 1.0, i.ask or i.bid or 0) if (i.ask or i.bid) else None
        c.execute("INSERT OR REPLACE INTO symbol_specs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (s, U.canonical_symbol(s), i.description, i.path, i.digits, i.point,
                  i.trade_contract_size, i.volume_min, i.volume_step, i.volume_max, i.spread, int(i.spread_float), i.swap_long, i.swap_short, i.swap_mode,
                  i.margin_initial, i.margin_maintenance, i.margin_hedged, i.currency_base, i.currency_profit, i.currency_margin, i.trade_calc_mode,
                  ai.leverage if ai else None, m1, time.strftime("%Y-%m-%dT%H:%M:%S")))
        for tf in a.tfs.split(","):
            last = c.execute("SELECT MAX(time_server) FROM rates WHERE symbol=? AND timeframe=?", (s, tf)).fetchone()[0]
            st = dt.datetime.utcfromtimestamp(last) if last else START; n = 0
            for x, y in chunks(tf, st, end):
                r = mt5.copy_rates_range(s, tfmap(mt5)[tf], x, y)
                if r is None or len(r) == 0: continue
                d = pd.DataFrame(r); d["ts_utc"] = U.to_utc_ms(pd.to_datetime(d.time, unit="s").values, tz=tz).values
                c.executemany("INSERT OR REPLACE INTO rates VALUES(?,?,?,?,?,?,?,?,?,?,?)", [(s, tf, int(t), None if pd.isna(u) else int(u), o, h, l, cl, tv, int(sp), rv)
                              for t, u, o, h, l, cl, tv, sp, rv in d[["time", "ts_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]].itertuples(index=False)])
                c.commit(); n += len(d)
            f, l, rows = c.execute("SELECT MIN(ts_utc), MAX(ts_utc), COUNT(*) FROM rates WHERE symbol=? AND timeframe=?", (s, tf)).fetchone()
            capped = int(rows >= ti.maxbars - 1); c.execute("INSERT OR REPLACE INTO coverage VALUES(?,?,?,?,?,?,?)", (s, tf, f, l, rows, capped,
                     f"maxbars={ti.maxbars}" if capped else "")); c.commit()
            print(s, tf, "new", n, "total", rows, "first", pd.to_datetime(f, unit="ms") if f else None, "CAPPED" if capped else "", flush=True)
            d = pd.read_sql("SELECT symbol raw_symbol, ts_utc, open, high, low, close, real_volume volume, tick_volume, spread_points FROM rates WHERE symbol=? AND timeframe=?", c, params=(s, tf))
            d["spread"] = d.pop("spread_points") * i.point
            if (d.volume == 0).all(): d["volume"] = d.tick_volume
            g, b, gaps = U.validate(d, "mt5", tf); U.upsert(g, b, gaps).close()
        if a.ticks_days:
            tk = mt5.copy_ticks_range(s, dt.datetime.now() - dt.timedelta(days=a.ticks_days), end, mt5.COPY_TICKS_ALL)
            if tk is not None and len(tk):
                t = pd.DataFrame(tk); t["ts_utc"] = U.to_utc_ms(pd.to_datetime(t.time_msc, unit="ms").values, tz=tz).values
                c.executemany("INSERT OR IGNORE INTO ticks VALUES(?,?,?,?,?,?,?,?)", [(s, int(a_), None if pd.isna(b_) else int(b_), bi, ak, la, vo, int(fl)) for a_, b_, bi, ak, la, vo, fl in
                              t[["time_msc", "ts_utc", "bid", "ask", "last", "volume", "flags"]].itertuples(index=False)]); c.commit()
                print(s, "ticks", len(t))
    mt5.shutdown(); print(U.refresh_coverage().query("source=='mt5'").to_string())

if __name__ == "__main__": main()
