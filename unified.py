"""Unified, validated multi-source market DB (data/unified/market_unified.db).
Every source importer writes to data/<source>/<source>_market.db in its own raw layout; this module normalises
symbols/timeframes/timestamps, validates rows, quarantines rejects and upserts into ONE canonical schema.
Models/features must read via `read_bars()` which checks SCHEMA_VERSION and fails loudly on mismatch."""
import re, sqlite3, pathlib, time
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UNIFIED_DB = DATA / "unified" / "market_unified.db"
SCHEMA_VERSION = "unified-v1"
TIMEFRAMES = {"M1": 60_000, "M5": 300_000, "M15": 900_000, "H1": 3_600_000, "H4": 14_400_000, "D1": 86_400_000}
TF_ALIASES = {"1m": "M1", "5m": "M5", "15m": "M15", "1h": "H1", "4h": "H4", "1d": "D1", "d1": "D1", "1D": "D1",
              "60": "H1", "240": "H4", "1": "M1", "5": "M5", "15": "M15", "D": "D1"}
BAR_COLS = ["source", "canonical_symbol", "raw_symbol", "timeframe", "ts_utc", "open", "high", "low", "close", "volume", "tick_volume", "spread"]
DDL = """
CREATE TABLE IF NOT EXISTS schema_info(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS bars(source TEXT NOT NULL, canonical_symbol TEXT NOT NULL, raw_symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
  ts_utc INTEGER NOT NULL, open REAL, high REAL, low REAL, close REAL, volume REAL, tick_volume REAL, spread REAL,
  PRIMARY KEY(source, canonical_symbol, timeframe, ts_utc)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS rejects(source TEXT, canonical_symbol TEXT, raw_symbol TEXT, timeframe TEXT, ts_raw TEXT, ts_utc INTEGER,
  open REAL, high REAL, low REAL, close REAL, volume REAL, reason TEXT, logged_at TEXT);
CREATE TABLE IF NOT EXISTS gaps(source TEXT, canonical_symbol TEXT, timeframe TEXT, gap_from INTEGER, gap_to INTEGER, missing_bars INTEGER,
  PRIMARY KEY(source, canonical_symbol, timeframe, gap_from));
CREATE TABLE IF NOT EXISTS symbol_map(source TEXT, raw_symbol TEXT, canonical_symbol TEXT, PRIMARY KEY(source, raw_symbol));
CREATE TABLE IF NOT EXISTS funding(source TEXT, canonical_symbol TEXT, raw_symbol TEXT, ts_utc INTEGER, rate REAL, relative_rate REAL,
  PRIMARY KEY(source, canonical_symbol, ts_utc));
CREATE TABLE IF NOT EXISTS coverage(source TEXT, canonical_symbol TEXT, timeframe TEXT, first_ts INTEGER, last_ts INTEGER, rows INTEGER,
  rejects INTEGER, gaps INTEGER, missing_bars INTEGER, updated TEXT, PRIMARY KEY(source, canonical_symbol, timeframe));
"""
# Kraken futures perpetual PI_XBTUSD is kept as its own canonical symbol (different instrument from spot).
SPECIAL = {"PI_XBTUSD": "BTCUSD_PERP", "PF_XBTUSD": "BTCUSD_PERP", "XXBTZUSD": "BTCUSD", "XBT/USD": "BTCUSD"}
_PREFIX = re.compile(r"^[A-Z0-9_]+:")                       # KRAKEN:, BITSTAMP:, OANDA:
_SUFFIX = re.compile(r"(\.(r|m|pro|ecn|raw|i|a|b|c|e|x|stp)|_i|_ecn|ecn|pro|#|\+|!)$", re.I)

def canonical_symbol(raw: str) -> str:
    s = str(raw).strip().upper()
    if s in SPECIAL: return SPECIAL[s]
    s = _PREFIX.sub("", s)
    if s in SPECIAL: return SPECIAL[s]
    for _ in range(3): s = _SUFFIX.sub("", s)
    s = s.replace("/", "").replace("-", "").replace("_", "").replace(" ", "")
    if len(s) == 7 and s[-1] == "M" and s[:6].isalpha(): s = s[:6]   # EURUSDm style micro suffix
    s = s.replace("XBT", "BTC")
    if s.endswith("USDT") and s.startswith("BTC"): s = s  # keep USDT distinct from USD
    return s

def canonical_tf(tf: str) -> str:
    t = str(tf).strip()
    if t.upper() in TIMEFRAMES: return t.upper()
    if t in TF_ALIASES: return TF_ALIASES[t]
    raise ValueError(f"unknown timeframe {tf!r}")

def to_utc_ms(ts, unit_hint=None, tz=None) -> pd.Series:
    """Accepts epoch s/ms/us or datetime strings. tz: source timezone for naive datetimes (e.g. broker server tz
    'Europe/Athens' = EET/EEST with DST, MT5's usual GMT+2/+3) -> converted to UTC."""
    s = pd.Series(ts)
    if np.issubdtype(s.dtype, np.number):
        v = s.astype("int64")
        unit = unit_hint or ("s" if v.abs().median() < 1e11 else "ms" if v.abs().median() < 1e14 else "us")
        dt = pd.Series(pd.to_datetime(v, unit=unit))
    else:
        dt = pd.Series(pd.to_datetime(s, utc=False, errors="coerce"))
    if tz is not None:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    elif getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    ms = (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)   # resolution-independent (pandas 2/3)
    return ms.where(dt.notna())

def connect(db=UNIFIED_DB):
    db = pathlib.Path(db); db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db, timeout=60); c.execute("PRAGMA journal_mode=WAL"); c.executescript(DDL)
    cur = dict(c.execute("SELECT k, v FROM schema_info").fetchall())
    if not cur:
        c.executemany("INSERT INTO schema_info VALUES(?,?)", [("version", SCHEMA_VERSION), ("created", time.strftime("%Y-%m-%dT%H:%M:%S")),
                      ("ts_convention", "ts_utc = epoch ms UTC, bar OPEN time"), ("encoding", "UTF-8")]); c.commit()
    elif cur.get("version") != SCHEMA_VERSION:
        raise RuntimeError(f"unified DB schema {cur.get('version')!r} != expected {SCHEMA_VERSION!r} ({db})")
    return c

def validate(df: pd.DataFrame, source: str, timeframe: str, spike_sigma: float = 25.0, spike_floor: float = 0.30):
    """df: raw_symbol, ts_utc (ms), open, high, low, close, volume, [tick_volume, spread]. Returns (good, rejects, gaps)."""
    d = df.copy(); d["source"] = source; d["timeframe"] = canonical_tf(timeframe)
    d["canonical_symbol"] = d["raw_symbol"].map(canonical_symbol)
    for c in ("tick_volume", "spread"):
        if c not in d: d[c] = np.nan
    num = ["open", "high", "low", "close", "volume", "tick_volume", "spread"]
    for c in num: d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    bm = TIMEFRAMES[d["timeframe"].iloc[0]] if len(d) else 0
    reason = pd.Series("", index=d.index)
    def flag(mask, why):
        nonlocal reason; m = mask & (reason == ""); reason[m] = why
    flag(d.ts_utc.isna(), "bad_timestamp")
    flag(d[["open", "high", "low", "close"]].isna().any(axis=1), "nan_price")
    flag((d[["open", "high", "low", "close"]] <= 0).any(axis=1), "non_positive_price")
    flag(d.high < d[["open", "close"]].max(axis=1) - 1e-12, "high_below_open_close")
    flag(d.low > d[["open", "close"]].min(axis=1) + 1e-12, "low_above_open_close")
    flag(d.volume < 0, "negative_volume")
    flag((d.ts_utc % bm) != 0 if bm else False, "ts_not_on_bar_boundary")
    flag((d.ts_utc < 946684800000) | (d.ts_utc > (time.time() + 86400) * 1000), "ts_out_of_range")
    d = d.sort_values("ts_utc", kind="mergesort")
    dup = d.duplicated(["canonical_symbol", "ts_utc"], keep="last"); flag(dup.reindex(d.index), "duplicate_ts")
    ok = reason.reindex(d.index) == ""
    # spike filter on clean rows: |log return| beyond spike_sigma robust sigmas AND beyond spike_floor, that reverts next bar
    g = d[ok].copy()
    if len(g) > 50:
        lr = np.log(g.close).groupby(g.canonical_symbol).diff()
        mad = lr.abs().groupby(g.canonical_symbol).transform(lambda x: x.rolling(500, min_periods=50).median()).bfill() * 1.4826
        nxt = lr.groupby(g.canonical_symbol).shift(-1)
        spike = (lr.abs() > np.maximum(spike_sigma * mad, spike_floor)) & (np.sign(nxt) == -np.sign(lr)) & (nxt.abs() > 0.5 * lr.abs())
        flag(spike.reindex(d.index, fill_value=False), "price_spike")
    reason = reason.reindex(d.index); good = d[reason == ""].copy(); bad = d[reason != ""].copy(); bad["reason"] = reason[reason != ""]
    for c in num: good[c] = good[c].round(10)
    gaps = []
    for sym, gg in good.groupby("canonical_symbol"):
        t = gg.ts_utc.values.astype("int64"); dt = np.diff(t)
        for i in np.nonzero(dt > bm)[0]:
            gaps.append((source, sym, good.timeframe.iloc[0], int(t[i]), int(t[i + 1]), int(dt[i] // bm - 1)))
    return good[BAR_COLS], bad, gaps

def upsert(good, bad, gaps, db=UNIFIED_DB):
    c = connect(db)
    good = good.astype({"ts_utc": "int64"})
    c.executemany(f"INSERT OR REPLACE INTO bars VALUES({','.join('?'*len(BAR_COLS))})",
                  [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in r) for r in good.itertuples(index=False, name=None)])
    if len(bad):
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        c.executemany("INSERT INTO rejects VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      [(r.source, r.canonical_symbol, r.raw_symbol, r.timeframe, str(r.ts_utc), None if pd.isna(r.ts_utc) else int(r.ts_utc),
                        r.open, r.high, r.low, r.close, r.volume, r.reason, now) for r in bad.itertuples()])
    c.executemany("INSERT OR REPLACE INTO gaps VALUES(?,?,?,?,?,?)", gaps)
    c.executemany("INSERT OR REPLACE INTO symbol_map VALUES(?,?,?)", list({(r[0], r[2], r[1]) for r in good[["source", "canonical_symbol", "raw_symbol"]].itertuples(index=False)}))
    c.commit(); return c

def refresh_coverage(db=UNIFIED_DB):
    c = connect(db); now = time.strftime("%Y-%m-%dT%H:%M:%S")
    c.execute("DELETE FROM coverage")
    c.execute("""INSERT INTO coverage SELECT b.source, b.canonical_symbol, b.timeframe, MIN(ts_utc), MAX(ts_utc), COUNT(*),
        (SELECT COUNT(*) FROM rejects r WHERE r.source=b.source AND r.canonical_symbol=b.canonical_symbol AND r.timeframe=b.timeframe),
        (SELECT COUNT(*) FROM gaps g WHERE g.source=b.source AND g.canonical_symbol=b.canonical_symbol AND g.timeframe=b.timeframe),
        (SELECT COALESCE(SUM(missing_bars),0) FROM gaps g WHERE g.source=b.source AND g.canonical_symbol=b.canonical_symbol AND g.timeframe=b.timeframe),
        ? FROM bars b GROUP BY 1,2,3""", (now,)); c.commit()
    return pd.read_sql("SELECT * FROM coverage ORDER BY 1,2,3", c)

def read_bars(source, canonical, timeframe, since_ms=None, db=UNIFIED_DB) -> pd.DataFrame:
    """The ONLY entry point for models/features. Fails loudly if schema/version/columns mismatch."""
    db = pathlib.Path(db)
    if not db.exists(): raise FileNotFoundError(f"unified DB missing: {db} (run python -m datahub.import_kraken)")
    c = sqlite3.connect(db)
    v = dict(c.execute("SELECT k, v FROM schema_info").fetchall()).get("version")
    if v != SCHEMA_VERSION: raise RuntimeError(f"unified DB version {v!r} != {SCHEMA_VERSION!r}")
    cols = [r[1] for r in c.execute("PRAGMA table_info(bars)")]
    if cols != BAR_COLS: raise RuntimeError(f"unified bars columns {cols} != {BAR_COLS}")
    q = "SELECT ts_utc ts, open, high, low, close, volume, tick_volume, spread FROM bars WHERE source=? AND canonical_symbol=? AND timeframe=?"
    p = [source, canonical, canonical_tf(timeframe)]
    if since_ms is not None: q += " AND ts_utc>=?"; p.append(int(since_ms))
    return pd.read_sql(q + " ORDER BY ts_utc", c, params=p)

def read_funding(source, canonical, db=UNIFIED_DB):
    c = connect(db); return pd.read_sql("SELECT ts_utc ts, rate, relative_rate FROM funding WHERE source=? AND canonical_symbol=? ORDER BY ts_utc", c, params=(source, canonical))
