"""cTrader Open API -> data/ctrader/ctrader_market.db -> unified DB.  READY TO RUN once credentials exist.
Needs in kraken-bridge/.env (never paste them in chat):
  CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID (ctidTraderAccountId, numeric),
  CTRADER_HOST=live|demo (default demo)
Create the app at https://openapi.ctrader.com/apps, then get an access token via its OAuth playground.
Uses ProtoOAGetTrendbarsReq (chunked by the API's max range per period, resume from last stored bar) for
M1,M5,M15,H1,H4,D1 since 2008, and ProtoOASymbolsListReq/ProtoOASymbolByIdReq for specs.
Trendbar prices are integers in 1/100000 units, low + deltaOpen/deltaHigh/deltaClose; utcTimestampInMinutes is UTC."""
import os, sys, time, sqlite3, datetime as dt
import pandas as pd
from . import unified as U
DB = U.DATA / "ctrader" / "ctrader_market.db"
NEED = ["CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID"]
PERIOD = {"M1": ("M1", 5), "M5": ("M5", 35), "M15": ("M15", 35), "H1": ("H1", 35), "H4": ("H4", 300), "D1": ("D1", 3000)}  # days per request
WANT = os.environ.get("CTRADER_SYMBOLS", "BTCUSD,ETHUSD,EURUSD,XAUUSD").split(",")

def env():
    try:
        from dotenv import load_dotenv; load_dotenv(U.ROOT / ".env")
    except ImportError: pass
    miss = [k for k in NEED if not os.environ.get(k)]
    if miss: print("cTrader blocked: missing env keys", miss, "in", U.ROOT / ".env"); sys.exit(3)

def main():
    env()
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (ProtoOAApplicationAuthReq, ProtoOAAccountAuthReq, ProtoOASymbolsListReq,
        ProtoOAGetTrendbarsReq, ProtoOASymbolByIdReq)
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
    from twisted.internet import reactor, defer
    host = EndPoints.PROTOBUF_LIVE_HOST if os.environ.get("CTRADER_HOST") == "live" else EndPoints.PROTOBUF_DEMO_HOST
    acc = int(os.environ["CTRADER_ACCOUNT_ID"]); cl = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
    DB.parent.mkdir(parents=True, exist_ok=True); c = sqlite3.connect(DB)
    c.executescript("""CREATE TABLE IF NOT EXISTS bars(raw_symbol TEXT, timeframe TEXT, ts_utc INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
                       PRIMARY KEY(raw_symbol, timeframe, ts_utc)) WITHOUT ROWID;
                       CREATE TABLE IF NOT EXISTS symbol_specs(raw_symbol TEXT PRIMARY KEY, symbol_id INTEGER, digits INTEGER, pip_position INTEGER, lot_size REAL,
                       swap_long REAL, swap_short REAL, min_volume REAL, max_volume REAL);""")
    def send(req): return cl.send(req)
    @defer.inlineCallbacks
    def run(_):
        r = ProtoOAApplicationAuthReq(); r.clientId = os.environ["CTRADER_CLIENT_ID"]; r.clientSecret = os.environ["CTRADER_CLIENT_SECRET"]; yield send(r)
        r = ProtoOAAccountAuthReq(); r.ctidTraderAccountId = acc; r.accessToken = os.environ["CTRADER_ACCESS_TOKEN"]; yield send(r)
        r = ProtoOASymbolsListReq(); r.ctidTraderAccountId = acc; res = Protobuf.extract((yield send(r)))
        ids = {s.symbolName: s.symbolId for s in res.symbol if U.canonical_symbol(s.symbolName) in WANT}
        r = ProtoOASymbolByIdReq(); r.ctidTraderAccountId = acc; r.symbolId.extend(ids.values()); spec = Protobuf.extract((yield send(r)))
        for s in spec.symbol:
            name = [k for k, v in ids.items() if v == s.symbolId][0]
            c.execute("INSERT OR REPLACE INTO symbol_specs VALUES(?,?,?,?,?,?,?,?,?)", (name, s.symbolId, s.digits, s.pipPosition, s.lotSize / 100, s.swapLong, s.swapShort, s.minVolume / 100, s.maxVolume / 100))
        for name, sid in ids.items():
            for tf, (p, days) in PERIOD.items():
                last = c.execute("SELECT MAX(ts_utc) FROM bars WHERE raw_symbol=? AND timeframe=?", (name, tf)).fetchone()[0]
                a = (last + 1) if last else int(dt.datetime(2008, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000); now = int(time.time() * 1000)
                while a < now:
                    b = min(a + days * 86400000, now); r = ProtoOAGetTrendbarsReq(); r.ctidTraderAccountId = acc; r.symbolId = sid
                    r.period = ProtoOATrendbarPeriod.Value(p); r.fromTimestamp = a; r.toTimestamp = b
                    tb = Protobuf.extract((yield send(r))).trendbar
                    rows = [(name, tf, t.utcTimestampInMinutes * 60000, (t.low + t.deltaOpen) / 1e5, (t.low + t.deltaHigh) / 1e5, t.low / 1e5, (t.low + t.deltaClose) / 1e5, float(t.volume)) for t in tb]
                    c.executemany("INSERT OR REPLACE INTO bars VALUES(?,?,?,?,?,?,?,?)", rows); c.commit(); a = b; yield defer.succeed(None)
                d = pd.read_sql("SELECT raw_symbol, ts_utc, open, high, low, close, volume FROM bars WHERE raw_symbol=? AND timeframe=?", c, params=(name, tf))
                g, bad, gaps = U.validate(d, "ctrader", tf); U.upsert(g, bad, gaps).close(); print(name, tf, len(g), "bars", flush=True)
        reactor.stop()
    cl.setConnectedCallback(lambda _c: run(None)); cl.setDisconnectedCallback(lambda _c, reason: print("disconnected", reason)); cl.startService(); reactor.run()

if __name__ == "__main__": main()
