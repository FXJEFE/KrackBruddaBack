"""Delete one source from the unified DB (bars/rejects/gaps/symbol_map/coverage). Usage: python -m datahub.purge_source tradingview"""
import sys
from . import unified as U
c = U.connect(); s = sys.argv[1]
for t in ("bars", "rejects", "gaps", "symbol_map", "coverage"): print(t, c.execute(f"DELETE FROM {t} WHERE source=?", (s,)).rowcount)
c.commit()
