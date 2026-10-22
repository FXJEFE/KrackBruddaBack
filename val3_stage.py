"""Post-training v3 stage, invoked by train_btc.py (guarded hook) or the nightly learn loop, always with .myenv311:
 1) refresh unified DB from btc_market.db (picks up live bars appended since the last run)
 2) val3 (purged/embargoed CV + rolling live holdout + GridSearchCV on hist+live) -> artifacts_v3/val3_report.json, promotion.json
 3) Obsidian vault + /graph HTML refresh (tools/vault_update.py)
Never touches artifacts_btc or the trader. Exit code 0 even if a step fails (logged to logs/val3_stage.log)."""
import sys, subprocess, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]; LOG = ROOT / "logs" / "val3_stage.log"
def run(args):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(args)}\n"); f.flush()
        r = subprocess.run([sys.executable, "-u"] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT); f.write(f"  -> rc {r.returncode}\n"); return r.returncode
if __name__ == "__main__":
    run(["-m", "datahub.import_kraken"]); run(["models/val3.py"] + sys.argv[1:]); run(["tools/vault_update.py"])
