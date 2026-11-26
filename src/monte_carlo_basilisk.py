"""Adapted from author-supplied process-isolated runner; not byte-identical.
Run in a working directory separate from archived data.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "rendezvous_basilisk.py"

def run_worker(seed):
    path = ROOT / f"_basilisk_worker_seed_{seed}.json"
    if path.exists():
        path.unlink()
    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(WORKER), "--seed", str(seed),
        "--noise-pos", "0.2", "--noise-vel", "0.0002", "--thrust-error", "0.01",
        "--direction-error", "0.005", "--worker-output", str(path)],
        cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode or not path.exists():
        raise RuntimeError(result.stderr or "Missing worker JSON")
    record = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    record["wall_time_seconds"] = time.perf_counter()-start
    record["execution_error"] = ""
    return record

def run_campaign(runs=100, start_seed=0):
    rows = []
    for seed in range(start_seed, start_seed+runs):
        try:
            row = run_worker(seed)
        except Exception as exc:
            row = dict(seed=seed, success=None, aborted=None,
                       termination_reason="worker execution failure", execution_error=str(exc))
        rows.append(row)
        pd.DataFrame(rows).to_csv(ROOT/"monte_carlo_basilisk_results.csv", index=False)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()
    run_campaign(args.runs, args.start_seed)
