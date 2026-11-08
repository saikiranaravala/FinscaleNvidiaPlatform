#!/usr/bin/env python3
"""Poll nvidia-smi during a training run and log GPU utilization/memory to CSV.

Run this alongside train.py (own terminal), then Ctrl+C when training finishes:
  python scripts/gpu_monitor.py --out out/gpu_node0.csv --node-id 0

For a two-node benchmark, run one instance per node with a different --node-id and
--out path, then feed both CSVs to scripts/benchmark_scaling.py.
"""
import argparse, csv, subprocess, sys, time
from pathlib import Path

QUERY = "timestamp,index,utilization.gpu,memory.used,memory.total"


def sample():
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    rows = []
    for line in out.splitlines():
        ts, idx, util, mem_used, mem_total = [p.strip() for p in line.split(",")]
        rows.append({"timestamp": ts, "gpu_index": idx, "utilization_pct": util,
                      "memory_used_mib": mem_used, "memory_total_mib": mem_total})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/gpu_monitor.csv"))
    ap.add_argument("--node-id", default="0")
    ap.add_argument("--interval-seconds", type=float, default=2.0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["node_id", "timestamp", "gpu_index", "utilization_pct", "memory_used_mib", "memory_total_mib"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Logging GPU stats for node {args.node_id} to {args.out}; Ctrl+C to stop.", file=sys.stderr)
        try:
            while True:
                for row in sample():
                    writer.writerow({"node_id": args.node_id, **row})
                f.flush()
                time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
