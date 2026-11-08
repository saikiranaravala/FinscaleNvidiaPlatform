#!/usr/bin/env python3
"""Compare 1-node vs 2-node training runs and compute scaling efficiency.

This does NOT launch training itself (a real 1-node and a real 2-node run have to
happen on actual hardware: single GPU for the 1-node baseline, scripts/launch_distributed.sh
on both nodes for the 2-node run). Point this at the resulting out/metrics.json files
and, optionally, the scripts/gpu_monitor.py CSVs captured during each run.

Usage:
  python scripts/benchmark_scaling.py \
    --single-node out/1node/metrics.json \
    --multi-node out/2node/metrics.json \
    --single-node-gpu-csv out/1node/gpu_monitor.csv \
    --multi-node-gpu-csv out/2node/gpu_node0.csv out/2node/gpu_node1.csv \
    --out out/scaling_report.json
"""
import argparse, csv, json, statistics, sys
from pathlib import Path

FIELDS = ["tokens_per_second_observed", "mean_step_time_seconds", "wall_time_seconds", "peak_memory_gb_observed"]


def load_metrics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_gpu_utilization(csv_paths):
    if not csv_paths:
        return None
    values = []
    for p in csv_paths:
        with Path(p).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    values.append(float(row["utilization_pct"]))
                except (KeyError, ValueError):
                    continue
    return statistics.mean(values) if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single-node", type=Path, required=True)
    ap.add_argument("--multi-node", type=Path, required=True)
    ap.add_argument("--single-node-gpu-csv", nargs="*", default=[])
    ap.add_argument("--multi-node-gpu-csv", nargs="*", default=[])
    ap.add_argument("--world-size", type=int, default=2, help="GPUs used in the multi-node run")
    ap.add_argument("--out", type=Path, default=Path("out/scaling_report.json"))
    args = ap.parse_args()

    single = load_metrics(args.single_node)
    multi = load_metrics(args.multi_node)

    report = {"single_node": {k: single.get(k) for k in FIELDS},
              "multi_node": {k: multi.get(k) for k in FIELDS},
              "world_size": args.world_size}

    single_tps = single.get("tokens_per_second_observed")
    multi_tps = multi.get("tokens_per_second_observed")
    if single_tps and multi_tps:
        ideal_tps = single_tps * args.world_size
        report["throughput_speedup_x"] = multi_tps / single_tps
        report["scaling_efficiency_pct"] = 100.0 * (multi_tps / ideal_tps)
    else:
        report["note"] = "tokens_per_second_observed missing from one of the metrics.json files; " \
                          "scaling_efficiency_pct not computed. Fall back to wall_time_seconds if needed."
        single_wt, multi_wt = single.get("wall_time_seconds"), multi.get("wall_time_seconds")
        if single_wt and multi_wt:
            report["wall_time_speedup_x"] = single_wt / multi_wt

    report["single_node_mean_gpu_utilization_pct"] = mean_gpu_utilization(args.single_node_gpu_csv)
    report["multi_node_mean_gpu_utilization_pct"] = mean_gpu_utilization(args.multi_node_gpu_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
