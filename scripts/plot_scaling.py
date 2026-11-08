#!/usr/bin/env python3
"""Render the 1-node vs 2-node scaling headline chart from scripts/benchmark_scaling.py output.

Usage:
  python scripts/plot_scaling.py --results out/scaling_report.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("out/scaling_report.json"))
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if not args.results.is_file():
        ap.error(f"Missing {args.results}; run scripts/benchmark_scaling.py first.")
    report = json.loads(args.results.read_text(encoding="utf-8"))

    single_tps = report["single_node"].get("tokens_per_second_observed")
    multi_tps = report["multi_node"].get("tokens_per_second_observed")
    if not (single_tps and multi_tps):
        ap.error("tokens_per_second_observed missing in the report; nothing to chart.")

    world_size = report.get("world_size", 2)
    ideal_tps = single_tps * world_size
    efficiency = report.get("scaling_efficiency_pct", 100.0 * multi_tps / ideal_tps)

    output_dir = args.output_dir or (args.results.parent / "charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["1 node", f"{world_size} nodes (actual)", f"{world_size} nodes (ideal linear)"]
    values = [single_tps, multi_tps, ideal_tps]
    colors = ["#64748B", "#0F9D8A", "#CBD5E1"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:,.0f} tok/s", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Training throughput (tokens/sec)")
    ax.set_title(f"FinScale | Multi-node FSDP scaling\n{efficiency:.1f}% scaling efficiency at {world_size} nodes",
                 fontsize=15, weight="bold", pad=18)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        target = output_dir / f"finscale_scaling.{ext}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print(f"Saved {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
