#!/usr/bin/env python3
"""Sample held-out excerpts for human labeling (see scripts/label_gold_set.py).

Usage:
  python scripts/sample_gold_set.py --input data/eval_raw.jsonl --n 200 --output data/gold/candidates.jsonl
"""
import argparse, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/eval_raw.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/gold/candidates.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    picked = rows[: args.n]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Sampled {len(picked)}/{len(rows)} records -> {args.output}")
    print("Next: python scripts/label_gold_set.py --input", args.output)


if __name__ == "__main__":
    main()
