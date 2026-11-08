#!/usr/bin/env python3
"""Generate a small, deterministic synthetic dataset for smoke-testing the training pipeline
(e.g. validating 2-node FSDP mechanics on throwaway cloud instances) without needing real SEC
data or a labeling model.

Running this with the same --n/--seed on multiple machines produces a byte-identical
data/train_raw.jsonl on each, so a distributed run doesn't need shared storage to agree on data.

NOTE: this validates pipeline mechanics only (splits, loss masking, FSDP, scoring code) — it
says nothing about real classification quality. Never report its eval numbers as results.

Usage:
  python scripts/make_synthetic_smoke_data.py --n 240 --output data/train_raw.jsonl
"""
import argparse, json, random
from pathlib import Path

CATEGORIES = ["market", "credit", "liquidity", "operational",
              "regulatory_legal", "cybersecurity", "supply_chain", "other"]
SEVERITIES = ["low", "medium", "high"]
TICKERS = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]

TEMPLATE = ("The Company faces {category} risk related to {noun}. {detail} "
            "Management considers this risk to be {severity}, given {reason}.")
NOUNS = {
    "market": "interest rate and equity price fluctuations",
    "credit": "counterparty default exposure",
    "liquidity": "short-term funding availability",
    "operational": "disruption to core business processes",
    "regulatory_legal": "changes in applicable law or litigation",
    "cybersecurity": "unauthorized access to information systems",
    "supply_chain": "dependence on a limited set of suppliers",
    "other": "general business uncertainty",
}
DETAILS = [
    "This has affected prior periods and may recur.",
    "No assurance can be given that mitigation efforts will succeed.",
    "The Company continues to monitor this exposure closely.",
]
REASONS = ["the magnitude of potential financial impact", "the likelihood of occurrence",
           "historical experience in the sector"]


def make_record(rng, ticker):
    category = rng.choice(CATEGORIES)
    severity = rng.choice(SEVERITIES)
    noun = NOUNS[category]
    sentence = TEMPLATE.format(category=category.replace("_", "/"), noun=noun,
                                detail=rng.choice(DETAILS), severity=severity, reason=rng.choice(REASONS))
    evidence = sentence.split(".")[0] + "."
    return {"ticker": ticker, "excerpt": sentence,
            "label": {"category": category, "severity": severity, "evidence": evidence,
                      "explanation": f"The excerpt describes {noun}, consistent with {category} risk at {severity} severity."}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output", type=Path, default=Path("data/train_raw.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(make_record(rng, TICKERS[i % len(TICKERS)]), ensure_ascii=False) + "\n")
    print(f"Wrote {args.n} synthetic records -> {args.output}")
    print("Mechanics-only data: do not report its eval numbers as results.")


if __name__ == "__main__":
    main()
