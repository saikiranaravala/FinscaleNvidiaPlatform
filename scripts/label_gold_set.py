#!/usr/bin/env python3
"""Interactive terminal tool for human-labeling a gold evaluation set.

For each candidate record (see scripts/sample_gold_set.py), shows the excerpt and the
teacher's (Qwen-7B/72B) existing label as a starting suggestion, then lets a human accept it
or override category/severity/evidence/explanation. Progress is saved after every record, so
the session can be stopped and resumed (already-labeled tickers+excerpts are skipped).

Usage:
  python scripts/label_gold_set.py --input data/gold/candidates.jsonl --output data/gold/heldout_human.jsonl
"""
import argparse, json
from pathlib import Path

CATEGORIES = ["market", "credit", "liquidity", "operational",
              "regulatory_legal", "cybersecurity", "supply_chain", "other"]
SEVERITIES = ["low", "medium", "high"]


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_choice(label, options, default):
    options_str = "/".join(f"[{o}]" if o == default else o for o in options)
    while True:
        raw = input(f"{label} ({options_str}, Enter for default): ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"  not one of {options}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/gold/candidates.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/gold/heldout_human.jsonl"))
    args = ap.parse_args()

    candidates = load_jsonl(args.input)
    done = load_jsonl(args.output)
    done_keys = {(d["ticker"], d["excerpt"]) for d in done}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    remaining = [c for c in candidates if (c["ticker"], c["excerpt"]) not in done_keys]
    print(f"{len(done)} already labeled, {len(remaining)} remaining. Ctrl+C any time to stop; progress is saved per record.")

    with args.output.open("a", encoding="utf-8") as out:
        for i, record in enumerate(remaining, 1):
            print(f"\n--- {i}/{len(remaining)} | {record['ticker']} ---")
            print(record["excerpt"][:1200])
            teacher = record.get("label", {})
            print(f"\nTeacher label: category={teacher.get('category')} severity={teacher.get('severity')}")
            print(f"Teacher evidence: {teacher.get('evidence')}")
            print(f"Teacher explanation: {teacher.get('explanation')}")

            category = prompt_choice("category", CATEGORIES, teacher.get("category") if teacher.get("category") in CATEGORIES else CATEGORIES[-1])
            severity = prompt_choice("severity", SEVERITIES, teacher.get("severity") if teacher.get("severity") in SEVERITIES else "medium")
            evidence = input(f"evidence (Enter to keep teacher's): ").strip() or teacher.get("evidence", "")
            explanation = input(f"explanation (Enter to keep teacher's): ").strip() or teacher.get("explanation", "")

            human_record = {
                "ticker": record["ticker"],
                "excerpt": record["excerpt"],
                "label": {"category": category, "severity": severity, "evidence": evidence, "explanation": explanation},
                "teacher_label": teacher,
            }
            out.write(json.dumps(human_record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"\nDone. Human-verified labels written to {args.output}")
    print("Next: python scripts/evaluate_against_gold.py --gold", args.output)


if __name__ == "__main__":
    main()
