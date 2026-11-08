#!/usr/bin/env python3
"""Score base, fine-tuned, and (optionally) the teacher model against HUMAN labels.

This is the fix for "accuracy only measures agreement with a 7B/72B teacher": the teacher's
own labels are never the ground truth here, scripts/label_gold_set.py's human-reviewed
labels are. Run scripts/sample_gold_set.py + scripts/label_gold_set.py first.

Usage:
  python scripts/evaluate_against_gold.py \
    --gold data/gold/heldout_human.jsonl --backend peft \
    --model Qwen/Qwen2.5-3B-Instruct --adapter out/adapters \
    --teacher-endpoint http://localhost:8000/v1 --teacher-model Qwen/Qwen2.5-72B-Instruct \
    --out-dir out/gold
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import SYSTEM_PROMPT, chat_example
from evaluate import score, score_generations, parse_json_output


def load_gold(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [(chat_example(r)["messages"], r["label"]) for r in records]
    return rows


def score_teacher(rows, endpoint, model_id, max_tokens, n_resamples):
    import requests
    endpoint = endpoint.rstrip("/")
    generations = []
    for msgs, gold in rows:
        resp = requests.post(
            f"{endpoint}/chat/completions",
            json={"model": model_id, "messages": msgs[:-1], "max_tokens": max_tokens, "temperature": 0},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        generations.append((gold, raw))
    return score_generations(generations, n_resamples)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=Path("data/gold/heldout_human.jsonl"))
    ap.add_argument("--backend", choices=("mlx", "peft"), default="peft")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", type=Path, default=Path("out/adapters"))
    ap.add_argument("--teacher-endpoint", default=None, help="vLLM OpenAI-compatible base URL for the 72B teacher")
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--bootstrap-resamples", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=Path("out/gold"))
    args = ap.parse_args()

    if not args.gold.exists():
        ap.error(f"{args.gold} not found; run scripts/sample_gold_set.py and scripts/label_gold_set.py first")
    rows = load_gold(args.gold)
    if not rows:
        ap.error(f"{args.gold} has no records")

    results = {"gold_file": str(args.gold), "examples": len(rows)}
    print(f"Scoring {len(rows)} human-labeled examples against base, fine-tuned"
          + (", and teacher" if args.teacher_endpoint else "") + "...")

    results["base"] = score(rows, args.backend, args.model, None, args.max_tokens, False, args.bootstrap_resamples)
    results["fine_tuned"] = score(rows, args.backend, args.model, args.adapter, args.max_tokens, False, args.bootstrap_resamples)
    if args.teacher_endpoint:
        results["teacher"] = score_teacher(rows, args.teacher_endpoint, args.teacher_model, args.max_tokens, args.bootstrap_resamples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gold_eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cols = ["Model", "Category accuracy (vs human)", "Category macro-F1", "Severity accuracy (vs human)"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for name, key in (("Base", "base"), ("Fine-tuned", "fine_tuned"), ("Teacher", "teacher")):
        m = results.get(key)
        if not m:
            continue
        lines.append(f"| {name} | {m['category_accuracy']:.1%} | {m['category_macro_f1']:.3f} | {m['severity_accuracy']:.1%} |")
    md = "\n".join(lines) + "\n"
    (args.out_dir / "gold_eval_results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
