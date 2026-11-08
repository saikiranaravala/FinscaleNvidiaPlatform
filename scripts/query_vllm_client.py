#!/usr/bin/env python3
"""Minimal client demonstrating per-customer LoRA hot-swap against scripts/serve_vllm.py.

Usage:
  python scripts/query_vllm_client.py --model customer_a --excerpt "..." --endpoint http://localhost:8000/v1
"""
import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import SYSTEM_PROMPT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="the --lora NAME registered in scripts/serve_vllm.py, or the base model id")
    ap.add_argument("--ticker", default="DEMO")
    ap.add_argument("--excerpt", required=True)
    args = ap.parse_args()

    user = f"Ticker: {args.ticker}\nFiling excerpt:\n{args.excerpt}"
    resp = requests.post(
        f"{args.endpoint.rstrip('/')}/chat/completions",
        json={"model": args.model,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
              "max_tokens": 384, "temperature": 0},
        timeout=60,
    )
    resp.raise_for_status()
    print(json.dumps(resp.json()["choices"][0]["message"], indent=2))


if __name__ == "__main__":
    main()
