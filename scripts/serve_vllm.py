#!/usr/bin/env python3
"""Launch a vLLM OpenAI-compatible server with hot-swappable per-customer LoRA adapters.

One base model (e.g. the fine-tuned 14B student) serves N customer-specific adapters, each
selected per-request by name — this is the "customer-specific fine-tuning, served cheaply"
half of FinScale's pitch. This script is a thin wrapper around `vllm serve`; adjust
--tensor-parallel-size for however many GPUs the base model needs on this box.

Usage:
  python scripts/serve_vllm.py --model Qwen/Qwen2.5-14B-Instruct \
    --lora customer_a=out/adapters_customer_a --lora customer_b=out/adapters_customer_b

Then query a specific customer's adapter (see scripts/query_vllm_client.py) by setting
"model": "customer_a" in the chat completion request.
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="base model all LoRA adapters are applied to")
    ap.add_argument("--lora", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable; registers one LoRA adapter under NAME, servable by passing model=NAME")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-loras", type=int, default=max(1, len(sys.argv)))
    args = ap.parse_args()

    if not args.lora:
        print("warning: no --lora adapters registered; serving the base model only", file=sys.stderr)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--enable-lora",
        "--max-loras", str(max(1, len(args.lora))),
    ]
    for entry in args.lora:
        cmd += ["--lora-modules", entry]

    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
