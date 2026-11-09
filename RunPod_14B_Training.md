# RunPod 14B Single-GPU Training

Goal: push the fine-tuning target from the repo's default `Qwen/Qwen2.5-3B-Instruct` up to
`Qwen/Qwen2.5-14B-Instruct` on a single RunPod A100 SXM (80GB) Pod, to exhaust this single-GPU
configuration's headroom before moving to the AWS 2-node FSDP path in `AWS_VALIDATION.md`. There
is no quantization support in this repo (no `bitsandbytes` in `requirements.txt`), so 14B LoRA
in bf16 (~28GB frozen weights) is the safe ceiling on 80GB; `Qwen2.5-32B-Instruct` (~64GB
weights) is a stretch goal only worth attempting after 14B is confirmed stable and
`gpu_monitor.py` shows real headroom left.

All commands below run **inside the RunPod Pod's terminal**, from the repo root
(`cd /workspace/FinscaleNvidiaPlatform` if using a mounted network volume).

## 0. Confirm the Pod's actual GPU and disk

```bash
nvidia-smi
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory/1e9)'
df -h /workspace 2>/dev/null; df -h /
```

Confirm VRAM reads ~80GB before committing to 14B — "A100 SXM" on RunPod is normally the 80GB
part, but don't assume it. Also check whether you have a mounted network volume at `/workspace`
(Volumes tab in the RunPod console) — the container disk alone (commonly 80GB) can fill up fast
once a 14B model's HF cache (~28GB) lands on it.

## 1. Environment bootstrap

If `.venv` doesn't exist yet:

```bash
export HF_HOME=/workspace/.cache/huggingface   # only if a volume is mounted at /workspace
bash runpod-start.sh
source .venv/bin/activate
python -m pip check
python -c 'import torch, transformers, datasets, peft; print(torch.cuda.is_available())'
python -m compileall -q train.py evaluate.py scripts
```

If `.venv` already exists from a prior session:

```bash
source .venv/bin/activate
export USE_TF=0 USE_FLAX=0
```

## 2. Check whether training data already exists

```bash
ls -lh data/train_raw.jsonl data/eval_raw.jsonl data/nvidia/heldout_test.jsonl 2>/dev/null
```

If all three are present from a prior run, skip to step 3. Otherwise regenerate:

```bash
export USE_TF=0 USE_FLAX=0
# review scripts/download_edgar.py's USER_AGENT first — SEC requires a properly identified one
python scripts/download_edgar.py
python -u scripts/make_data_nvidia.py
python scripts/validate_data.py
python scripts/prepare_heldout.py
```

`validate_data.py` must pass (it asserts the train/eval ticker sets are disjoint) before
proceeding — don't train on data that failed this check.

## 3. Start GPU monitoring

Open a **second terminal** into the same Pod and start this just before kicking off training in
step 4:

```bash
source .venv/bin/activate
python scripts/gpu_monitor.py --out out/gpu_node0.csv --node-id 0
```

Leave it running; `Ctrl+C` it once training finishes. This CSV is the evidence for whether
there's room to push batch size further or attempt the 32B stretch afterward — don't skip it
and guess from `nvidia-smi` glances instead.

## 4. Train — Qwen2.5-14B-Instruct

Back in the first terminal:

```bash
source .venv/bin/activate
export USE_TF=0 USE_FLAX=0
mkdir -p out
python -u train.py --backend peft \
  --input data/train_raw.jsonl --data-dir data/nvidia \
  --model Qwen/Qwen2.5-14B-Instruct --batch-size 4 --grad-accum-steps 2 \
  --max-seq-length 1024 --lora-r 16 --lora-alpha 32 \
  --eval-steps 25 --out-dir out 2>&1 | tee out/train_peft.log
```

Notes on these flags relative to the repo defaults:
- `--max-seq-length 1024` instead of the default 4096 — the source excerpts are 60–220 words
  (~150–400 tokens), so 4096 is wasted headroom. Trading unused context length for batch size
  is a better use of the extra VRAM a 14B model still leaves free.
- `--batch-size 4 --grad-accum-steps 2` instead of `1`/`4` — with 14B's ~28GB of frozen weights
  leaving ~50GB free, a real per-device batch size uses the GPU's parallelism better than relying
  entirely on gradient accumulation.
- `--lora-r 16 --lora-alpha 32` kept at the repo's current defaults — LoRA adapter size barely
  affects memory regardless of base model size, so there's no need to change these for 14B.

**While it runs**, watch the `gpu_monitor.py` terminal's CSV growth (or `nvidia-smi` directly):
- If memory stays well under 80GB throughout → stop (`Ctrl+C` is safe, nothing destructive has
  happened), and either raise `--batch-size` further or move on to the 32B stretch test (see
  step 7).
- If you see a CUDA OOM traceback → drop `--batch-size` to 2, then 1, before retrying; if still
  OOM at `--batch-size 1`, drop `--max-seq-length` to 512 next.

## 5. Evaluate

Run internal and held-out evaluation into **separate** `--out-dir`s — they silently overwrite
each other's `eval_results.*` otherwise:

```bash
python -u evaluate.py --backend peft --model Qwen/Qwen2.5-14B-Instruct \
  --adapter out/adapters --test data/nvidia/test.jsonl --out-dir out/internal

python -u evaluate.py --backend peft --model Qwen/Qwen2.5-14B-Instruct \
  --adapter out/adapters --test data/nvidia/heldout_test.jsonl --out-dir out/heldout
```

## 6. Chart

```bash
python -m pip install matplotlib
python scripts/plot_evaluation.py --results out/heldout/eval_results.json
ls -lh out/heldout/charts/
```

Reads real numbers from `eval_results.json` — never hardcode metric values into a chart script
or this doc. Treat any numbers from this run as preliminary/small-sample, same caveat as the
rest of this repo's results.

## 7. Decision point: push further, or move to AWS 2-node

Based on `out/gpu_node0.csv`'s peak memory from step 4:

- **Comfortably under 80GB** (e.g. well below ~65GB peak) → there's headroom left on this single
  GPU. Try raising `--batch-size` further, or attempt `Qwen2.5-32B-Instruct` as a stretch test
  (expect to need `--batch-size 1 --max-seq-length 512` or lower — ~64GB of frozen weights alone
  leaves little margin, so this may still OOM; that's an expected outcome of "exhausting" the
  config, not a failure to fix).
- **Near 80GB already** → 14B is this Pod's practical ceiling. Stop tuning model size and instead
  use this run as the single-node baseline (`out/metrics.json`) you'll need for
  `scripts/benchmark_scaling.py`'s scaling-efficiency comparison once you start the AWS 2-node
  work in `AWS_VALIDATION.md`.

Either way, keep `out/metrics.json` and `out/gpu_node0.csv` from whichever run you settle on —
`AWS_VALIDATION.md`'s step 10 (`benchmark_scaling.py`) wants a real 1-node baseline to compare
the 2-node FSDP run against, and this is it.

## Troubleshooting

| Symptom | Action |
|---|---|
| `CUDA out of memory` at `--batch-size 4` | Retry at `--batch-size 2`, then `1`; then reduce `--max-seq-length`. |
| `nvidia-smi` shows ~80GB but training fails immediately | Check for a stray process still holding memory from a prior killed run (`nvidia-smi` lists PIDs); kill it before retrying. |
| Container disk fills up mid-download | No network volume mounted, or `HF_HOME` not pointed at it — check `df -h` and the Volumes tab. |
| `validate_data.py` fails on ticker overlap | A company leaked between `train_raw.jsonl`/`eval_raw.jsonl` — don't train until this passes; re-run `make_data_nvidia.py` if needed. |
| Evaluation results overwritten | Confirm you used separate `--out-dir out/internal` and `--out-dir out/heldout`. |
