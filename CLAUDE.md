# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FinScale fine-tunes **Qwen/Qwen2.5-3B-Instruct** with LoRA (PyTorch + Transformers + PEFT) to classify financial-risk disclosures from SEC filings into structured JSON (`category`, `severity`, `evidence`, `explanation`), then compares base vs. fine-tuned model performance. The pipeline runs on NVIDIA GPUs (RunPod or Lambda Labs); there is no local/CPU training path. `train.py` also supports an `mlx` backend (Apple Silicon) but PEFT/CUDA is the one actively maintained for this repo.

**Status:** results so far are small-sample/preliminary, not production benchmarks. Don't treat numbers in README.md or `results/` as validated.

## Environment setup

This only works on a CUDA GPU box (RunPod or Lambda Labs) with CUDA-enabled PyTorch preinstalled — `requirements.txt` deliberately does **not** install/replace PyTorch.

```bash
nvidia-smi
python3 -c 'import torch; assert torch.cuda.is_available()'
```

RunPod bootstrap (creates `.venv` with `--system-site-packages`, installs `requirements.txt`, runs `pip check`):
```bash
bash runpod-start.sh
source .venv/bin/activate
```

Lambda Labs:
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export USE_TF=0 USE_FLAX=0
```

Always set `USE_TF=0 USE_FLAX=0` before running training/eval to avoid optional TF/Flax imports.

## Common commands

Sanity-check the stack after install:
```bash
python -m pip check
python -c 'import torch, transformers, datasets, peft; print(torch.cuda.is_available())'
python -m compileall -q train.py evaluate.py scripts
```

Data pipeline (run in order):
```bash
python scripts/download_edgar.py        # scrape SEC EDGAR 10-K "Item 1A Risk Factors" -> data/raw/*.txt
python -u scripts/make_data_nvidia.py    # label chunks with Qwen2.5-7B-Instruct -> data/train_raw.jsonl, data/eval_raw.jsonl
python scripts/validate_data.py          # check schema + train/eval ticker (company) isolation
python scripts/prepare_heldout.py        # data/eval_raw.jsonl -> data/nvidia/heldout_test.jsonl (chat format)
```

Train (also creates the internal train/valid/test split as a side effect):
```bash
python -u train.py --backend peft \
  --input data/train_raw.jsonl --data-dir data/nvidia \
  --model Qwen/Qwen2.5-3B-Instruct --epochs 1 --batch-size 1 \
  --max-seq-length 2048 --learning-rate 1e-5 --lora-r 8 --lora-alpha 16 \
  --eval-steps 25 --out-dir out 2>&1 | tee out/train_peft.log
```

Evaluate (run internal and held-out separately into different `--out-dir`s — they will silently overwrite each other's `eval_results.*` otherwise):
```bash
python -u evaluate.py --backend peft --model Qwen/Qwen2.5-3B-Instruct \
  --adapter out/adapters --test data/nvidia/test.jsonl --out-dir out/internal

python -u evaluate.py --backend peft --model Qwen/Qwen2.5-3B-Instruct \
  --adapter out/adapters --test data/nvidia/heldout_test.jsonl --out-dir out/heldout
```

Chart generation from real eval output (never hardcode numbers into a chart script):
```bash
python -m pip install matplotlib
python scripts/plot_evaluation.py --results out/heldout/eval_results.json
```

There is no formal test suite; `python -m compileall` plus `validate_data.py` are the closest things to CI checks in this repo.

## Architecture

**Pipeline stages**, each stage's output is the next stage's input:

1. `scripts/download_edgar.py` — scrapes SEC EDGAR 10-K filings for a hardcoded `COMPANIES` ticker→CIK map, extracts the "Item 1A Risk Factors" section via regex/BeautifulSoup, writes `data/raw/{TICKER}_{date}.txt`.
2. `scripts/make_data_nvidia.py` — loads **Qwen2.5-7B-Instruct** (the labeling model, distinct from the 3B model being fine-tuned), chunks each raw file into 60–220 word excerpts, prompts the 7B model to emit a JSON label, and validates each label (`evidence` must be a verbatim substring of the excerpt — an anti-hallucination filter). Shuffles tickers with `seed=7` and reserves roughly 1/5 of companies (min 2) as held-out, writing `data/train_raw.jsonl` and `data/eval_raw.jsonl`. **Splitting is company-level**, not record-level — this is what guarantees no ticker leaks between train and held-out eval.
3. `scripts/validate_data.py` — re-validates schema/categories/severities on both raw files and asserts the train/eval ticker sets are disjoint (fails loudly if a company leaked).
4. `train.py` (`write_splits`/`stratified_split`) — reads `data/train_raw.jsonl`, dedupes exact-duplicate records, and does a **second, independent, record-level, category-stratified** 80/10/10 split (seed 42) into `data/nvidia/{train,valid,test}.jsonl`. This internal split is unrelated to the company-level held-out split — don't conflate "test.jsonl" (internal, record-level) with "heldout_test.jsonl" (held-out companies). Each record is converted to the chat schema via `chat_example()`: a fixed `SYSTEM_PROMPT`, a user turn with ticker+excerpt, and an assistant turn containing the canonical JSON label.
5. `scripts/prepare_heldout.py` — reuses `train.py`'s `parse_records`/`chat_example` to convert `data/eval_raw.jsonl` into the same chat schema, writing `data/nvidia/heldout_test.jsonl`.
6. `train.py --backend peft` (`run_peft`) — loads the base model in bf16/fp16 depending on GPU support, wraps it with a PEFT `LoraConfig` (`target_modules="all-linear"`), trains with HF `Trainer`, saves the adapter to `out/adapters/` and training metrics to `out/metrics.json`.
7. `evaluate.py` — loads the base model, generates on every test row, then loads the same base model again with the LoRA adapter applied (`PeftModel.from_pretrained`) and regenerates, so base and fine-tuned are compared on identical inputs in one run. Scores valid-JSON rate, category accuracy, category macro-F1 (computed manually in `macro_f1()`, not sklearn), and severity accuracy. Writes `eval_results.json` (includes full `per_example` detail with raw model output) and a Markdown table `eval_results.md`.
8. `scripts/plot_evaluation.py` — reads an `eval_results.json`, never hardcodes metric values, and renders a grouped bar chart (PNG + PDF) comparing base vs. fine-tuned.

**Shared contract:** `train.py`'s `SYSTEM_PROMPT`, `parse_records()`, and `chat_example()` are the single source of truth for the chat message schema and are imported directly by `scripts/prepare_heldout.py` — don't duplicate or fork this logic elsewhere.

**Two distinct train/test boundaries to keep straight:**
- Company-level: `data/train_raw.jsonl` vs `data/eval_raw.jsonl` (set at data-generation time in `make_data_nvidia.py`, enforced by `validate_data.py`).
- Record-level: `data/nvidia/{train,valid,test}.jsonl` (set inside `train.py` at training time from `train_raw.jsonl` only).

`scripts/format_data.py` and `scripts/make_data.py` are the older/MLX-era counterparts of the `*_nvidia.py` scripts and a simpler chat formatter — not part of the current NVIDIA pipeline's command sequence in README.md, but present for reference/Apple Silicon use.

## Repository structure and generated files

Git tracks directory skeletons with `.gitkeep` only. `data/*`, `out/*` (except `out/adapters/.gitkeep`), and `*.log` are gitignored — generated JSONL datasets, LoRA adapters, checkpoints, and training logs are never committed. **Do not create empty JSONL files or fake adapter files to fill out the tree** — if a script needs its input and the file is missing, that means an earlier pipeline stage hasn't been run yet, not that a placeholder should be created.

`results/` holds previously-captured eval JSON/Markdown/charts that *are* committed as record-of-experiment (e.g. `results/heldout/eval_results.json`) — these are historical snapshots, distinct from the gitignored live `out/` working directory.

## Conventions specific to this repo

- Never commit Hugging Face tokens, SEC API credentials, or real adapters/datasets.
- `scripts/download_edgar.py` has a hardcoded `USER_AGENT` with an email — SEC requires a properly identified User-Agent; update it before running rather than leaving the placeholder.
- When adding to the data/eval pipeline, preserve the company-level vs. record-level split distinction above; a change that lets held-out companies leak into the record-level split (or vice versa) invalidates the generalization claim the held-out eval exists to test.
- `evaluate.py` and `train.py` both support `--backend mlx` (Apple Silicon) and `--backend peft` (CUDA/NVIDIA) — default is `mlx`, so NVIDIA invocations must always pass `--backend peft` explicitly.
