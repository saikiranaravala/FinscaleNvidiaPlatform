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

Train single-GPU (also creates the internal train/valid/test split as a side effect; defaults are `--learning-rate 2e-4 --lora-r 16 --lora-alpha 32 --epochs 3`):
```bash
python -u train.py --backend peft \
  --input data/train_raw.jsonl --data-dir data/nvidia \
  --model Qwen/Qwen2.5-3B-Instruct --batch-size 1 --grad-accum-steps 4 \
  --max-seq-length 2048 --eval-steps 25 --out-dir out 2>&1 | tee out/train_peft.log
```

Train distributed (2 nodes, FSDP) — run the same command on both nodes, only `NODE_RANK` differs:
```bash
MASTER_ADDR=<node0-ip> NODE_RANK=0 bash scripts/launch_distributed.sh   # on node 0
MASTER_ADDR=<node0-ip> NODE_RANK=1 bash scripts/launch_distributed.sh   # on node 1
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

1. `scripts/download_edgar.py` — resolves a ~150-ticker, sector-diverse default list (or `--tickers-file`) to CIKs dynamically via SEC's own `company_tickers.json` map (cached at `data/.cik_map.json`), scrapes 10-K filings, extracts the "Item 1A Risk Factors" section via regex/BeautifulSoup, writes `data/raw/{TICKER}_{date}.txt`.
2. `scripts/make_data_nvidia.py` — labels chunked excerpts (60–220 words) with either a locally-loaded **Qwen2.5-7B-Instruct** (default) or, via `--labeler-endpoint`, a vLLM-served **Qwen2.5-72B-Instruct** teacher (`LocalLabeler` vs `VLLMLabeler`) — the 72B path is what actually needs two nodes, since its bf16 weights (~144GB) don't fit on one 128GB GB10. Validates each label (`evidence` must be a verbatim substring of the excerpt — an anti-hallucination filter). Shuffles tickers with `seed=7` and reserves roughly 1/5 of companies (min 2) as held-out, writing `data/train_raw.jsonl` and `data/eval_raw.jsonl`. **Splitting is company-level**, not record-level — this is what guarantees no ticker leaks between train and held-out eval.
3. `scripts/validate_data.py` — re-validates schema/categories/severities on both raw files and asserts the train/eval ticker sets are disjoint (fails loudly if a company leaked).
4. `train.py` (`write_splits`/`stratified_split`) — reads `data/train_raw.jsonl`, dedupes exact-duplicate records, optionally oversamples minority categories in the train split only (`--oversample-train`, `oversample_minority()`), and does a **second, independent, record-level, category-stratified** 80/10/10 split (seed 42) into `data/nvidia/{train,valid,test}.jsonl`. This internal split is unrelated to the company-level held-out split — don't conflate "test.jsonl" (internal, record-level) with "heldout_test.jsonl" (held-out companies). Each record is converted to the chat schema via `chat_example()`: a fixed `SYSTEM_PROMPT`, a user turn with ticker+excerpt, and an assistant turn containing the canonical JSON label.
5. `scripts/prepare_heldout.py` — reuses `train.py`'s `parse_records`/`chat_example` to convert `data/eval_raw.jsonl` into the same chat schema, writing `data/nvidia/heldout_test.jsonl`.
6. `train.py --backend peft` (`run_peft`) — loads the base model in bf16/fp16 depending on GPU support, wraps it with a PEFT `LoraConfig` (`target_modules="all-linear"`), trains with HF `Trainer`. Loss is masked to assistant tokens only (`tokenize()` sets prompt-token labels to -100; `CausalLMCollator` pads labels with -100 instead of HF's default same-as-input_ids collator). With `--distributed`, enables FSDP full-shard (`fsdp_config.transformer_layer_cls_to_wrap`) instead of calling `.to("cuda")` directly — launch both ranks identically via `scripts/launch_distributed.sh` (torchrun, `MASTER_ADDR`/`NODE_RANK` env vars). Saves the adapter to `out/adapters/` and training metrics (including `distributed`/`world_size`) to `out/metrics.json`, gated on `trainer.is_world_process_zero()` so multi-node ranks don't race on the write.
7. `evaluate.py` — `generate_predictions()` loads a model once (base or base+adapter via `PeftModel.from_pretrained`) and generates on every row; `score()` = `generate_predictions()` + `score_generations()`, the latter factored out so `scripts/evaluate_against_gold.py` can feed it predictions from a different source (a vLLM-served teacher over HTTP) and get the same metrics. Reports valid-JSON rate, category accuracy, category macro-F1 and weighted-F1, a per-category confusion matrix, and a 95% bootstrap CI (`bootstrap_ci()`) on every rate metric. `--guided-json` constrains decoding to the label JSON schema via `lm-format-enforcer` if installed (`build_guided_logits_processor()`), else falls back to unconstrained decoding with a warning. Writes `eval_results.json` (full `per_example` detail with raw model output) and `eval_results.md`.
8. `scripts/plot_evaluation.py` — reads an `eval_results.json`, never hardcodes metric values, and renders a grouped bar chart (PNG + PDF) comparing base vs. fine-tuned.
9. **Gold-set correctness check** (`scripts/sample_gold_set.py` → `scripts/label_gold_set.py` → `scripts/evaluate_against_gold.py`) — samples held-out excerpts, walks a human through an interactive (resumable) terminal review that starts from the teacher's label, then scores base/fine-tuned/teacher against the *human* labels instead of the teacher's — fixing the gap where category accuracy otherwise only measures agreement with a 7B/72B teacher, not correctness.
10. **Scaling benchmark** (`scripts/gpu_monitor.py` per node → `scripts/benchmark_scaling.py` → `scripts/plot_scaling.py`) — compares a 1-node `out/metrics.json` against a 2-node one, computing tokens/sec speedup and `scaling_efficiency_pct` against ideal linear scaling, plus mean GPU utilization from the per-node CSVs.
11. **Serving** — `scripts/serve_vllm.py --model <base> --lora customer_a=out/adapters_a --lora customer_b=out/adapters_b` serves one base model via vLLM with hot-swappable per-customer LoRA adapters selected per-request by `model` name (`scripts/query_vllm_client.py` is a minimal example client). `scripts/dashboard.py --log out/train_peft.log --gpu-csv ... --eval-results ...` is a stdlib-only (no extra deps) live page at `http://localhost:7777` polling every 2s for both nodes' GPU utilization, the loss curve, and a gold-vs-base-vs-fine-tuned comparison table.

See `PIPELINE.md` for the same pipeline stages laid out end-to-end with explicit reads/writes per step — useful as a second reference when `grep`-ing for which script produces a given file.

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
- `fuzzball/*.yaml` and `containers/finscale-aarch64.def` (with `requirements-aarch64.txt` — no `bitsandbytes`; bf16/fp16 LoRA only) are **unvalidated templates** (written without access to a live Fuzzball cluster or an aarch64 build host) — adapt and run `fuzzball job validate` / a real `apptainer build` before relying on them; don't treat their presence as proof they work.
- `evaluate.py --guided-json` and `requirements-optional.txt`'s `lm-format-enforcer` are optional; code must keep working (with a stderr warning) when that package isn't installed.
- `AWS_VALIDATION.md` is an alternative to the Fuzzball path for proving the 2-node FSDP distributed-training mechanics (`train.py --distributed` + `scripts/launch_distributed.sh`) work on real, physically separate GPUs — it provisions 2x `g5.2xlarge` on AWS EC2, trains a tiny model on `scripts/make_synthetic_smoke_data.py`'s deterministic synthetic data (so no SEC data or shared storage is needed — same seed+n produces a byte-identical file on each node), and tears the instances down afterward. It proves the distributed code path is correct, **not** real throughput/scaling-efficiency/accuracy numbers at hackathon scale — don't quote its numbers as results.
- See `FinScale Hackathon Readiness Review.md` for the rationale behind the multi-node/FSDP, gold-set, and benchmark additions, and README.md section 9 for the commands.
