# FinScale Pipeline Sequence

End-to-end pipeline, stage by stage, with what each step reads/writes.

## 1. Scrape SEC filings — `scripts/download_edgar.py`
- Hardcoded `COMPANIES` dict maps tickers → CIK numbers (AAPL, MSFT, JPM, XOM, PFE, TSLA, AMZN, GS, WMT, BA, NFLX, INTC).
- For each company, fetches up to 2 recent 10-K filings from SEC EDGAR's JSON API, downloads the filing HTML, and extracts the **"Item 1A Risk Factors"** section via regex (finds the longest span between "Item 1A" and the next "Item 1B"/"Item 2" heading).
- Skips sections under 5,000 chars (too short to be real).
- **Output:** `data/raw/{TICKER}_{date}.txt`

## 2. Auto-label with a bigger model — `scripts/make_data_nvidia.py`
- Loads **Qwen2.5-7B-Instruct** (a separate, larger "labeling" model — not the 3B model being fine-tuned).
- Chunks each raw text file into 60–220 word excerpts.
- Prompts the 7B model to classify each excerpt into JSON (`category`, `severity`, `evidence`, `explanation`).
- Validates each response: category/severity must be in the allowed sets, and critically, `evidence` must be a **verbatim substring of the excerpt** (anti-hallucination filter) — invalid labels are dropped.
- Shuffles tickers with `seed=7`, reserves ~1/5 of companies (min 2) as held-out **at the company level** — this is the key design point: held-out separation happens here, by ticker, not later by record.
- **Output:** `data/train_raw.jsonl` (train companies) and `data/eval_raw.jsonl` (held-out companies)

## 3. Validate — `scripts/validate_data.py`
- Re-checks schema (required fields, valid category/severity) on both files.
- Asserts the ticker sets in `train_raw.jsonl` and `eval_raw.jsonl` are **disjoint** — fails loudly (`SystemExit(1)`) if any company leaked into both.

## 4. Convert held-out set to chat format — `scripts/prepare_heldout.py`
- Imports `parse_records` and `chat_example` directly from `train.py` (shared schema, not duplicated).
- Converts `data/eval_raw.jsonl` → `data/nvidia/heldout_test.jsonl` in the same chat-message format `evaluate.py` expects.

## 5. Train — `train.py --backend peft`
- First does its **own, independent, second split**: reads `data/train_raw.jsonl`, dedupes exact duplicates, and does a **record-level**, category-stratified 80/10/10 split (`seed=42`) into `data/nvidia/{train,valid,test}.jsonl`. This split has nothing to do with the company-level held-out split from step 2 — it's purely for internal train/validation/test during fine-tuning.
- Each record becomes a chat example via `chat_example()`: fixed `SYSTEM_PROMPT` + user turn (ticker + excerpt) + assistant turn (JSON label).
- Loads Qwen2.5-3B-Instruct in bf16/fp16, wraps with a PEFT `LoraConfig` (`target_modules="all-linear"`), trains with HF `Trainer`.
- **Output:** `out/adapters/` (LoRA weights) and `out/metrics.json`

## 6. Evaluate — `evaluate.py --backend peft`
- Run twice, into separate `--out-dir`s, against two different test files:
  - **Internal test:** `--test data/nvidia/test.jsonl` → `out/internal/` (record-level holdout from step 5)
  - **Held-out company test:** `--test data/nvidia/heldout_test.jsonl` → `out/heldout/` (company-level holdout from step 2 — tests generalization to unseen companies)
- Within one run, it loads the base model, generates on every row, then applies the LoRA adapter to the same base model and regenerates — so base vs. fine-tuned are compared on identical inputs.
- Scores valid-JSON rate, category accuracy, category macro-F1 (hand-computed, not sklearn), severity accuracy.
- **Output:** `eval_results.json` (includes raw per-example model output) and `eval_results.md`

## 7. Visualize — `scripts/plot_evaluation.py`
- Reads an `eval_results.json` and renders a grouped bar chart (base vs. fine-tuned) as PNG + PDF — never hardcodes numbers, always reads from the actual JSON.

## Key invariant

There are two separate train/test boundaries — never conflate them:

- **Company-level** split (`train_raw.jsonl` vs `eval_raw.jsonl`, fixed in step 2): tests generalization to unseen companies.
- **Record-level** split (`train/valid/test.jsonl`, fixed in step 5): normal ML train/val/test hygiene within the training companies only.

## Hackathon additions (see README.md section 9 and `FinScale Hackathon Readiness Review.md`)

Optional stages that plug into the sequence above without changing it:

- **Step 2 can target a 72B teacher over vLLM** instead of a local 7B model
  (`make_data_nvidia.py --labeler-endpoint ... --labeler-model Qwen/Qwen2.5-72B-Instruct`) —
  this is the one piece of the pipeline that actually needs two nodes.
- **Step 5 (train.py) can run distributed** across two nodes via FSDP
  (`--distributed`, launched with `scripts/launch_distributed.sh`), masks prompt tokens out
  of the loss, and can oversample minority categories in the train split (`--oversample-train`).
- **A parallel gold-set track** (`sample_gold_set.py` → `label_gold_set.py` →
  `evaluate_against_gold.py`) scores base/fine-tuned/teacher against human-verified labels
  instead of the teacher's own labels — this is what turns "accuracy" into a correctness
  claim rather than a teacher-agreement claim.
- **A scaling-benchmark track** (`gpu_monitor.py` per node → `benchmark_scaling.py` →
  `plot_scaling.py`) compares a 1-node and a 2-node training run's `metrics.json` and GPU-
  utilization CSVs to produce the scaling-efficiency headline chart.
- **Serving**: `serve_vllm.py` hosts the fine-tuned model with hot-swappable per-customer
  LoRA adapters; `dashboard.py` is a stdlib-only live view of both nodes' GPU utilization,
  the loss curve, and a gold-vs-base-vs-fine-tuned comparison table for the demo.
- **Infra**: `containers/finscale-aarch64.def` (Apptainer, aarch64/GB10) and `fuzzball/*.yaml`
  (workflow templates) are unvalidated — written without access to a live cluster — adapt
  and validate against the actual Fuzzball CLI/build host before the event.
