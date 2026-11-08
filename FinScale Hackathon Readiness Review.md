# FinScale Hackathon Readiness Review

Nov 8, 2026 · @Sai kiran

## Verdict

As it stands, the FinScale repo probably won't win, because it doesn't use the second node. The CIQ hackathon judges one thing above all: did the workload actually need two PGX (GB10) nodes? A single-GPU LoRA fine-tune of a 3B model fits easily on one node.

The skeleton is right: the data pipeline, the company-level held-out split, the anti-hallucination filter and the base-vs-fine-tuned eval. The fix is to change the scale and the story, not to start over.

## The core gaps

| # | Gap | Why it hurts |
| --- | --- | --- |
| 1 | Nothing is distributed | `train.py` runs HF `Trainer` on one GPU at batch size 1. No `torchrun`, FSDP, DeepSpeed, NCCL or Fuzzball job, so it doesn't address the challenge. |
| 2 | Workload too small for a cluster | 3B parameters, 12 companies, about 49 held-out records. Judges will ask why this needs two nodes. |
| 3 | Weak results | Held-out category accuracy fell from 55.1% to 53.1%. The valid-JSON gain (53% to 100%) is easy to dismiss, since constrained decoding gives 100% without fine-tuning. |
| 4 | Labels come from Qwen-7B | Accuracy measures agreement with a 7B teacher, not correctness. The student can't meaningfully surpass a weak teacher. |
| 5 | Not built for the hardware | GB10 is ARM64 (Grace CPU). The pinned requirements target x86 Lambda/RunPod images and could cost Saturday morning. |

## The reframe: big model teaches, small model serves

Use both nodes for something one node can't do. This is a real enterprise pattern for banks.

1. **Distributed labeling.** Replace the 7B labeler with Qwen2.5-72B-Instruct. In bf16 its weights are about 144 GB: too big for one GB10 (128 GB), but they fit across two. Serve it with vLLM using tensor or pipeline parallelism and label thousands of excerpts. This is the clearest proof of what the second node unlocks.
2. **Multi-node fine-tuning.** Train LoRA on a 14B student with FSDP across both nodes, keeping the 3B model as the baseline. Measure one-node versus two-node throughput.
3. **Cheap local serving.** The fine-tuned 14B approaches the 72B teacher's quality at a fraction of the cost and runs on one box. That is FinScale's pitch: big-model quality, private and affordable.

If 72B labeling is too slow on-site, pre-label most of the data before the event. Run a live labeling batch at the venue for the demo and the throughput numbers.

## Improvements, in priority order

### Must do (decides whether you place)

- [ ] **Make training distributed.** Launch with `torchrun --nnodes=2 --nproc-per-node=1` and enable FSDP via `TrainingArguments` (`fsdp="full_shard auto_wrap"`) or an `accelerate` config. In `run_peft`, drop `.to("cuda")` and let FSDP place the model.
- [ ] **Write Fuzzball workflows** for labeling, training and evaluation, and show them on a slide. CIQ is judging.
- [ ] **Build an aarch64 Apptainer container** from an NGC PyTorch image and freeze requirements inside it. Keep bf16 LoRA as the guaranteed path; don't depend on `bitsandbytes` on ARM/Blackwell.
- [ ] **Add a benchmark script.** Log tokens/sec, step time, GPU utilization and memory for 1 node vs 2 nodes, and compute scaling efficiency. This is your headline chart.
- [ ] **Expand the data before the event.** Grow `COMPANIES` from 12 to 100–300 tickers across sectors, 2–3 years each. Bring the raw files on disk; venue Wi-Fi and SEC rate limits are risks.

### Should do (fixes the weak results)

- [ ] **Human-verified gold set.** Hand-label 150–200 held-out excerpts this week. Report accuracy for base 3B, fine-tuned 14B and the 72B teacher against human labels.
- [ ] **Fix hyperparameters.** Learning rate 1e-5 is low for LoRA; try 1e-4 to 2e-4, r=16–32, 2–3 epochs, and a larger effective batch via gradient accumulation.
- [ ] **Train only on assistant tokens.** Mask prompt tokens out of the loss so learning isn't spent on the system prompt and excerpt.
- [ ] **Handle class imbalance.** Report a per-category confusion matrix; upsample rare categories or add weighted F1.
- [ ] **Report uncertainty.** Add bootstrap confidence intervals to `evaluate.py`.

### Nice to have (polish)

- [ ] **Serve with vLLM LoRA hot-swap**, so one base model serves per-customer adapters. This maps to FinScale's customer-specific fine-tuning.
- [ ] **Use guided JSON decoding** at inference and say so openly. Lead with category and severity gains instead of JSON validity.
- [ ] **Live dashboard** showing both nodes' GPU utilization, the loss curve, and excerpt vs base output vs fine-tuned output vs verified label.

## What the pitch should claim

The pitch rests on three claims the current repo can't make yet, all reachable in five days if the plan is locked now:

- A 72B teacher that doesn't fit on one node, served across two.
- A 14B student trained across two nodes with X% scaling efficiency.
- The student reaches Y% of the teacher's accuracy on human-verified held-out companies, running privately on one box.

Next step: write the FSDP changes to `run_peft`, a two-node `torchrun` launcher, a Fuzzball workflow and the benchmark script, working from the actual `train.py`.
