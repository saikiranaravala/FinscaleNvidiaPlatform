#!/usr/bin/env python3
"""Prepare chat-formatted FinScale splits and fine-tune with MLX or PEFT.

Run from the project root. MLX is the Apple-Silicon default; --backend peft
selects a Transformers/PEFT training path for CUDA/NVIDIA development.
"""
from __future__ import annotations
import argparse, collections, json, math, os, random, re, subprocess, sys, time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a financial-risk analyst. Classify the provided filing excerpt. "
    "Return exactly one JSON object and no markdown, with the keys category, "
    "severity, evidence, explanation. category must be one of: market, credit, "
    "liquidity, operational, regulatory_legal, cybersecurity, supply_chain, other. "
    "severity must be low, medium, or high. evidence must be a verbatim span "
    "from the excerpt; explanation should briefly justify the classification."
)


def parse_records(path: Path):
    records, errors = [], []
    for n, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: {e.msg}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {n}: expected JSON object")
            continue
        if not all(k in obj for k in ("ticker", "excerpt", "label")):
            errors.append(f"line {n}: missing one of ticker/excerpt/label")
            continue
        if not isinstance(obj["label"], dict) or not isinstance(obj["excerpt"], str) or not obj["excerpt"].strip():
            errors.append(f"line {n}: invalid label or empty excerpt")
            continue
        label = obj["label"]
        if not isinstance(label.get("category"), str) or not isinstance(label.get("severity"), str):
            errors.append(f"line {n}: label requires string category and severity")
            continue
        records.append(obj)
    return records, errors


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def chat_example(record):
    user = f"Ticker: {record['ticker']}\nFiling excerpt:\n{record['excerpt']}"
    target = {k: record["label"].get(k) for k in ("category", "severity", "evidence", "explanation")}
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}


def stratified_split(records, seed=42):
    """Deterministic per-category split; exact duplicates are removed beforehand."""
    groups = collections.defaultdict(list)
    for r in records:
        groups[str(r["label"]["category"])].append(r)
    rng = random.Random(seed)
    parts = {"train": [], "valid": [], "test": []}
    for category in sorted(groups):
        group = groups[category][:]
        rng.shuffle(group)
        n = len(group)
        if n == 1:
            counts = [1, 0, 0]
        elif n == 2:
            counts = [1, 0, 1]
        else:
            # Largest-remainder allocation, with at least one item per split for n>=3.
            raw = [0.8*n, 0.1*n, 0.1*n]
            counts = [math.floor(v) for v in raw]
            for ix in sorted(range(3), key=lambda j: (raw[j]-counts[j], -j), reverse=True)[:n-sum(counts)]:
                counts[ix] += 1
            for ix in range(3):
                if counts[ix] == 0:
                    donor = max((j for j in range(3) if counts[j] > 1), key=lambda j: counts[j])
                    counts[donor] -= 1
                    counts[ix] += 1
        a, b = counts[0], counts[0]+counts[1]
        for name, subset in zip(parts, (group[:a], group[a:b], group[b:])):
            parts[name].extend(subset)
    for name in parts:
        rng.shuffle(parts[name])
    return parts


def oversample_minority(train_rows, seed, cap_multiplier=3):
    """Duplicate under-represented categories in the train split toward the majority count.

    Each category is duplicated at most `cap_multiplier`x its own size, so a single rare
    category can't be blown up to dominate the dataset just because another category is huge.
    """
    groups = collections.defaultdict(list)
    for r in train_rows:
        groups[str(r["label"]["category"])].append(r)
    if not groups:
        return train_rows
    majority = max(len(v) for v in groups.values())
    rng = random.Random(seed)
    out = []
    for rows in groups.values():
        target = min(majority, len(rows) * cap_multiplier)
        out.extend(rows)
        if len(rows) < target:
            out.extend(rng.choices(rows, k=target - len(rows)))
    rng.shuffle(out)
    return out


def write_splits(source_path: Path, data_dir: Path, seed: int, oversample: bool = False):
    parsed, errors = parse_records(source_path)
    seen, deduped = set(), []
    duplicates = 0
    for row in parsed:
        key = canonical(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key); deduped.append(row)
    if len(deduped) < 3:
        raise ValueError(f"Need at least 3 unique valid records to create splits; found {len(deduped)}")
    splits = stratified_split(deduped, seed)
    if oversample:
        splits["train"] = oversample_minority(splits["train"], seed + 1)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (data_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(chat_example(r), ensure_ascii=False) + "\n")
    return {"records_read": len(parsed)+len(errors), "valid_records": len(parsed), "unique_records": len(deduped),
            "exact_duplicates_removed": duplicates, "invalid_lines": errors, "train_oversampled": oversample,
            "split_counts": {k: len(v) for k,v in splits.items()},
            "split_categories": {k: dict(collections.Counter(str(r['label']['category']) for r in v)) for k,v in splits.items()}}


def run_mlx(args, data_dir, out_dir, prep):
    adapter = out_dir / "adapters"
    adapter.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--model", args.model,
           "--train", "--data", str(data_dir), "--adapter-path", str(adapter),
           "--iters", str(args.iters), "--batch-size", str(args.batch_size),
           "--num-layers", str(args.lora_layers), "--learning-rate", str(args.learning_rate)]
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if args.max_seq_length:
        cmd += ["--max-seq-length", str(args.max_seq_length)]
    log_path = out_dir / "train.log"
    start = time.perf_counter(); step_times=[]; observed_tps=[]; peak=[]
    step_re = re.compile(r"(?:Iter(?:ation)?\s*)?(\d+)\s*[:|].*?(?:time|sec(?:onds)?)\s*[=:]?\s*([0-9.]+)\s*s", re.I)
    tps_re = re.compile(r"([0-9.]+)\s*tokens?/s", re.I)
    mem_re = re.compile(r"(?:peak\s+memory|memory)\s*[:=]\s*([0-9.]+)\s*(GB|GiB|MB|MiB)", re.I)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: "+" ".join(cmd)+"\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="")
            log.write(line); log.flush()
            m=step_re.search(line)
            if m: step_times.append(float(m.group(2)))
            m=tps_re.search(line)
            if m: observed_tps.append(float(m.group(1)))
            m=mem_re.search(line)
            if m:
                v=float(m.group(1)); peak.append(v*(1024 if m.group(2).lower().startswith("m") else 1))
        rc=proc.wait()
    elapsed=time.perf_counter()-start
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)
    # Report measured console metrics where emitted. The fallback step time is total wall time / requested iterations;
    # token throughput is omitted unless MLX prints it (no fabricated token counter).
    metrics={**prep, "backend":"mlx", "model":args.model, "iterations_requested":args.iters,
             "batch_size":args.batch_size, "lora_layers":args.lora_layers, "learning_rate":args.learning_rate,
             "wall_time_seconds":elapsed, "step_time_seconds_observed":step_times,
             "mean_step_time_seconds":sum(step_times)/len(step_times) if step_times else elapsed/args.iters,
             "step_time_source":"mlx log" if step_times else "wall time / requested iterations (approximation)",
             "tokens_per_second_observed":observed_tps[-1] if observed_tps else None,
             "peak_memory_gb_observed":peak[-1] if peak else None,
             "peak_memory_source":"mlx log" if peak else None,
             "training_log":str(log_path), "adapter_path":str(adapter), "command":cmd}
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")


class CausalLMCollator:
    """Pads input_ids/attention_mask with the pad token and labels with -100.

    Unlike DataCollatorForLanguageModeling(mlm=False), this does not overwrite
    pre-masked labels with a straight copy of input_ids, so prompt tokens stay
    excluded from the loss (see tokenize() in run_peft).
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)}


def run_peft(args, data_dir, out_dir, prep):
    """Optional portable NVIDIA route; lazily imports the torch/Transformers stack.

    Supports single-GPU and multi-node FSDP (launch via scripts/launch_distributed.sh
    with --distributed so this process doesn't fight FSDP's own device placement).
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    model_id=args.model
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("PEFT backend requires a CUDA-enabled NVIDIA GPU. Check nvidia-smi and PyTorch installation.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if not args.distributed:
        model.to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model=get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                           target_modules="all-linear", task_type="CAUSAL_LM"))

    max_len = args.max_seq_length or 4096
    def tokenize(ex):
        # Mask the system+user prompt out of the loss so training signal is spent only on the
        # assistant's JSON label, not on re-learning the fixed instructions and filing excerpt.
        input_ids_list, labels_list, attn_list = [], [], []
        for messages in ex["messages"]:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            full_ids = tokenizer(full_text, truncation=True, max_length=max_len)["input_ids"]
            prompt_len = min(len(tokenizer(prompt_text, truncation=True, max_length=max_len)["input_ids"]), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            input_ids_list.append(full_ids); labels_list.append(labels); attn_list.append([1] * len(full_ids))
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attn_list}

    ds=load_dataset("json", data_files={k:str(data_dir/f"{k}.jsonl") for k in ("train","valid")})
    ds=ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

    fsdp_kwargs = {}
    if args.distributed:
        fsdp_kwargs = {"fsdp": "full_shard auto_wrap",
                       "fsdp_config": {"transformer_layer_cls_to_wrap": args.fsdp_layer_cls,
                                        "backward_prefetch": "backward_pre"}}
    training_args=TrainingArguments(output_dir=str(out_dir/"peft_checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, logging_steps=1,
        gradient_checkpointing=True, report_to="none", logging_dir=str(out_dir/"tb"), **fsdp_kwargs)
    trainer=Trainer(model=model,args=training_args,train_dataset=ds["train"],eval_dataset=ds["valid"],
                    data_collator=CausalLMCollator(tokenizer.pad_token_id))
    start=time.perf_counter(); torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    result=trainer.train(); elapsed=time.perf_counter()-start
    adapter=out_dir/"adapters"
    if trainer.is_world_process_zero():
        adapter.mkdir(parents=True,exist_ok=True); model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
        metrics={**prep,"backend":"peft","model":model_id,"wall_time_seconds":elapsed,
           "distributed":args.distributed,"world_size":int(os.environ.get("WORLD_SIZE",1)),
           "train_metrics":result.metrics,"mean_step_time_seconds":elapsed/max(1,result.global_step),
           "tokens_per_second_observed":result.metrics.get("train_tokens_per_second"),
           "peak_memory_gb_observed":torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None,
           "adapter_path":str(adapter)}
        (out_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,default=str)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,default=Path("data/train_raw.jsonl"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/nvidia"))
    #ap.add_argument("--data-dir",type=Path,default=Path("data/mlx"))
    ap.add_argument("--out-dir",type=Path,default=Path("out"))
    ap.add_argument("--backend",choices=("mlx","peft"),default="mlx",help="mlx for Apple Silicon; peft for PyTorch/PEFT on NVIDIA")
    ap.add_argument("--model",default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--iters",type=int,default=100)
    ap.add_argument("--epochs",type=float,default=3); ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--grad-accum-steps",type=int,default=4,help="PEFT only: effective batch = batch-size * grad-accum-steps * world-size")
    ap.add_argument("--lora-layers",type=int,default=8); ap.add_argument("--lora-r",type=int,default=16)
    ap.add_argument("--lora-alpha",type=int,default=32); ap.add_argument("--lora-dropout",type=float,default=0.05)
    ap.add_argument("--learning-rate",type=float,default=2e-4); ap.add_argument("--max-seq-length",type=int,default=4096)
    ap.add_argument("--eval-steps",type=int,default=25); ap.add_argument("--grad-checkpoint",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--oversample-train",action=argparse.BooleanOptionalAction,default=False,
                     help="duplicate under-represented categories in the train split to reduce class imbalance")
    ap.add_argument("--distributed",action=argparse.BooleanOptionalAction,default=False,
                     help="PEFT only: enable FSDP full-shard training; launch via scripts/launch_distributed.sh")
    ap.add_argument("--fsdp-layer-cls",default="Qwen2DecoderLayer",
                     help="transformer decoder layer class to auto-wrap under FSDP; matches the --model family")
    args=ap.parse_args()
    if not args.input.exists(): ap.error(f"input not found: {args.input}")
    records, errors=parse_records(args.input)
    # Split preparation re-reads source intentionally through helper for uniform diagnostics.
    prep=write_splits(args.input,args.data_dir,args.seed,oversample=args.oversample_train)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/"split_report.json").write_text(json.dumps(prep,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(prep,indent=2))
    if prep["invalid_lines"]: print(f"Skipped {len(prep['invalid_lines'])} malformed/invalid lines; see out/split_report.json",file=sys.stderr)
    if args.backend=="mlx": run_mlx(args,args.data_dir,args.out_dir,prep)
    else: run_peft(args,args.data_dir,args.out_dir,prep)

if __name__=="__main__": main()
