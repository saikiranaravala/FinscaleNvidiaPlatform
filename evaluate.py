#!/usr/bin/env python3
"""Evaluate base and fine-tuned FinScale classifiers on data/mlx/test.jsonl."""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["market", "credit", "liquidity", "operational",
                                                  "regulatory_legal", "cybersecurity", "supply_chain", "other"]},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["category", "severity", "evidence", "explanation"],
}


def parse_json_output(text):
    text=(text or "").strip()
    try:
        obj=json.loads(text)
        return obj if isinstance(obj,dict) else None
    except Exception:
        # Accept a single JSON object surrounded by incidental prose/fences for field metrics,
        # but count it as invalid JSON because the model did not return a clean object.
        cleaned=re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        a,b=cleaned.find("{"),cleaned.rfind("}")
        if a>=0 and b>a:
            try:
                obj=json.loads(cleaned[a:b+1]); return obj if isinstance(obj,dict) else None
            except Exception: pass
    return None


def read_test(path):
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: row=json.loads(line)
        except Exception as e: raise ValueError(f"Invalid test JSONL line {n}: {e}")
        msgs=row.get("messages")
        if not isinstance(msgs,list) or len(msgs)<3: raise ValueError(f"Line {n}: expected chat messages list")
        try: gold=json.loads(msgs[-1]["content"])
        except Exception as e: raise ValueError(f"Line {n}: final assistant message must contain JSON label: {e}")
        rows.append((msgs, gold))
    return rows


def macro_f1(truth, pred, labels):
    vals=[]
    for label in labels:
        tp=sum(t==label and p==label for t,p in zip(truth,pred))
        fp=sum(t!=label and p==label for t,p in zip(truth,pred))
        fn=sum(t==label and p!=label for t,p in zip(truth,pred))
        denom=2*tp+fp+fn
        vals.append((2*tp/denom) if denom else 0.0)
    return sum(vals)/len(vals) if vals else 0.0


def weighted_f1(truth, pred, labels):
    """F1 per label weighted by the label's support in truth; downweights rare categories less
    than macro-F1 would, so it tracks what a class-imbalanced dataset actually rewards."""
    scores, weights = [], []
    for label in labels:
        tp=sum(t==label and p==label for t,p in zip(truth,pred))
        fp=sum(t!=label and p==label for t,p in zip(truth,pred))
        fn=sum(t==label and p!=label for t,p in zip(truth,pred))
        denom=2*tp+fp+fn
        scores.append((2*tp/denom) if denom else 0.0)
        weights.append(sum(t==label for t in truth))
    total=sum(weights)
    return sum(s*w for s,w in zip(scores,weights))/total if total else 0.0


def confusion_matrix(truth, pred, labels):
    matrix={t:{p:0 for p in labels} for t in labels}
    for t,p in zip(truth,pred):
        if t in matrix:
            matrix[t][p]=matrix[t].get(p,0)+1
    return matrix


def bootstrap_ci(values, metric_fn, n_resamples=300, alpha=0.05, seed=0):
    """Nonparametric bootstrap: resample `values` with replacement n_resamples times and take the
    alpha/2 and 1-alpha/2 percentiles of metric_fn over the resamples. Returns (low, point, high)."""
    n=len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point=metric_fn(values)
    rng=random.Random(seed)
    samples=sorted(metric_fn([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    lo=samples[int((alpha/2)*n_resamples)]
    hi=samples[min(n_resamples-1, int((1-alpha/2)*n_resamples))]
    return (lo, point, hi)


def build_guided_logits_processor(tokenizer):
    """Optional grammar-constrained JSON decoding via lm-format-enforcer, if installed.
    Returns None (unconstrained decoding) when the optional dependency is missing."""
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    except ImportError:
        return None
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(LABEL_SCHEMA))


def generate_predictions(rows, backend, model_id, adapter, max_tokens, guided_json=False):
    """Load the model once and generate raw completions for every (messages, gold) row.

    Shared by evaluate.py's base-vs-fine-tuned scoring and scripts/evaluate_against_gold.py,
    so there is exactly one code path that talks to mlx_lm / transformers+peft.
    """
    if backend=="mlx":
        from mlx_lm import load, generate
        model,tokenizer=load(model_id,adapter_path=str(adapter) if adapter else None)
        processor=None
        if guided_json:
            print("guided JSON decoding is not supported on the mlx backend; decoding unconstrained", file=sys.stderr)
    else:
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from peft import PeftModel
        tokenizer=AutoTokenizer.from_pretrained(model_id)
        model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto",device_map="auto")
        if adapter: model=PeftModel.from_pretrained(model,str(adapter))
        processor=build_guided_logits_processor(tokenizer) if guided_json else None
        if guided_json and processor is None:
            print("guided JSON decoding requested but lm-format-enforcer is not installed "
                  "(pip install lm-format-enforcer); decoding unconstrained", file=sys.stderr)
    results=[]
    for msgs,gold in rows:
        prompt=tokenizer.apply_chat_template(msgs[:-1],tokenize=False,add_generation_prompt=True)
        if backend=="mlx":
            raw=generate(model,tokenizer,prompt=prompt,max_tokens=max_tokens)
        else:
            import torch
            inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
            gen_kwargs=dict(max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            if processor is not None: gen_kwargs["prefix_allowed_tokens_fn"]=processor
            with torch.inference_mode(): out=model.generate(**inputs,**gen_kwargs)
            raw=tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        results.append((gold, raw))
    return results


def score_generations(generations, n_resamples=300):
    """Compute all metrics from pre-generated (gold, raw_output) pairs.

    Factored out of score() so a caller with its own generation path (e.g.
    scripts/evaluate_against_gold.py calling a vLLM-served teacher over HTTP) can still get
    the same confusion-matrix/bootstrap-CI metrics without going through generate_predictions.
    """
    cat_true=[]; cat_pred=[]; sev_true=[]; sev_pred=[]; detail=[]
    for gold, raw in generations:
        pred=parse_json_output(raw)
        try:
            candidate=json.loads((raw or "").strip())
            clean_valid=isinstance(candidate,dict)
        except Exception: clean_valid=False
        cat_true.append(gold.get("category")); sev_true.append(gold.get("severity"))
        cat_pred.append(pred.get("category") if pred else None); sev_pred.append(pred.get("severity") if pred else None)
        detail.append({"gold_category":gold.get("category"),"pred_category":cat_pred[-1],"gold_severity":gold.get("severity"),
                        "pred_severity":sev_pred[-1],"valid_json":clean_valid,"raw_output":raw})
    cats=sorted(set(cat_true)); sevs=sorted(set(sev_true))
    valid_flags=[d["valid_json"] for d in detail]
    cat_pairs=list(zip(cat_true,cat_pred)); sev_pairs=list(zip(sev_true,sev_pred))
    acc_lo,acc_pt,acc_hi=bootstrap_ci(cat_pairs, lambda ps: sum(a==b for a,b in ps)/len(ps) if ps else 0.0, n_resamples)
    sev_lo,sev_pt,sev_hi=bootstrap_ci(sev_pairs, lambda ps: sum(a==b for a,b in ps)/len(ps) if ps else 0.0, n_resamples)
    f1_lo,f1_pt,f1_hi=bootstrap_ci(cat_pairs, lambda ps: macro_f1([a for a,_ in ps],[b for _,b in ps],cats), n_resamples)
    valid_lo,valid_pt,valid_hi=bootstrap_ci(valid_flags, lambda vs: sum(vs)/len(vs) if vs else 0.0, n_resamples)
    return {"examples":len(generations),
       "valid_json_count":sum(valid_flags),"valid_json_rate":valid_pt,"valid_json_rate_ci":[valid_lo,valid_hi],
       "category_accuracy":acc_pt,"category_accuracy_ci":[acc_lo,acc_hi],
       "category_macro_f1":f1_pt,"category_macro_f1_ci":[f1_lo,f1_hi],
       "category_weighted_f1":weighted_f1(cat_true,cat_pred,cats),
       "category_confusion_matrix":confusion_matrix(cat_true,cat_pred,cats),
       "severity_accuracy":sev_pt,"severity_accuracy_ci":[sev_lo,sev_hi],
       "category_labels_scored":cats,"per_example":detail}


def score(rows, backend, model_id, adapter, max_tokens, guided_json=False, n_resamples=300):
    generations=generate_predictions(rows, backend, model_id, adapter, max_tokens, guided_json)
    return score_generations(generations, n_resamples)


def render_markdown(base, tuned):
    cols=["Model","Valid JSON","Category accuracy","Category macro-F1","Category weighted-F1","Severity accuracy"]
    lines=["| " + " | ".join(cols) + " |","|"+"|".join(["---"]*len(cols))+"|"]
    for name,m in (("Base",base),("Fine-tuned",tuned)):
        lines.append(
            f"| {name} "
            f"| {m['valid_json_rate']:.1%} (95% CI {m['valid_json_rate_ci'][0]:.1%}-{m['valid_json_rate_ci'][1]:.1%}) "
            f"| {m['category_accuracy']:.1%} (95% CI {m['category_accuracy_ci'][0]:.1%}-{m['category_accuracy_ci'][1]:.1%}) "
            f"| {m['category_macro_f1']:.3f} (95% CI {m['category_macro_f1_ci'][0]:.3f}-{m['category_macro_f1_ci'][1]:.3f}) "
            f"| {m['category_weighted_f1']:.3f} "
            f"| {m['severity_accuracy']:.1%} (95% CI {m['severity_accuracy_ci'][0]:.1%}-{m['severity_accuracy_ci'][1]:.1%}) |")
    md="\n".join(lines)+"\n"
    for name, m in (("Base", base), ("Fine-tuned", tuned)):
        labels=m["category_labels_scored"]
        if not labels: continue
        md += f"\n### {name} category confusion matrix (rows = gold, columns = predicted)\n\n"
        md += "| gold \\ pred | " + " | ".join(labels) + " |\n"
        md += "|" + "---|" * (len(labels)+1) + "\n"
        for t in labels:
            row=m["category_confusion_matrix"].get(t, {})
            md += "| " + t + " | " + " | ".join(str(row.get(p,0)) for p in labels) + " |\n"
    return md


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test",type=Path,default=Path("data/mlx/test.jsonl")); ap.add_argument("--backend",choices=("mlx","peft"),default="mlx")
    ap.add_argument("--model",default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--adapter",type=Path,default=Path("out/adapters"))
    ap.add_argument("--out-dir",type=Path,default=Path("out")); ap.add_argument("--max-tokens",type=int,default=384)
    ap.add_argument("--guided-json",action="store_true",
                     help="constrain generation to the label JSON schema via lm-format-enforcer, if installed (peft backend only)")
    ap.add_argument("--bootstrap-resamples",type=int,default=300)
    args=ap.parse_args()
    if not args.test.exists(): ap.error(f"test split not found: {args.test}; run train.py first")
    rows=read_test(args.test)
    if not rows: ap.error("test split has no examples")
    base=score(rows,args.backend,args.model,None,args.max_tokens,args.guided_json,args.bootstrap_resamples)
    tuned=score(rows,args.backend,args.model,args.adapter,args.max_tokens,args.guided_json,args.bootstrap_resamples)
    results={"backend":args.backend,"model":args.model,"adapter":str(args.adapter),"test_file":str(args.test),
             "guided_json":args.guided_json,"base":base,"fine_tuned":tuned}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/"eval_results.json").write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    md=render_markdown(base,tuned)
    (args.out_dir/"eval_results.md").write_text(md,encoding="utf-8")
    print(md,end="")

if __name__=="__main__": main()
