import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm

CATEGORIES = [
    "market", "credit", "liquidity", "operational",
    "regulatory_legal", "cybersecurity", "supply_chain", "other"
]
SEVERITIES = ["low", "medium", "high"]

SYSTEM = (
    "You are a financial risk analyst. Classify the risk disclosure excerpt. "
    "Respond with ONLY a JSON object with keys: "
    '"category" (one of ' + ", ".join(CATEGORIES) + '), '
    '"severity" (one of low, medium, high), '
    '"evidence" (a short verbatim quote from the excerpt, max 25 words), '
    '"explanation" (1-2 sentences justifying the label, grounded in evidence).'
)

CHUNKS_PER_FILE = 25
MIN_WORDS, MAX_WORDS = 60, 220
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def chunk_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, current = [], []

    for paragraph in paragraphs:
        if len(paragraph.split()) > MAX_WORDS:
            if current:
                words = sum(len(x.split()) for x in current)
                if MIN_WORDS <= words <= MAX_WORDS:
                    chunks.append(" ".join(current))
                current = []
            continue

        current.append(paragraph)
        words = sum(len(x.split()) for x in current)

        if words >= MIN_WORDS:
            if words <= MAX_WORDS:
                chunks.append(" ".join(current))
            current = []

    return chunks


def parse_label(raw, excerpt):
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None

    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    category = result.get("category")
    severity = result.get("severity")
    evidence = result.get("evidence", "").strip()
    explanation = result.get("explanation", "").strip()

    if category not in CATEGORIES or severity not in SEVERITIES:
        return None
    if not evidence or not explanation:
        return None

    normalize = lambda s: re.sub(r"\s+", " ", s.lower()).strip(" .\"'")
    if normalize(evidence) not in normalize(excerpt):
        return None

    if len(evidence.split()) > 25:
        return None

    return {
        "category": category,
        "severity": severity,
        "evidence": evidence,
        "explanation": explanation
    }


class LocalLabeler:
    """Labels excerpts with a HF Transformers model loaded in-process (the original path).

    Fine for a 7B teacher on one GPU; a 72B teacher won't fit in one GB10's memory this way
    — use VLLMLabeler against a vLLM server instead (see --labeler-endpoint).
    """

    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading local labeling model:", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.torch = torch

    def label(self, excerpt):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Excerpt:\n{excerpt}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=250, do_sample=False,
                                          pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class VLLMLabeler:
    """Labels excerpts via an OpenAI-compatible vLLM server (e.g. Qwen2.5-72B-Instruct served
    with tensor/pipeline parallelism across both GB10 nodes). This is what actually needs the
    second node: the 72B teacher's weights (~144GB in bf16) don't fit on one 128GB GB10.
    """

    def __init__(self, endpoint, model_id):
        import requests
        self.requests = requests
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id

    def label(self, excerpt):
        resp = self.requests.post(
            f"{self.endpoint}/chat/completions",
            json={
                "model": self.model_id,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Excerpt:\n{excerpt}"}],
                "max_tokens": 250,
                "temperature": 0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeler-endpoint", default=None,
                    help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1) for a vLLM-served teacher; "
                         "omit to load --labeler-model locally with Transformers instead")
    ap.add_argument("--labeler-model", default=DEFAULT_LOCAL_MODEL,
                    help="model id; Qwen2.5-7B-Instruct fits one GPU locally, Qwen2.5-72B-Instruct needs --labeler-endpoint")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    files = sorted(Path("data/raw").glob("*.txt"))
    if not files:
        raise SystemExit(
            "No SEC text files found in data/raw. "
            "Run python scripts/download_edgar.py first."
        )

    if args.labeler_endpoint:
        print(f"Labeling via vLLM endpoint {args.labeler_endpoint} with model {args.labeler_model}")
        labeler = VLLMLabeler(args.labeler_endpoint, args.labeler_model)
    else:
        labeler = LocalLabeler(args.labeler_model)

    by_company = {}
    for file in files:
        company = file.stem.split("_")[0]
        by_company.setdefault(company, []).append(file)

    tickers = sorted(by_company)
    random.shuffle(tickers)

    n_eval = max(2, len(tickers) // 5)
    eval_tickers = set(tickers[:n_eval])

    print("Held-out evaluation companies:", sorted(eval_tickers))

    rows = {"train": [], "eval": []}

    for ticker, company_files in by_company.items():
        split = "eval" if ticker in eval_tickers else "train"

        for file in company_files:
            chunks = chunk_text(file.read_text(errors="replace"))
            random.shuffle(chunks)

            for excerpt in tqdm(chunks[:CHUNKS_PER_FILE], desc=file.name):
                raw = labeler.label(excerpt)
                label = parse_label(raw, excerpt)
                if label is None:
                    continue

                rows[split].append({
                    "ticker": ticker,
                    "excerpt": excerpt,
                    "label": label
                })

    Path("data").mkdir(exist_ok=True)

    for split, items in rows.items():
        output_path = Path(f"data/{split}_raw.jsonl")
        with output_path.open("w") as output_file:
            for item in items:
                output_file.write(json.dumps(item) + "\n")

        print(f"{split}: {len(items)} records -> {output_path}")

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
