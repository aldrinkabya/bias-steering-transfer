import datetime
import importlib.metadata
import json
import os
import time

import pandas as pd
import torch

from src.data import load_split, permutations
from src.metrics import bbq_metrics
from src.mmlu import load_questions, mmlu_prompt
from src.plan import run_tag
from src.scoring import aggregate, bbq_prompt, letter_probs, letter_tokens
from src.steering import steer, steering_vector

RUNS_DIR = "results/runs"
VECTORS_DIR = "results/vectors"
PACKAGES = ["torch", "transformers", "accelerate", "bitsandbytes", "gptqmodel", "hqq"]


def _versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _bbq(model, tokenizer, run, batch_size):
    items = load_split(run["split"])
    perms = [permutations(it) for it in items]
    prompts = [bbq_prompt(it, p["options"], tokenizer, debias=run["cond"] == "prompt") for it, ps in zip(items, perms)
               for p in ps]
    letters = letter_tokens(tokenizer, bbq_prompt(items[0], [items[0][f"ans{i}"] for i in range(3)], tokenizer))
    probs = letter_probs(model, tokenizer, prompts, letters, batch_size)
    records = [aggregate(it, ps, probs[3 * i:3 * i + 3]) for i, (it, ps) in enumerate(zip(items, perms))]
    return records, bbq_metrics(records)


def _mmlu(model, tokenizer, batch_size):
    questions = load_questions()
    prompts = [mmlu_prompt(q, tokenizer) for q in questions]
    probs = letter_probs(model, tokenizer, prompts, letter_tokens(tokenizer, prompts[0], n_letters=4), batch_size)
    preds = probs.argmax(dim=-1).tolist()
    records = [{"mmlu_id": q["mmlu_id"], "subject": q["subject"], "answer": q["answer"], "pred": p,
                "is_correct": p == q["answer"]} for q, p in zip(questions, preds)]
    return records, {"mmlu_acc": sum(r["is_correct"] for r in records) / len(records)}


def evaluate(model, tokenizer, name, prec, run, batch_size=16, out_dir=RUNS_DIR, vectors_dir=VECTORS_DIR):
    tag = run_tag(name, prec, **run)
    csv_path, meta_path = (os.path.join(out_dir, tag + ext) for ext in (".csv", ".json"))
    if os.path.exists(csv_path) and os.path.exists(meta_path):
        return
    os.makedirs(out_dir, exist_ok=True)
    vec = None
    if run["cond"] in ("steer_native", "steer_transfer", "random"):
        vec = steering_vector(run["cond"], name, prec, run["vector"], run["layer"], run["alpha"], vectors_dir, run["seed"])

    start = time.time()
    with steer(model, run["layer"], vec):
        records, metrics = _mmlu(model, tokenizer, batch_size) if run["split"] == "mmlu" else \
            _bbq(model, tokenizer, run, batch_size)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    meta = {"tag": tag, "model": name, "prec": prec, **run, "metrics": metrics,
            "steering_norm": float(torch.linalg.vector_norm(vec)) if vec is not None else None,
            "seconds": round(time.time() - start, 1), "versions": _versions(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{tag}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)))
