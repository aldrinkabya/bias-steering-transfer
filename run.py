import argparse
import glob
import json
import os

from src.data import load_split
from src.evaluate import RUNS_DIR, VECTORS_DIR, evaluate
from src.models import PRECISIONS, load_model
from src.plan import load_experiment, plan
from src.selection import SELECTION_DIR, apply_rules, selection_path, setting
from src.steering import VECTORS, extract, vector_path

STAGES = ["extract", "sweep", "select", "test", "mmlu"]


def select(model, prec, exp):
    metas = []
    for path in glob.glob(os.path.join(RUNS_DIR, f"{model}_{prec}_*.json")):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["split"] == "dev":
            metas.append(meta)
    base = next(m for m in metas if m["cond"] == "base")
    rules = exp["selection"]
    out = {}
    for vector in VECTORS:
        cands = [m for m in metas if m["cond"] == "steer_native" and m["vector"] == vector]
        if prec != "fp16":
            layer, _ = setting(model, vector)
            cands = [c for c in cands if c["layer"] == layer]
        out[vector] = apply_rules(base, cands, rules["max_context_acc_drop"], rules["min_committed"])
        main = out[vector]["main"]
        print(f"{model}/{prec} {vector}: layer {main['layer']}, alpha {main['alpha']} "
              f"({out[vector]['n_within_limit']}/{out[vector]['n_candidates']} within the accuracy limit)")
    os.makedirs(SELECTION_DIR, exist_ok=True)
    with open(selection_path(model, prec), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run the experiment stages for one model at one precision.")
    parser.add_argument("--model", required=True, choices=["mistral", "llama", "qwen"])
    parser.add_argument("--prec", required=True, choices=PRECISIONS)
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    exp = load_experiment()
    batch_size = args.batch_size or exp["batch_size"].get(args.prec, exp["batch_size"]["default"])
    stages = [s for s in STAGES if s in args.stages]
    if args.prec != "fp16" and not os.path.exists(selection_path(args.model, "fp16")):
        raise SystemExit(f"run all stages for {args.model}/fp16 first")

    model = tokenizer = None
    if set(stages) - {"select"}:
        model, tokenizer = load_model(args.model, args.prec, args.gpu)

    for stage in stages:
        if stage == "extract":
            for vector in VECTORS:
                n_layers = model.config.num_hidden_layers
                if not all(os.path.exists(vector_path(VECTORS_DIR, args.model, args.prec, vector, L))
                           for L in range(n_layers)):
                    extract(model, tokenizer, load_split("extract"), vector, VECTORS_DIR, args.model, args.prec,
                            batch_size=max(1, batch_size // 2))
        elif stage == "select":
            select(args.model, args.prec, exp)
        else:
            runs = plan(args.model, args.prec, stage, exp)
            print(f"{args.model}/{args.prec} {stage}: {len(runs)} runs")
            for run in runs:
                evaluate(model, tokenizer, args.model, args.prec, run, batch_size)


if __name__ == "__main__":
    main()
