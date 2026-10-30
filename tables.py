import argparse
import json
import os

import numpy as np
import pandas as pd

from src.plan import load_experiment

MODELS = {"mistral": "Mistral-7B", "llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B"}
VERSIONS = {"fp16": "FP16", "int8": "INT8", "nf4": "NF4", "gptq4": "GPTQ-4", "hqq4": "HQQ-4", "hqq3": "HQQ-3",
            "hqq2": "HQQ-2"}
QUANTIZED = list(VERSIONS)[1:]
SHARES = ["p_st", "p_oth", "p_unk", "bias_score", "committed_st", "context_acc"]
HEADER = ["Stereo.", "Other", "Unknown", "Bias score", "Committed st.", "Context acc."]


def pp(x):
    s = f"{100 * x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def setting(sel_dir, model, vector):
    with open(os.path.join(sel_dir, f"{model}_fp16.json"), encoding="utf-8") as f:
        s = json.load(f)[vector]
    s = s["committed"] if vector == "cs" else s["main"]
    return s["layer"], s["alpha"]


def at(df, layer, alpha):
    return df[(df.layer == layer) & np.isclose(df.alpha, alpha)]


def quantization_table(con, changes):
    rows = []
    for model in MODELS:
        for prec in QUANTIZED:
            r = con[(con.kind == "quantization") & (con.model == model) & (con.prec == prec)]
            if r.empty:
                continue
            r = r.iloc[0]
            ch = changes[(changes.model == model) & (changes.prec == prec)].iloc[0]
            rows.append([MODELS[model], VERSIONS[prec], *(pp(r[f"d_{m}"]) for m in SHARES), f"{100 * ch.changed:.1f}"])
    return markdown(["Model", "Version", *HEADER, "Answers changed (%)"], rows)


def transfer_table(tr, sel_dir):
    rows = []
    for model in ("mistral", "llama"):
        for vector, name in (("abst", "ABS"), ("cs", "CS")):
            layer, alpha = setting(sel_dir, model, vector)
            g = at(tr[(tr.model == model) & (tr.vector == vector)], layer, alpha)
            for prec in QUANTIZED:
                r = g[g.prec == prec]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append([MODELS[model], name, f"{layer}/{alpha:g}", VERSIONS[prec], f"{r.cosine:.2f}",
                             f"{r.tr_bias_score:.2f}", f"[{r.tr_bias_score_lo:.2f}, {r.tr_bias_score_hi:.2f}]"])
    return markdown(["Model", "Vector", "Layer/dose", "Version", "Cosine", "Transfer ratio", "95% CI"], rows)


def steering_table(con, sel_dir, seeds):
    spec = [("mistral", "fp16", "Prompt", "prompt", None), ("mistral", "fp16", "ABS", "steer_native", "abst"),
            ("mistral", "fp16", "CS", "steer_native", "cs"), ("llama", "fp16", "Prompt", "prompt", None),
            ("llama", "fp16", "ABS", "steer_native", "abst"), ("llama", "fp16", "CS", "steer_native", "cs")]
    for prec in ("int8", "nf4", "gptq4", "hqq4", "hqq3"):
        if prec == "hqq3":
            spec.append(("llama", prec, "ABS (T)", "steer_transfer", "abst"))
        spec += [("llama", prec, "CS", "steer_native", "cs"), ("llama", prec, "CS (T)", "steer_transfer", "cs")]
    rows = []
    for model, prec, label, kind, vector in spec:
        g = con[(con.model == model) & (con.prec == prec)]
        rank = "-"
        if vector:
            layer, alpha = setting(sel_dir, model, vector)
            g = at(g[g.vector == vector], layer, alpha)
            rnd = g[(g.kind == "random") & g.seed.isin(seeds)]
            r = g[g.kind == kind].iloc[0]
            rank = 1 + int((rnd.d_committed_st < r.d_committed_st).sum())
        else:
            r = g[g.kind == kind].iloc[0]
        rows.append([MODELS[model], VERSIONS[prec], label, *(pp(r[f"d_{m}"]) for m in SHARES), rank])
    return markdown(["Model", "Version", "Steering", *HEADER, f"Rank (of {len(seeds) + 1})"], rows)


def main():
    parser = argparse.ArgumentParser(description="Print the paper's tables from results/summary.")
    parser.add_argument("--summary", default="results/summary")
    parser.add_argument("--selection-dir", default="results/selection")
    args = parser.parse_args()
    read = lambda name: pd.read_csv(os.path.join(args.summary, f"{name}.csv"))
    con = read("contrasts")
    seeds = load_experiment()["random"]["seeds"]
    out = ("## Table 1. Quantization alone (change from FP16, pp)\n\n" + quantization_table(con, read("answer_changes"))
           + "\n\n## Table 2. Transfer of the FP16 vector\n\n" + transfer_table(read("transfer"), args.selection_dir)
           + "\n\n## Table 3. Steering (change from the unsteered model of the same version, pp)\n\n"
           + steering_table(con, args.selection_dir, seeds) + "\n")
    with open(os.path.join(args.summary, "tables.md"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
