import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.evaluate import RUNS_DIR, VECTORS_DIR
from src.metrics import METRICS, answer_changes, bbq_metrics
from src.models import PRECISIONS
from src.plan import load_experiment
from src.selection import SELECTION_DIR
from src.stats import differences, holm, transfer_ratios, wald_p
from src.steering import VECTORS, cosine, load_vector, vector_path

TESTED = ["bias_score", "committed_st", "context_acc"]
STEERED = ["steer_native", "steer_transfer"]

_cache = {}


def records(path):
    if path not in _cache:
        _cache[path] = pd.read_csv(path).to_dict("records")
    return _cache[path]


def load_runs(runs_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        csv = path[:-5] + ".csv"
        metrics = meta["metrics"] if meta["split"] == "mmlu" else bbq_metrics(records(csv))
        rows.append({k: meta[k] for k in ("tag", "model", "prec", "cond", "split", "vector", "layer", "alpha", "seed")}
                    | {"csv": csv} | metrics)
    return pd.DataFrame(rows)


def load_selections(sel_dir):
    out = {}
    for path in glob.glob(os.path.join(sel_dir, "*.json")):
        model, prec = os.path.basename(path)[:-5].split("_", 1)
        with open(path, encoding="utf-8") as f:
            out[(model, prec)] = json.load(f)
    return out


def contrast(kind, a, b):
    row = {"kind": kind, "model": b.model, "prec": b.prec, "vector": b.vector, "layer": b.layer, "alpha": b.alpha,
           "seed": b.seed, "run_a": a.tag, "run_b": b.tag}
    for m, (d, lo, hi, se) in differences(records(a.csv), records(b.csv), METRICS).items():
        row.update({f"d_{m}": d, f"d_{m}_lo": lo, f"d_{m}_hi": hi})
        if m in TESTED:
            row[f"p_{m}"] = wald_p(d, se)
    return row


def same_setting(r, s):
    return s is not None and int(r.layer) == s["layer"] and np.isclose(r.alpha, s["alpha"])


def family(r, sel, holm_cfg):
    if r.model not in holm_cfg["models"] or r.kind == "random":
        return None
    for name, f in holm_cfg["version_families"].items():
        if r.model == f["model"] and r.prec in f["precisions"]:
            return name
    if r.kind in ("quantization", "prompt"):
        return "main"
    if same_setting(r, sel[(r.model, "fp16")][r.vector]["main"]) or \
            (r.kind == "steer_native" and same_setting(r, sel[(r.model, r.prec)][r.vector]["main"])):
        return "main"
    if r.vector == "cs" and same_setting(r, sel[(r.model, "fp16")]["cs"].get("committed")):
        return "cs"
    return None


def contrasts(test, sel, holm_cfg):
    rows = []
    for (model, prec), g in test.groupby(["model", "prec"]):
        base = g[g.cond == "base"].iloc[0]
        if prec != "fp16":
            fp16 = test[(test.model == model) & (test.prec == "fp16") & (test.cond == "base")].iloc[0]
            rows.append(contrast("quantization", fp16, base))
        for r in g[g.cond != "base"].itertuples():
            rows.append(contrast(r.cond, base, r))
    out = pd.DataFrame(rows)
    out["family"] = [family(r, sel, holm_cfg) for r in out.itertuples()]
    for m in TESTED:
        out[f"holm_p_{m}"] = np.nan
    for _, g in out[out.family.notna()].groupby("family"):
        cells = [(i, m) for m in TESTED for i in g.index]
        adjusted = holm([1.0 if not np.isfinite(out.at[i, f"p_{m}"]) else out.at[i, f"p_{m}"] for i, m in cells])
        for (i, m), p in zip(cells, adjusted):
            out.at[i, f"holm_p_{m}"] = p
    return out


def transfer(test, vectors_dir):
    rows = []
    for t in test[test.cond == "steer_transfer"].itertuples():
        g = test[(test.model == t.model) & (test.prec == t.prec)]
        native = g[(g.cond == "steer_native") & (g.vector == t.vector) & (g.layer == t.layer) & np.isclose(g.alpha, t.alpha)]
        if native.empty:
            continue
        base = g[g.cond == "base"].iloc[0]
        tr = transfer_ratios(records(base.csv), records(native.iloc[0].csv), records(t.csv), ["bias_score", "committed_st"])
        fp16, q = (load_vector(vectors_dir, t.model, p, t.vector, int(t.layer)) for p in ("fp16", t.prec))
        row = {"model": t.model, "prec": t.prec, "vector": t.vector, "layer": t.layer, "alpha": t.alpha,
               "cosine": cosine(fp16["unit_vector"], q["unit_vector"]),
               "norm_ratio": q["residual_norm"] / fp16["residual_norm"]}
        for m, (p, lo, hi, _) in tr.items():
            row.update({f"tr_{m}": p, f"tr_{m}_lo": lo, f"tr_{m}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def random_null(con):
    rnd = con[con.kind == "random"]
    rows = []
    for e in con[con.kind.isin(STEERED)].itertuples():
        seeds = rnd[(rnd.model == e.model) & (rnd.prec == e.prec) & (rnd.vector == e.vector) & (rnd.layer == e.layer)
                    & np.isclose(rnd.alpha, e.alpha)]
        if seeds.empty:
            continue
        row = {"model": e.model, "prec": e.prec, "kind": e.kind, "vector": e.vector, "layer": e.layer, "alpha": e.alpha,
               "run": e.run_b, "n_seeds": len(seeds)}
        for m in METRICS:
            d, null = getattr(e, f"d_{m}"), seeds[f"d_{m}"].to_numpy()
            row.update({f"{m}_delta": d, f"{m}_null_min": null.min(), f"{m}_null_max": null.max(),
                        f"{m}_rank": int(1 + (null < d).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def per_category(test, con):
    csv = dict(zip(test.tag, test.csv))
    rows = []
    for c in con[con.kind.isin(STEERED)].itertuples():
        a, b = records(csv[c.run_a]), records(csv[c.run_b])
        for cat in sorted({r["category"] for r in a}):
            d = differences([r for r in a if r["category"] == cat], [r for r in b if r["category"] == cat],
                            ["committed_st", "context_acc"])
            rows.append({"run": c.run_b, "category": cat,
                         **{f"d_{m}{s}": v for m, x in d.items() for s, v in zip(("", "_lo", "_hi"), x[:3])}})
    return pd.DataFrame(rows)


def answer_change_rates(test):
    rows = []
    for (model, prec), g in test[test.cond == "base"].groupby(["model", "prec"]):
        if prec == "fp16":
            continue
        fp16 = test[(test.model == model) & (test.prec == "fp16") & (test.cond == "base")].iloc[0]
        rows.append({"model": model, "prec": prec, **answer_changes(records(fp16.csv), records(g.iloc[0].csv))})
    return pd.DataFrame(rows)


def mmlu(runs):
    m = runs[runs.split == "mmlu"].copy()
    base = m[m.cond == "base"].set_index(["model", "prec"]).mmlu_acc
    m["d_mmlu_acc"] = [r.mmlu_acc - base[(r.model, r.prec)] for r in m.itertuples()]
    return m[["tag", "model", "prec", "cond", "vector", "layer", "alpha", "mmlu_acc", "d_mmlu_acc"]]


def usability(sel):
    rows = []
    for (model, prec), s in sorted(sel.items()):
        if prec == "fp16":
            continue
        for vector in VECTORS:
            v = s[vector]
            rows.append({"model": model, "prec": prec, "vector": vector, "n_within_limit": v["n_within_limit"],
                         "n_candidates": v["n_candidates"], "alpha": v["main"]["alpha"],
                         "d_context_acc": v["main"]["context_acc"] - v["base"]["context_acc"]})
    return pd.DataFrame(rows)


def vector_tables(runs, vectors_dir):
    cos_rows, overlap_rows = [], []
    for model in sorted(runs.model.unique()):
        precs = [p for p in PRECISIONS if p in set(runs[runs.model == model].prec)]
        n_layers = len(glob.glob(os.path.join(vectors_dir, f"{model}_fp16_abst_L*.pt")))
        for prec in precs:
            for L in range(n_layers):
                v = {vec: load_vector(vectors_dir, model, prec, vec, L)["unit_vector"] for vec in VECTORS
                     if os.path.exists(vector_path(vectors_dir, model, prec, vec, L))}
                if len(v) == 2:
                    overlap_rows.append({"model": model, "prec": prec, "layer": L, "cosine": cosine(v["abst"], v["cs"])})
                if prec != "fp16":
                    for vec in v:
                        ref = load_vector(vectors_dir, model, "fp16", vec, L)["unit_vector"]
                        cos_rows.append({"model": model, "prec": prec, "vector": vec, "layer": L,
                                         "cosine": cosine(ref, v[vec])})
    return pd.DataFrame(cos_rows), pd.DataFrame(overlap_rows)


def main():
    parser = argparse.ArgumentParser(description="Summarise all runs into results/summary.")
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--vectors-dir", default=VECTORS_DIR)
    parser.add_argument("--selection-dir", default=SELECTION_DIR)
    parser.add_argument("--out-dir", default="results/summary")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_runs(args.runs_dir)
    sel = load_selections(args.selection_dir)
    test = runs[runs.split == "test"]
    con = contrasts(test, sel, load_experiment()["holm"])
    cos, overlap = vector_tables(runs, args.vectors_dir)
    tables = {
        "runs": runs.drop(columns="csv"),
        "contrasts": con,
        "transfer": transfer(test, args.vectors_dir),
        "random_null": random_null(con),
        "per_category": per_category(test, con),
        "answer_changes": answer_change_rates(test),
        "mmlu": mmlu(runs),
        "usability": usability(sel),
        "cosine_layers": cos,
        "vector_overlap": overlap,
    }
    for name, df in tables.items():
        df.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    print(f"summarised {len(runs)} runs into {args.out_dir}/")


if __name__ == "__main__":
    main()
