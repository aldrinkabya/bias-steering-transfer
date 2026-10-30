import yaml

from src.models import model_config
from src.selection import load_selection, setting
from src.steering import VECTORS


def load_experiment(path="configs/experiment.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_tag(model, prec, cond, split, vector=None, layer=None, alpha=None, seed=None):
    tag = f"{model}_{prec}_{cond}"
    if vector:
        tag += f"_{vector}"
    tag += f"_{split}"
    if layer is not None:
        tag += f"_L{layer}_a{alpha}"
    if seed is not None:
        tag += f"_s{seed}"
    return tag


def _run(cond, split, vector=None, layer=None, alpha=None, seed=None):
    return {"cond": cond, "split": split, "vector": vector, "layer": layer, "alpha": alpha, "seed": seed}


def _steer(prec, split, vector, layer, alpha):
    runs = [_run("steer_native", split, vector, layer, alpha)]
    if prec != "fp16":
        runs.append(_run("steer_transfer", split, vector, layer, alpha))
    return runs


def sweep_runs(model, prec, exp):
    runs = [_run("base", "dev")]
    for vector in VECTORS:
        if prec == "fp16":
            grid = [(L, a) for L in model_config(model)["layers"] for a in exp["selection"]["alphas"]]
        else:
            layer, alpha = setting(model, vector)
            grid = [(layer, round(f * alpha, 6)) for f in exp["selection"]["dose_factors"]]
        runs += [_run("steer_native", "dev", vector, L, a) for L, a in grid]
    return runs


def eval_runs(model, prec, split, exp):
    """Test (BBQ) or MMLU runs for one model and precision."""
    test = split == "test"
    runs = [_run("base", split)] + ([_run("prompt", split)] if test else [])

    rnd = exp["random"]
    seeds = rnd["seeds"] if prec in rnd["all_seeds_on"] else rnd["seeds"][:1]
    for vector in VECTORS:
        layer, alpha = setting(model, vector)
        native = load_selection(model, prec)[vector]["main"]["alpha"] if prec != "fp16" else alpha
        runs.append(_run("steer_native", split, vector, layer, native))
        runs += _steer(prec, split, vector, layer, alpha)
        if test:
            runs += [_run("random", split, vector, layer, alpha, s) for s in seeds]

    cs = exp["cs_rule"].get(model, {})
    if prec in cs.get("precisions", []):
        layer, alpha = setting(model, "cs", "committed")
        runs += _steer(prec, split, "cs", layer, alpha)
        if test:
            runs += [_run("random", split, "cs", layer, alpha, s) for s in cs["random_seeds"].get(prec, [])]

    for d in exp["doses"]:
        if d["model"] != model or prec not in d["precisions"] or not (test or d.get("mmlu")):
            continue
        layer, alpha = setting(model, d["vector"], "committed" if d["vector"] == "cs" else "main")
        for f in d["factors"]:
            runs += _steer(prec, split, d["vector"], layer, round(f * alpha, 6))

    unique = {}
    for r in runs:
        unique.setdefault(run_tag(model, prec, **r), r)
    return list(unique.values())


def plan(model, prec, stage, exp):
    if stage == "sweep":
        return sweep_runs(model, prec, exp)
    return eval_runs(model, prec, stage, exp)
