import json
import os

SELECTION_DIR = "results/selection"


def selection_path(model, prec):
    return os.path.join(SELECTION_DIR, f"{model}_{prec}.json")


def load_selection(model, prec):
    path = selection_path(model, prec)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run the select stage for {model}/{prec} first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summary(run):
    m = run["metrics"]
    return {"layer": run["layer"], "alpha": run["alpha"], **{k: m[k] for k in
            ("bias_score", "committed_st", "context_acc", "n_committed")}}


def apply_rules(base, candidates, max_drop, min_committed):
    """Main rule: bias score closest to zero with at most `max_drop` loss of context accuracy (if no candidate
    qualifies, the smallest dose). Committed-stereotyping rule: committed stereotyping closest to 50% among
    qualifying candidates with at least `min_committed` committed answers. Ties go to the smaller dose, then layer."""
    floor = base["metrics"]["context_acc"] - max_drop
    ok = [c for c in candidates if c["metrics"]["context_acc"] >= floor]
    pick = lambda pool, key: min(pool, key=lambda c: (key(c["metrics"]), c["alpha"], c["layer"]))
    pool = ok or [c for c in candidates if c["alpha"] == min(c["alpha"] for c in candidates)]
    main = pick(pool, lambda m: abs(m["bias_score"]))
    committed = [c for c in ok if c["metrics"]["n_committed"] >= min_committed]
    out = {
        "main": _summary(main),
        "n_candidates": len(candidates),
        "n_within_limit": len(ok),
        "base": {k: base["metrics"][k] for k in ("bias_score", "committed_st", "context_acc")},
    }
    if committed:
        out["committed"] = _summary(pick(committed, lambda m: abs(2 * m["committed_st"] - 1)))
    return out


def setting(model, vector, rule="main"):
    """Layer and dose selected on FP16; ABS uses the main rule, CS the committed-stereotyping rule."""
    sel = load_selection(model, "fp16")[vector]
    s = sel[rule]
    return s["layer"], s["alpha"]
