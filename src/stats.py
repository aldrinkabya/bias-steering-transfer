import numpy as np
from scipy import stats

from src.metrics import FIELDS, from_counts, item_counts

N_RESAMPLES = 2000
SEED = 2026


def align(runs):
    """Restrict runs to their common items, in one order, and return each item's template cluster."""
    maps = [{(r["category"], int(r["example_id"])): r for r in run} for run in runs]
    keys = sorted(set.intersection(*[set(m) for m in maps]))
    aligned = [[m[k] for k in keys] for m in maps]
    clusters = [(r["category"], str(r["question_index"])) for r in aligned[0]]
    return aligned, clusters


def cluster_counts(runs):
    aligned, clusters = align(runs)
    ids = {c: i for i, c in enumerate(sorted(set(clusters)))}
    out = np.zeros((len(runs), len(ids), len(FIELDS)))
    for k, run in enumerate(aligned):
        for r, c in zip(run, clusters):
            out[k, ids[c]] += item_counts(r)
    return out


def bootstrap(runs, statistics, n_resamples=N_RESAMPLES, seed=SEED):
    """Template-clustered bootstrap. statistics maps a name to a function of the runs' metrics;
    returns {name: (point, 95% CI low, high, SE)}."""
    counts = cluster_counts(runs)
    rng = np.random.default_rng(seed)
    n = counts.shape[1]
    weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_resamples)])
    point = [from_counts(c.sum(0)) for c in counts]
    boot = [from_counts(weights @ c) for c in counts]
    out = {}
    for name, stat in statistics.items():
        vals = np.asarray(stat(boot), dtype=float)
        vals = vals[~np.isnan(vals)]
        p = float(stat(point))
        if vals.size == 0:
            out[name] = (p, np.nan, np.nan, np.nan)
        else:
            out[name] = (p, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.std(vals, ddof=1)))
    return out


def differences(a, b, metrics, **kw):
    """metric(b) - metric(a) for each metric."""
    return bootstrap([a, b], {m: (lambda r, m=m: r[1][m] - r[0][m]) for m in metrics}, **kw)


def transfer_ratios(base, native, transfer, metrics, **kw):
    """(base - transfer) / (base - native) for each metric; resamples with a zero denominator are dropped."""
    def ratio(r, m):
        den = r[0][m] - r[1][m]
        num = r[0][m] - r[2][m]
        return np.where(np.abs(den) > 1e-9, num / np.where(np.abs(den) > 1e-9, den, 1), np.nan)
    return bootstrap([base, native, transfer], {m: (lambda r, m=m: ratio(r, m)) for m in metrics}, **kw)


def wald_p(estimate, se):
    if not np.isfinite(se) or se <= 0:
        return np.nan
    return float(2 * stats.norm.sf(abs(estimate) / se))


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p_values[i] * (len(p_values) - rank)))
        adjusted[i] = running
    return adjusted
