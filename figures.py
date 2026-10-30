import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tables import MODELS, QUANTIZED, VERSIONS, at, setting

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY = "#8c8b86"
MARKERS = {"int8": "v", "nf4": "s", "gptq4": "P", "hqq4": "D", "hqq3": "^", "hqq2": "X"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def transfer_figure(summary, sel_dir, path):
    """(a, b) per-layer cosine between the FP16 and the re-extracted vector; (c) transfer ratio on the bias score."""
    cos = pd.read_csv(os.path.join(summary, "cosine_layers.csv"))
    tr = pd.read_csv(os.path.join(summary, "transfer.csv"))
    fig = plt.figure(figsize=(4.8, 3.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    for k, (model, vector, name) in enumerate([("mistral", "abst", "ABS"), ("llama", "cs", "CS")]):
        ax = fig.add_subplot(gs[0, k])
        g = cos[(cos.model == model) & (cos.vector == vector)]
        for c, prec in zip(COLORS, QUANTIZED):
            gp = g[g.prec == prec]
            ax.plot(gp.layer, gp.cosine, color=c, marker=MARKERS[prec], ms=2.5, markevery=4, lw=1.2, label=VERSIONS[prec])
        ax.axvline(setting(sel_dir, model, vector)[0], color="#52514e", lw=0.6, ls="--")
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine to FP16 vector")
        ax.set_title(f"({'ab'[k]}) {MODELS[model]}, {name} vector")
        if k == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=6, frameon=False,
                       bbox_to_anchor=(0.5, 1.01))

    ax = fig.add_subplot(gs[1, :])
    lo_clip, hi_clip = -0.5, 2.5
    series = [(m, v) for m in ("mistral", "llama") for v in ("abst", "cs")]
    handles = []
    for model, vector in series:
        color = COLORS[0] if model == "mistral" else COLORS[1]
        marker = "s" if vector == "abst" else "D"
        g = at(tr[(tr.model == model) & (tr.vector == vector)], *setting(sel_dir, model, vector))
        for r in g.itertuples():
            present = [s for s in series if not at(tr[(tr.model == s[0]) & (tr.vector == s[1]) & (tr.prec == r.prec)],
                                                   *setting(sel_dir, *s)).empty]
            x = QUANTIZED.index(r.prec) + (present.index((model, vector)) - (len(present) - 1) / 2) * 0.13
            ax.vlines(x, np.clip(r.tr_bias_score_lo, lo_clip, hi_clip), np.clip(r.tr_bias_score_hi, lo_clip, hi_clip),
                      color=color, lw=1.0)
            y = r.tr_bias_score
            if lo_clip <= y <= hi_clip:
                ax.plot(x, y, marker, color=color, ms=3.8)
            else:
                ax.plot(x, hi_clip if y > hi_clip else lo_clip, "^" if y > hi_clip else "v", color=color, ms=4)
        handles.append(Line2D([], [], marker=marker, color=color, ls="", ms=4.5,
                              label=f"{MODELS[model].split('-')[0]} {'ABS' if vector == 'abst' else 'CS'}"))
    for y in (0, 1):
        ax.axhline(y, color="#52514e", lw=0.6)
    ax.set_xticks(range(len(QUANTIZED)))
    ax.set_xticklabels([VERSIONS[p] for p in QUANTIZED])
    ax.set_ylim(lo_clip - 0.05, hi_clip + 0.05)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("transfer ratio")
    ax.set_title("(c) Transfer ratio on the bias score (1 = transferred vector as effective as re-extracted)")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.955))
    fig.savefig(path)
    plt.close(fig)


def answers_figure(summary, sel_dir, path):
    """Llama: answer shares on ambiguous questions and context accuracy, per version and steering condition."""
    runs = pd.read_csv(os.path.join(summary, "runs.csv"))
    runs = runs[(runs.model == "llama") & (runs.split == "test")]
    spec = [("fp16", "no steering", "base", None), ("fp16", "prompt", "prompt", None),
            ("fp16", "ABS", "steer_native", "abst"), ("fp16", "CS", "steer_native", "cs")]
    for prec in ("nf4", "hqq4", "hqq3"):
        spec += [(prec, "no steering", "base", None), (prec, "CS native", "steer_native", "cs"),
                 (prec, "CS transferred", "steer_transfer", "cs")]
    rows, ys, y, prev = [], [], 0.0, None
    for prec, label, cond, vector in spec:
        g = runs[(runs.prec == prec) & (runs.cond == cond)]
        if vector:
            g = at(g[g.vector == vector], *setting(sel_dir, "llama", vector))
        rows.append((prec, label, g.iloc[0]))
        y += 0.45 if prev is not None and prec != prev else 0
        ys.append(y)
        y, prev = y + 1, prec
    ys = np.array(ys)

    fig = plt.figure(figsize=(4.8, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.0], wspace=0.14)
    ax, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    segments = [("stereotyped person", "p_st", COLORS[1]), ("other person", "p_oth", COLORS[0]),
                ("“unknown”", "p_unk", COLORS[2])]
    for yy, (_, _, r) in zip(ys, rows):
        left = 0.0
        for _, key, color in segments:
            v = 100 * r[key]
            ax.barh(yy, v, left=left, height=0.78, color=color, lw=0)
            if v >= 6:
                ax.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white", weight="bold")
            left += v
        acc = 100 * r["context_acc"]
        axc.barh(yy, acc, height=0.78, color=GREY, lw=0)
        axc.text(acc - 3, yy, f"{acc:.0f}", ha="right", va="center", fontsize=6.5, color="white", weight="bold")
    for a in (ax, axc):
        a.set_ylim(ys[-1] + 0.6, ys[0] - 0.6)
        a.set_xlim(0, 100)
        a.grid(axis="y", visible=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in rows])
    axc.set_yticks(ys)
    axc.set_yticklabels([])
    axc.tick_params(axis="y", length=0)
    for prec in dict.fromkeys(p for p, _, _ in rows):
        yy = ys[[p == prec for p, _, _ in rows]].mean()
        ax.text(-0.33, yy, VERSIONS[prec], transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=7, weight="bold", rotation=90)
    ax.set_title("(a) Ambiguous questions: answers (%)")
    axc.set_title("(b) Context accuracy (%)")
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, _, c in segments], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.42, -0.08))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw the paper's result figures from results/summary.")
    parser.add_argument("--summary", default="results/summary")
    parser.add_argument("--selection-dir", default="results/selection")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, fn in (("transfer", transfer_figure), ("answers", answers_figure)):
        for ext in ("pdf", "png"):
            fn(args.summary, args.selection_dir, os.path.join(args.out_dir, f"{name}.{ext}"))
    print(f"figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
