"""
Regenerate the two model-comparison figures directly from the current Table IV
values (R2 for regression, AUC-ROC for classification), so they stay consistent
with the HPO-parity re-benchmark:

  - foundation_ranking.png            : per-task rank heatmap (Fig. foundation_ranking)
  - gnn_vs_foundation_comparison.png  : 2-panel bar comparison (Fig. foundation_comparison)

Values below are the authoritative Table IV cells (GNN-Best from the HPO sweep;
each baseline the best of a 50-trial validation-based head search). Baselines are
sourced from results/baseline_hpo_parity/baseline_hpo_parity_results.json; the GNN
column is unchanged from the main benchmark.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(BASE, "paper_final", "images")
os.makedirs(IMG, exist_ok=True)

MODELS = ["GNN (Ours)", "ChemBERTa", "Morgan-FP", "MolE-FP", "MolCLR"]

# --- Table IV values (rows aligned to MODELS) ---
# Canonical 10-seed MEDIAN values (leak-free fair comparison; all families).
# Source: results/fair_model_family/table_IV_median_canonical.csv
REG = {  # regression R^2 (GNN = 10-seed median, molecule-level dedup split)
    "Caco2":    [0.482, 0.413, 0.322, 0.127, 0.232],
    "Half-Life":[-0.010, -1.918, -6.341, -3.776, -0.672],
    "CL-Hep":   [-0.373, -0.224, -0.017, -0.106, -0.113],
    "CL-Mic":   [0.093, -0.010, 0.169, 0.056, 0.125],
}
CLF = {  # classification AUC-ROC
    "Tox21":    [0.718, 0.714, 0.706, 0.698, 0.722],
    "hERG":     [0.754, 0.747, 0.767, 0.735, 0.657],
}

MCOLORS = {"GNN (Ours)": "#1f77b4", "ChemBERTa": "#ff7f0e", "Morgan-FP": "#2ca02c",
           "MolE-FP": "#9467bd", "MolCLR": "#d62728"}

sns.set_style("white")
plt.rcParams.update({"figure.dpi": 300, "savefig.facecolor": "white",
                     "axes.facecolor": "white", "figure.facecolor": "white"})


def _ranks(row):
    """1 = best (highest value)."""
    order = sorted(range(len(row)), key=lambda j: row[j], reverse=True)
    r = [0] * len(row)
    for i, j in enumerate(order):
        r[j] = i + 1
    return r


def foundation_ranking():
    rank_cmap = LinearSegmentedColormap.from_list(
        "rank", ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    reg_names = list(REG.keys())
    arrA = np.array([_ranks(REG[d]) for d in reg_names])
    sns.heatmap(arrA, annot=True, fmt="d", cmap=rank_cmap, vmin=1, vmax=5, ax=axes[0],
                xticklabels=MODELS, yticklabels=reg_names,
                cbar_kws={"label": "Rank"}, linewidths=0.8, linecolor="white")
    axes[0].set_title("(A) ADME Rankings by R$^2$", fontweight="bold")

    clf_names = list(CLF.keys())
    arrB = np.array([_ranks(CLF[d]) for d in clf_names])
    sns.heatmap(arrB, annot=True, fmt="d", cmap=rank_cmap, vmin=1, vmax=5, ax=axes[1],
                xticklabels=MODELS, yticklabels=clf_names,
                cbar_kws={"label": "Rank"}, linewidths=0.8, linecolor="white")
    axes[1].set_title("(B) Toxicity Rankings by AUC-ROC", fontweight="bold")

    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Model Rankings per Task  (1 = Best, 5 = Worst)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(IMG, "foundation_ranking.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def gnn_vs_foundation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    w = 0.16
    YMIN = -1.5  # clip extreme-negative bars for readability

    # (A) ADME R^2
    ax = axes[0]
    reg_names = list(REG.keys())
    x = np.arange(len(reg_names))
    clipped = []
    for i, m in enumerate(MODELS):
        vals = [REG[d][i] for d in reg_names]
        disp = [max(v, YMIN) for v in vals]
        ax.bar(x + (i - 2) * w, disp, w, color=MCOLORS[m], label=m,
               edgecolor="white", linewidth=0.4)
        for d, v in zip(reg_names, vals):
            if v < YMIN:
                clipped.append(f"{m}/{d} R$^2$={v:.2f}")
    ax.axhline(0, color="#d62728", ls="--", lw=1, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(reg_names)
    ax.set_ylim(YMIN, 0.7)
    ax.set_ylabel("Test R$^2$"); ax.set_title("(A) ADME Regression: R$^2$", fontweight="bold")
    if clipped:
        ax.annotate("Off scale (< " + str(YMIN) + "): " + "; ".join(clipped),
                    xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # (B) Toxicity AUC-ROC
    ax = axes[1]
    clf_names = list(CLF.keys())
    x = np.arange(len(clf_names))
    for i, m in enumerate(MODELS):
        vals = [CLF[d][i] for d in clf_names]
        ax.bar(x + (i - 2) * w, vals, w, color=MCOLORS[m], label=m,
               edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="#d62728", ls="--", lw=1, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(clf_names)
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("Test AUC-ROC"); ax.set_title("(B) Toxicity: AUC-ROC", fontweight="bold")
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper right")

    for ax in axes:
        sns.despine(ax=ax)
    fig.suptitle("Fair model-family comparison (10-seed median)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(IMG, "gnn_vs_foundation_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    foundation_ranking()
    gnn_vs_foundation()
