"""
Generate reviewer-requested toxicity figures from the saved predictions:

  1. Reliability diagram (calibration curve) for Tox21 and hERG, with the
     Expected Calibration Error (ECE) annotated. Uses the pooled five-seed
     test predictions.
  2. Per-seed metric stability plot: ROC-AUC, PR-AUC, F1, MCC with mean and
     standard-deviation error bars across the five seeds (trustworthy-AI
     style robustness view).

Inputs : toxicity_extended_metrics.json, predictions/<ds>_pooled_predictions.npz
Outputs: images/toxicity_reliability.png, images/toxicity_metric_stability.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# This script lives in paper_final/reproduction/: repo root is two levels up,
# and the paper's image directory is one level up (paper_final/images).
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "toxicity_extended"
PRED_DIR = DATA_DIR / "predictions"
IMG_DIR = HERE.parent / "images"  # figures belong with the paper (paper_final/images)
IMG_DIR.mkdir(exist_ok=True)

DATASETS = [("tox21", "Tox21 (NR-AR)"), ("herg", "hERG")]
COLORS = {"tox21": "#d1495b", "herg": "#2e86ab"}


def reliability_curve(y_true, y_prob, n_bins=10):
    """Equal-width binning reliability curve + Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    conf, acc, weight = [], [], []
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        conf.append(bin_conf)
        acc.append(bin_acc)
        weight.append(mask.sum() / n)
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return np.array(conf), np.array(acc), np.array(weight), ece


def make_reliability():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (ds, label) in zip(axes, DATASETS):
        data = np.load(PRED_DIR / f"{ds}_pooled_predictions.npz")
        y_true = data["y_true"].astype(int)
        y_prob = data["y_prob"].astype(float)
        conf, acc, weight, ece = reliability_curve(y_true, y_prob, n_bins=10)

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5,
                label="Perfect calibration")
        ax.plot(conf, acc, "o-", color=COLORS[ds], lw=2, markersize=7,
                label=f"Model (ECE = {ece:.3f})")
        # bubble size proportional to bin population
        ax.scatter(conf, acc, s=weight * 1500, color=COLORS[ds],
                   alpha=0.25, zorder=1)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability", fontsize=12,
                      fontweight="bold")
        ax.set_ylabel("Observed positive fraction", fontsize=12,
                      fontweight="bold")
        ax.set_title(f"{label}  (n={len(y_true)}, pos rate="
                     f"{y_true.mean():.3f})", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10, frameon=True)

    fig.suptitle("Reliability diagrams (pooled over 10 seeds)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "toxicity_reliability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def make_stability(summary):
    # ROC-AUC is already reported in the multi-seed table; here we show the
    # newly added imbalance-aware metrics with across-seed error bars.
    metrics = [("pr_auc", "PR-AUC"), ("f1", "F1"), ("mcc", "MCC")]
    fig, ax = plt.subplots(figsize=(9, 5.2))

    x = np.arange(len(metrics))
    width = 0.38
    offsets = {"tox21": -width / 2, "herg": width / 2}

    for ds, label in DATASETS:
        s = summary[ds]
        means = [s[f"{m}_mean"] for m, _ in metrics]
        stds = [s[f"{m}_std"] for m, _ in metrics]
        ax.bar(x + offsets[ds], means, width, yerr=stds, capsize=5,
               color=COLORS[ds], alpha=0.85,
               label=f"{label} ({s['best_algo']})",
               error_kw=dict(ecolor="black", lw=1.3))
        for xi, m, sd in zip(x + offsets[ds], means, stds):
            ax.text(xi, m + sd + 0.015, f"{m:.2f}", ha="center",
                    fontsize=9, fontweight="bold")

    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=12,
                       fontweight="bold")
    ax.set_ylabel("Score (mean $\\pm$ std over 10 seeds)", fontsize=12,
                  fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Toxicity classification metric stability across seeds",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, frameon=True)
    fig.tight_layout()
    out = IMG_DIR / "toxicity_metric_stability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    with open(DATA_DIR / "toxicity_extended_metrics.json") as f:
        summary = json.load(f)
    make_reliability()
    make_stability(summary)


if __name__ == "__main__":
    main()
