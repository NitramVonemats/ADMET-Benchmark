"""
Clean HPO-comparison figure (Fig.~\ref{fig:algorithm_performance}).

Reads the leak-free re-run summary (results/hpo_comparison_clean/summary.json) --
NOT the deprecated runs/ directory -- and shows each optimizer's validation-selected
test metric on every endpoint, so the reader can see directly that no optimizer is
consistently ahead of Random Search. Regression R2 is clipped for readability.

Output -> paper_final/images/01_algorithm_performance.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMM = os.path.join(ROOT, "results", "hpo_comparison_clean", "summary.json")
OUT = os.path.join(ROOT, "paper_final", "images", "01_algorithm_performance.png")

ALGOS = ["random", "pso", "abc", "ga", "sa", "hc", "tpe"]
ALGO_LBL = {"random": "Random", "pso": "PSO", "abc": "ABC", "ga": "GA",
            "sa": "SA", "hc": "HC", "tpe": "TPE"}
REG = [("Caco2_Wang", "Caco-2"), ("Half_Life_Obach", "Half-Life"),
       ("Clearance_Hepatocyte_AZ", "CL-Hep"), ("Clearance_Microsome_AZ", "CL-Mic")]
CLF = [("tox21", "Tox21"), ("herg", "hERG")]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def grouped(ax, summ, datasets, clip, title, ylabel, hline):
    n_alg = len(ALGOS)
    w = 0.8 / len(datasets)
    x = np.arange(n_alg)
    for di, (ds, lbl) in enumerate(datasets):
        vals = [np.clip(summ[ds][a]["test"], *clip) for a in ALGOS]
        ax.bar(x + di * w, vals, w, label=lbl, color=COLORS[di], edgecolor="black", linewidth=0.3)
    ax.axhline(hline, ls="--", c="gray", lw=0.8)
    ax.set_xticks(x + w * (len(datasets) - 1) / 2)
    ax.set_xticklabels([ALGO_LBL[a] for a in ALGOS], rotation=0, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, ncol=len(datasets), loc="lower center")
    ax.grid(axis="y", ls=":", alpha=0.4)


def main():
    summ = json.load(open(SUMM))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    grouped(a1, summ, REG, (-1.5, 0.65),
            "(A) Regression: test R$^2$ by optimizer (clipped at $-1.5$)", "R$^2$", 0.0)
    grouped(a2, summ, CLF, (0.5, 0.9),
            "(B) Classification: test AUC-ROC by optimizer", "AUC-ROC", 0.5)
    fig.suptitle("HPO optimizers vs. Random Search (validation-selected, single test; "
                 "clean leak-free re-run)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
