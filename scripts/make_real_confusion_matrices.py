"""
Real confusion matrices for the two toxicity endpoints (Fig. S3).

Replaces the earlier synthesized/placeholder figure: this trains each endpoint's
validation-selected GNN configuration (from fair_gnn_summary.json) on the canonical
split at seed 42, runs the trained model on the held-out test set, and plots the
ACTUAL 2x2 confusion matrix at the standard 0.5 decision threshold. No counts are
fabricated.

Output -> paper_final/images/confusion_matrices.png
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from optimized_gnn import (  # noqa: E402
    OptimizedGNNConfig, train_model, build_loaders, prepare_dataset, resolve_device,
)

SEED = 42
IMG = os.path.join(ROOT, "paper_final", "images", "confusion_matrices.png")
SUMM = os.path.join(ROOT, "results", "fair_model_family", "fair_gnn_summary.json")
LABELS = {"tox21": "Tox21 (NR-AR)", "herg": "hERG"}


def cfg_for(ds):
    c = json.load(open(SUMM))[ds]["selected_config"]
    return OptimizedGNNConfig(hidden_dim=c["hidden_dim"], num_layers=c["num_layers"],
                              head_dims=c["head_dims"], lr=c["lr"],
                              weight_decay=c["weight_decay"])


def test_confusion(ds, device):
    dcache = prepare_dataset(ds, val_fraction=0.1, seed=SEED, verbose=False)
    res = train_model(dataset_name=ds, config=cfg_for(ds), epochs=50, patience=12,
                      device=device, seed=SEED, dataset_cache=dcache,
                      evaluate_test=True, return_model=True, verbose=False)
    model = res["model"].to(device).eval()
    _, _, test_loader, _, _ = build_loaders(dataset_name=ds, dataset_cache=dcache,
                                            return_cache=True, verbose=False)
    logits, trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits.append(model(data).cpu())
            trues.append(data.y.cpu())
    probs = 1 / (1 + np.exp(-torch.cat(logits).numpy()))
    y = torch.cat(trues).numpy().astype(int)
    pred = (probs >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]]), res["test_metrics"]


def main():
    device = resolve_device("auto")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ds, tag in zip(axes, ["tox21", "herg"], ["(A)", "(B)"]):
        cm, tm = test_confusion(ds, device)
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{tag} {LABELS[ds]}\nAUC={tm['auc_roc']:.3f}, F1={tm['f1']:.3f} "
                     f"(test, thr=0.5, seed {SEED})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        print(f"{ds}: cm={cm.tolist()} AUC={tm['auc_roc']:.3f} F1={tm['f1']:.3f}")
    fig.tight_layout()
    fig.savefig(IMG, dpi=300, bbox_inches="tight", facecolor="white")
    print("Wrote", IMG)


if __name__ == "__main__":
    main()
