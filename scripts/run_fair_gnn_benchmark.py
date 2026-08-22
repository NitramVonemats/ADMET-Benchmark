"""
Fair, leak-free GNN model-family benchmark (Experiment A).

Replaces the contaminated "GNN-Best" column (which was the max-over-optimizers
of the TEST metric, ~300 test-evaluated configs) with a clean, budget-matched
protocol identical to the baselines:

  per endpoint:
    - 50-trial random search over the GNN search space (Table II),
    - each trial trained with the shared GraphConv pipeline (optimized_gnn.train_model),
    - the config is selected on the VALIDATION metric only
      (regression: min val RMSE; classification: max val AUC-ROC),
    - the selected config is then evaluated ONCE on the held-out test split.

Full per-trial history (config, val, test) is stored so the selection is
auditable and reproducible. Test metrics are recorded for every trial but NEVER
used to choose the config (selection uses validation only) -> no leakage.

Outputs -> results/fair_model_family/gnn_random_<dataset>.json  (raw, gitignored)
           results/fair_model_family/fair_gnn_summary.json      (canonical configs+metrics)
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from optimized_gnn import (  # noqa: E402
    OptimizedGNNConfig, train_model, is_classification_dataset, resolve_device,
    prepare_dataset,
)

DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]

# Table II search space
HIDDEN = [64, 96, 128, 192, 256, 384, 512]
LAYERS = [3, 4, 5, 6, 7]
HEAD1 = [128, 192, 256, 384, 512]
HEAD2 = [64, 96, 128, 192, 256]
HEAD3 = [32, 48, 64, 96, 128]
LR_LOG = (-4.0, -2.0)      # 1e-4 .. 1e-2 log-uniform
WD_LOG = (-6.0, -2.0)      # 1e-6 .. 1e-2 log-uniform

N_TRIALS = 50
EPOCHS = 50
PATIENCE = 12
OUT = PROJECT_ROOT / "results" / "fair_model_family"

# Per-dataset config-sampling seeds (distinct so configs differ across datasets;
# fixed so the search is reproducible).
DS_SEED = {d: 1000 + i for i, d in enumerate(DATASETS)}


def sample_config(rng):
    h1 = rng.choice(HEAD1)
    h2 = rng.choice([x for x in HEAD2 if x <= h1] or HEAD2)
    h3 = rng.choice([x for x in HEAD3 if x <= h2] or HEAD3)
    return dict(
        hidden_dim=int(rng.choice(HIDDEN)),
        num_layers=int(rng.choice(LAYERS)),
        head_dims=[int(h1), int(h2), int(h3)],
        lr=float(10 ** rng.uniform(*LR_LOG)),
        weight_decay=float(10 ** rng.uniform(*WD_LOG)),
    )


def run_dataset(ds, device):
    is_clf = is_classification_dataset(ds)
    rng = np.random.RandomState(DS_SEED[ds])
    # Build the canonical split ONCE (seed=42, val_fraction=0.1); it is identical for
    # every trial (selection varies only the architecture, not the partition), so we
    # reuse it and avoid reloading/re-parsing the dataset on each of the 50 trials.
    dcache = prepare_dataset(ds, val_fraction=0.1, seed=42, verbose=False)
    trials = []
    for t in range(N_TRIALS):
        cfg_d = sample_config(rng)
        cfg = OptimizedGNNConfig(**cfg_d)
        res = train_model(dataset_name=ds, config=cfg, epochs=EPOCHS,
                          patience=PATIENCE, device=device, seed=42,
                          dataset_cache=dcache,
                          evaluate_test=True, return_model=False, verbose=False)
        vm, tm = res["val_metrics"], res["test_metrics"]
        val_sel = (vm["auc_roc"] if is_clf else vm["rmse"])
        trials.append({"config": cfg_d, "val_metrics": vm, "test_metrics": tm,
                       "val_select": float(val_sel)})
        disp = (f"val_auc={vm['auc_roc']:.4f} test_auc={tm['auc_roc']:.4f}" if is_clf
                else f"val_rmse={vm['rmse']:.4f} test_r2={tm['r2']:.4f}")
        print(f"  [{ds:24}] trial {t+1:2}/{N_TRIALS}: {disp} "
              f"h{cfg_d['hidden_dim']}L{cfg_d['num_layers']}")

    # Select on VALIDATION only: classification -> max val AUC; regression -> min val RMSE.
    if is_clf:
        best = max(trials, key=lambda x: x["val_select"])
    else:
        best = min(trials, key=lambda x: x["val_select"])

    payload = {"dataset": ds, "task": "classification" if is_clf else "regression",
               "n_trials": N_TRIALS, "epochs": EPOCHS, "patience": PATIENCE,
               "train_seed": 42, "sampling_seed": DS_SEED[ds],
               "selection_metric": ("val_auc_roc" if is_clf else "val_rmse"),
               "selected_config": best["config"],
               "selected_val_metrics": best["val_metrics"],
               "selected_test_metrics": best["test_metrics"],
               "trials": trials}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"gnn_random_{ds}.json", "w") as f:
        json.dump(payload, f, indent=2)
    hv = (best["test_metrics"]["auc_roc"] if is_clf else best["test_metrics"]["r2"])
    print(f"  -> {ds}: SELECTED test {'AUC' if is_clf else 'R2'}={hv:.4f} "
          f"cfg={best['config']}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    device = resolve_device("auto")
    print(f"Fair GNN benchmark | device={device} | {N_TRIALS} trials/endpoint "
          f"(validation-selected, single test)")
    summary = {}
    for ds in args.datasets:
        p = run_dataset(ds, device)
        is_clf = p["task"] == "classification"
        summary[ds] = {
            "selected_config": p["selected_config"],
            "test": (p["selected_test_metrics"]["auc_roc"] if is_clf
                     else p["selected_test_metrics"]["r2"]),
            "val_select": p["selected_val_metrics"].get("auc_roc" if is_clf else "rmse"),
            "task": p["task"],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "fair_gnn_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== FAIR GNN SUMMARY (validation-selected, single test) ===")
    for ds, s in summary.items():
        print(f"  {ds:24} {'AUC' if s['task']=='classification' else 'R2'}={s['test']:+.4f}")
    print(f"Saved -> {OUT/'fair_gnn_summary.json'}")


if __name__ == "__main__":
    main()
