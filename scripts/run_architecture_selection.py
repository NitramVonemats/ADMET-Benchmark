"""
Preliminary GNN architecture-selection screen (reproducible, no hardcoded numbers).

Trains eight message-passing backbones under a single fixed configuration and the
same scaffold-split data pipeline used by the main study, so the backbone choice
for the ADMET benchmark is supported by real, independently computed evidence.

Architectures : GraphConv, GCN, GAT, GraphSAGE, GIN, TAG, SGC, TransformerConv
Datasets      : Caco2_Wang, Half_Life_Obach, Clearance_Hepatocyte_AZ,
                Clearance_Microsome_AZ (regression); tox21, herg (classification)
Seeds         : multiple, results reported as mean +/- std across seeds.

The data preparation, feature extraction, evaluation, and training-epoch logic are
imported directly from optimized_gnn.py; only the graph-convolution backbone is made
configurable here. Every reported metric is produced by the model on the held-out
scaffold-split test set. Outputs are written to results/architecture_selection/
(gitignored), from which the supplementary figures/tables are regenerated.
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GraphConv, GCNConv, GATConv, SAGEConv, GINConv, TAGConv, SGConv,
    TransformerConv, global_mean_pool, global_max_pool,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimized_gnn import (  # noqa: E402
    OptimizedGNNConfig,
    prepare_dataset,
    build_loaders,
    train_epoch,
    evaluate,
    evaluate_classification,
    resolve_device,
    is_classification_dataset,
)

ARCHITECTURES = [
    "GraphConv", "GCN", "GAT", "GraphSAGE", "GIN", "TAG", "SGC", "Transformer",
]
REGRESSION_DATASETS = [
    "Caco2_Wang", "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ",
]
CLASSIFICATION_DATASETS = ["tox21", "herg"]
SEEDS = [42, 123, 456]

# Attention heads for GAT / TransformerConv; concat=False keeps output dim = hidden.
HEADS = 4

OUT_DIR = PROJECT_ROOT / "results" / "architecture_selection"


def build_conv(arch: str, in_ch: int, out_ch: int) -> nn.Module:
    """Return a PyG convolution layer whose output dimension is out_ch."""
    if arch == "GraphConv":
        return GraphConv(in_ch, out_ch)
    if arch == "GCN":
        return GCNConv(in_ch, out_ch, add_self_loops=True)
    if arch == "GAT":
        return GATConv(in_ch, out_ch, heads=HEADS, concat=False)
    if arch == "GraphSAGE":
        return SAGEConv(in_ch, out_ch)
    if arch == "GIN":
        mlp = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch))
        return GINConv(mlp, train_eps=True)
    if arch == "TAG":
        return TAGConv(in_ch, out_ch, K=3)
    if arch == "SGC":
        return SGConv(in_ch, out_ch, K=2)
    if arch == "Transformer":
        return TransformerConv(in_ch, out_ch, heads=HEADS, concat=False)
    raise ValueError(f"Unknown architecture: {arch}")


class ConfigurableBackbone(nn.Module):
    """Stacked message-passing backbone with a selectable convolution type.

    Mirrors optimized_gnn.OptimalGraphBackbone (BatchNorm + ReLU + residual from
    layer 1 onward); only the convolution operator varies with `arch`.
    """

    def __init__(self, arch, input_dim=8, hidden_dim=128, num_layers=5, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.convs.append(build_conv(arch, in_channels, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                self.dropouts.append(nn.Dropout(dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = F.relu(norm(conv(x, edge_index)))
            if self.dropout > 0:
                h = self.dropouts[i](h)
            x = x + h if i > 0 else h
        return x


class ConfigurableGNN(nn.Module):
    """Backbone + concat(mean,max) readout + MLP head (mirrors OptimizedMolecularGNN)."""

    def __init__(self, arch, input_dim=8, hidden_dim=128, num_layers=5,
                 adme_dim=15, head_dims=(256, 128, 64), dropout=0.0):
        super().__init__()
        self.backbone = ConfigurableBackbone(arch, input_dim, hidden_dim, num_layers, dropout)
        self.readout_dim = hidden_dim * 2
        combined_dim = self.readout_dim + adme_dim
        layers, current = [], combined_dim
        for hidden in head_dims:
            layers += [nn.Linear(current, int(hidden)), nn.BatchNorm1d(int(hidden)), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, data):
        emb = self.backbone(data)
        pooled = torch.cat([global_mean_pool(emb, data.batch),
                            global_max_pool(emb, data.batch)], dim=-1)
        adme = data.adme_features
        if adme.dim() == 1:
            adme = adme.unsqueeze(0)
        combined = torch.cat([pooled, adme], dim=-1) if adme.numel() > 0 else pooled
        return self.head(combined).squeeze(-1)


def train_one(dataset_name, arch, seed, epochs, patience, device, cfg, cache):
    """Train a single (architecture, dataset, seed) and return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    is_clf = is_classification_dataset(dataset_name)
    train_loader, val_loader, test_loader, (mu, sigma), cache = build_loaders(
        dataset_name=dataset_name, batch_train=cfg.batch_train, batch_eval=cfg.batch_eval,
        val_fraction=cfg.val_fraction, seed=seed, dataset_cache=cache,
        return_cache=True, verbose=False,
    )
    is_log_transformed = cache.get("is_log_transformed", False)

    sample = cache["train"][0]
    input_dim = int(sample.x.size(-1))
    adme_dim = int(sample.adme_features.numel())
    # Match the main pipeline's per-dataset regularization treatment.
    dropout = 0.5 if dataset_name == "Caco2_Wang" else 0.0
    label_noise = 0.05 if (dataset_name == "Caco2_Wang" and not is_clf) else 0.0

    model = ConfigurableGNN(
        arch=arch, input_dim=input_dim, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, adme_dim=adme_dim, head_dims=cfg.head_dims,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("-inf") if is_clf else float("inf")
    best_state, no_improve = None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, device, cfg.max_grad_norm,
                    label_noise=label_noise, is_classification=is_clf)
        vm = (evaluate_classification(model, val_loader, device) if is_clf
              else evaluate(model, val_loader, device, mu, sigma))
        cur = vm["f1"] if is_clf else vm["rmse"]
        improved = cur > best if is_clf else cur < best
        if improved:
            best, best_state, no_improve = cur, deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics (used for config selection in search mode).
    val_final = (evaluate_classification(model, val_loader, device) if is_clf
                 else evaluate(model, val_loader, device, mu, sigma))

    if is_clf:
        tm = evaluate_classification(model, test_loader, device)
        metrics = {"auc_roc": tm["auc_roc"], "f1": tm["f1"], "accuracy": tm["accuracy"],
                   "precision": tm.get("precision"), "recall": tm.get("recall"),
                   "val_auc_roc": val_final["auc_roc"], "val_f1": val_final["f1"]}
    else:
        tm = evaluate(model, test_loader, device, mu, sigma, is_log_transformed)
        metrics = {"rmse": tm["rmse"], "mae": tm["mae"], "r2": tm["r2"],
                   "val_rmse": val_final["rmse"], "val_r2": val_final["r2"]}
    metrics.update({"train_time": time.time() - t0, "n_params": int(n_params),
                    "stopped_epoch": epoch})
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


# Small hyperparameter grid for the best-of-K validation-selected screen. Configs
# span the Table II search space; each architecture is evaluated at its best
# validation config so the backbone comparison reflects achievable (HPO-selected)
# performance rather than a single arbitrary configuration.
CONFIG_GRID = [
    dict(hidden_dim=128, num_layers=5, head_dims=(256, 128, 64), lr=1e-3),
    dict(hidden_dim=96,  num_layers=3, head_dims=(192, 96, 64),  lr=2e-3),
    dict(hidden_dim=192, num_layers=5, head_dims=(384, 192, 96), lr=5e-4),
    dict(hidden_dim=256, num_layers=4, head_dims=(384, 128, 64), lr=1e-3),
    dict(hidden_dim=64,  num_layers=6, head_dims=(128, 96, 64),  lr=1e-3),
]


def search_one(dataset_name, arch, seed, epochs, patience, device, cache):
    """Best-of-K: train each grid config, select by validation, return its test metrics."""
    is_clf = is_classification_dataset(dataset_name)
    trials = []
    for gc in CONFIG_GRID:
        cfg = OptimizedGNNConfig(**gc)
        m = train_one(dataset_name, arch, seed, epochs, patience, device, cfg, cache)
        m["config"] = gc
        trials.append(m)
    # Select by validation: AUC (higher) for classification, RMSE (lower) for regression.
    if is_clf:
        best = max(trials, key=lambda m: (m.get("val_auc_roc")
                   if m.get("val_auc_roc") is not None and np.isfinite(m["val_auc_roc"]) else -1))
    else:
        best = min(trials, key=lambda m: (m.get("val_rmse")
                   if m.get("val_rmse") is not None and np.isfinite(m["val_rmse"]) else 1e18))
    best = dict(best)
    best["n_configs_tried"] = len(trials)
    return best


def aggregate(per_seed, keys):
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed if m.get(k) is not None], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"{k}_mean"] = float(vals.mean())
            out[f"{k}_std"] = float(vals.std(ddof=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=REGRESSION_DATASETS + CLASSIFICATION_DATASETS)
    ap.add_argument("--architectures", nargs="+", default=ARCHITECTURES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--mode", choices=["fixed", "search"], default="fixed",
                    help="fixed: one config, multi-seed. search: best-of-K by validation.")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick sanity run: 1 arch subset, few epochs.")
    args = ap.parse_args()

    if args.smoke:
        args.architectures = ["GraphConv", "GAT"]
        args.datasets = ["herg"]
        args.seeds = [42]
        args.epochs = 5
        args.patience = 5

    device = resolve_device("auto")
    cfg = OptimizedGNNConfig()  # base config (val_fraction, batch sizes, grad clip)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | mode={args.mode} | datasets={args.datasets} | "
          f"archs={args.architectures} | seeds={args.seeds} | epochs={args.epochs}")

    results = {}
    total_runs = 0
    search_seed = args.seeds[0]
    for ds in args.datasets:
        is_clf = is_classification_dataset(ds)
        metric_keys = (["auc_roc", "f1", "accuracy"] if is_clf else ["rmse", "mae", "r2"])
        # Validation metrics are computed in train_one but must be persisted so the
        # eight-architecture ranking can be reproduced WITHOUT touching the test set.
        val_keys = (["val_auc_roc", "val_f1"] if is_clf else ["val_rmse", "val_r2"])
        results[ds] = {"task": "classification" if is_clf else "regression",
                       "architectures": {}}
        seed_caches = {}
        for arch in args.architectures:
            if args.mode == "search":
                # Best-of-K validation-selected config, evaluated at seed_search.
                if search_seed not in seed_caches:
                    seed_caches[search_seed] = prepare_dataset(
                        dataset_name=ds, val_fraction=cfg.val_fraction,
                        seed=search_seed, verbose=False)
                best = search_one(ds, arch, search_seed, args.epochs, args.patience,
                                  device, seed_caches[search_seed])
                total_runs += best["n_configs_tried"]
                # Store under the same *_mean keys the downstream generator expects.
                entry = {"selected_config": best["config"], "seed": search_seed,
                         "n_configs_tried": best["n_configs_tried"],
                         "train_time_mean": best["train_time"],
                         "n_params_mean": best["n_params"]}
                for k in metric_keys:
                    entry[f"{k}_mean"] = best[k]
                    entry[f"{k}_std"] = 0.0
                # Persist the validation metrics that governed config selection, so
                # the cross-architecture ranking can be computed on validation only.
                for vk in val_keys:
                    if best.get(vk) is not None:
                        entry[vk] = float(best[vk])
                results[ds]["architectures"][arch] = entry
                disp = (f"AUC={best['auc_roc']:.4f} F1={best['f1']:.4f}" if is_clf
                        else f"RMSE={best['rmse']:.4f} R2={best['r2']:.4f}")
                print(f"  [{ds:24}] {arch:12} best/{best['n_configs_tried']}: {disp} "
                      f"cfg=h{best['config']['hidden_dim']}L{best['config']['num_layers']}"
                      f"lr{best['config']['lr']:.0e} ({best['n_params']:,}p)")
            else:
                per_seed = []
                for seed in args.seeds:
                    if seed not in seed_caches:
                        seed_caches[seed] = prepare_dataset(
                            dataset_name=ds, val_fraction=cfg.val_fraction,
                            seed=seed, verbose=False)
                    m = train_one(ds, arch, seed, args.epochs, args.patience,
                                  device, cfg, seed_caches[seed])
                    m["seed"] = seed
                    per_seed.append(m)
                    total_runs += 1
                    disp = (f"AUC={m['auc_roc']:.4f} F1={m['f1']:.4f}" if is_clf
                            else f"RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")
                    print(f"  [{ds:24}] {arch:12} seed={seed}: {disp} "
                          f"({m['train_time']:.1f}s, {m['n_params']:,}p)")
                agg = aggregate(per_seed, metric_keys + val_keys + ["train_time", "n_params"])
                results[ds]["architectures"][arch] = {"per_seed": per_seed, **agg}

    payload = {"config": {"mode": args.mode, "epochs": args.epochs,
                          "patience": args.patience, "seeds": args.seeds,
                          "search_seed": search_seed, "attention_heads": HEADS,
                          "config_grid": CONFIG_GRID if args.mode == "search" else None,
                          "base": {"batch_train": cfg.batch_train,
                                   "val_fraction": cfg.val_fraction}},
               "total_runs": total_runs, "results": results}
    suffix = "_smoke" if args.smoke else ("_search" if args.mode == "search" else "")
    out_json = OUT_DIR / f"architecture_selection{suffix}.json"
    # Merge into any existing payload so per-dataset / per-architecture chunked runs
    # accumulate (each run updates only the datasets/architectures it covered).
    if out_json.exists():
        try:
            prev = json.load(open(out_json))
            merged = prev.get("results", {})
            for ds, dres in results.items():
                if ds in merged and "architectures" in merged[ds]:
                    merged[ds]["architectures"].update(dres["architectures"])
                else:
                    merged[ds] = dres
            payload["results"] = merged
            payload["total_runs"] = prev.get("total_runs", 0) + total_runs
        except Exception:
            pass
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nTotal runs (this chunk): {total_runs}\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
