"""Descriptor ablation for the deployed GraphConv GNN.

For each benchmark endpoint we take the best HPO-selected configuration used in the
main study (Table VI) and train the GraphConv backbone twice under the SAME config
and seed: once WITH the auxiliary physicochemical descriptors (the paper's setting)
and once WITHOUT them (adme_dim=0, descriptors dropped before the MLP head). The
gap between the two isolates the contribution of the auxiliary descriptor input from
the graph representation itself, addressing the input-asymmetry disclosure in the
review. No data, split, or training-protocol change other than the descriptor toggle.

Outputs: results/descriptor_ablation/descriptor_ablation.json
"""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from optimized_gnn import (  # noqa: E402
    OptimizedGNNConfig,
    build_loaders,
    train_epoch,
    evaluate,
    evaluate_classification,
    resolve_device,
    is_classification_dataset,
)

# Best HPO config per dataset (Table VI best algo), read from runs/.
BEST_ALGO = {
    "Caco2_Wang": "random", "Half_Life_Obach": "pso",
    "Clearance_Hepatocyte_AZ": "random", "Clearance_Microsome_AZ": "random",
    "tox21": "sa", "herg": "abc",
}
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 7, 99, 2718]
OUT_DIR = PROJECT_ROOT / "results" / "descriptor_ablation"


class GraphConvBackbone(nn.Module):
    """GraphConv stack with BatchNorm + ReLU + residual from layer 1 (mirrors main)."""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=5, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            self.convs.append(GraphConv(in_ch, hidden_dim))
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


class AblationGNN(nn.Module):
    """GraphConv GNN with a toggle for the auxiliary physicochemical descriptors."""

    def __init__(self, input_dim=8, hidden_dim=128, num_layers=5, adme_dim=15,
                 head_dims=(256, 128, 64), dropout=0.0, use_descriptors=True):
        super().__init__()
        self.use_descriptors = use_descriptors
        self.backbone = GraphConvBackbone(input_dim, hidden_dim, num_layers, dropout)
        readout_dim = hidden_dim * 2
        combined_dim = readout_dim + (adme_dim if use_descriptors else 0)
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
        if self.use_descriptors:
            adme = data.adme_features
            if adme.dim() == 1:
                adme = adme.unsqueeze(0)
            if adme.numel() > 0:
                pooled = torch.cat([pooled, adme], dim=-1)
        return self.head(pooled).squeeze(-1)


def train_variant(dataset_name, seed, cfg, use_descriptors, epochs, patience, device, cache):
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
    is_log = cache.get("is_log_transformed", False)
    sample = cache["train"][0]
    input_dim = int(sample.x.size(-1))
    adme_dim = int(sample.adme_features.numel())
    dropout = 0.5 if dataset_name == "Caco2_Wang" else 0.0
    label_noise = 0.05 if (dataset_name == "Caco2_Wang" and not is_clf) else 0.0

    model = AblationGNN(input_dim=input_dim, hidden_dim=cfg.hidden_dim,
                        num_layers=cfg.num_layers, adme_dim=adme_dim,
                        head_dims=cfg.head_dims, dropout=dropout,
                        use_descriptors=use_descriptors).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("-inf") if is_clf else float("inf")
    best_state, no_improve, t0 = None, 0, time.time()
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, opt, device, cfg.max_grad_norm,
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

    if is_clf:
        tm = evaluate_classification(model, test_loader, device)
        out = {"auc_roc": tm["auc_roc"], "f1": tm["f1"], "primary": tm["auc_roc"]}
    else:
        tm = evaluate(model, test_loader, device, mu, sigma, is_log)
        out = {"rmse": tm["rmse"], "r2": tm["r2"], "primary": tm["r2"]}
    out.update({"adme_dim": adme_dim, "train_time": time.time() - t0, "stopped_epoch": epoch})
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def load_best_cfg(dataset_name):
    # Canonical validation-selected config from the fair, leak-free GNN benchmark
    # (scripts/run_fair_gnn_benchmark.py). Replaces the earlier per-optimizer runs/,
    # which were selected via test-based optimizer picking and stored no trial history.
    fair = json.load(open(PROJECT_ROOT / "results" / "fair_model_family" / "fair_gnn_summary.json"))
    bp = fair[dataset_name]["selected_config"]
    return OptimizedGNNConfig(
        hidden_dim=bp["hidden_dim"], num_layers=bp["num_layers"],
        head_dims=bp["head_dims"], lr=bp["lr"], weight_decay=bp.get("weight_decay", 0.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(BEST_ALGO.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=12)
    args = ap.parse_args()

    device = resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "descriptor_ablation.json"
    # Merge into existing results so a chunked rerun keeps untouched endpoints.
    results = {}
    if out_file.exists():
        try:
            results = json.loads(out_file.read_text())
        except Exception:
            results = {}
    for ds in args.datasets:
        cfg = load_best_cfg(ds)
        is_clf = is_classification_dataset(ds)
        metric_name = "AUC" if is_clf else "R2"
        cache = None
        with_vals, without_vals = [], []
        rec = {"metric": metric_name, "with": [], "without": []}
        for seed in args.seeds:
            w = train_variant(ds, seed, cfg, True, args.epochs, args.patience, device, cache)
            wo = train_variant(ds, seed, cfg, False, args.epochs, args.patience, device, cache)
            with_vals.append(w["primary"]); without_vals.append(wo["primary"])
            rec["with"].append(w); rec["without"].append(wo)
            print(f"[{ds}] seed={seed} adme_dim={w['adme_dim']} "
                  f"with={w['primary']:.4f} without={wo['primary']:.4f} "
                  f"delta={w['primary']-wo['primary']:+.4f}", flush=True)
        rec["with_median"] = float(np.median(with_vals))
        rec["without_median"] = float(np.median(without_vals))
        rec["delta_median"] = rec["with_median"] - rec["without_median"]
        results[ds] = rec
        print(f"==> {ds}: with={rec['with_median']:.4f} without={rec['without_median']:.4f} "
              f"delta={rec['delta_median']:+.4f}", flush=True)
    out_file.write_text(json.dumps(results, indent=2))
    print("Saved", out_file, flush=True)


if __name__ == "__main__":
    main()
