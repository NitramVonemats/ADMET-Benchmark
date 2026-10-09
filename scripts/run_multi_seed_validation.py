"""
Multi-Seed Validation Script

Retrains the best HPO-selected configuration per dataset across multiple seeds to
characterise across-seed stability of the final GNN.

IMPORTANT (revision): this script now uses the SAME GraphConv training pipeline as
every other table in the paper, by calling optimized_gnn.train_model
(OptimalGraphBackbone / GraphConv). Previously it defined its own self-contained
GCNConv model, which was inconsistent with the paper's stated backbone and with the
model-comparison, HPO, and imbalance-metric tables. Regression metric is the
original-scale RMSE returned by optimized_gnn.evaluate (identical definition to the
rest of the study); classification metric is AUC-ROC.

Outputs -> results/multi_seed/ (gitignored); the released copy lives under
artifacts/multi_seed/.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from optimized_gnn import (  # noqa: E402
    OptimizedGNNConfig,
    train_model,
    resolve_device,
    is_classification_dataset,
)

# ============== CONFIGURATION ==============

DATASETS = {
    'Caco2_Wang': 'regression',
    'Half_Life_Obach': 'regression',
    'Clearance_Hepatocyte_AZ': 'regression',
    'Clearance_Microsome_AZ': 'regression',
    'tox21': 'classification',
    'herg': 'classification',
}

# Extended from 5 to 10 seeds (reviewer request) for the endpoints on which the
# main conclusions depend; kept uniform across all datasets for consistency.
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 7, 99, 2718]
MAX_EPOCHS = 50
PATIENCE = 12
OUTPUT_DIR = os.path.join(project_root, 'results', 'multi_seed')

# Canonical GNN configs = validation-selected best of the fair 50-trial random
# search. Loaded LIVE from results/fair_model_family/fair_gnn_summary.json (the
# single source of truth) so this script can never drift from the benchmark.
# dropout is handled inside train_model (0.5 for Caco2_Wang, else 0.0).
def _load_best_params():
    p = os.path.join(project_root, 'results', 'fair_model_family', 'fair_gnn_summary.json')
    summ = json.load(open(p))
    return {ds: {'hidden_dim': c['selected_config']['hidden_dim'],
                 'num_layers': c['selected_config']['num_layers'],
                 'lr': c['selected_config']['lr'],
                 'weight_decay': c['selected_config']['weight_decay'],
                 'head_dims': c['selected_config']['head_dims']}
            for ds, c in summ.items()}


BEST_PARAMS = _load_best_params()


def run_single_seed(dataset_name, seed, params, device):
    """Train the best-HPO GraphConv config for one seed; return test metrics."""
    cfg = OptimizedGNNConfig(
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        head_dims=params['head_dims'],
        lr=params['lr'],
        weight_decay=params['weight_decay'],
    )
    result = train_model(
        dataset_name=dataset_name, config=cfg,
        epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, seed=seed,
        evaluate_test=True, return_model=False, verbose=False,
    )
    tm = result['test_metrics']
    if is_classification_dataset(dataset_name):
        return {'auc': float(tm['auc_roc']), 'f1': float(tm['f1']),
                'accuracy': float(tm['accuracy'])}
    # Original-scale RMSE and R2 (identical definition to the rest of the study).
    return {'rmse_orig': float(tm['rmse']), 'mae_orig': float(tm['mae']),
            'r2': float(tm['r2'])}


def _stats(values):
    a = np.array(values, dtype=float)
    finite = a[np.isfinite(a)]
    base = finite if finite.size else a
    n = max(base.size, 1)
    return {
        'mean': float(np.mean(base)), 'std': float(np.std(base)),
        'median': float(np.median(base)),
        'min': float(np.min(base)), 'max': float(np.max(base)),
        'ci_lower': float(np.mean(base) - 1.96 * np.std(base) / np.sqrt(n)),
        'ci_upper': float(np.mean(base) + 1.96 * np.std(base) / np.sqrt(n)),
    }


def run_multi_seed_validation(dataset_name, task_type, device):
    print(f"\n{'='*60}\nMulti-Seed Validation: {dataset_name} ({task_type})")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}\n{'='*60}")
    params = BEST_PARAMS[dataset_name]

    per_seed = []
    for seed in SEEDS:
        m = run_single_seed(dataset_name, seed, params, device)
        per_seed.append(m)
        if task_type == 'classification':
            print(f"  seed {seed:5d}: AUC={m['auc']:.4f} F1={m['f1']:.4f}")
        else:
            print(f"  seed {seed:5d}: RMSE={m['rmse_orig']:.4f} R2={m['r2']:.4f}")

    summary = {'dataset': dataset_name, 'task_type': task_type, 'n_seeds': len(SEEDS)}
    if task_type == 'classification':
        aucs = [r['auc'] for r in per_seed]
        s = _stats(aucs)
        summary.update({'auc_values': aucs, 'auc_mean': s['mean'], 'auc_std': s['std'],
                        'auc_median': s['median'], 'auc_min': s['min'], 'auc_max': s['max'],
                        'f1_values': [r['f1'] for r in per_seed],
                        'ci_lower': s['ci_lower'], 'ci_upper': s['ci_upper']})
        print(f"  -> AUC median {s['median']:.4f} [{s['min']:.4f}-{s['max']:.4f}], "
              f"mean {s['mean']:.4f} +/- {s['std']:.4f}")
    else:
        rmses = [r['rmse_orig'] for r in per_seed]
        r2s = [r['r2'] for r in per_seed]
        s = _stats(rmses)
        rs = _stats(r2s)
        summary.update({'rmse_orig_values': rmses, 'rmse_orig_mean': s['mean'],
                        'rmse_orig_std': s['std'], 'rmse_orig_median': s['median'],
                        'rmse_orig_min': s['min'], 'rmse_orig_max': s['max'],
                        'r2_values': r2s, 'r2_median': rs['median'], 'r2_mean': rs['mean'],
                        'r2_std': rs['std'], 'r2_min': rs['min'], 'r2_max': rs['max'],
                        'ci_lower': s['ci_lower'], 'ci_upper': s['ci_upper']})
        print(f"  -> RMSE median {s['median']:.4f} [{s['min']:.4f}-{s['max']:.4f}]; "
              f"R2 median {rs['median']:.4f} [{rs['min']:.4f}-{rs['max']:.4f}]")
    return summary


def main():
    print("=" * 60)
    print("Multi-Seed Validation (GraphConv pipeline, 10 seeds)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = resolve_device("auto")
    print(f"Device: {device}")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()))
    a = ap.parse_args()

    out_path = os.path.join(OUTPUT_DIR, 'multi_seed_results_fixed.json')
    # Merge into any existing results so a partial (chunked) rerun preserves the
    # endpoints it did not touch.
    all_results = {}
    if os.path.exists(out_path):
        try:
            all_results = json.load(open(out_path))
        except Exception:
            all_results = {}
    for dataset_name in a.datasets:
        task_type = DATASETS[dataset_name]
        try:
            all_results[dataset_name] = run_multi_seed_validation(dataset_name, task_type, device)
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    for name, r in all_results.items():
        if r['task_type'] == 'classification':
            rows.append({'Dataset': name, 'Task': 'classification',
                         'AUC_median': r['auc_median'], 'AUC_min': r['auc_min'],
                         'AUC_max': r['auc_max'], 'AUC_mean': r['auc_mean'],
                         'AUC_std': r['auc_std']})
        else:
            rows.append({'Dataset': name, 'Task': 'regression',
                         'RMSE_median': r['rmse_orig_median'], 'RMSE_min': r['rmse_orig_min'],
                         'RMSE_max': r['rmse_orig_max'], 'R2_median': r['r2_median']})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'multi_seed_summary_fixed.csv'), index=False)

    print("\n" + "=" * 60 + "\nFINAL SUMMARY\n" + "=" * 60)
    print(df.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
