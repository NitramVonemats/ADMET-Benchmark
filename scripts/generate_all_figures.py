#!/usr/bin/env python3
"""
Generate ALL paper-sources-2 figures from existing HPO JSON and foundation CSV data.
NO training — read-only from existing results.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

# ============================================================
# PATHS
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(BASE, "runs")
FOUNDATION_CSV = os.path.join(
    BASE, "archive", "old_experiments", "history", "old_results",
    "foundation_comparison_UPDATED_20260129_200243.csv"
)
OUT_PAPER1 = os.path.join(BASE, "paper_final", "images")
OUT_FIG = os.path.join(BASE, "figures", "paper-sources-2")

os.makedirs(OUT_PAPER1, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
REGRESSION_DATASETS = [
    "Caco2_Wang", "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ"
]
CLASSIFICATION_DATASETS = ["herg", "tox21"]
ALL_DATASETS = REGRESSION_DATASETS + CLASSIFICATION_DATASETS
ALGORITHMS = ["abc", "ga", "hc", "pso", "random", "sa"]
ALGO_LABELS = {
    "abc": "ABC", "ga": "GA", "hc": "HC",
    "pso": "PSO", "random": "Random", "sa": "SA"
}
DATASET_SHORT = {
    "Caco2_Wang": "Caco2", "Half_Life_Obach": "Half-Life",
    "Clearance_Hepatocyte_AZ": "CL-Hep",
    "Clearance_Microsome_AZ": "CL-Mic",
    "herg": "hERG", "tox21": "Tox21"
}

# ============================================================
# STYLE
# ============================================================
sns.set_style("white")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.facecolor': 'white',
})

blues_cmap = plt.cm.Blues
algo_colors = {
    a: blues_cmap(0.3 + 0.7 * i / (len(ALGORITHMS) - 1))
    for i, a in enumerate(ALGORITHMS)
}

# ============================================================
# DATA LOADING
# ============================================================
def load_hpo_data():
    data = {}
    for ds in ALL_DATASETS:
        data[ds] = {}
        for algo in ALGORITHMS:
            fpath = os.path.join(RUNS, ds, f"hpo_{ds}_{algo}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    data[ds][algo] = json.load(f)
            else:
                print(f"  WARNING: Missing {fpath}")
    return data

def load_foundation_csv():
    return pd.read_csv(FOUNDATION_CSV)

print("Loading data...")
hpo_data = load_hpo_data()
foundation_df = load_foundation_csv()
print("Data loaded.\n")

# ============================================================
# HELPERS
# ============================================================
def save_fig(fig, name):
    for d in [OUT_PAPER1, OUT_FIG]:
        out = os.path.join(d, name)
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> {name}")
    plt.close(fig)


def style_ax(ax):
    ax.grid(False)
    ax.set_facecolor('white')
    sns.despine(ax=ax)


# ============================================================
# FIGURE 1 — Regression Performance  (01_algorithm_performance.png)
# 2x2: (A) RMSE, (B) R2, (C) Time, (D) RMSE-vs-Time scatter
# ============================================================
def fig1_regression_performance():
    print("[1/8] Regression Performance...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rmse, r2, ttime = {}, {}, {}
    for ds in REGRESSION_DATASETS:
        rmse[ds], r2[ds], ttime[ds] = [], [], []
        for algo in ALGORITHMS:
            tm = hpo_data[ds][algo]['final_training']['test_metrics']
            rmse[ds].append(tm['rmse'])
            r2[ds].append(tm['r2'])
            ttime[ds].append(hpo_data[ds][algo]['final_training']['train_time'])

    x = np.arange(len(REGRESSION_DATASETS))
    w = 0.13
    labels = [ALGO_LABELS[a] for a in ALGORITHMS]
    short = [DATASET_SHORT[d] for d in REGRESSION_DATASETS]

    # (A) Test RMSE
    ax = axes[0, 0]
    for i, algo in enumerate(ALGORITHMS):
        vals = [rmse[ds][i] for ds in REGRESSION_DATASETS]
        ax.bar(x + (i - 2.5) * w, vals, w, color=algo_colors[algo], label=labels[i],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('Test RMSE'); ax.set_title('(A) Test RMSE by Algorithm')
    ax.legend(fontsize=7, ncol=3, frameon=False); style_ax(ax)

    # (B) Test R2
    ax = axes[0, 1]
    for i, algo in enumerate(ALGORITHMS):
        vals = [r2[ds][i] for ds in REGRESSION_DATASETS]
        ax.bar(x + (i - 2.5) * w, vals, w, color=algo_colors[algo], label=labels[i],
               edgecolor='white', linewidth=0.4)
    ax.axhline(0, color='#d62728', ls='--', lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('Test R\u00b2'); ax.set_title('(B) Test R\u00b2 by Algorithm')
    ax.legend(fontsize=7, ncol=3, frameon=False); style_ax(ax)

    # (C) Training Time
    ax = axes[1, 0]
    for i, algo in enumerate(ALGORITHMS):
        vals = [ttime[ds][i] for ds in REGRESSION_DATASETS]
        ax.bar(x + (i - 2.5) * w, vals, w, color=algo_colors[algo], label=labels[i],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('Training Time (s)'); ax.set_title('(C) Training Time by Algorithm')
    ax.legend(fontsize=7, ncol=3, frameon=False); style_ax(ax)

    # (D) Performance vs Time scatter
    ax = axes[1, 1]
    for i, algo in enumerate(ALGORITHMS):
        ts = [ttime[ds][i] for ds in REGRESSION_DATASETS]
        rs = [rmse[ds][i] for ds in REGRESSION_DATASETS]
        ax.scatter(ts, rs, color=algo_colors[algo], label=labels[i],
                   s=70, edgecolors='black', linewidth=0.5, zorder=3)
    ax.set_xlabel('Training Time (s)'); ax.set_ylabel('Test RMSE')
    ax.set_title('(D) Performance vs Training Time')
    ax.legend(fontsize=7, ncol=2, frameon=False); style_ax(ax)

    fig.suptitle('GNN Regression Performance across ADME Datasets',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, '01_algorithm_performance.png')


# ============================================================
# FIGURE 2 — Classification Performance  (05_classification_performance.png)
# ============================================================
def fig2_classification_performance():
    print("[2/8] Classification Performance...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    auc, f1, ttime = {}, {}, {}
    for ds in CLASSIFICATION_DATASETS:
        auc[ds], f1[ds], ttime[ds] = [], [], []
        for algo in ALGORITHMS:
            tm = hpo_data[ds][algo]['final_training']['test_metrics']
            auc[ds].append(tm['auc_roc'])
            f1[ds].append(tm['f1'])
            ttime[ds].append(hpo_data[ds][algo]['final_training']['train_time'])

    x = np.arange(len(CLASSIFICATION_DATASETS))
    w = 0.13
    labels = [ALGO_LABELS[a] for a in ALGORITHMS]
    short = [DATASET_SHORT[d] for d in CLASSIFICATION_DATASETS]

    # (A) AUC-ROC
    ax = axes[0, 0]
    for i, algo in enumerate(ALGORITHMS):
        vals = [auc[ds][i] for ds in CLASSIFICATION_DATASETS]
        ax.bar(x + (i - 2.5) * w, vals, w, color=algo_colors[algo], label=labels[i],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('Test AUC-ROC'); ax.set_title('(A) Test AUC-ROC by Algorithm')
    ax.legend(fontsize=7, ncol=3, frameon=False); style_ax(ax)

    # (B) F1
    ax = axes[0, 1]
    for i, algo in enumerate(ALGORITHMS):
        vals = [f1[ds][i] for ds in CLASSIFICATION_DATASETS]
        ax.bar(x + (i - 2.5) * w, vals, w, color=algo_colors[algo], label=labels[i],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('Test F1 Score'); ax.set_title('(B) Test F1 Score by Algorithm')
    ax.legend(fontsize=7, ncol=3, frameon=False); style_ax(ax)

    # (C) Training Time
    ax = axes[1, 0]
    for i, algo in enumerate(ALGORITHMS):
        vals = [ttime[ds][i] for ds in CLASSIFICATION_DATASETS]
        ax.bar(x + (i - 2.5) * w, vals, w, color=algo_colors[algo], label=labels[i],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('Training Time (s)'); ax.set_title('(C) Training Time by Algorithm')
    ax.legend(fontsize=7, ncol=3, frameon=False); style_ax(ax)

    # (D) AUC vs F1 scatter
    ax = axes[1, 1]
    markers = {'herg': 'o', 'tox21': 's'}
    for ds in CLASSIFICATION_DATASETS:
        for i, algo in enumerate(ALGORITHMS):
            ax.scatter(auc[ds][i], f1[ds][i],
                       color=algo_colors[algo], marker=markers[ds],
                       s=90, edgecolors='black', linewidth=0.5, zorder=3)
    h_algo = [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor=algo_colors[a], markersize=8, label=ALGO_LABELS[a])
              for a in ALGORITHMS]
    h_ds = [Line2D([0], [0], marker='o', color='gray', ls='None', ms=8, label='hERG'),
            Line2D([0], [0], marker='s', color='gray', ls='None', ms=8, label='Tox21')]
    ax.legend(handles=h_algo + h_ds, fontsize=7, ncol=2, frameon=False)
    ax.set_xlabel('Test AUC-ROC'); ax.set_ylabel('Test F1 Score')
    ax.set_title('(D) AUC-ROC vs F1 Trade-off'); style_ax(ax)

    fig.suptitle('GNN Classification Performance on Toxicity Datasets',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, '05_classification_performance.png')


# ============================================================
# FIGURE 3 — HPO Convergence Curves  (hpo_convergence_curves.png)
# 2x3 — one per dataset, best-so-far val metric vs epoch
# ============================================================
def fig3_convergence():
    print("[3/8] HPO Convergence Curves...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, ds in enumerate(ALL_DATASETS):
        ax = axes[idx // 3, idx % 3]
        is_reg = ds in REGRESSION_DATASETS

        for algo in ALGORITHMS:
            hist = hpo_data[ds][algo]['final_training']['history']
            epochs = [h['epoch'] for h in hist]
            key = 'val_rmse' if is_reg else 'val_auc_roc'
            raw = [h[key] for h in hist]

            # best-so-far
            bsf = []
            if is_reg:
                best = float('inf')
                for v in raw:
                    best = min(best, v); bsf.append(best)
            else:
                best = -float('inf')
                for v in raw:
                    best = max(best, v); bsf.append(best)

            ax.plot(epochs, bsf, color=algo_colors[algo],
                    label=ALGO_LABELS[algo], lw=1.5, alpha=0.85)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Val RMSE' if is_reg else 'Best Val AUC-ROC')
        ax.set_title(DATASET_SHORT[ds], fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, ncol=2, frameon=False)
        style_ax(ax)

    fig.suptitle(
        'Training Convergence: Best-so-far Validation Metric per Epoch\n'
        '(Per-trial HPO search history not stored in JSON; showing final-training curves)',
        fontsize=12, fontweight='bold', y=1.03)
    fig.tight_layout()
    save_fig(fig, 'hpo_convergence_curves.png')


# ============================================================
# FIGURE 4 — Confusion Matrices  (confusion_matrices.png)
# ============================================================
def fig4_confusion_matrices():
    print("[4/8] Confusion Matrices...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = [
        ('tox21', 'sa',  'Tox21 (SA — best GNN)', 1453),
        ('herg',  'abc', 'hERG (ABC — best GNN)',  132),
    ]

    for idx, (ds, algo, title, n_total) in enumerate(configs):
        ax = axes[idx]
        tm = hpo_data[ds][algo]['final_training']['test_metrics']
        prec  = tm['precision']
        rec   = tm['recall']
        acc   = tm['accuracy']
        f1v   = tm['f1']
        auc_v = tm['auc_roc']

        # Reconstruct CM from precision, recall, accuracy, N
        # P_pos = N*(1-acc) / (rec/prec - 2*rec + 1)
        denom = rec / prec - 2 * rec + 1
        P_pos = int(round(n_total * (1 - acc) / denom))
        TP = int(round(rec * P_pos))
        FN = P_pos - TP
        FP = int(round(TP / prec - TP)) if prec > 0 else 0
        TN = n_total - TP - FN - FP

        cm = np.array([[TN, FP], [FN, TP]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='white')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'{title}\nAcc={acc:.3f}  F1={f1v:.3f}  AUC={auc_v:.3f}')
        ax.set_facecolor('white')

    fig.suptitle('Confusion Matrices for Best GNN Classification Models',
                 fontsize=14, fontweight='bold', y=1.04)
    fig.tight_layout()
    save_fig(fig, 'confusion_matrices.png')


# ============================================================
# FIGURE 5 — GNN vs Foundation  (gnn_vs_foundation_comparison.png)
# ============================================================
def fig5_gnn_vs_foundation():
    print("[5/8] GNN vs Foundation Comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fdf = foundation_df[foundation_df['status'] == 'success'].copy()
    models = ['GNN-Best', 'Morgan-FP', 'ChemBERTa', 'MolE-FP', 'MolCLR']
    model_labels = {
        'GNN-Best': 'GNN (Ours)', 'Morgan-FP': 'Morgan-FP',
        'ChemBERTa': 'ChemBERTa', 'MolE-FP': 'MolE-FP', 'MolCLR': 'MolCLR'
    }
    teal = plt.cm.GnBu
    mcols = {m: teal(0.3 + 0.7 * i / (len(models) - 1)) for i, m in enumerate(models)}

    def _val(row_df, col):
        if len(row_df) == 0:
            return np.nan
        v = row_df.iloc[0][col]
        return v if pd.notna(v) else np.nan

    x_reg = np.arange(len(REGRESSION_DATASETS))
    x_cls = np.arange(len(CLASSIFICATION_DATASETS))
    w = 0.15

    # (A) ADME RMSE
    ax = axes[0, 0]
    for i, m in enumerate(models):
        vals = [_val(fdf[(fdf['dataset'] == ds) & (fdf['model'] == m)], 'test_rmse')
                for ds in REGRESSION_DATASETS]
        vals = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(x_reg + (i - 2) * w, vals, w, color=mcols[m], label=model_labels[m],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x_reg)
    ax.set_xticklabels([DATASET_SHORT[d] for d in REGRESSION_DATASETS], fontsize=9)
    ax.set_ylabel('Test RMSE'); ax.set_title('(A) ADME Regression: RMSE')
    ax.legend(fontsize=7, ncol=2, frameon=False); style_ax(ax)
    ax.annotate('*Caco2: GNN RMSE in original-space;\nfoundation models in log-space',
                xy=(0.02, 0.97), xycoords='axes fraction', fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))

    # (B) ADME R2
    ax = axes[0, 1]
    for i, m in enumerate(models):
        vals = [_val(fdf[(fdf['dataset'] == ds) & (fdf['model'] == m)], 'test_r2')
                for ds in REGRESSION_DATASETS]
        vals = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(x_reg + (i - 2) * w, vals, w, color=mcols[m], label=model_labels[m],
               edgecolor='white', linewidth=0.4)
    ax.axhline(0, color='#d62728', ls='--', lw=1, alpha=0.7)
    ax.set_xticks(x_reg)
    ax.set_xticklabels([DATASET_SHORT[d] for d in REGRESSION_DATASETS], fontsize=9)
    ax.set_ylabel('Test R\u00b2'); ax.set_title('(B) ADME Regression: R\u00b2')
    ax.legend(fontsize=7, ncol=2, frameon=False); style_ax(ax)

    # (C) Toxicity AUC-ROC
    ax = axes[1, 0]
    for i, m in enumerate(models):
        vals = [_val(fdf[(fdf['dataset'] == ds) & (fdf['model'] == m)], 'test_auc')
                for ds in CLASSIFICATION_DATASETS]
        vals = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(x_cls + (i - 2) * w, vals, w, color=mcols[m], label=model_labels[m],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x_cls)
    ax.set_xticklabels([DATASET_SHORT[d] for d in CLASSIFICATION_DATASETS])
    ax.set_ylabel('Test AUC-ROC'); ax.set_title('(C) Toxicity: AUC-ROC')
    ax.legend(fontsize=7, ncol=2, frameon=False); style_ax(ax)

    # (D) Toxicity F1
    ax = axes[1, 1]
    for i, m in enumerate(models):
        vals = [_val(fdf[(fdf['dataset'] == ds) & (fdf['model'] == m)], 'test_f1')
                for ds in CLASSIFICATION_DATASETS]
        vals = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(x_cls + (i - 2) * w, vals, w, color=mcols[m], label=model_labels[m],
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x_cls)
    ax.set_xticklabels([DATASET_SHORT[d] for d in CLASSIFICATION_DATASETS])
    ax.set_ylabel('Test F1 Score'); ax.set_title('(D) Toxicity: F1 Score')
    ax.legend(fontsize=7, ncol=2, frameon=False); style_ax(ax)

    fig.suptitle('GNN vs Foundation Models Comparison',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, 'gnn_vs_foundation_comparison.png')


# ============================================================
# FIGURE 6 — Foundation Ranking Heatmap  (foundation_ranking.png)
# ============================================================
def fig6_foundation_ranking():
    print("[6/8] Foundation Rankings Heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fdf = foundation_df[foundation_df['status'] == 'success'].copy()
    models = ['GNN-Best', 'Morgan-FP', 'ChemBERTa', 'MolE-FP', 'MolCLR']
    mlabels = [m.replace('GNN-Best', 'GNN (Ours)') for m in models]

    rank_cmap = LinearSegmentedColormap.from_list(
        'rank', ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'])

    def rank_rows(datasets, metric, higher_better):
        matrix = []
        for ds in datasets:
            raw = []
            for m in models:
                r = fdf[(fdf['dataset'] == ds) & (fdf['model'] == m)]
                raw.append(r.iloc[0][metric] if len(r) > 0 and pd.notna(r.iloc[0][metric]) else np.nan)
            valid = [(v, j) for j, v in enumerate(raw) if not np.isnan(v)]
            valid.sort(reverse=higher_better)
            ranks = [np.nan] * len(raw)
            for ri, (_, j) in enumerate(valid):
                ranks[j] = ri + 1
            matrix.append(ranks)
        return np.array(matrix)

    # (A) ADME by RMSE
    ax = axes[0]
    arr = rank_rows(REGRESSION_DATASETS, 'test_rmse', higher_better=False)
    sns.heatmap(arr, annot=True, fmt='.0f', cmap=rank_cmap, vmin=1, vmax=5,
                ax=ax, xticklabels=mlabels,
                yticklabels=[DATASET_SHORT[d] for d in REGRESSION_DATASETS],
                cbar_kws={'label': 'Rank'}, linewidths=0.8, linecolor='white')
    ax.set_title('(A) ADME Rankings by RMSE', fontweight='bold')
    ax.set_facecolor('white')

    # (B) Toxicity by AUC-ROC
    ax = axes[1]
    arr = rank_rows(CLASSIFICATION_DATASETS, 'test_auc', higher_better=True)
    sns.heatmap(arr, annot=True, fmt='.0f', cmap=rank_cmap, vmin=1, vmax=5,
                ax=ax, xticklabels=mlabels,
                yticklabels=[DATASET_SHORT[d] for d in CLASSIFICATION_DATASETS],
                cbar_kws={'label': 'Rank'}, linewidths=0.8, linecolor='white')
    ax.set_title('(B) Toxicity Rankings by AUC-ROC', fontweight='bold')
    ax.set_facecolor('white')

    fig.suptitle('Foundation Model Rankings  (1 = Best, 5 = Worst)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'foundation_ranking.png')


# ============================================================
# FIGURE 7 — Parameter Sensitivity Heatmap  (param_sensitivity_heatmap.png)
# ============================================================
def fig7_param_sensitivity():
    print("[7/8] Parameter Sensitivity Heatmap...")

    param_names = ['hidden_dim', 'num_layers', 'lr', 'weight_decay']
    param_labels = ['Hidden Dim', 'Num Layers', 'Learning Rate', 'Weight Decay']

    corr_matrix = []
    for ds in ALL_DATASETS:
        pvals = {p: [] for p in param_names}
        metric = []
        for algo in ALGORITHMS:
            bp = hpo_data[ds][algo]['search']['best_params']
            for p in param_names:
                pvals[p].append(bp[p])
            # performance: negative best_val_rmse already means
            # higher = better for classification (stored as -auc)
            # for regression lower rmse = better -> negate
            metric.append(-hpo_data[ds][algo]['search']['best_val_rmse'])

        row = []
        for p in param_names:
            if len(set(pvals[p])) > 1:
                c, _ = stats.pearsonr(pvals[p], metric)
                row.append(c)
            else:
                row.append(0.0)
        corr_matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(np.array(corr_matrix), annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=param_labels,
                yticklabels=[DATASET_SHORT[d] for d in ALL_DATASETS],
                cbar_kws={'label': 'Pearson r'},
                linewidths=0.6, linecolor='white')
    ax.set_title('Hyperparameter \u2013 Performance Correlation\n'
                 '(Positive = higher value \u2192 better metric)',
                 fontsize=13, fontweight='bold')
    ax.set_facecolor('white')
    fig.tight_layout()
    save_fig(fig, 'param_sensitivity_heatmap.png')


# ============================================================
# FIGURE 8 — Sankey Diagram  (sankey-diagram.png)
# ============================================================
def fig8_sankey():
    print("[8/8] Sankey Diagram...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.3, 9.2)
    ax.set_ylim(-0.5, 13)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    bc = plt.cm.Blues

    # --- helpers ---
    def box(x, y, w, h, label, color, fs=9):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           fc=color, ec='#333333', lw=1.0)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label,
                ha='center', va='center', fontsize=fs, fontweight='bold',
                color='black' if sum(color[:3]) > 1.5 else 'white')

    def flow(x1, y1_bot, h1, x2, y2_bot, h2, color, alpha=0.22):
        verts = [
            (x1, y1_bot),
            ((x1 + x2) / 2, y1_bot),
            ((x1 + x2) / 2, y2_bot),
            (x2, y2_bot),
            (x2, y2_bot + h2),
            ((x1 + x2) / 2, y2_bot + h2),
            ((x1 + x2) / 2, y1_bot + h1),
            (x1, y1_bot + h1),
            (x1, y1_bot),
        ]
        codes = [Path.MOVETO] + [Path.CURVE4] * 3 + \
                [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(
            Path(verts, codes), fc=color, alpha=alpha, ec='none'))

    # --- data ---
    total = 11805;  adme_tot = 3892;  tox_tot = 7913
    adme_ds = [('Caco2_Wang', 910), ('Half_Life_Obach', 667),
               ('CL_Hepatocyte', 1213), ('CL_Microsome', 1102)]
    tox_ds  = [('Tox21', 7258), ('hERG', 655)]

    S = 12.0 / total   # vertical scale
    gap = 0.35

    # Level 0 — TDC
    tdc_h = total * S
    tdc_y = (13 - tdc_h) / 2
    box(0, tdc_y, 2.2, tdc_h, f'TDC Repository\n({total:,})', bc(0.25), 12)

    # Level 1 — ADME / Toxicity
    adme_h = adme_tot * S;  tox_h = tox_tot * S
    block_h = adme_h + tox_h + gap
    base_y = (13 - block_h) / 2
    tox_y  = base_y
    adme_y = base_y + tox_h + gap

    box(3.6, adme_y, 2.0, adme_h, f'ADME\n({adme_tot:,})', bc(0.40), 11)
    box(3.6, tox_y,  2.0, tox_h,  f'Toxicity\n({tox_tot:,})', bc(0.50), 11)
    flow(2.2, adme_y, adme_h, 3.6, adme_y, adme_h, bc(0.35))
    flow(2.2, tox_y,  tox_h,  3.6, tox_y,  tox_h,  bc(0.45))

    # Level 2 — individual datasets
    def place_children(parent_y, parent_h, children, x_left, x_right, intensity_start):
        total_child = sum(c[1] for c in children)
        total_gaps  = gap * (len(children) - 1)
        child_scale = (parent_h - total_gaps) / total_child if total_child else S
        cy = parent_y + parent_h
        for i, (name, size) in enumerate(children):
            h = size * child_scale
            cy -= h
            ci = intensity_start + 0.12 * i
            box(x_right, cy, 2.0, h, f'{name}\n({size:,})', bc(ci), 9)
            flow(x_left, cy, h, x_right, cy, h, bc(ci))
            cy -= gap

    place_children(adme_y, adme_h, adme_ds, 5.6, 7.0, 0.45)
    place_children(tox_y,  tox_h,  tox_ds,  5.6, 7.0, 0.58)

    ax.set_title('Dataset Flow: TDC Repository \u2192 Task Categories → Datasets',
                 fontsize=14, fontweight='bold', pad=15)
    save_fig(fig, 'sankey-diagram.png')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  GENERATING ALL PAPER FIGURES")
    print("=" * 60)
    fig1_regression_performance()
    fig2_classification_performance()
    fig3_convergence()
    fig4_confusion_matrices()
    fig5_gnn_vs_foundation()
    fig6_foundation_ranking()
    fig7_param_sensitivity()
    fig8_sankey()
    print("\n" + "=" * 60)
    print("  ALL 8 FIGURES GENERATED SUCCESSFULLY")
    print(f"  Output: {OUT_PAPER1}")
    print(f"  Output: {OUT_FIG}")
    print("=" * 60)
