# Task-Dependent Performance of GNNs and Pretrained Models in ADMET Prediction

Martin Stamenov, Mila Gjurovska, Viktorija Vodilovska, Ilinka Ivanoska
Faculty of Computer Science and Engineering, Ss Cyril and Methodius University, Skopje, N. Macedonia

Code, canonical splits, and every number reported in the manuscript. The paper
benchmarks graph neural networks (GNN), frozen pretrained encoders (ChemBERTa,
MolCLR) and fingerprint baselines (Morgan-FP, MolE-FP) on six Therapeutics Data
Commons (TDC) ADMET endpoints under scaffold-split evaluation, with a matched
50-trial validation-selected tuning budget per model family and the median over ten
seeds. A separate study compares seven hyperparameter-optimization (HPO) strategies
against Random Search.

**Main findings.** No model family dominates: on most endpoints the leading
families differ by less than the across-seed range. The GNN is clearly best only on
Caco-2 (R² ≈ 0.48) and a controlled ablation shows that this edge comes entirely from
its auxiliary RDKit descriptors, not from message passing. Morgan fingerprints are
competitive on hERG (AUC ≈ 0.77) and give the best median on both clearance
endpoints; MolCLR is best on Tox21. Half-Life and Hepatocyte clearance are not
predictable from structure alone (all families near-zero or negative R²). No HPO
metaheuristic consistently outperforms Random Search at a 50-trial budget.

---

## 1. Evaluation protocol

| Item | Setting |
|---|---|
| Datasets | 6 TDC ADMET endpoints (4 ADME regression, 2 toxicity classification) |
| Split | TDC Bemis–Murcko scaffold split (train/test); validation carved from the scaffold-train side at the **molecule level** (10% of unique canonical SMILES, seed 42) → ≈70/8/22 |
| Selection | Validation only; test split used once for the final estimate |
| Budget | 50 trials per family: GNN = Random Search over architecture + training hyperparameters; baselines = search over the MLP head on frozen features |
| Robustness | 10 seeds `[42, 123, 456, 789, 1011, 2022, 3033, 7, 99, 2718]`, reported as **median [min–max]** |
| Targets | Half-Life, Hepatocyte and Microsomal clearance: `log(max(y, 1e-3))` then train-only z-score; Caco-2: z-score only |
| GNN | Stacked GraphConv + BatchNorm + ReLU + residuals, 8 atom features, no bond features, mean‖max pooling, concatenated with 15 RDKit descriptors (7 for Caco-2), MLP head; Adam, batch 32, MSE / BCE-with-logits, grad-clip 1.0, 50 epochs, patience 12; Caco-2 adds dropout 0.5 + label noise σ = 0.05 |
| Hardware | Intel Core i7-8700K, 16 GB RAM, NVIDIA RTX 3060 |

### Table I – dataset statistics

| Dataset | Type | TDC | Used | Metric |
|---|---|---:|---:|---|
| Caco2_Wang | Regr. | 910 | 819 | RMSE / R² |
| Half_Life_Obach | Regr. | 667 | 601 | RMSE / R² |
| Clearance_Hepatocyte_AZ | Regr. | 1,213 | 1,092 | RMSE / R² |
| Clearance_Microsome_AZ | Regr. | 1,102 | 992 | RMSE / R² |
| Tox21 (NR-AR) | Class. | 7,258 | 6,533 | AUC-ROC (+ PR-AUC) |
| hERG | Class. | 655 | 590 | AUC-ROC |
| **Total** | | **11,805** | **10,627** | |

"Used" = molecules retained after removing invalid SMILES / failed graph conversion
(= train + val + test). Tox21 positives: 4.2% of the full corpus, 4.9% of the scaffold
test set. Verified by `scripts/verify_splits.py`.

---

## 2. Results

### Table II – fair model-family comparison (median over 10 seeds, test split)

R² for regression, AUC-ROC for classification; **bold** = highest median.
Per-seed [min, max] in `artifacts/fair_model_family/table_IV_median_canonical.csv`.

| Task | Metric | GNN | ChemBERTa | Morgan-FP | MolE-FP | MolCLR |
|---|---|---:|---:|---:|---:|---:|
| *Group 1 – structure-driven* | | | | | | |
| hERG | AUC | 0.754 | 0.747 | **0.767** | 0.735 | 0.657 |
| Caco-2 | R² | **0.482** | 0.413 | 0.322 | 0.127 | 0.232 |
| *Group 2 – partially structure-driven* | | | | | | |
| Tox21 (NR-AR) | AUC | 0.718 | 0.714 | 0.706 | 0.698 | **0.722** |
| Clearance Mic. | R² | 0.093 | −0.010 | **0.169** | 0.056 | 0.125 |
| *Group 3 – weakly learnable from structure* | | | | | | |
| Half-Life | R² | **−0.010** | −1.918 | −6.341 | −3.776 | −0.672 |
| Clearance Hep. | R² | −0.373 | −0.224 | **−0.017** | −0.106 | −0.113 |

Per-seed ranges: GNN hERG [0.704, 0.809], Tox21 [0.695, 0.749], Caco-2 [0.111, 0.553],
Half-Life [−0.109, 0.281], Clear. Mic. [−0.652, 0.202], Clear. Hep. [−414.0, 0.061].

### Table III – across-seed stability of the GNN (median [min–max], 10 seeds)

| Dataset | Metric | Median [min–max] |
|---|---|---|
| Caco2_Wang | RMSE | 0.0027 [0.0025–0.0035] |
| Half_Life_Obach | RMSE | 21.80 [18.40–22.86] |
| Clearance_Hepatocyte_AZ | RMSE | 56.22 [46.53–977.95] |
| Clearance_Microsome_AZ | RMSE | 41.03 [38.48–55.37] |
| Tox21 (NR-AR) | AUC | 0.718 [0.695–0.749] |
| hERG | AUC | 0.754 [0.704–0.809] |

The validation-selected configuration diverges on a minority of seeds for the two
clearance endpoints (strongly negative R²), which is why medians are reported.

### Descriptor ablation (GNN with vs. without the RDKit descriptors, median of 10 seeds)

| Endpoint | With | Without | Δ |
|---|---:|---:|---:|
| Caco-2 (R²) | 0.455 | −0.513 | **+0.968** |
| Half-Life (R²) | 0.037 | 0.013 | +0.024 |
| Clearance Hep. (R²) | −0.354 | −0.040 | −0.314 |
| Clearance Mic. (R²) | 0.135 | 0.094 | +0.041 |
| Tox21 (AUC) | 0.708 | 0.697 | +0.011 |
| hERG (AUC) | 0.763 | 0.761 | +0.002 |

The GNN's Caco-2 advantage is descriptor-driven; on the toxicity endpoints the
descriptors add essentially nothing.

### ChemBERTa partial fine-tuning (last 2 transformer layers, 33.8% trainable, canonical split)

| Endpoint | Frozen (Table II) | Fine-tuned |
|---|---:|---:|
| hERG (AUC) | 0.747 | 0.770 |
| Tox21 (AUC) | 0.714 | 0.735 |
| Caco-2 (R²) | 0.413 | 0.390 |
| Half-Life (R²) | −1.918 | −0.081 |
| Clearance Hep. (R²) | −0.224 | −0.246 |
| Clearance Mic. (R²) | −0.010 | 0.009 |

Fine-tuning neither rescues the pharmacokinetic endpoints nor overturns the
frozen-feature ranking.

### Imbalance-aware toxicity metrics (GNN, mean ± std over 10 seeds, threshold tuned on validation)

| Endpoint | ROC-AUC | PR-AUC | F1 | MCC | Positives (test) |
|---|---:|---:|---:|---:|---:|
| Tox21 (NR-AR) | 0.722 ± 0.019 | 0.417 ± 0.038 | 0.444 ± 0.063 | 0.493 ± 0.071 | 4.9% |
| hERG | 0.770 ± 0.049 | 0.886 ± 0.032 | 0.791 ± 0.090 | 0.373 ± 0.092 | 73.5% |

Seed-42 confusion matrices at the 0.5 threshold: Tox21 F1 = 0.505, hERG F1 = 0.850
(`paper_final/images/confusion_matrices.png`).

### HPO optimizers vs. Random Search (clean, leak-free re-run; n = 6 endpoints)

Each optimizer's validation-selected configuration is evaluated once on test; paired
per-endpoint differences vs. Random Search (R² for regression, AUC for
classification; positive favours the optimizer). 95% CIs are 10,000-resample
percentile bootstraps.

| Optimizer | W/T/L | Sign-test p | Norm. effect [95% CI] | Clip [−1,1] mean [95% CI] | Floor-0 mean [95% CI] |
|---|:---:|---:|---|---|---|
| PSO | 2/0/4 | 0.69 | −0.16 [−1.35, +1.25] | +0.174 [−0.084, +0.612] | +0.033 [−0.036, +0.124] |
| ABC | 5/0/1 | 0.22 | +0.52 [−0.45, +1.38] | +0.026 [−0.292, +0.311] | +0.032 [+0.000, +0.091] |
| GA | 3/0/3 | 1.00 | +0.01 [−1.41, +1.40] | +0.169 [−0.067, +0.536] | +0.020 [−0.040, +0.086] |
| SA | 1/0/5 | 0.22 | −0.77 [−1.82, +0.55] | −0.135 [−0.471, +0.171] | −0.018 [−0.031, −0.006] |
| HC | 3/0/3 | 1.00 | +0.02 [−1.13, +1.37] | +0.034 [−0.440, +0.556] | +0.014 [−0.021, +0.062] |
| TPE | 2/0/4 | 0.69 | +0.01 [−1.02, +1.16] | +0.105 [−0.075, +0.329] | +0.027 [−0.024, +0.114] |

Note: on hERG, ABC (0.81001) and Random Search (0.80987) differ by 0.0001, i.e. a
tie at three decimals; counting it as a tie gives ABC 4/1/1 (p = 0.375). Either way no
optimizer is significant and every scale-free / clipped interval spans zero.

Per-optimizer test performance (regression R², classification AUC-ROC):

| Dataset | PSO | ABC | GA | SA | HC | Random | TPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Caco2_Wang | 0.508 | 0.553 | 0.450 | 0.519 | 0.519 | 0.552 | 0.515 |
| Half_Life_Obach | 0.119 | 0.211 | 0.193 | −0.285 | 0.037 | 0.035 | 0.274 |
| Clearance_Hep. | −0.269 | −0.762 | −0.219 | −1.351 | −0.989 | −0.114 | −0.283 |
| Clearance_Mic. | 0.235 | −0.387 | 0.058 | −0.529 | 0.130 | −3.288 | −0.365 |
| Tox21 (NR-AR) | 0.739 | 0.757 | 0.730 | 0.739 | 0.758 | 0.743 | 0.726 |
| hERG | 0.738 | 0.810 | 0.830 | 0.771 | 0.777 | 0.810 | 0.787 |

Optimizer implementation (`scripts/run_hpo_comparison_clean.py`, NiaPy 2.6.1 /
Optuna 4.7.0): 50 trials per optimizer and dataset (7 × 6 × 50 = 2,100 GNN training
runs); PSO / ABC / GA with `population_size=10`; SA and HC with NiaPy defaults; TPE via
`optuna.samplers.TPESampler` (Optuna default pruner). Search space:

| Hyperparameter | Space |
|---|---|
| Hidden dimension | {64, 96, 128, 192, 256, 384, 512} |
| Number of layers | {3, 4, 5, 6, 7} |
| Learning rate | [1e-4, 1e-2] log-uniform |
| Weight decay | [1e-6, 1e-2] log-uniform |
| MLP head layers 1 / 2 / 3 | {128…512} / {64…256} / {32…128}, sorted descending |

### Architecture screen (backbone selection, validation only)

Eight message-passing backbones (GraphConv, GCN, GAT, GraphSAGE, GIN, TAG, SGC,
TransformerConv) × 6 endpoints × best-of-5 configurations = 240 runs. Ranked by
validation performance no backbone dominates (average regression rank 3.0–6.0;
validation AUC 0.78–0.85 on Tox21, 0.88–0.92 on hERG; Caco-2 validation R²
0.51–0.63). GraphConv is mid-ranked (regression rank 4.25 of 8) and was adopted as a
single representative backbone. Full tables:
`artifacts/architecture_selection/table_S1_classification.tex`,
`table_S2_regression_ranking.tex`, `architecture_selection_summary.csv`.

---

## 3. Where every number comes from

`results/` (gitignored, local) is the working output tree; `artifacts/` is the
committed, released copy of the canonical result files (identical content). Every
paper element traces to one artifact and one generator script.

| Paper element | Artifact | Generator |
|---|---|---|
| Table I (TDC / Used) | `splits/<dataset>.json` | `scripts/canonical_split.py`, `scripts/verify_splits.py` |
| Table II, GNN column | `artifacts/fair_model_family/table_IV_median_canonical.csv`, `artifacts/multi_seed/multi_seed_results_fixed.json` | `scripts/run_fair_gnn_benchmark.py` (50-trial validation-selected Random Search → `fair_gnn_summary.json`) → `scripts/run_multi_seed_validation.py` |
| Table II, baseline columns | `artifacts/baseline_hpo_parity/baseline_hpo_parity_results.json` (50-trial head search), `baseline_multiseed.json` (10 seeds) | `scripts/run_baseline_hpo_parity.py` → `scripts/run_baseline_multiseed.py` |
| Table III (multi-seed) | `artifacts/multi_seed/multi_seed_results_fixed.json` | `scripts/run_multi_seed_validation.py` |
| Descriptor ablation | `artifacts/descriptor_ablation/descriptor_ablation.json` | `scripts/run_descriptor_ablation.py` |
| ChemBERTa fine-tuning | `artifacts/chemberta_finetune/chemberta_ft_<dataset>_results_fixed.json` | `scripts/run_chemberta_finetune.py` |
| Imbalance metrics, Figs. S3/S4 | `artifacts/toxicity_extended/toxicity_extended_metrics.json` | `scripts/toxicity_extended_metrics.py` → `scripts/make_toxicity_figures.py`, `scripts/make_real_confusion_matrices.py` |
| HPO vs. Random Search (Supp. Table S3) | `artifacts/hpo_comparison_clean/robust_analysis.json` | `scripts/run_hpo_comparison_clean.py` → `scripts/analyze_hpo_robust.py` |
| Per-optimizer test metrics (Supp. Table S5), Fig. 3 | `artifacts/hpo_comparison_clean/summary.json` (+ 42 per-arm `hpo_<dataset>_<algo>.json` with full trial histories) | `scripts/run_hpo_comparison_clean.py`, `scripts/make_hpo_clean_figure.py` |
| Architecture screen (Supp. Tables S1/S2, Figs. S1/S2) | `artifacts/architecture_selection/architecture_selection_search.json` | `scripts/run_architecture_selection.py --mode search --epochs 50 --patience 12` → `scripts/make_architecture_figures.py` |
| Fig. 2 (model rankings) | Table II values | `scripts/regenerate_comparison_figures.py` |

Figures referenced by the manuscript live in `paper_final/images/`
(`architecture.png` = Fig. 1, `foundation_ranking.png` = Fig. 2,
`01_algorithm_performance.png` = Fig. 3, `arch_selection_summary.png` = Fig. S1,
`arch_comparison_all_datasets.png` = Fig. S2, `confusion_matrices.png` = Fig. S3,
`toxicity_metric_stability.png` = Fig. S4).

### Audit

```bash
python scripts/verify_splits.py       # no train/val/test leakage; all families share the test split
python scripts/validate_results.py    # artifacts agree with each other (uses results/ if present, else artifacts/)
python scripts/pre_submission_audit.py
```

### Regenerating from scratch (GPU; run sequentially)

```bash
python scripts/canonical_split.py                  # 0. canonical splits -> splits/
python scripts/run_fair_gnn_benchmark.py           # 1. GNN 50-trial search  -> results/fair_model_family/
python scripts/run_multi_seed_validation.py        # 2. GNN 10 seeds         -> results/multi_seed/
python scripts/run_baseline_hpo_parity.py          # 3. baselines (needs ChemBERTa HF weights + MolCLR checkpoint in external/MolCLR)
python scripts/run_baseline_multiseed.py
python scripts/run_descriptor_ablation.py          # 4. ablation
python scripts/run_chemberta_finetune.py           # 5. partial fine-tuning
python scripts/run_hpo_comparison_clean.py         # 6. 7 optimizers x 6 datasets x 50 trials
python scripts/analyze_hpo_robust.py
python scripts/toxicity_extended_metrics.py        # 7. imbalance metrics + figures
python scripts/make_toxicity_figures.py
python scripts/make_real_confusion_matrices.py
python scripts/run_architecture_selection.py --mode search --epochs 50 --patience 12   # 8. backbone screen
python scripts/make_architecture_figures.py
python scripts/regenerate_comparison_figures.py    # 9. figures
python scripts/make_hpo_clean_figure.py
```

GPU runs are seeded (seed 42 for selection) but not bit-deterministic; the canonical
numbers are the ones in `artifacts/`.

---

## 4. Repository layout

```
optimized_gnn.py          GraphConv model, molecule-level split, training/evaluation (shared by every runner)
scripts/                  canonical runners, analysis and figure scripts (see table above)
artifacts/                released copies of the canonical result files
splits/                   canonical scaffold / molecule-level splits (JSON, one per endpoint)
paper_final/images/       manuscript figures
datasets/, data/          TDC data (downloaded on first use)
external/MolCLR/          MolCLR pretrained GIN checkpoint
optimization/, src/       HPO framework and legacy modules
archive/                  superseded experiments, scripts and documentation (not used by the paper)
requirements.txt, requirements-lock.txt
```

`runs/` holds the deprecated pre-audit HPO results (test-selected) and is not consumed
by any canonical script. `MolE-FP` is Morgan fingerprints + a learned projection
inspired by MolE, not the pretrained MolE encoder. Only the GNN receives the auxiliary
RDKit descriptors; the Caco-2 ablation quantifies this asymmetry.

## 5. Environment

Python 3.12.9, torch 2.10.0+cu126, torch-geometric 2.7.0, rdkit 2025.09.3,
transformers 4.57.6, optuna 4.7.0, niapy 2.6.1 — exact pins in `requirements-lock.txt`.

```bash
pip install -r requirements.txt
```

## License

MIT
