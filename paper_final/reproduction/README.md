# Reproduction guide — *ADMET-Bench: Task-Dependent Performance of GNNs and Pretrained Models in ADMET Prediction*

Primary manuscript: `paper_final/main_short.tex` + `paper_final/supplementary_materials_short.tex`.
Every number traces to one committed artifact and one generator script. All model
families are trained, selected, and evaluated on **one canonical split**
(`scripts/canonical_split.py`): a TDC Bemis–Murcko scaffold split for train/test and a
**molecule-level** validation carve (replicate measurements of a canonical SMILES never
straddle train/validation). Selection uses validation only; the held-out test split is
used once.

## One-command audit

```bash
python scripts/pre_submission_audit.py     # runs the two checks below + static checks
# or individually:
python scripts/verify_splits.py            # A. no train/val/test leakage; families share test
python scripts/validate_results.py         # B. artifacts agree with the manuscript numbers
```
All three exit non-zero on any failure.

## Table → artifact → generator map

| Paper element | Canonical artifact | Generator |
|---|---|---|
| **Table III** dataset stats (TDC / Used) | `results/fair_model_family/*` counts; verified by `verify_splits.py` (Used = train+val+test) | `optimized_gnn.prepare_dataset` / `scripts/canonical_split.py` |
| **Table IV** model-family comparison, GNN column (10-seed median) | `results/fair_model_family/table_IV_median_canonical.csv`; `results/multi_seed/multi_seed_results_fixed.json` | GNN config: `scripts/run_fair_gnn_benchmark.py` (50-trial, validation-selected random search) → medians: `scripts/run_multi_seed_validation.py` (10 seeds) |
| **Table IV** baseline columns (ChemBERTa/MolCLR/Morgan/MolE) | `results/baseline_hpo_parity/baseline_multiseed.json` | `scripts/run_baseline_hpo_parity.py` → `scripts/run_baseline_multiseed.py` (50-trial MLP-head search on frozen embeddings, 10 seeds) |
| **Table V** multi-seed stability | `results/multi_seed/multi_seed_results_fixed.json` | `scripts/run_multi_seed_validation.py` (10 seeds `[42,123,456,789,1011,2022,3033,7,99,2718]`) |
| **Ablation** (Caco-2 descriptor Δ) | `results/descriptor_ablation/descriptor_ablation.json` | `scripts/run_descriptor_ablation.py` (with/without RDKit descriptors, 10 seeds) |
| **Fine-tuning** (ChemBERTa-FT) | `results/chemberta_finetune/chemberta_ft_*_results_fixed.json` | `scripts/run_chemberta_finetune.py` (last-2-layer partial fine-tune, canonical split, train-only normalization) |
| **Supp. Table S3** HPO vs Random (robust paired stats) | `results/hpo_comparison_clean/robust_analysis.json` | `scripts/run_hpo_comparison_clean.py` → `scripts/analyze_hpo_robust.py` |
| **Supp. Table S5** per-optimizer test metrics | `results/hpo_comparison_clean/summary.json` (42 arms) | `scripts/run_hpo_comparison_clean.py` |
| **Imbalance metrics + Figs S3/S4** | `results/toxicity_extended/toxicity_extended_metrics.json` | `paper_final/reproduction/toxicity_extended_metrics.py` → `make_toxicity_figures.py`, `scripts/make_real_confusion_matrices.py` |
| **Supp. Tables S1/S2 + arch figures** | `results/architecture_selection/*` | `scripts/run_architecture_selection.py --mode search --epochs 50 --patience 12` → `scripts/make_architecture_figures.py` |

## Regenerating from scratch

All GPU runs are deterministic (seed 42 for selection). Reruns must be **foreground**;
each command is self-contained.

```bash
# 0. Canonical splits (single source of truth) — writes splits/*.json, verifies leak-free
python scripts/canonical_split.py

# 1. Fair GNN benchmark: 50-trial validation-selected random search per endpoint
python scripts/run_fair_gnn_benchmark.py           # -> results/fair_model_family/fair_gnn_summary.json

# 2. Table IV GNN column: 10-seed medians of the selected configs (loaded live from step 1)
python scripts/run_multi_seed_validation.py        # -> results/multi_seed/

# 3. Baselines (needs ChemBERTa HF weights + local MolCLR GIN checkpoint)
python scripts/run_baseline_hpo_parity.py
python scripts/run_baseline_multiseed.py           # -> results/baseline_hpo_parity/

# 4. Descriptor ablation (10 seeds)
python scripts/run_descriptor_ablation.py          # -> results/descriptor_ablation/

# 5. ChemBERTa partial fine-tuning on the canonical split (all six endpoints)
python scripts/run_chemberta_finetune.py           # -> results/chemberta_finetune/

# 6. HPO optimizer comparison (7 optimizers x 6 datasets x 50 trials) + robust stats
python scripts/run_hpo_comparison_clean.py         # -> results/hpo_comparison_clean/
python scripts/analyze_hpo_robust.py               # -> robust_analysis.json (Table S3)

# 7. Imbalance-aware toxicity metrics + figures
python paper_final/reproduction/toxicity_extended_metrics.py
python paper_final/reproduction/make_toxicity_figures.py
python scripts/make_real_confusion_matrices.py     # real test confusion matrices (Fig S3)

# 8. Architecture screen (Supp. S1/S2)
python scripts/run_architecture_selection.py --mode search --epochs 50 --patience 12
python scripts/make_architecture_figures.py

# 9. Main comparison + HPO figures
python scripts/regenerate_comparison_figures.py    # Table IV figures
python scripts/make_hpo_clean_figure.py            # Fig: HPO optimizer comparison

# 10. Audit everything
python scripts/pre_submission_audit.py
```

Core model shared by all runners: `optimized_gnn.py` (repo root, GraphConv backbone).
Exact environment: `requirements-lock.txt` (Python 3.12.9, torch 2.10.0+cu126,
torch-geometric 2.7.0, rdkit 2025.09.3, transformers 4.57.6).

## Known caveats (disclosed in the paper)

- **`runs/` is deprecated.** It held pre-audit, test-selected HPO results and is no
  longer consumed by any canonical generator; the clean re-run lives in
  `results/hpo_comparison_clean/`. `pre_submission_audit.py` enforces this.
- **Descriptor asymmetry**: only the GNN receives the auxiliary structure-derived RDKit
  descriptors (15; a 7-subset for Caco-2); the pretrained/fingerprint baselines do not.
  The Caco-2 ablation quantifies this (Δ ≈ +0.97).
- **Baseline validation partition** differs from the GNN's (baselines use TDC's native
  scaffold `valid`); all families are evaluated on the **same** scaffold test set.
- **MolE-FP is a placeholder**: Morgan fingerprints + a learned projection inspired by
  MolE, **not** the pretrained MolE encoder (disclosed in Section III-C).
- **Raw pre-audit results** are preserved under `results/_raw_backup_preaudit/` and never
  overwritten.
