# MANU Project -- Complete Documentation

**Framework for Benchmarking and Optimization of Small Molecule Foundation Models for ADMET**

**Authors:** Martin, Mila, Adrian, Viktorija, Ilinka
**Version:** 1.0 (March 2026)
**Target Journal:** Bioinformatics

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Motivation](#2-background-and-motivation)
3. [Datasets](#3-datasets)
4. [Model Architecture](#4-model-architecture)
5. [Hyperparameter Optimization Framework](#5-hyperparameter-optimization-framework)
6. [Foundation Model Baselines](#6-foundation-model-baselines)
7. [Experimental Setup](#7-experimental-setup)
8. [Results and Analysis](#8-results-and-analysis)
9. [Multi-Seed Validation](#9-multi-seed-validation)
10. [Statistical Significance Testing](#10-statistical-significance-testing)
11. [Foundation Model Comparison](#11-foundation-model-comparison)
12. [Discussion](#12-discussion)
13. [Codebase Architecture](#13-codebase-architecture)
14. [API Reference](#14-api-reference)
15. [Reproducibility Guide](#15-reproducibility-guide)
16. [Troubleshooting](#16-troubleshooting)
17. [Appendices](#17-appendices)

---

# 1. Introduction

## 1.1 Project Overview

MANU is a systematic benchmarking framework for evaluating hyperparameter optimization (HPO) strategies applied to Graph Neural Networks (GNNs) for molecular property prediction. The project focuses on ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties, which are critical in early-stage drug discovery.

The framework evaluates **seven HPO algorithms** across **six datasets** from the Therapeutics Data Commons (TDC), producing over **2,100 model training runs**. It additionally compares optimized GNNs against **four foundation model baselines** and validates results with **5-seed statistical testing**.

## 1.2 Research Questions

1. **Which HPO algorithm should practitioners choose** for GNN-based molecular property prediction under scaffold-split evaluation?
2. **How does scaffold-split evaluation change optimizer rankings** compared to random-split settings?
3. **Can task-specific GNNs with systematic HPO match or exceed frozen pretrained foundation models?**
4. **What are the fundamental limits** of structure-only models for complex pharmacokinetic properties?

## 1.3 Key Contributions

- A **reproducible benchmarking framework** evaluating 7 HPO algorithms across 6 ADMET datasets under scaffold-split evaluation.
- **2,100+ training runs** with systematic analysis of optimization behavior and convergence.
- **Empirical comparison** between optimized task-specific GNNs and pretrained molecular foundation models.
- **Multi-seed validation** (5 seeds) with 95% confidence intervals and Wilcoxon signed-rank significance tests.
- **Practical recommendations** for HPO strategy selection under limited compute budgets.

---

# 2. Background and Motivation

## 2.1 ADMET Prediction in Drug Discovery

Accurate prediction of ADMET properties is essential for early-stage drug discovery. These properties determine whether a drug candidate will be absorbed into the bloodstream, distributed to the target tissue, metabolized and eliminated safely, and free of toxic effects. Laboratory measurement of these properties is expensive and time-consuming, motivating computational approaches.

**Key challenges:**
- Datasets are small (hundreds to low thousands of compounds).
- Properties depend on complex biological mechanisms beyond molecular structure alone.
- Evaluation protocol (random vs. scaffold split) dramatically affects reported performance.

## 2.2 Graph Neural Networks for Molecules

GNNs naturally represent molecules as graphs (atoms as nodes, bonds as edges). Unlike fingerprint-based methods that use fixed-length vectors, GNNs learn task-specific representations directly from the molecular graph through message-passing operations.

**GNN backbone used in this project:**
- **Architecture:** Graph Convolutional Network (GCN/GraphConv)
- **Node features:** 8 atom features (atomic number, degree, charge, hybridization, aromaticity, ring membership, H-count, atomic mass)
- **Edge features:** Bond type and aromaticity flag (used for graph construction but not message-passing)
- **Readout:** Concatenated global mean and max pooling
- **Prediction head:** 3-layer MLP with batch normalization and dropout

## 2.3 The Scaffold-Split Challenge

Standard random splitting of datasets creates an unrealistically optimistic evaluation: training and test molecules share similar scaffolds (core structures), allowing the model to memorize scaffold-specific patterns. **Scaffold splitting** (Bemis-Murcko) separates molecules by their core ring structures, forcing the model to generalize to chemically novel compounds.

This project uses scaffold splitting exclusively, producing results that more faithfully reflect real drug discovery performance.

## 2.4 Foundation Models in Molecular ML

Pretrained molecular models (ChemBERTa, MolCLR, MolE) learn general chemical representations from large unlabeled datasets, then transfer to downstream tasks. We evaluate whether these pretrained representations outperform task-specific GNNs.

---

# 3. Datasets

## 3.1 Overview

All six datasets are sourced from the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) ADMET benchmark.

| Dataset | Type | Task | Molecules | Primary Metric | Difficulty |
|---------|------|------|-----------|----------------|------------|
| Caco2_Wang | ADME | Regression | 910 | RMSE, R^2 | Moderate |
| Half_Life_Obach | PK | Regression | 667 | RMSE, R^2 | Very Hard |
| Clearance_Hepatocyte_AZ | ADME | Regression | 1,213 | RMSE, R^2 | Impossible |
| Clearance_Microsome_AZ | ADME | Regression | 1,102 | RMSE, R^2 | Weak |
| Tox21 (NR-AR) | Toxicity | Classification | 7,258 | AUC-ROC | Moderate |
| hERG | Toxicity | Classification | 655 | AUC-ROC | Good |

**Total molecules across all datasets:** 11,805

## 3.2 ADME Regression Tasks

### 3.2.1 Caco-2 Permeability (Caco2_Wang)

**Target property:** Apparent permeability (P_app) measured in Caco-2 cell monolayers.

**Biological significance:** Caco-2 cells form a monolayer that mimics the human intestinal epithelium. High permeability (P_app > 10^-6 cm/s) indicates good oral bioavailability. This is the gold-standard in vitro surrogate for human intestinal absorption.

**Dataset characteristics:**
- **Size:** 910 compounds
- **Source:** Wang et al. (curated dataset)
- **Target units:** log(cm/s) -- already log-transformed
- **Preprocessing:** Z-score normalization only (no additional log transform)
- **Train/Val/Test split:** ~637/73/182 (scaffold split)

**Why moderately predictable (R^2 = 0.48):**
Permeability is governed primarily by molecular properties that are well-encoded in the graph structure:
- Molecular size and lipophilicity (Lipinski-like features)
- Hydrogen-bond donor/acceptor count
- Polar surface area

**Why not perfectly predictable:**
- In vitro-in vivo correlation is imperfect
- Active transport (efflux by P-gp) is hard to predict from structure alone
- Inter-laboratory variability in Caco-2 assays

### 3.2.2 Plasma Half-Life (Half_Life_Obach)

**Target property:** Elimination half-life (t_1/2) in human plasma, measured in hours.

**Biological significance:** Half-life determines dosing frequency. Drugs with very short half-lives require multiple daily doses; those with very long half-lives carry accumulation risk.

**Dataset characteristics:**
- **Size:** 667 FDA-approved drugs
- **Source:** Obach et al. database
- **Target range:** 0.5--100+ hours (highly skewed)
- **Preprocessing:** log(y + 1e-3) transformation followed by z-score normalization
- **Train/Val/Test split:** ~466/53/135 (scaffold split)

**Why almost unpredictable (R^2 = 0.004):**
Half-life is a **composite pharmacokinetic parameter** influenced by:
1. Metabolic clearance (partially structure-dependent)
2. Volume of distribution (protein binding, tissue accumulation)
3. Renal elimination (not predictable from structure)
4. Individual patient variability (genetics, age, disease)

Only factor 1 is partially encoded in the molecular structure. R^2 near zero means structure explains essentially none of the variance.

### 3.2.3 Hepatocyte Clearance (Clearance_Hepatocyte_AZ)

**Target property:** Intrinsic clearance measured in freshly isolated human hepatocytes (uL/min/10^6 cells).

**Biological significance:** Hepatocyte clearance captures both Phase I (CYP-mediated) and Phase II (conjugation) metabolic routes. It is the most physiologically relevant in vitro metabolic stability assay and directly correlates with in vivo half-life.

**Dataset characteristics:**
- **Size:** 1,213 proprietary compounds from AstraZeneca
- **Source:** Distributed via TDC
- **Preprocessing:** log(y + 1e-3) + z-score normalization
- **Train/Val/Test split:** ~849/97/243 (scaffold split)

**Why impossible to predict (R^2 = -1.02):**
- Metabolic enzyme kinetics are extremely complex
- Different CYP isoforms (2D6, 3A4, etc.) have different substrate specificities
- Phase II conjugation reactions add another layer of complexity
- Individual donor variation in enzyme expression
- Proprietary AZ data may have higher noise than public datasets

**R^2 < 0 means the model is worse than predicting the mean for all molecules.** This represents a fundamental limit of structure-only models for hepatic metabolism prediction.

### 3.2.4 Microsomal Clearance (Clearance_Microsome_AZ)

**Target property:** Intrinsic clearance in human liver microsomes (uL/min/mg protein).

**Biological significance:** Microsomal clearance measures primarily CYP P450-mediated oxidative metabolism (Phase I only). It is simpler than hepatocyte clearance because microsomes lack Phase II enzymes, cofactors, and cellular organization.

**Dataset characteristics:**
- **Size:** 1,102 compounds from AstraZeneca
- **Source:** Distributed via TDC
- **Preprocessing:** log(y + 1e-3) + z-score normalization
- **Train/Val/Test split:** ~771/88/221 (scaffold split)

**Why weakly predictable (R^2 = 0.19):**
- Simpler than hepatocyte clearance (Phase I only)
- Still subject to CYP isoform complexity and donor variation
- 19% of variance explained -- marginal but non-trivial

## 3.3 Toxicity Classification Tasks

### 3.3.1 Tox21 Androgen Receptor (Tox21 NR-AR)

**Target property:** Nuclear receptor androgen receptor (AR) activation (binary: active/inactive).

**Biological significance:** Part of the Tox21 multi-assay toxicology panel. Identifies compounds that disrupt androgen signaling -- a key mechanism in endocrine disruption and reproductive toxicity.

**Dataset characteristics:**
- **Size:** 7,258 compounds (largest dataset in the benchmark)
- **Source:** Tox21 initiative (NIH/EPA/NCATS)
- **Class distribution:** Only ~3.5% active (positive) -- **severe class imbalance**
- **Mitigation:** Positive-class weighting in BCE loss (weight = neg/pos ratio ~28x)
- **Train/Val/Test split:** ~5080/580/1453 (scaffold split)

**Performance challenges:**
- With 96.5% negative class, a naive "always predict inactive" classifier achieves 96.5% accuracy but 0% recall.
- AUC-ROC (0.742) is more informative than accuracy (96.2%) for this imbalanced task.
- F1 score (0.455) reflects the difficulty of detecting rare active compounds.

### 3.3.2 hERG Cardiotoxicity (hERG)

**Target property:** hERG potassium channel blockade (binary: blocker/non-blocker).

**Biological significance:** hERG channel blockade is the primary mechanism of drug-induced QT prolongation and fatal cardiac arrhythmias. The hERG assay is a **mandatory safety screen** in preclinical drug development.

**Dataset characteristics:**
- **Size:** 655 compounds
- **Source:** TDC
- **Class distribution:** ~31% blockers, 69% non-blockers (reasonably balanced)
- **Train/Val/Test split:** ~458/52/132 (scaffold split)

**Why well-predictable (AUC = 0.825):**
- Clear structural alerts for hERG blockade (specific pharmacophore patterns)
- Well-studied biology with 20+ years of structure-activity relationship (SAR) data
- Reasonably balanced classes
- Binary endpoint with strong structural signal

**Clinical utility:** This model is suitable for virtual screening to deprioritize potentially cardiotoxic compounds early in drug development.

## 3.4 Data Splitting Protocol

**Method:** Bemis-Murcko scaffold splitting via TDC's built-in splitter.

**Procedure:**
1. Extract Bemis-Murcko scaffolds for all molecules
2. Partition scaffolds so that molecules with distinct core structures appear in different folds
3. TDC provides train/test split; we further subdivide train 90/10 (train/val) with seed 42
4. **Final proportions:** approximately 80/10/10 (train/val/test)

**Key property:** The test set contains molecules with **structurally novel scaffolds** not seen during training. This prevents memorization of scaffold-specific patterns and produces more realistic performance estimates.

## 3.5 Target Transformations

| Task Type | Transformation | Details |
|-----------|---------------|---------|
| Caco2_Wang | Z-score only | Targets already in log(cm/s) |
| Half_Life, Clearances | log(y + 1e-3) + Z-score | Log transform reduces skewness |
| Tox21, hERG | None | Binary labels (0/1) |

All normalization parameters (mean, std) are computed **exclusively on the training fold** and reused for validation/test to prevent data leakage.

## 3.6 Data Access

```python
from tdc.single_pred import ADME, Tox

# Load individual datasets
caco2 = ADME(name='Caco2_Wang')
half_life = ADME(name='Half_Life_Obach')
hep_clear = ADME(name='Clearance_Hepatocyte_AZ')
mic_clear = ADME(name='Clearance_Microsome_AZ')
tox21 = Tox(name='Tox21', label_name='NR-AR')
herg = Tox(name='hERG')
```

Cached CSV files are stored in `datasets/adme/` and `datasets/toxicity/`.

---

# 4. Model Architecture

## 4.1 GNN Backbone

The GNN backbone is implemented in `src/core/optimized_gnn.py` and uses the **GCN (GraphConv) architecture** from PyTorch Geometric.

### 4.1.1 Architecture Diagram

```
Input: Molecular Graph (SMILES -> RDKit -> PyG Data)
  |
  v
[Atom Feature Extraction] (8 features per atom)
  |-- Atomic number (one-hot or integer)
  |-- Degree
  |-- Formal charge
  |-- Hybridization (sp, sp2, sp3, sp3d, sp3d2)
  |-- Is aromatic
  |-- Is in ring
  |-- Total hydrogen count
  |-- Atomic mass (normalized)
  |
  v
[GCN Layer 1] -> BatchNorm -> ReLU
  |
  v
[GCN Layer 2] -> BatchNorm -> ReLU
  |
  ... (num_layers total, configurable 3-7)
  |
  v
[GCN Layer N] -> BatchNorm -> ReLU
  |
  v
[Readout: concat(global_mean_pool, global_max_pool)]
  |  -> produces 2 * hidden_dim features
  |
  v
[MLP Head]
  |-- Linear(2*hidden_dim, head_dim_1) -> ReLU -> Dropout
  |-- Linear(head_dim_1, head_dim_2) -> ReLU -> Dropout
  |-- Linear(head_dim_2, head_dim_3) -> ReLU -> Dropout
  |-- Linear(head_dim_3, output_dim)
  |
  v
Output: 1 (regression) or 2 (classification logits)
```

### 4.1.2 Key Design Decisions

**Why GCN?**
Preliminary architecture comparison tests evaluated 8 GNN architectures (GCN, GAT, GIN, SGC, TAG, Transformer, GraphSAGE, basic Graph) across all datasets. GCN was selected based on:
- Average rank of 2.7 across datasets
- High stability (low variance across runs)
- Good balance between expressiveness and computational efficiency

**Why concatenated mean+max pooling?**
- Mean pooling captures the average atom contribution
- Max pooling captures the most salient atom (outlier detection)
- Concatenation provides both signals to the MLP head

**Why 3-layer MLP head?**
- Provides sufficient capacity for nonlinear prediction
- Batch normalization and dropout between layers prevent overfitting
- Head dimensions are part of the HPO search space

### 4.1.3 Molecular Featurization

Each molecule is converted from SMILES to a PyTorch Geometric `Data` object:

```python
# Atom (node) features (8-dimensional):
features = [
    atom.GetAtomicNum(),           # Atomic number
    atom.GetDegree(),              # Number of bonds
    atom.GetFormalCharge(),        # Formal charge
    atom.GetHybridization(),      # sp, sp2, sp3...
    atom.GetIsAromatic(),         # Boolean
    atom.IsInRing(),              # Boolean
    atom.GetTotalNumHs(),         # Hydrogen count
    atom.GetMass() / 100.0        # Normalized mass
]

# Bond (edge) features:
# - Bond type (single, double, triple, aromatic)
# - Is aromatic flag
# Edges are bidirectional (both directions added)
```

## 4.2 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | Default PyTorch parameters |
| Batch size (train) | 32 | Fixed across all experiments |
| Batch size (eval) | 64 | Larger for faster inference |
| Max epochs (HPO) | 50 | Per trial during HPO |
| Early stopping patience | 12 | Epochs without improvement |
| Loss (regression) | MSE | Mean Squared Error |
| Loss (classification) | BCE with logits | Binary Cross-Entropy |
| Class weighting | Yes (Tox21) | weight = neg_count / pos_count |
| Gradient clipping | max_norm = 1.0 | Prevents gradient explosion |
| LR scheduler | ReduceLROnPlateau | patience=10, factor=0.5 |

## 4.3 Model Selection Protocol

For each algorithm-dataset pair:
1. Run 50 HPO trials
2. Select the hyperparameter configuration with the **best validation metric** (lowest RMSE for regression, highest AUC for classification)
3. Retrain with the selected configuration on the full train+val set
4. Evaluate **once** on the held-out test set
5. Report test metrics

**Critical:** The test set is never touched during HPO. Model selection is performed exclusively on the validation split.

---

# 5. Hyperparameter Optimization Framework

## 5.1 Overview

The HPO framework is implemented in the `optimization/` directory and uses **NiaPy** for metaheuristic algorithms and **Optuna** for TPE.

### Architecture

```
optimization/
|-- space.py          # Search space definition (bounds, decode_vector)
|-- problem.py        # NiaPy Problem wrapper (HyperParamProblem)
|-- runner.py         # Orchestrates HPO runs, saves results
|-- algorithms/       # Individual algorithm wrappers
|   |-- pso.py        # Particle Swarm Optimization
|   |-- genetic.py    # Genetic Algorithm
|   |-- abc.py        # Artificial Bee Colony
|   |-- simulated_annealing.py
|   |-- hill_climbing.py
|   `-- random_search.py
|-- foundation_problem.py  # HPO for foundation model baselines
|-- foundation_runner.py   # Runs foundation model HPO
`-- foundation_space.py    # Foundation model search space
```

## 5.2 Search Space

The search space is defined in `optimization/space.py` and consists of **7 continuous dimensions** that are decoded into discrete GNN hyperparameters.

### 5.2.1 Hyperparameter Definitions

| Dim | Hyperparameter | Choices/Range | Type |
|-----|----------------|---------------|------|
| 0 | Hidden dimensions | {64, 96, 128, 192, 256, 384, 512} | Discrete (nearest) |
| 1 | Number of layers | {3, 4, 5, 6, 7} | Discrete (nearest) |
| 2 | MLP head layer 1 | {128, 192, 256, 384, 512} | Discrete (nearest) |
| 3 | MLP head layer 2 | {64, 96, 128, 192, 256} | Discrete (nearest) |
| 4 | MLP head layer 3 | {32, 48, 64, 96, 128} | Discrete (nearest) |
| 5 | log10(learning rate) | [-4.0, -2.0] (1e-4 to 1e-2) | Continuous |
| 6 | log10(weight decay) | [-6.0, -2.0] (1e-6 to 1e-2) | Continuous |

**Total configuration space:** 7 x 5 x 5 x 5 x 5 = 4,375 discrete hidden/layer/head combinations, times continuous LR/WD -- effectively ~250,000+ configurations.

### 5.2.2 Encoding/Decoding

NiaPy algorithms operate on continuous vectors. The `decode_vector()` function maps each continuous dimension to the nearest discrete choice:

```python
def decode_vector(x, _=None):
    hidden_dim = nearest(HIDDEN_CHOICES, x[0])    # Snap to {64,...,512}
    num_layers = nearest(LAYER_CHOICES, x[1])      # Snap to {3,...,7}
    head_dims = (
        nearest(HEAD1_CHOICES, x[2]),              # Snap to {128,...,512}
        nearest(HEAD2_CHOICES, x[3]),              # Snap to {64,...,256}
        nearest(HEAD3_CHOICES, x[4]),              # Snap to {32,...,128}
    )
    head_dims = tuple(sorted(head_dims, reverse=True))  # Ensure monotonic decrease
    lr = 10.0 ** clip(x[5], -4, -2)
    weight_decay = 10.0 ** clip(x[6], -6, -2)
    return {"hidden_dim": hidden_dim, "num_layers": num_layers,
            "head_dims": head_dims, "lr": lr, "weight_decay": weight_decay}
```

### 5.2.3 TPE Difference

TPE (Optuna) uses a slightly different search space: it additionally optimizes **dropout in [0.0, 0.5]** and uses a **fixed MLP head [256, 128, 64]**. This means TPE results are not directly comparable with NiaPy-based algorithms.

## 5.3 Algorithm Details

### 5.3.1 Random Search (Baseline)

**Implementation:** Uniform random sampling from the search space.

**Key properties:**
- No adaptation or learning between trials
- Each trial is independent
- Surprisingly effective when the hyperparameter space is relatively smooth or when only a subset of hyperparameters matter
- Zero optimizer overhead

**Config:** 50 trials, uniform sampling.

### 5.3.2 Particle Swarm Optimization (PSO)

**Implementation:** NiaPy `ParticleSwarmAlgorithm`.

**Principle:** Maintains a swarm of particles that explore the search space. Each particle remembers its personal best position and is attracted toward the global best position found by any particle.

**Config:**
- Population size: 16 particles
- C1 (cognitive coefficient): 2.0
- C2 (social coefficient): 2.0
- Inertia weight w: 0.7
- Budget: 50 evaluations (~3 rounds of 16 particles)

**Behavior:** Rapid initial convergence (identifies near-optimal regions within 10-15 trials), then focused exploitation.

### 5.3.3 Genetic Algorithm (GA)

**Implementation:** NiaPy `GeneticAlgorithm`.

**Principle:** Maintains a population of candidate solutions that evolve through selection (tournament), crossover (combining two parents), and mutation (random perturbation).

**Config:**
- Population size: 16 individuals
- Mutation rate: 0.1
- Crossover rate: 0.8
- Budget: 50 evaluations (~3 generations)

**Behavior:** Mixed results due to limited population and generations. Crossover of continuous-encoded parameters may not align well with the discrete nature of the underlying search space.

### 5.3.4 Artificial Bee Colony (ABC)

**Implementation:** NiaPy `ArtificialBeeColonyAlgorithm`.

**Principle:** Mimics honeybee foraging with three bee types:
- **Employed bees:** Exploit known food sources (current solutions)
- **Onlooker bees:** Choose promising sources based on probability
- **Scout bees:** Abandon exhausted sources and explore randomly

**Config:**
- Colony size: 16
- Limit: 50 (abandonment threshold)
- Budget: 50 evaluations

**Behavior:** Good at exploring diverse regions of the search space. Won on hERG classification (AUC = 0.825).

### 5.3.5 Simulated Annealing (SA)

**Implementation:** NiaPy `SimulatedAnnealing`.

**Principle:** Inspired by metallurgical annealing. Accepts both improvements and strategic downhill moves (worse solutions) with probability that decreases over time (temperature schedule).

**Config:**
- Initial temperature T0: 50
- Cooling rate alpha: 0.99
- Budget: 50 evaluations

**Behavior:** Excels on challenging tasks with many local optima. Won on Tox21 classification (AUC = 0.742). Gradual convergence across all trials rather than rapid initial exploration.

### 5.3.6 Hill Climbing (HC)

**Implementation:** NiaPy `HillClimbAlgorithm`.

**Principle:** Greedy local search. Starts from a random configuration and only moves to a neighboring configuration if it improves the objective.

**Config:**
- Budget: 50 evaluations
- No escape mechanism from local optima

**Behavior:** Fast convergence on smooth landscapes but prone to premature convergence on complex, multimodal spaces. Achieved surprisingly strong performance on hERG (AUC = 0.821).

### 5.3.7 Tree-structured Parzen Estimator (TPE)

**Implementation:** Optuna `TPESampler`.

**Principle:** Bayesian optimization that models the distributions of "good" and "bad" hyperparameter configurations separately (Parzen estimators). Proposes new configurations by maximizing the ratio l(x)/g(x).

**Config:**
- 10 startup random trials
- 40 TPE-guided trials
- Median pruning of underperforming trials
- Search space: 8 dimensions (adds dropout, uses fixed head)

**Behavior:** Best sample efficiency on the hardest task (Clearance_Hepatocyte, RMSE = 52.16). Pruning saves compute by terminating bad trials early.

## 5.4 NiaPy Problem Wrapper

The `HyperParamProblem` class in `optimization/problem.py` wraps the GNN training pipeline as a NiaPy optimization problem:

```python
class HyperParamProblem(Problem):
    def __init__(self, dataset_name, base_config, epochs, patience, seed, device):
        lower, upper = bounds()
        super().__init__(dimension=len(lower), lower=lower, upper=upper)
        # ... setup dataset, device, etc.

    def _evaluate(self, x):
        hp = decode_vector(x)           # Decode continuous vector
        config = replace(base_config, **hp)  # Create config
        result = train_model(...)        # Train GNN
        if classification:
            return -result['val_auc']    # Minimize negative AUC
        else:
            return result['val_rmse']    # Minimize RMSE
```

## 5.5 Result Storage

HPO results are stored as JSON files in `runs/<dataset>/<algo>.json`:

```json
{
  "dataset": "Caco2_Wang",
  "algorithm": "pso",
  "n_trials": 50,
  "seed": 42,
  "best_params": {
    "hidden_dim": 96,
    "num_layers": 5,
    "head_dims": [512, 96, 96],
    "lr": 5.685e-03,
    "weight_decay": 9.185e-04
  },
  "best_val_rmse": 0.452,
  "final_training": {
    "test_metrics": {
      "rmse_orig": 0.0031,
      "rmse_log": 0.452,
      "mae_log": 0.350,
      "r2": 0.45
    }
  },
  "all_trials": [ ... ]
}
```

---

# 6. Foundation Model Baselines

## 6.1 Overview

Four foundation model baselines contextualize GNN performance:

| Model | Type | Representation | Parameters |
|-------|------|----------------|------------|
| Morgan-FP | Fingerprint | ECFP4 (2048-bit) | ~50K (MLP only) |
| ChemBERTa | Transformer | RoBERTa on SMILES | ~85M (frozen) |
| MolE-FP | GNN+pretrained | Multi-task pretrained GNN | ~5M (frozen) |
| MolCLR | GNN+contrastive | Contrastive graph learning | ~2M (frozen) |

## 6.2 Morgan Fingerprints (Morgan-FP)

**Approach:** Extended-Connectivity Fingerprints (ECFP4) of radius 2, producing 2048-bit binary vectors. An MLP predictor is trained on these fixed representations.

**Advantages:** Simple, interpretable, no GPU needed, fast inference.
**Limitations:** Fixed representation, cannot learn task-specific features.

## 6.3 ChemBERTa

**Approach:** RoBERTa model pretrained on 77 million SMILES strings from ZINC-15 using masked language modeling. The encoder is **frozen** and used as a feature extractor; a trainable MLP head is added for prediction.

**Two settings evaluated:**
1. **ChemBERTa (frozen):** Encoder weights frozen, only MLP head trained.
2. **ChemBERTa-FT:** Limited fine-tuning with small learning rate sweep.

**Key finding:** ChemBERTa exhibits scaffold-split sensitivity. On Tox21, validation AUC = 0.896 but test AUC drops to 0.73. ChemBERTa-FT shows even worse degradation: test AUC = 0.482 (worse than random), likely due to overfitting scaffold-specific SMILES patterns during fine-tuning.

## 6.4 MolE-FP

**Approach:** Multi-task pretrained GNN that learns molecular embeddings across multiple prediction tasks simultaneously. We use the learned embeddings as fixed-length fingerprints with an MLP predictor.

## 6.5 MolCLR

**Approach:** Contrastive graph learning with three augmentation strategies: atom masking, bond deletion, and subgraph removal. Learns representations where augmentations of the same molecule have similar embeddings.

**Key finding:** MolCLR degrades to near-random performance on Tox21 (test AUC = 0.538) and hERG (test AUC = 0.504), suggesting that contrastive graph pretraining does not transfer well to these endpoints under scaffold split.

## 6.6 Fairness Note

The comparison between GNNs and foundation models is **not head-to-head:**
- GNNs receive a full 50-trial HPO budget per dataset.
- Foundation models are evaluated with frozen encoders and limited tuning.
- This reflects a practical resource-constrained setting where practitioners may use pretrained models out-of-the-box.

---

# 7. Experimental Setup

## 7.1 Hardware

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7-8700K @ 3.70 GHz |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| OS | Windows 10 Pro |

## 7.2 Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.8 | Runtime |
| PyTorch | >= 2.0 | Deep learning framework |
| PyTorch Geometric | >= 2.4 | Graph neural networks |
| RDKit | >= 2023.03 | Molecular featurization |
| PyTDC | >= 0.4 | Dataset access |
| NiaPy | >= 2.0 | Metaheuristic algorithms |
| Optuna | >= 3.0 | TPE optimization |
| scikit-learn | >= 1.3 | Evaluation metrics |
| Transformers | >= 4.30 | ChemBERTa (optional) |
| matplotlib | >= 3.7 | Visualization |
| seaborn | >= 0.12 | Statistical plots |
| scipy | >= 1.10 | Statistical tests |

## 7.3 Experiment Matrix

| Factor | Values | Count |
|--------|--------|-------|
| Datasets | 6 | 6 |
| HPO algorithms | Random, PSO, ABC, GA, SA, HC | 6 |
| Trials per run | 50 | -- |
| **Total NiaPy HPO runs** | | **36** |
| TPE runs | 6 datasets | **6** |
| **Total HPO runs** | | **42** |
| Multi-seed seeds | 42, 123, 456, 789, 1011 | 5 |
| Multi-seed runs | 6 datasets x 5 seeds | **30** |
| Foundation model evaluations | 4 models x 6 datasets | **24** |

**Total model evaluations:** 2,100+ (HPO) + 30 (multi-seed) + 24 (foundation) = **2,150+**

## 7.4 Compute Time

| Experiment | Approximate Time |
|------------|-----------------|
| HPO runs (42 x 50 trials) | ~40 hours |
| Multi-seed validation | ~2 hours |
| Foundation model baselines | ~3 hours |
| **Total** | **~45 hours** |

## 7.5 Random Seeds

- **HPO seed:** 42 (for all splitting and random initialization)
- **Multi-seed validation:** {42, 123, 456, 789, 1011}
- **All stochastic operations** are seeded for reproducibility (Python, NumPy, PyTorch, CUDA)

---

# 8. Results and Analysis

## 8.1 HPO Algorithm Performance

### 8.1.1 Complete Results Table

**ADME Regression (Test RMSE -- lower is better):**

| Dataset | PSO | ABC | GA | SA | HC | Random | TPE |
|---------|-----|-----|----|----|-----|--------|-----|
| Caco2_Wang | 0.0031 | 0.0029 | 0.0031 | 0.0029 | 0.0030 | **0.0027** | 0.0030 |
| Half_Life_Obach | **21.66** | **21.66** | **21.66** | 23.70 | 24.52 | 22.31 | 22.34 |
| Clearance_Hepatocyte_AZ | 70.21 | 72.04 | 71.34 | 72.04 | 72.04 | 68.22 | **52.16** |
| Clearance_Microsome_AZ | 42.76 | 42.29 | 42.29 | 40.94 | 41.63 | **38.75** | 44.34 |

**Toxicity Classification (Test AUC-ROC -- higher is better):**

| Dataset | PSO | ABC | GA | SA | HC | Random | TPE |
|---------|-----|-----|----|----|-----|--------|-----|
| Tox21 (NR-AR) | 0.692 | 0.735 | 0.735 | **0.742** | 0.652 | 0.713 | 0.705 |
| hERG | 0.747 | **0.825** | 0.747 | 0.802 | 0.821 | 0.747 | 0.772 |

### 8.1.2 Winner Summary

| Algorithm | Wins (NiaPy only) | Datasets Won |
|-----------|-------------------|--------------|
| Random Search | 3/6 | Caco2, Clearance_Hepatocyte, Clearance_Microsome |
| PSO / ABC / GA | 1/6 | Half_Life (three-way tie: RMSE = 21.66) |
| SA | 1/6 | Tox21 |
| ABC | 1/6 | hERG |
| HC | 0/6 | -- |
| TPE | 1/6 | Clearance_Hepatocyte (if included; different search space) |

### 8.1.3 Key Observations

**Random Search dominance on regression:**
Random Search won on 3/4 regression tasks among NiaPy algorithms. This suggests:
- The hyperparameter landscape for regression tasks may be relatively smooth with few local optima
- Under a 50-trial budget, random exploration covers the space efficiently
- Scaffold-split evaluation introduces noise that reduces the advantage of adaptive methods

**Three-way tie on Half_Life:**
PSO, ABC, and GA all achieved RMSE = 21.66 on Half_Life_Obach, suggesting they converged to the **same hyperparameter configuration**. This indicates a clear global optimum in this search space for this dataset.

**Metaheuristic advantage on classification:**
SA and ABC won on the two classification tasks, suggesting that metaheuristics are more beneficial when:
- The objective function is non-smooth (class imbalance creates a noisy loss landscape)
- The relationship between hyperparameters and AUC is more complex than for RMSE

**TPE excels on the hardest task:**
TPE achieved the best overall result on Clearance_Hepatocyte (RMSE = 52.16 vs next best 68.22). However, TPE uses a different search space (includes dropout, uses fixed head), so this is not directly comparable.

## 8.2 Best Configurations Per Dataset

### Caco2_Wang (Best: Random Search)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 96 |
| Num layers | 5 |
| Head dims | [512, 96, 96] |
| Learning rate | 5.685e-03 |
| Weight decay | 9.185e-04 |
| **Test RMSE** | **0.0027** |
| **R^2** | **0.482** |

### Half_Life_Obach (Best: PSO/ABC/GA)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 256 |
| Num layers | 4 |
| Head dims | [384, 96, 64] |
| Learning rate | 1.834e-03 |
| Weight decay | 1.077e-03 |
| **Test RMSE** | **21.66** |
| **R^2** | **0.004** |

### Clearance_Hepatocyte_AZ (Best NiaPy: Random Search)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 64 |
| Num layers | 3 |
| Head dims | [192, 128, 64] |
| Learning rate | 1.040e-03 |
| Weight decay | 4.268e-03 |
| **Test RMSE** | **68.22** |
| **R^2** | **-1.019** |

### Clearance_Microsome_AZ (Best: Random Search)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 384 |
| Num layers | 4 |
| Head dims | [192, 128, 96] |
| Learning rate | 2.871e-03 |
| Weight decay | 1.216e-03 |
| **Test RMSE** | **38.75** |
| **R^2** | **0.191** |

### Tox21 NR-AR (Best: SA)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 384 |
| Num layers | 5 |
| Head dims | [512, 192, 48] |
| Learning rate | 2.954e-03 |
| Weight decay | 1.713e-03 |
| **Test AUC** | **0.742** |
| **F1** | **0.455** |

### hERG (Best: ABC)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 512 |
| Num layers | 5 |
| Head dims | [384, 192, 48] |
| Learning rate | 8.938e-03 |
| Weight decay | 1.108e-03 |
| **Test AUC** | **0.825** |
| **F1** | **0.809** |

## 8.3 Convergence Analysis

### PSO and ABC
- **Rapid initial convergence:** Identify near-optimal configurations within 10-15 trials
- **Focused exploitation:** Later trials refine around the best regions
- Most effective on smooth landscapes (Caco2, Half_Life)

### SA
- **Gradual convergence:** Consistent improvement across all 50 trials
- **No premature convergence:** Temperature schedule prevents getting stuck
- Best for challenging, multimodal objectives (Tox21)

### Random Search
- **No convergence curve:** Each trial is independent
- **Uniform coverage:** Explores the full space without bias
- Probability of finding a good configuration increases linearly with trials

### HC
- **Fast initial convergence but plateaus early**
- Prone to getting stuck in local optima
- Performance varies dramatically depending on initial random configuration

## 8.4 Scaffold Split Impact

A central finding of this work is that **scaffold-split evaluation changes the relative effectiveness of optimization strategies**. Under scaffold splits:

- The validation landscape becomes noisier due to structural distribution shift
- The advantage of adaptive metaheuristic strategies is reduced
- Simple approaches (Random Search) remain highly competitive
- Previously reported optimizer superiority (from random-split papers) may not hold

This suggests that HPO algorithm selection should be validated under the specific evaluation protocol used in the final application.

---

# 9. Multi-Seed Validation

## 9.1 Methodology

To assess robustness, we retrain the best HPO-selected configuration per dataset across **five random seeds:** {42, 123, 456, 789, 1011}.

For each seed, we:
1. Re-initialize all random number generators (Python, NumPy, PyTorch, CUDA)
2. Retrain the model from scratch with the fixed hyperparameters
3. Evaluate on the same held-out test set

## 9.2 Results

| Dataset | Task | Metric | Mean +/- Std | 95% CI |
|---------|------|--------|-------------|--------|
| Caco2_Wang | Regr. | RMSE | 0.0033 +/- 0.0005 | (0.0027, 0.0039) |
| Half_Life_Obach | Regr. | RMSE | 20.05 +/- 1.17 | (18.61, 21.50) |
| Clearance_Hepatocyte_AZ | Regr. | RMSE | 52.37 +/- 2.87 | (48.81, 55.93) |
| Clearance_Microsome_AZ | Regr. | RMSE | 53.46 +/- 13.56 | (36.63, 70.30) |
| Tox21 (NR-AR) | Class. | AUC | 0.711 +/- 0.012 | (0.696, 0.727) |
| hERG | Class. | AUC | 0.805 +/- 0.022 | (0.778, 0.832) |

**Confidence intervals** are computed using the t-distribution with 4 degrees of freedom:

```
CI = mean +/- t_{0.975, df=4} * (std / sqrt(5))
   = mean +/- 2.776 * (std / sqrt(5))
```

## 9.3 Interpretation

- **Low variance datasets:** Caco2 (CV=15%), Hepatocyte (CV=5.5%), Tox21 (CV=1.7%), hERG (CV=2.7%) show stable performance across seeds.
- **High variance dataset:** Clearance_Microsome (CV=25%) shows substantial seed sensitivity, with std = 13.56 on a mean of 53.46.
- **All CIs exclude zero/chance performance**, confirming that the models are learning meaningful patterns (except Hepatocyte, where the CI for R^2 would include values near zero).

## 9.4 Multi-Seed vs. Single-Seed HPO Results

The multi-seed means may differ from the single best HPO trial because:
1. HPO selects the best validation metric across 50 trials (selection bias)
2. Multi-seed retrains a fixed configuration, averaging out initialization effects
3. Some single-trial results may be lucky outliers

---

# 10. Statistical Significance Testing

## 10.1 Methodology

We use the **Wilcoxon signed-rank test** to compare each HPO algorithm against Random Search across all six datasets. The test asks: "Does algorithm X systematically improve over Random Search?"

- **Paired comparison:** For each dataset, we compare algorithm X's test metric against Random Search's test metric
- **Two-sided test** with significance level alpha = 0.05
- **Effect size:** Rank-biserial correlation (r)

## 10.2 Results

| Algorithm | W/T/L vs Random | p-value | Effect (r) | Significant? |
|-----------|-----------------|---------|------------|-------------|
| PSO | 4/0/2 | 0.844 | 0.57 | No |
| ABC | 4/0/2 | 0.844 | 0.57 | No |
| GA | 4/0/2 | 0.844 | 0.57 | No |
| SA | 5/0/1 | 0.156 | 0.86 | No |
| HC | 3/0/3 | 0.688 | 0.62 | No |

W/T/L = Wins/Ties/Losses against Random Search.

## 10.3 Interpretation

**No algorithm achieves statistical significance** over Random Search at p < 0.05. This reinforces our main finding: under a 50-trial budget with scaffold-split evaluation, Random Search is a strong baseline that is not systematically outperformed.

SA comes closest (p = 0.156, 5 wins out of 6), suggesting that with more datasets or a larger HPO budget, SA might demonstrate significant improvement.

---

# 11. Foundation Model Comparison

## 11.1 Complete Results

### Regression Tasks

| Model | Caco2 RMSE | Caco2 R^2 | Half_Life RMSE | Hepatocyte RMSE | Microsome RMSE |
|-------|-----------|----------|---------------|----------------|----------------|
| **GNN-Best** | 0.0027* | 0.481 | **21.66** | 68.22 | **38.75** |
| Morgan-FP | 0.614 | 0.200 | 22.12 | **48.36** | 40.36 |
| ChemBERTa | 0.496 | 0.478 | 27.39 | **47.31** | 42.56 |
| MolE-FP | 0.670 | 0.047 | 25.01 | **47.22** | 41.79 |
| MolCLR | 0.713 | -0.079 | 21.97 | 48.71 | 43.33 |

*Caco2 GNN RMSE (0.0027) is in original log(cm/s) units; foundation model RMSEs are in z-score-normalized space. R^2 is used for comparison.

### Classification Tasks

| Model | Tox21 AUC | Tox21 F1 | hERG AUC | hERG F1 |
|-------|-----------|----------|----------|---------|
| **GNN-Best** | **0.742** | **0.455** | **0.825** | **0.809** |
| Morgan-FP | 0.722 | 0.310 | 0.611 | 0.847 |
| ChemBERTa | 0.728 | 0.330 | 0.770 | 0.873 |
| MolE-FP | 0.675 | 0.391 | 0.672 | 0.857 |
| MolCLR | 0.538 | 0.0 | 0.504 | 0.847 |

## 11.2 Key Findings

**GNN wins on toxicity:** The optimized GNN outperforms all frozen foundation models on both toxicity tasks. Task-specific graph convolution patterns are particularly valuable for learning structural alerts.

**Foundation models win on Hepatocyte clearance:** All foundation models (ChemBERTa, Morgan-FP, MolE-FP) outperform the GNN on the hardest regression task. While none achieve meaningful R^2, pretrained representations provide a more stable starting point than task-specific graph learning.

**Caco2 tie:** GNN and ChemBERTa achieve similar R^2 (~0.48), suggesting both capture comparable structure-property relationships for permeability.

**MolCLR failure:** MolCLR degrades to near-random on both classification tasks under scaffold split, suggesting contrastive graph pretraining does not transfer well to these endpoints.

**ChemBERTa scaffold sensitivity:** Validation AUC = 0.896 but test AUC = 0.73 on Tox21. Fine-tuned ChemBERTa is even worse (test AUC = 0.482), demonstrating catastrophic scaffold-split overfitting.

---

# 12. Discussion

## 12.1 Practical Recommendations

Based on our comprehensive evaluation, we provide the following recommendations for practitioners:

| Scenario | Recommendation | Reasoning |
|----------|---------------|-----------|
| **Regression, limited budget** | Start with Random Search | Wins 3/4 ADME tasks, zero optimizer overhead |
| **Regression, larger budget** | PSO or TPE | PSO converges faster; TPE is more sample-efficient |
| **Classification / toxicity** | SA or ABC | Better at navigating noisy, imbalanced loss landscapes |
| **Very hard metabolic tasks** | TPE + consider foundation models | TPE's pruning saves compute; frozen encoders may help |
| **Quick baseline** | Morgan-FP + MLP | Simple, fast, no GPU, competitive on some tasks |
| **Production screening** | GNN with HPO for toxicity | Best AUC on hERG (mandatory safety screen) |

## 12.2 When to Use Metaheuristics

Metaheuristic HPO algorithms (PSO, ABC, GA, SA) are most beneficial when:
- The task has a non-smooth objective landscape (classification with imbalance)
- The compute budget allows > 50 trials
- Population size can be matched to the number of parallel workers
- The search space has many local optima

They are **less beneficial** when:
- The evaluation protocol introduces high noise (scaffold split)
- The budget is limited (< 50 trials, especially for GA which needs generations)
- The hyperparameter landscape is relatively smooth (ADME regression)

## 12.3 Limitations

1. **Fixed GNN architecture:** Only GCN was tested; GAT, GIN, and other architectures may perform differently.
2. **Limited HPO budget:** 50 trials may not be sufficient for GA (needs more generations) or PSO (needs more swarm iterations).
3. **Frozen foundation models:** Full fine-tuning with equal HPO budget would be a fairer comparison.
4. **Single-seed HPO:** Multi-seed validation was only done for the best configuration, not for the entire HPO process.
5. **Small datasets:** Some datasets (hERG: 655, Half_Life: 667) have fewer than 1,000 compounds, introducing high variance.

## 12.4 Future Work

- **Multi-seed HPO:** Run the entire 50-trial HPO process multiple times to quantify optimizer stability.
- **Full fine-tuning of foundation models:** Compare GNNs against fine-tuned ChemBERTa with equal HPO budget.
- **Larger datasets:** Evaluate on datasets with > 10,000 compounds to see if metaheuristics gain more advantage.
- **3D and multi-modal features:** Incorporate 3D conformer information, protein binding data, or quantum chemical descriptors.
- **Class imbalance strategies:** Explore SMOTE for graphs, focal loss, and curriculum learning for Tox21.

---

# 13. Codebase Architecture

## 13.1 Directory Structure

```
MANU/
|
|-- optimized_gnn.py                # Legacy entry point (imports from src/)
|
|-- src/                            # Core source code
|   |-- __init__.py
|   |-- core/
|   |   |-- __init__.py
|   |   |-- optimized_gnn.py        # Main GNN: model, featurization, training
|   |   `-- model_comparison.py     # Model comparison utilities
|   `-- utils/
|       |-- __init__.py
|       `-- check_progress.py       # Progress monitoring
|
|-- optimization/                   # HPO framework
|   |-- __init__.py
|   |-- space.py                    # Search space: bounds(), decode_vector()
|   |-- problem.py                  # NiaPy Problem: HyperParamProblem
|   |-- runner.py                   # HPO runner: train_with_best_to_summary()
|   |-- foundation_problem.py       # Foundation model HPO wrapper
|   |-- foundation_runner.py
|   |-- foundation_space.py
|   `-- algorithms/
|       |-- __init__.py
|       |-- pso.py
|       |-- genetic.py
|       |-- abc.py
|       |-- simulated_annealing.py
|       |-- hill_climbing.py
|       `-- random_search.py
|
|-- scripts/                        # Experiment execution scripts
|   |-- run_hpo_50_trials.py        # Main HPO script
|   |-- run_tpe_benchmark.py        # TPE via Optuna
|   |-- run_multi_seed_validation.py # 5-seed validation
|   |-- run_chemberta_finetune.py   # ChemBERTa fine-tuning
|   |-- run_complete_foundation_benchmark.py
|   |-- create_hpo_visualizations.py
|   |-- create_foundation_comparison_plots.py
|   |-- statistical_significance_tests.py
|   |-- generate_confusion_matrices.py
|   |-- generate_convergence_curves.py
|   |-- generate_param_sensitivity.py
|   `-- analyses/                   # Detailed analysis scripts
|       |-- analyze_gnn_results.py
|       |-- benchmark_foundation_models.py
|       |-- feature_label_correlation_analysis.py
|       |-- label_distribution_analysis.py
|       `-- tanimoto_similarity_analysis.py
|
|-- datasets/                       # Raw data (CSV)
|   |-- adme/                       # Caco2, Half_Life, Hepatocyte, Microsome
|   `-- toxicity/                   # Tox21, hERG, ClinTox
|
|-- runs/                           # Raw HPO results (JSON)
|   |-- Caco2_Wang/                 # 6 algorithm results per dataset
|   |-- Half_Life_Obach/
|   |-- Clearance_Hepatocyte_AZ/
|   |-- Clearance_Microsome_AZ/
|   |-- tox21/
|   `-- herg/
|
|-- results/                        # Processed results
|   |-- multi_seed/                 # 5-seed validation
|   |-- tpe_benchmark/              # TPE results
|   |-- foundation_benchmark/       # Foundation model CSV
|   |-- chemberta_finetune/         # ChemBERTa results
|   |-- figures/                    # Generated tables/figures
|   `-- hpo/                        # Organized HPO results
|
|-- paper_1/                        # LaTeX manuscript
|   |-- main.tex
|   |-- refs.bib
|   `-- images/                     # Paper figures (PNG)
|
|-- external/MolCLR/                # MolCLR pretrained checkpoints
|-- figures/paper/                  # Generated LaTeX tables
|-- archive/                        # Old experiments, scripts, docs
|-- config/                         # Benchmark configuration files
|-- requirements.txt
`-- setup.py
```

## 13.2 Data Flow

```
SMILES strings (CSV)
    |
    v
[RDKit: Chem.MolFromSmiles] -- Parse molecules
    |
    v
[Featurization: atom/bond features] -- 8 atom features
    |
    v
[PyG Data objects] -- Graph representation
    |
    v
[DataLoader] -- Batched graphs (batch_size=32)
    |
    v
[GCN Encoder] -- Message passing (num_layers)
    |
    v
[Pooling: mean+max concat] -- Graph-level representation
    |
    v
[MLP Head] -- 3-layer prediction
    |
    v
[Loss: MSE or BCE] -- Backpropagation
    |
    v
[Validation metric] -- Model selection
    |
    v
[Test evaluation] -- Final reported metrics
```

## 13.3 Key Classes and Functions

### `OptimizedGNNConfig` (dataclass)
Configuration container for all model hyperparameters:
- `hidden_dim`, `num_layers`, `head_dims`, `lr`, `weight_decay`
- `batch_train`, `batch_eval`, `val_fraction`
- `scheduler_patience`, `scheduler_factor`, `max_grad_norm`

### `prepare_dataset(dataset_name, val_fraction, seed)`
Loads a TDC dataset, applies scaffold splitting, featurizes molecules, and creates train/val/test DataLoaders.

### `train_model(dataset_name, config, epochs, patience, ...)`
Trains a GNN model with the given configuration. Returns a dictionary of validation and test metrics.

### `HyperParamProblem` (NiaPy Problem)
Wraps the GNN training pipeline as a minimization problem for NiaPy optimizers. Decodes continuous vectors into hyperparameters, trains the model, and returns the validation loss.

### `bounds()` and `decode_vector(x)`
Define the continuous search space boundaries and map continuous vectors to discrete hyperparameter configurations.

---

# 14. API Reference

## 14.1 Core Functions

### `prepare_dataset`

```python
def prepare_dataset(
    dataset_name: str,        # One of: 'Caco2_Wang', 'Half_Life_Obach', etc.
    val_fraction: float = 0.1, # Validation fraction (from train set)
    seed: int = 42,           # Random seed for splitting
    verbose: bool = True       # Print dataset statistics
) -> dict:
    """Load and prepare a TDC dataset with scaffold splitting.

    Returns:
        dict with keys:
        - 'train_loader': DataLoader for training
        - 'val_loader': DataLoader for validation
        - 'test_loader': DataLoader for testing
        - 'y_mean', 'y_std': Normalization statistics
        - 'input_dim': Number of node features
        - 'output_dim': 1 (regression) or 2 (classification)
    """
```

### `train_model`

```python
def train_model(
    dataset_name: str,
    config: OptimizedGNNConfig,
    epochs: int = 50,
    patience: int = 12,
    device: str = 'auto',
    seed: int = 42,
    dataset_cache: dict = None,
    evaluate_test: bool = True
) -> dict:
    """Train a GNN model and return metrics.

    Returns:
        dict with keys:
        - 'val_rmse' / 'val_auc': Validation metric
        - 'test_metrics': dict of test RMSE, MAE, R^2, AUC, F1, etc.
        - 'training_history': List of per-epoch losses
        - 'best_epoch': Epoch of best validation metric
    """
```

### `resolve_device`

```python
def resolve_device(device: str = 'auto') -> str:
    """Resolve device string to actual device.
    'auto' -> 'cuda' if available, else 'cpu'
    """
```

## 14.2 Optimization Functions

### `bounds`

```python
def bounds(model_name=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower_bound, upper_bound) arrays for the 7-dim search space."""
```

### `decode_vector`

```python
def decode_vector(x: np.ndarray, model_name=None) -> dict:
    """Decode a 7-dim continuous vector into GNN hyperparameters.

    Returns:
        dict with keys: hidden_dim, num_layers, head_dims, lr, weight_decay
    """
```

## 14.3 HPO Execution

### `train_with_best_to_summary`

```python
def train_with_best_to_summary(
    best: dict,               # Best hyperparameters from HPO
    dataset: str,             # Dataset name
    epochs: int,              # Training epochs
    patience: int,            # Early stopping patience
    batch_train: int = 32,
    batch_eval: int = 64,
    seed: int = 42,
    out_dir: str = 'runs',
    algo_name: str = 'random',
    best_val_rmse_from_search: float = 0.0,
    trials: int = 50,
    device: str = 'auto'
) -> str:
    """Train with best HPO hyperparameters and save JSON summary.

    Returns:
        Path to the saved JSON summary file.
    """
```

---

# 15. Reproducibility Guide

## 15.1 Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/NitramVonemats/MANU_Project.git
cd MANU_Project

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 5. Verify all imports
python scripts/check_imports.py
```

## 15.2 Running Experiments

### Full HPO Benchmark (all 42 runs, ~40 hours)

```bash
python scripts/run_hpo_50_trials.py
```

This runs all 6 NiaPy algorithms on all 6 datasets with 50 trials each. Results are saved to `runs/<dataset>/hpo_<dataset>_<algo>.json`.

### Single Algorithm/Dataset

```python
# In Python:
from optimization.runner import train_with_best_to_summary
from optimization.algorithms.pso import run_pso_hpo

best_params, best_val = run_pso_hpo(
    dataset_name='Caco2_Wang',
    n_trials=50,
    seed=42
)
```

### TPE Benchmark

```bash
python scripts/run_tpe_benchmark.py
```

Results are saved to `results/tpe_benchmark/tpe_<dataset>_results.json`.

### Foundation Model Baselines

```bash
# All foundation models (Morgan-FP, ChemBERTa, MolE-FP, MolCLR)
python scripts/run_complete_foundation_benchmark.py

# ChemBERTa fine-tuning only
python scripts/run_chemberta_finetune.py
```

### Multi-Seed Validation

```bash
python scripts/run_multi_seed_validation.py
```

Retrains the best HPO configuration per dataset across 5 seeds {42, 123, 456, 789, 1011}. Results saved to `results/multi_seed/`.

### Generate Figures

```bash
python scripts/create_hpo_visualizations.py
python scripts/create_foundation_comparison_plots.py
python scripts/generate_confusion_matrices.py
python scripts/generate_convergence_curves.py
python scripts/generate_param_sensitivity.py
```

## 15.3 Verifying Results

### Check HPO results match the paper

```python
import json, glob

for path in sorted(glob.glob('runs/*/hpo_*.json')):
    with open(path) as f:
        data = json.load(f)
    algo = data.get('algorithm', path.split('_')[-1].replace('.json',''))
    dataset = data.get('dataset', path.split('/')[1])
    metrics = data['final_training']['test_metrics']
    print(f"{dataset}/{algo}: RMSE={metrics.get('rmse_orig','N/A')}, "
          f"AUC={metrics.get('auc','N/A')}")
```

### Verify multi-seed results

```python
import json

with open('results/multi_seed/multi_seed_results_fixed.json') as f:
    results = json.load(f)

for dataset, data in results.items():
    if data['task_type'] == 'regression':
        print(f"{dataset}: RMSE = {data['rmse_orig_mean']:.4f} +/- {data['rmse_orig_std']:.4f}")
    else:
        print(f"{dataset}: AUC = {data['auc_mean']:.4f} +/- {data['auc_std']:.4f}")
```

## 15.4 Seed Management

All experiments use controlled seeding:

```python
import random, numpy as np, torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

# 16. Troubleshooting

## 16.1 Common Issues

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size or hidden dimensions:
```python
config = OptimizedGNNConfig(hidden_dim=128, batch_train=16)
```

### TDC Download Fails

```
ConnectionError: Failed to download dataset
```

**Solution:** Use cached CSV files in `datasets/`:
```python
# The datasets are already downloaded and stored locally
# Check datasets/adme/ and datasets/toxicity/
```

### RDKit Import Error

```
ImportError: No module named 'rdkit'
```

**Solution:** Install RDKit via conda (recommended) or pip:
```bash
conda install -c conda-forge rdkit
# or
pip install rdkit
```

### NiaPy Version Compatibility

```
AttributeError: module 'niapy' has no attribute ...
```

**Solution:** Ensure NiaPy >= 2.0:
```bash
pip install "niapy>=2.0"
```

### Optuna Not Found (TPE)

```
ModuleNotFoundError: No module named 'optuna'
```

**Solution:** Optuna is optional; install if you need TPE:
```bash
pip install optuna
```

## 16.2 Performance Issues

### Slow Training
- Verify GPU is being used: `print(torch.cuda.is_available())`
- Check GPU utilization: `nvidia-smi`
- Reduce num_layers or hidden_dim for faster trials

### Inconsistent Results
- Ensure seeds are set before each experiment
- Check PyTorch deterministic mode is enabled
- Note that CUDA operations may have non-deterministic behavior

---

# 17. Appendices

## Appendix A: Complete Algorithm Configurations

### PSO Configuration

```python
from niapy.algorithms.basic import ParticleSwarmAlgorithm

algo = ParticleSwarmAlgorithm(
    population_size=16,
    C1=2.0,          # Cognitive coefficient
    C2=2.0,          # Social coefficient
    w=0.7,           # Inertia weight
    seed=42
)
```

### SA Configuration

```python
from niapy.algorithms.basic import SimulatedAnnealing

algo = SimulatedAnnealing(
    temperature=50,        # Initial temperature T0
    cooling_factor=0.99,   # Cooling rate alpha
    seed=42
)
```

### GA Configuration

```python
from niapy.algorithms.basic import GeneticAlgorithm

algo = GeneticAlgorithm(
    population_size=16,
    mutation_rate=0.1,
    crossover_rate=0.8,
    seed=42
)
```

### ABC Configuration

```python
from niapy.algorithms.basic import ArtificialBeeColonyAlgorithm

algo = ArtificialBeeColonyAlgorithm(
    population_size=16,
    limit=50,
    seed=42
)
```

### TPE Configuration (Optuna)

```python
import optuna

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=42
    ),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50)
```

## Appendix B: Evaluation Metrics

### Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| RMSE | sqrt(mean((y_pred - y_true)^2)) | Average prediction error (in target units) |
| MAE | mean(abs(y_pred - y_true)) | Median-like error, less sensitive to outliers |
| R^2 | 1 - SS_res / SS_tot | Fraction of variance explained (1=perfect, 0=mean predictor, <0=worse than mean) |

### Classification Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| AUC-ROC | Area under ROC curve | Discrimination ability (0.5=random, 1=perfect) |
| F1 | 2 * (precision * recall) / (precision + recall) | Harmonic mean of precision and recall |
| Accuracy | correct / total | Fraction correct (misleading with imbalanced classes) |

## Appendix C: Dataset Predictability Ranking

From most to least predictable:

```
1. hERG          (AUC = 0.825)    -- Strong structural signal
2. Tox21 NR-AR   (AUC = 0.742)    -- Moderate, limited by imbalance
3. Caco2_Wang    (R^2 = 0.482)    -- Moderate, structure captures transport
4. Clearance_Microsome (R^2 = 0.191) -- Weak, CYP complexity
5. Half_Life     (R^2 = 0.004)    -- Near-zero, composite PK parameter
6. Clearance_Hepatocyte (R^2 = -1.02) -- Impossible, worse than mean
```

**Key insight:** Toxicity endpoints (binary, structure-mediated) are more predictable than PK endpoints (continuous, multi-factorial) using structure-only models.

## Appendix D: File Inventory

### HPO Result Files (runs/)

| Directory | Files | Contents |
|-----------|-------|---------|
| runs/Caco2_Wang/ | 6 JSON | PSO, ABC, GA, SA, HC, Random |
| runs/Half_Life_Obach/ | 6 JSON | PSO, ABC, GA, SA, HC, Random |
| runs/Clearance_Hepatocyte_AZ/ | 6 JSON | PSO, ABC, GA, SA, HC, Random |
| runs/Clearance_Microsome_AZ/ | 6 JSON | PSO, ABC, GA, SA, HC, Random |
| runs/tox21/ | 6 JSON | PSO, ABC, GA, SA, HC, Random |
| runs/herg/ | 6 JSON | PSO, ABC, GA, SA, HC, Random |

### TPE Result Files (results/tpe_benchmark/)

| File | Dataset |
|------|---------|
| tpe_Caco2_Wang_results.json | Caco2 |
| tpe_Half_Life_Obach_results.json | Half_Life |
| tpe_Clearance_Hepatocyte_AZ_results.json | Hepatocyte |
| tpe_Clearance_Microsome_AZ_results.json | Microsome |
| tpe_tox21_results.json | Tox21 |
| tpe_herg_results.json | hERG |

### Foundation Model Results

| File | Contents |
|------|---------|
| results/foundation_benchmark/foundation_comparison_COMPLETE.csv | All foundation model results |
| results/chemberta_finetune/chemberta_finetune_summary_fixed.csv | ChemBERTa FT summary |
| results/chemberta_finetune/chemberta_ft_*_results_fixed.json | Per-dataset ChemBERTa FT |

### Multi-Seed Results

| File | Contents |
|------|---------|
| results/multi_seed/multi_seed_results_fixed.json | All 5-seed results |
| results/multi_seed/multi_seed_summary_fixed.csv | Summary CSV |
| results/multi_seed/multi_seed_table.tex | LaTeX table |

## Appendix E: Paper Figures Reference

All paper figures are stored in `paper_1/images/`:

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | sankey-diagram.png | Dataset overview (data flow, class distribution) |
| Fig. 2 | Benchmark-Architecture-GNN.png | Benchmarking framework architecture |
| Fig. 3 | 01_algorithm_performance.png | HPO algorithm performance comparison |
| Fig. 4 | hpo_convergence_curves.png | Convergence trajectories |
| Fig. 5 | confusion_matrices.png | Toxicity confusion matrices |
| Fig. 6 | multi_seed_boxplots.png | Multi-seed validation distributions |
| Fig. 7 | gnn_vs_foundation_comparison.png | GNN vs foundation models |
| Fig. 8 | foundation_ranking.png | Foundation model ranking heatmap |

## Appendix F: GNN Architecture Selection (Preliminary)

Before the main HPO study, a preliminary comparison of 8 GNN architectures was conducted:

| Architecture | Average Rank | Stability |
|-------------|-------------|-----------|
| GCN (GraphConv) | 2.7 | High |
| GAT | 3.2 | Medium |
| GIN | 3.5 | Medium |
| GraphSAGE | 4.0 | Medium |
| TAG | 4.5 | Low |
| SGC | 5.0 | High |
| Transformer | 5.5 | Low |
| Basic Graph | 6.0 | Medium |

GCN was selected for all HPO experiments based on its best average rank and high stability. These preliminary results are archived in `archive/gnn_architecture_comparison/`.

## Appendix G: Glossary

| Term | Definition |
|------|-----------|
| ADMET | Absorption, Distribution, Metabolism, Excretion, Toxicity |
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve |
| BCE | Binary Cross-Entropy loss |
| CYP | Cytochrome P450 (drug-metabolizing enzymes) |
| ECFP | Extended-Connectivity Fingerprint |
| GCN | Graph Convolutional Network |
| GNN | Graph Neural Network |
| hERG | human Ether-a-go-go Related Gene (potassium channel) |
| HPO | Hyperparameter Optimization |
| MLP | Multi-Layer Perceptron |
| MSE | Mean Squared Error |
| NiaPy | Nature-Inspired Algorithms in Python |
| PK | Pharmacokinetics |
| PSO | Particle Swarm Optimization |
| R^2 | Coefficient of Determination |
| RMSE | Root Mean Squared Error |
| SA | Simulated Annealing |
| SAR | Structure-Activity Relationship |
| SMILES | Simplified Molecular Input Line Entry System |
| TDC | Therapeutics Data Commons |
| TPE | Tree-structured Parzen Estimator |

## Appendix H: Detailed Per-Seed Results

### Caco2_Wang (5-Seed RMSE in log-space)

| Seed | RMSE (log) | MAE (log) | RMSE (orig) |
|------|-----------|----------|-------------|
| 42 | 0.840 | 0.671 | 0.0041 |
| 123 | 0.594 | 0.449 | 0.0034 |
| 456 | 0.432 | 0.347 | 0.0026 |
| 789 | 0.577 | 0.434 | 0.0032 |
| 1011 | 0.543 | 0.420 | 0.0030 |
| **Mean** | **0.597 +/- 0.134** | **0.464 +/- 0.109** | **0.0033 +/- 0.0005** |

### Half_Life_Obach (5-Seed RMSE in log-space)

| Seed | RMSE (log) | MAE (log) | RMSE (orig) |
|------|-----------|----------|-------------|
| 42 | 1.133 | 0.874 | 19.51 |
| 123 | 1.383 | 1.104 | 21.83 |
| 456 | 1.146 | 0.903 | 19.15 |
| 789 | 1.149 | 0.866 | 19.22 |
| 1011 | 1.537 | 1.235 | 20.56 |
| **Mean** | **1.269 +/- 0.163** | **0.996 +/- 0.148** | **20.05 +/- 1.17** |

### Clearance_Hepatocyte_AZ (5-Seed RMSE in log-space)

| Seed | RMSE (log) | MAE (log) | RMSE (orig) |
|------|-----------|----------|-------------|
| 42 | 1.391 | 1.117 | 53.92 |
| 123 | 1.408 | 1.126 | 56.32 |
| 456 | 1.220 | 1.015 | 48.74 |
| 789 | 1.350 | 1.117 | 52.46 |
| 1011 | 1.308 | 1.079 | 50.40 |
| **Mean** | **1.335 +/- 0.067** | **1.091 +/- 0.041** | **52.37 +/- 2.87** |

### Clearance_Microsome_AZ (5-Seed RMSE in log-space)

| Seed | RMSE (log) | MAE (log) | RMSE (orig) |
|------|-----------|----------|-------------|
| 42 | 1.396 | 1.174 | 53.74 |
| 123 | 1.185 | 1.028 | 41.02 |
| 456 | 1.782 | 1.531 | 78.85 |
| 789 | 1.357 | 1.172 | 51.30 |
| 1011 | 1.176 | 0.944 | 42.40 |
| **Mean** | **1.379 +/- 0.220** | **1.170 +/- 0.201** | **53.46 +/- 13.56** |

### Tox21 NR-AR (5-Seed AUC)

| Seed | AUC-ROC |
|------|---------|
| 42 | 0.726 |
| 123 | 0.724 |
| 456 | 0.693 |
| 789 | 0.705 |
| 1011 | 0.710 |
| **Mean** | **0.711 +/- 0.012** |

### hERG (5-Seed AUC)

| Seed | AUC-ROC |
|------|---------|
| 42 | 0.809 |
| 123 | 0.791 |
| 456 | 0.817 |
| 789 | 0.837 |
| 1011 | 0.772 |
| **Mean** | **0.805 +/- 0.022** |

## Appendix I: Foundation Model Detailed Results

### Regression Tasks -- Complete Metrics

#### Caco2_Wang

| Model | Test RMSE | Test R^2 | Test MAE | Val RMSE | Val R^2 | Embed Time (s) | Train Time (s) |
|-------|-----------|----------|----------|----------|---------|----------------|----------------|
| Morgan-FP | 0.614 | 0.200 | 0.488 | 0.617 | 0.355 | 1.66 | 0.43 |
| ChemBERTa | 0.496 | 0.478 | 0.379 | 0.667 | 0.246 | 33.06 | 0.40 |
| MolE-FP | 0.670 | 0.047 | 0.536 | 0.698 | 0.174 | 1.71 | 0.45 |
| MolCLR | 0.713 | -0.079 | 0.576 | -- | -- | -- | -- |
| GNN-Best | 0.0027* | 0.481 | 0.0020* | -- | -- | -- | -- |

*GNN RMSE/MAE in original log(cm/s) units; foundation models in z-score space.

#### Half_Life_Obach

| Model | Test RMSE | Test R^2 | Test MAE | Embed Time (s) | Train Time (s) |
|-------|-----------|----------|----------|----------------|----------------|
| Morgan-FP | 22.12 | -0.039 | 9.81 | 0.96 | 0.13 |
| ChemBERTa | 27.39 | -0.594 | 17.15 | 24.12 | 0.16 |
| MolE-FP | 25.01 | -0.329 | 14.44 | 0.94 | 0.13 |
| MolCLR | 21.97 | -0.025 | 8.93 | -- | -- |
| GNN-Best | **21.66** | **0.004** | 9.13 | -- | -- |

#### Clearance_Hepatocyte_AZ

| Model | Test RMSE | Test R^2 | Test MAE | Embed Time (s) | Train Time (s) |
|-------|-----------|----------|----------|----------------|----------------|
| Morgan-FP | **48.36** | -0.015 | 38.07 | 1.61 | 0.32 |
| ChemBERTa | **47.31** | **0.029** | 40.67 | 24.90 | 0.25 |
| MolE-FP | **47.22** | **0.032** | 38.00 | 1.74 | 0.22 |
| MolCLR | 48.71 | -0.030 | 41.87 | -- | -- |
| GNN-Best | 68.22 | -1.019 | 35.21 | -- | -- |

#### Clearance_Microsome_AZ

| Model | Test RMSE | Test R^2 | Test MAE | Embed Time (s) | Train Time (s) |
|-------|-----------|----------|----------|----------------|----------------|
| Morgan-FP | 40.36 | 0.122 | 29.42 | 1.62 | 0.44 |
| ChemBERTa | 42.56 | 0.024 | 31.51 | 20.52 | 0.30 |
| MolE-FP | 41.79 | 0.059 | 30.26 | 1.59 | 0.33 |
| MolCLR | 43.33 | -0.012 | 33.89 | -- | -- |
| GNN-Best | **38.75** | **0.191** | 28.26 | -- | -- |

### Classification Tasks -- Complete Metrics

#### Tox21 NR-AR

| Model | Test AUC | Test F1 | Test Acc | Val AUC | Embed Time (s) | Train Time (s) |
|-------|----------|---------|----------|---------|----------------|----------------|
| Morgan-FP | 0.722 | 0.310 | 0.960 | 0.877 | 8.92 | 1.36 |
| ChemBERTa | 0.728 | 0.330 | 0.955 | **0.896** | 159.98 | 1.43 |
| MolE-FP | 0.675 | 0.391 | 0.961 | 0.859 | 9.22 | 1.81 |
| MolCLR | 0.538 | 0.000 | 0.951 | -- | -- | -- |
| GNN-Best | **0.742** | **0.455** | **0.962** | -- | -- | -- |

Note: ChemBERTa's val AUC (0.896) vs test AUC (0.728) shows scaffold-split degradation.

#### hERG

| Model | Test AUC | Test F1 | Test Acc | Val AUC | Embed Time (s) | Train Time (s) |
|-------|----------|---------|----------|---------|----------------|----------------|
| Morgan-FP | 0.611 | 0.847 | 0.735 | 0.655 | 0.88 | 0.10 |
| ChemBERTa | 0.770 | 0.873 | 0.788 | 0.797 | 16.85 | 0.21 |
| MolE-FP | 0.672 | 0.857 | 0.765 | 0.676 | 1.09 | 0.26 |
| MolCLR | 0.504 | 0.847 | 0.735 | -- | -- | -- |
| GNN-Best | **0.825** | 0.809 | 0.735 | -- | -- | -- |

## Appendix J: Convergence Behavior Analysis

### Trials-to-Best Analysis

How many trials does each algorithm need to find its final best configuration?

| Algorithm | Caco2 | Half_Life | Hep_Clear | Micro_Clear | Tox21 | hERG | Average |
|-----------|-------|-----------|-----------|-------------|-------|------|---------|
| PSO | 8 | 12 | 15 | 11 | 14 | 9 | 11.5 |
| ABC | 11 | 14 | 18 | 16 | 12 | 7 | 13.0 |
| GA | 14 | 11 | 22 | 19 | 13 | 16 | 15.8 |
| SA | 23 | 18 | 31 | 27 | 35 | 21 | 25.8 |
| HC | 6 | 4 | 8 | 5 | 9 | 3 | 5.8 |
| Random | 31 | 22 | 41 | 38 | 28 | 33 | 32.2 |

**Key observations:**
- HC converges fastest (avg 5.8 trials) but to local optima
- PSO and ABC find good solutions within 11-13 trials
- SA uses most trials (25.8 avg) but avoids local optima
- Random Search needs the most trials (32.2 avg) but finds globally competitive solutions

### Budget Efficiency

If we had only 20 trials instead of 50:

| Algorithm | Would lose best on N datasets | Impact |
|-----------|-------------------------------|--------|
| PSO | 1/6 | Minor (most found within 15) |
| ABC | 1/6 | Minor |
| GA | 3/6 | Moderate (needs more generations) |
| SA | 4/6 | Significant (needs full cooling) |
| HC | 0/6 | None (converges very early) |
| Random | 5/6 | Severe (statistical coverage drops) |
| TPE | 2/6 | Moderate (needs startup + guided) |

**Recommendation:** For budgets < 20 trials, use PSO or ABC. For budgets > 50, SA may become more competitive.

## Appendix K: Hyperparameter Sensitivity

### Most Influential Hyperparameters

Based on analysis of all 2,100 trials:

1. **Learning rate** -- Most influential across all datasets. The optimal range is [1e-3, 1e-2] for most tasks.
2. **Hidden dimensions** -- Second most influential. Larger models (256-512) tend to perform better on classification; smaller models (64-128) on some regression tasks.
3. **Number of layers** -- Moderate influence. 4-5 layers is the sweet spot; 3 layers may underfit, 7 layers may overfit on small datasets.
4. **Weight decay** -- Moderate influence. Optimal values are dataset-dependent.
5. **MLP head dimensions** -- Least influential. The head mainly needs sufficient capacity; the specific configuration matters less.

### Hyperparameter Correlations

| Hyperparameter pair | Correlation | Interpretation |
|--------------------|-------------|---------------|
| hidden_dim x num_layers | -0.31 | Wider models work with fewer layers |
| lr x weight_decay | 0.22 | Higher LR needs more regularization |
| hidden_dim x lr | -0.15 | Larger models need smaller LR |
| num_layers x lr | 0.08 | Weak relationship |

## Appendix L: ChemBERTa Scaffold-Split Analysis

### Validation vs. Test Performance

| Dataset | Val AUC | Test AUC | Gap | Interpretation |
|---------|---------|----------|-----|---------------|
| Tox21 | 0.896 | 0.728 | -0.168 | Moderate degradation |
| hERG | 0.797 | 0.770 | -0.027 | Minor degradation |

### ChemBERTa-FT (Fine-Tuned) -- Catastrophic Overfitting

| Dataset | Val AUC | Test AUC | Gap | Interpretation |
|---------|---------|----------|-----|---------------|
| Tox21 | ~0.95 | 0.482 | -0.47 | Catastrophic (worse than random!) |
| hERG | ~0.85 | 0.777 | -0.07 | Moderate degradation |

**Root cause:** Fine-tuning allows the model to memorize scaffold-specific SMILES token patterns in the training set. Under scaffold split, the test set contains novel scaffolds with different token distributions, causing the model to fail.

**Implication:** Fine-tuning SMILES transformers requires careful regularization (lower LR, early stopping, dropout) and validation under scaffold-split conditions. Frozen encoders are safer but sacrifice performance.

## Appendix M: Comparison with TDC Leaderboard

As of March 2026, representative TDC leaderboard results for comparison:

### Caco2_Wang (MAE, lower is better)

| Rank | Method | MAE |
|------|--------|-----|
| 1 | MapLight | 0.2780 |
| 2 | RDKit2D+MLP | 0.3010 |
| 3 | AttentiveFP | 0.3120 |
| -- | **MANU GNN-Best** | **~0.002*** |

*Note: Direct comparison is not possible due to different evaluation protocols (TDC uses their standard splits; we use scaffold split with different seed).

### hERG (AUC, higher is better)

| Rank | Method | AUC |
|------|--------|-----|
| 1 | MapLight | 0.880 |
| 2 | AttentiveFP | 0.860 |
| -- | **MANU GNN-Best** | **0.825** |

Our GNN achieves competitive performance on hERG without specialized architectures, demonstrating the value of systematic HPO.

## Appendix N: Compute Cost Breakdown

### Per-Dataset HPO Time (approximate)

| Dataset | Avg Trial Time | 50 Trials | 6 Algorithms | Total |
|---------|---------------|-----------|-------------|-------|
| Caco2_Wang | ~18s | ~15 min | ~90 min | 1.5 hr |
| Half_Life_Obach | ~22s | ~18 min | ~110 min | 1.8 hr |
| Clearance_Hepatocyte_AZ | ~7s | ~6 min | ~35 min | 0.6 hr |
| Clearance_Microsome_AZ | ~55s | ~45 min | ~270 min | 4.5 hr |
| Tox21 (NR-AR) | ~200s | ~170 min | ~1020 min | 17 hr |
| hERG | ~35s | ~30 min | ~175 min | 2.9 hr |
| **Total** | | | | **~28 hr** |

Additional time for TPE, foundation models, multi-seed: ~17 hours.

**Total project compute: ~45 hours on a single RTX 3060.**

### GPU Memory Usage

| Dataset | Batch Size | Peak GPU Memory |
|---------|-----------|----------------|
| Caco2 | 32 | ~1.2 GB |
| Half_Life | 32 | ~0.9 GB |
| Clearances | 32 | ~1.5 GB |
| Tox21 | 32 | ~3.8 GB |
| hERG | 32 | ~0.8 GB |
| ChemBERTa | 16 | ~8.5 GB |

---

*Document generated: 2026-03-23*
*MANU Project v1.0*
*Total: ~2,100 lines, 17 sections, 14 appendices*
