"""
Confidence-interval-based comparison of HPO optimizers vs Random Search.

Replaces the previous p-value / Wilcoxon signed-rank analysis.

For each optimizer, we compute the *paired* per-dataset improvement relative to
Random Search, evaluated on the same six datasets. Because the raw metrics live
on incompatible scales (RMSE ranging from ~0.003 to ~70, AUC ~0.7), each paired
difference is expressed as a *relative* improvement so that values are
dimensionless and comparable across datasets.

Direction (positive = better than Random Search):
  - RMSE (lower better): improvement = (RandomSearch_RMSE - Optimizer_RMSE) / RandomSearch_RMSE
  - AUC/R2/F1 (higher better): improvement = (Optimizer_metric - RandomSearch_metric) / RandomSearch_metric

Reported per optimizer:
  - win/tie/loss count (sign of the paired difference)
  - mean paired (relative) improvement across datasets
  - bootstrap 95% confidence interval for the mean paired improvement
  - effect size (one-sample Cohen's d of the paired improvements vs 0)
  - interpretation based on whether the CI includes zero

No p-values are computed.
"""

import numpy as np

N_BOOT = 10_000
SEED = 42

# Per-dataset test metrics of the single best trial per optimizer, taken from the
# logged final_training.test_metrics (runs/<dataset>/hpo_<dataset>_<algo>.json).
# Regression uses RMSE; classification uses F1 (the primary metric for the paired
# optimizer comparison, matching the paper's methodology and Table tab:ci_comparison).
# Random Search is the baseline; TPE is excluded because it optimizes a slightly
# different search space, matching the original significance table.
# metric_dir: 'lower' = lower is better (RMSE), 'higher' = higher is better (F1)
datasets = [
    # name,            metric_dir, random,  PSO,    ABC,    GA,     SA,     HC
    ("Caco2_Wang",      "lower",  0.0027, 0.0031, 0.0029, 0.0031, 0.0029, 0.0030),
    ("Half_Life_Obach", "lower",  22.31,  21.66,  21.66,  21.66,  23.70,  24.52),
    ("Clearance_Hep",   "lower",  68.22,  70.21,  72.04,  71.34,  72.04,  72.04),
    ("Clearance_Mic",   "lower",  38.75,  42.76,  42.29,  42.29,  40.94,  41.63),
    ("Tox21_NR-AR",     "higher", 0.4694, 0.4301, 0.4632, 0.4632, 0.4554, 0.4330),
    ("hERG",            "higher", 0.8333, 0.8571, 0.8087, 0.8571, 0.8586, 0.8298),
]

optimizers = ["PSO", "ABC", "GA", "SA", "HC"]


def relative_improvement(metric_dir, random_val, opt_val):
    """Positive = optimizer better than Random Search."""
    if metric_dir == "lower":   # RMSE
        return (random_val - opt_val) / abs(random_val)
    else:                       # AUC / R2 / F1
        return (opt_val - random_val) / abs(random_val)


def interpret(lo, hi, mean):
    if lo <= 0.0 <= hi:
        return "CI includes zero"
    return "CI excludes zero (better)" if mean > 0 else "CI excludes zero (worse)"


def cohen_d(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0
    return x.mean() / sd


rng = np.random.default_rng(SEED)

print(f"{'Algo':5} {'W/T/L':7} {'MeanImpr%':>10} {'95% CI (%)':>22} {'d':>7}  Interpretation")
print("-" * 80)

for j, opt in enumerate(optimizers):
    diffs = []
    for row in datasets:
        name, mdir, rnd = row[0], row[1], row[2]
        opt_val = row[3 + j]
        diffs.append(relative_improvement(mdir, rnd, opt_val))
    diffs = np.array(diffs, dtype=float)

    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))

    mean = diffs.mean()

    # bootstrap over datasets
    n = len(diffs)
    boot_means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    d = cohen_d(diffs)
    interp = interpret(lo, hi, mean)

    print(f"{opt:5} {f'{wins}/{ties}/{losses}':7} {mean*100:10.2f} "
          f"{f'[{lo*100:+.2f}, {hi*100:+.2f}]':>22} {d:7.2f}  {interp}")

print()
print("Per-dataset relative improvements (%), positive = better than Random Search:")
header = "Dataset".ljust(16) + "".join(o.rjust(9) for o in optimizers)
print(header)
for j_row, row in enumerate(datasets):
    name, mdir, rnd = row[0], row[1], row[2]
    cells = []
    for j, opt in enumerate(optimizers):
        imp = relative_improvement(mdir, rnd, row[3 + j]) * 100
        cells.append(f"{imp:+8.2f}")
    print(name.ljust(16) + "".join(c.rjust(9) for c in cells))
