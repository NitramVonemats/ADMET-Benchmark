"""
Clean, verifiable HPO-algorithm comparison (Experiment B).

Re-runs 7 optimizers x 6 datasets x 50 trials from scratch (the old runs/*.json had
no trial history and an identical-config anomaly). Consistent with the VERIFIED fair
GNN benchmark (scripts/run_fair_gnn_benchmark.py):
  - shared GraphConv pipeline via optimized_gnn.train_model (epochs=50, patience=12,
    train seed=42, single dataset split per endpoint),
  - objective = VALIDATION metric only (regression: min val RMSE; classification:
    max val AUC-ROC); test recorded per trial but NEVER used for selection,
  - full per-trial history stored (config, val_metrics, test_metrics),
  - distinct, reproducible per-(dataset,optimizer) candidate seed.

Optimizers: random (reproduces the fair benchmark), pso, abc, ga, sa, hc (NiaPy), tpe (Optuna).

Outputs -> results/hpo_comparison_clean/hpo_<dataset>_<algo>.json,
           results/hpo_comparison_clean/summary.json,
           results/hpo_comparison_clean/ci_analysis.json  (paired vs Random Search)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from optimized_gnn import (  # noqa: E402
    OptimizedGNNConfig, train_model, is_classification_dataset, resolve_device, prepare_dataset,
)
from run_fair_gnn_benchmark import HIDDEN, LAYERS, HEAD1, HEAD2, HEAD3, DS_SEED, sample_config  # noqa: E402

DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]
ALGOS = ["random", "pso", "abc", "ga", "sa", "hc", "tpe"]
ALGO_OFFSET = {"random": 0, "pso": 7, "abc": 13, "ga": 21, "sa": 29, "hc": 37, "tpe": 43}
N_TRIALS, EPOCHS, PATIENCE, TRAIN_SEED = 50, 50, 12, 42
OUT = PROJECT_ROOT / "results" / "hpo_comparison_clean"


def _idx(v, lst):
    return lst[min(int(v * len(lst)), len(lst) - 1)]


def decode(x):
    h1 = _idx(x[4], HEAD1)
    h2 = _idx(x[5], [a for a in HEAD2 if a <= h1] or HEAD2)
    h3 = _idx(x[6], [a for a in HEAD3 if a <= h2] or HEAD3)
    return dict(hidden_dim=int(_idx(x[0], HIDDEN)), num_layers=int(_idx(x[1], LAYERS)),
                head_dims=[int(h1), int(h2), int(h3)],
                lr=float(10 ** (-4 + x[2] * 2)), weight_decay=float(10 ** (-6 + x[3] * 4)))


def eval_cfg(ds, cfg_d, device, is_clf, cache):
    res = train_model(dataset_name=ds, config=OptimizedGNNConfig(**cfg_d), epochs=EPOCHS,
                      patience=PATIENCE, device=device, seed=TRAIN_SEED, dataset_cache=cache,
                      evaluate_test=True, return_model=False, verbose=False)
    vm, tm = res["val_metrics"], res["test_metrics"]
    return vm, tm, float(vm["auc_roc"] if is_clf else vm["rmse"])


def run_random(ds, device, is_clf, seed, cache):
    rng = np.random.RandomState(seed)
    tr = []
    for _ in range(N_TRIALS):
        cfg = sample_config(rng)
        vm, tm, vs = eval_cfg(ds, cfg, device, is_clf, cache)
        tr.append({"config": cfg, "val_metrics": vm, "test_metrics": tm, "val_select": vs})
    return tr


def run_niapy(ds, algo, device, is_clf, seed, cache):
    from niapy.task import Task
    from niapy.problems import Problem
    ALGO_CLS = _niapy_algos()
    if algo not in ALGO_CLS:
        raise RuntimeError(f"NiaPy algorithm '{algo}' unavailable in this build")
    tr = []

    class P(Problem):
        def __init__(self):
            super().__init__(dimension=7, lower=0.0, upper=1.0)

        def _evaluate(self, x):
            cfg = decode(x)
            vm, tm, vs = eval_cfg(ds, cfg, device, is_clf, cache)
            tr.append({"config": cfg, "val_metrics": vm, "test_metrics": tm, "val_select": vs})
            return (-vm["auc_roc"]) if is_clf else vm["rmse"]

    algo_obj = ALGO_CLS[algo](seed=seed)
    task = Task(problem=P(), max_evals=N_TRIALS)
    algo_obj.run(task)
    return tr[:N_TRIALS]


def _niapy_algos():
    out = {}
    try:
        from niapy.algorithms.basic import ParticleSwarmOptimization, ArtificialBeeColonyAlgorithm, GeneticAlgorithm
        out["pso"] = lambda seed: ParticleSwarmOptimization(population_size=10, seed=seed)
        out["abc"] = lambda seed: ArtificialBeeColonyAlgorithm(population_size=10, seed=seed)
        out["ga"] = lambda seed: GeneticAlgorithm(population_size=10, seed=seed)
    except Exception as e:
        print("  [niapy] basic import issue:", e)
    for name, mods in [("sa", "SimulatedAnnealing"), ("hc", "HillClimbAlgorithm")]:
        for loc in ("niapy.algorithms.other", "niapy.algorithms.basic"):
            try:
                m = __import__(loc, fromlist=[mods])
                out[name] = (lambda cls: (lambda seed: cls(seed=seed)))(getattr(m, mods))
                break
            except Exception:
                continue
    return out


def run_tpe(ds, device, is_clf, seed, cache):
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr = []

    def obj(trial):
        x = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(7)]
        cfg = decode(x)
        vm, tm, vs = eval_cfg(ds, cfg, device, is_clf, cache)
        tr.append({"config": cfg, "val_metrics": vm, "test_metrics": tm, "val_select": vs})
        return vm["auc_roc"] if is_clf else -vm["rmse"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return tr


def select(trials, is_clf):
    return (max(trials, key=lambda t: t["val_select"]) if is_clf
            else min(trials, key=lambda t: t["val_select"]))


def run_all(datasets, algos):
    device = resolve_device("auto")
    OUT.mkdir(parents=True, exist_ok=True)
    # Merge into any existing summary so per-arm / chunked reruns accumulate.
    summary = {}
    sp = OUT / "summary.json"
    if sp.exists():
        try:
            summary = json.load(open(sp))
        except Exception:
            summary = {}
    for ds in datasets:
        is_clf = is_classification_dataset(ds)
        cache = prepare_dataset(dataset_name=ds, val_fraction=0.1, seed=TRAIN_SEED, verbose=False)
        for algo in algos:
            t0 = time.time()
            seed = DS_SEED[ds] + ALGO_OFFSET[algo]
            try:
                if algo == "random":
                    trials = run_random(ds, device, is_clf, seed, cache)
                elif algo == "tpe":
                    trials = run_tpe(ds, device, is_clf, seed, cache)
                else:
                    trials = run_niapy(ds, algo, device, is_clf, seed, cache)
            except Exception as e:
                print(f"  [{ds}/{algo}] FAILED: {e}")
                continue
            best = select(trials, is_clf)
            sel_test = (best["test_metrics"]["auc_roc"] if is_clf else best["test_metrics"]["r2"])
            payload = {"dataset": ds, "algo": algo, "task": "classification" if is_clf else "regression",
                       "n_trials": len(trials), "candidate_seed": seed, "train_seed": TRAIN_SEED,
                       "selection_metric": "val_auc_roc" if is_clf else "val_rmse",
                       "selected_config": best["config"], "selected_val_metrics": best["val_metrics"],
                       "selected_test_metrics": best["test_metrics"], "trials": trials}
            json.dump(payload, open(OUT / f"hpo_{ds}_{algo}.json", "w"), indent=2)
            summary.setdefault(ds, {})[algo] = {"test": sel_test,
                                                 "val_select": best["val_select"]}
            print(f"  [{ds:24}/{algo:6}] {len(trials)} trials, "
                  f"selected test={sel_test:+.4f}  ({time.time()-t0:.0f}s)")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    return summary


def analyze():
    """Paired comparison of each optimizer vs Random Search on selected-config TEST."""
    summ = json.load(open(OUT / "summary.json"))
    algos = [a for a in ALGOS if a != "random"]
    rng = np.random.RandomState(0)
    result = {}
    for algo in algos:
        diffs = []
        wtl = [0, 0, 0]
        for ds in DATASETS:
            if ds not in summ or algo not in summ[ds] or "random" not in summ[ds]:
                continue
            is_clf = is_classification_dataset(ds)
            a = summ[ds][algo]["test"]; r = summ[ds]["random"]["test"]
            # normalize so positive favors the optimizer
            d = (a - r) if is_clf else (r - a) * -1 if False else (a - r)
            # regression test metric is R2 (higher better) -> algo - random too
            d = a - r
            diffs.append(d)
            if abs(d) < 1e-6:
                wtl[1] += 1
            elif d > 0:
                wtl[0] += 1
            else:
                wtl[2] += 1
        diffs = np.array(diffs, dtype=float)
        if len(diffs) == 0:
            continue
        boot = [np.mean(diffs[rng.randint(0, len(diffs), len(diffs))]) for _ in range(10000)]
        dz = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0
        result[algo] = {"n_datasets": int(len(diffs)), "mean_diff": float(diffs.mean()),
                        "win_tie_loss": wtl, "ci95": [float(np.percentile(boot, 2.5)),
                        float(np.percentile(boot, 97.5))], "d_z": dz,
                        "note": "diff = optimizer_test - random_test (per endpoint metric); "
                                "positive favors optimizer; selection was validation-only"}
    json.dump(result, open(OUT / "ci_analysis.json", "w"), indent=2)
    print("\n=== HPO vs Random Search (paired, n=6 datasets, test metric) ===")
    for a, r in result.items():
        print(f"  {a:5}: mean_diff={r['mean_diff']:+.4f} CI95={[round(c,3) for c in r['ci95']]} "
              f"W/T/L={r['win_tie_loss']} d_z={r['d_z']:+.2f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.analyze_only:
        analyze(); return
    print(f"Clean HPO comparison | {args.algos} x {args.datasets} | {N_TRIALS} trials each")
    run_all(args.datasets, args.algos)
    analyze()
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
