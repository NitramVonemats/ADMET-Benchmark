"""
10-seed robustness of the fair baseline configs (for a median-based Table IV).

Takes each baseline's validation-selected config from run_baseline_hpo_parity.py and
retrains its MLP head over the same 10 seeds used for the GNN robustness experiment,
on the cached frozen embeddings. Reports median [min-max] test R2/AUC per (baseline,
dataset). Selection is unchanged (validation, seed 42); only the final evaluation is
repeated across seeds -> a robust, seed-averaged comparison consistent with the GNN.

Output -> results/baseline_hpo_parity/baseline_multiseed.json
"""
import json
import os
import sys

import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import run_baseline_hpo_parity as P  # noqa: E402

SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 7, 99, 2718]
PARITY = os.path.join(PROJECT_ROOT, "results", "baseline_hpo_parity",
                      "baseline_hpo_parity_results.json")
OUT = os.path.join(PROJECT_ROOT, "results", "baseline_hpo_parity", "baseline_multiseed.json")


def main():
    recs = json.load(open(PARITY))
    split_cache = {}
    out = []
    for r in recs:
        ds, bl, cfg = r["dataset"], r["baseline"], r["best_cfg"]
        is_clf = P.is_classification(ds)
        splits = split_cache.setdefault(ds, P.prep_split(ds))
        Xtr, Xva, Xte = P.get_cached_embeddings(bl, ds, splits, "cpu")
        ytr, yte = splits["train"][1], splits["test"][1]
        xs = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
        hls = tuple(cfg["hidden_layer_sizes"])
        kw = dict(hidden_layer_sizes=hls, learning_rate_init=cfg["learning_rate_init"],
                  alpha=cfg["alpha"], max_iter=500, early_stopping=True)
        vals = []
        for s in SEEDS:
            try:
                if is_clf:
                    m = MLPClassifier(random_state=s, **kw).fit(Xtr_s, ytr.astype(int))
                    vals.append(float(roc_auc_score(yte.astype(int), m.predict_proba(Xte_s)[:, 1])))
                else:
                    ys = StandardScaler().fit(ytr.reshape(-1, 1))
                    m = MLPRegressor(random_state=s, **kw).fit(Xtr_s, ys.transform(ytr.reshape(-1, 1)).flatten())
                    pte = ys.inverse_transform(m.predict(Xte_s).reshape(-1, 1)).flatten()
                    vals.append(float(r2_score(yte, pte)))
            except Exception as e:
                print("  warn", ds, bl, s, e)
        med = float(np.median(vals))
        out.append({"dataset": ds, "baseline": bl,
                    "task": "classification" if is_clf else "regression",
                    "median": med, "min": float(min(vals)), "max": float(max(vals)),
                    "n_seeds": len(vals), "values": vals})
        print(f"  {ds:24} {bl:11} {'AUC' if is_clf else 'R2'} median {med:+.3f} "
              f"[{min(vals):+.3f}, {max(vals):+.3f}]")
    json.dump(out, open(OUT, "w"), indent=2)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
