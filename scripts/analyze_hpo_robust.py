"""
Robust, scale-free comparison of each HPO optimizer vs Random Search.

Motivation: on the six ADMET endpoints the per-endpoint test metrics live on
incomparable scales (R2 for regression -- occasionally pathologically negative --
and AUC for classification). A raw mean R2/AUC difference is therefore dominated
by a single pathological endpoint and is NOT a sound basis for a "which optimizer
wins" claim. This script reports, for each optimizer vs Random Search:

  1. Win / Tie / Loss over the six endpoints (direction only, fully scale-free).
  2. Exact two-sided sign test p-value on wins-vs-losses (ties dropped).
  3. A scale-free DIRECTION-NORMALIZED paired effect: per endpoint the optimizer-
     minus-random difference is divided by the spread (std) of ALL seven
     optimizers' test metrics on that endpoint, then averaged over the six
     endpoints, with a paired percentile bootstrap CI (n=6).
  4. Sensitivity: the mean paired difference recomputed three ways -- raw,
     R2 clipped to [-1,1], and R2 floored at 0 (AUC untouched) -- each with a
     paired bootstrap CI, to show the conclusion does not hinge on the
     pathological-R2 scale.

Reads  results/hpo_comparison_clean/summary.json
Writes results/hpo_comparison_clean/robust_analysis.json
"""
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUMM = os.path.join(ROOT, "results", "hpo_comparison_clean", "summary.json")
OUT = os.path.join(ROOT, "results", "hpo_comparison_clean", "robust_analysis.json")

DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]
CLASSIFICATION = {"tox21", "herg"}
ALGOS = ["random", "pso", "abc", "ga", "sa", "hc", "tpe"]
BOOT = 10000


def sign_test_p(wins, losses):
    """Exact two-sided binomial sign test (p=0.5), ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def boot_ci(diffs, rng):
    if len(diffs) == 0:
        return [float("nan"), float("nan")]
    idx = rng.randint(0, len(diffs), (BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def clip_r2(v, ds, lo, hi):
    return v if ds in CLASSIFICATION else float(np.clip(v, lo, hi))


def floor_r2(v, ds):
    return v if ds in CLASSIFICATION else max(0.0, v)


def main():
    summ = json.load(open(SUMM))
    # per-endpoint spread of the 7 optimizers' test metric (for scale-free effect)
    spread = {}
    for ds in DATASETS:
        vals = [summ[ds][a]["test"] for a in ALGOS if a in summ.get(ds, {})]
        spread[ds] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0

    rng = np.random.RandomState(0)
    out = {"n_endpoints": len(DATASETS), "bootstrap": BOOT,
           "note": "diff = optimizer - random on each endpoint's own test metric "
                   "(R2 for regression, AUC for classification); positive favors optimizer. "
                   "Selection was validation-only.",
           "optimizers": {}}
    print(f"{'algo':5} {'W/T/L':7} {'sign_p':>7} "
          f"{'norm_effect [95% CI]':>26} {'clip[-1,1] mean [CI]':>26} {'floor0 mean [CI]':>24}")
    for algo in [a for a in ALGOS if a != "random"]:
        raw, clipped, floored, normed = [], [], [], []
        w = t = l = 0
        for ds in DATASETS:
            if ds not in summ or algo not in summ[ds] or "random" not in summ[ds]:
                continue
            a = summ[ds][algo]["test"]
            r = summ[ds]["random"]["test"]
            d = a - r
            raw.append(d)
            clipped.append(clip_r2(a, ds, -1, 1) - clip_r2(r, ds, -1, 1))
            floored.append(floor_r2(a, ds) - floor_r2(r, ds))
            normed.append(d / spread[ds] if spread[ds] > 0 else 0.0)
            if abs(d) < 1e-9:
                t += 1
            elif d > 0:
                w += 1
            else:
                l += 1
        raw = np.array(raw); clipped = np.array(clipped)
        floored = np.array(floored); normed = np.array(normed)
        rec = {
            "win_tie_loss": [w, t, l],
            "sign_test_p": sign_test_p(w, l),
            "norm_effect_mean": float(normed.mean()),
            "norm_effect_ci95": boot_ci(normed, rng),
            "sensitivity": {
                "raw_mean_diff": float(raw.mean()), "raw_ci95": boot_ci(raw, rng),
                "clip_-1_1_mean_diff": float(clipped.mean()), "clip_-1_1_ci95": boot_ci(clipped, rng),
                "floor_0_mean_diff": float(floored.mean()), "floor_0_ci95": boot_ci(floored, rng),
            },
        }
        out["optimizers"][algo] = rec
        ne, nc = rec["norm_effect_mean"], rec["norm_effect_ci95"]
        cm, cc = rec["sensitivity"]["clip_-1_1_mean_diff"], rec["sensitivity"]["clip_-1_1_ci95"]
        fm, fc = rec["sensitivity"]["floor_0_mean_diff"], rec["sensitivity"]["floor_0_ci95"]
        print(f"{algo:5} {w}/{t}/{l:<3} {rec['sign_test_p']:7.3f} "
              f"{ne:+6.2f} [{ne and nc[0]:+.2f},{nc[1]:+.2f}]   "
              f"{cm:+6.3f} [{cc[0]:+.3f},{cc[1]:+.3f}]   "
              f"{fm:+6.3f} [{fc[0]:+.3f},{fc[1]:+.3f}]")

    # aggregate: does ANY optimizer show a CI excluding zero on ANY measure?
    any_sig = []
    for algo, rec in out["optimizers"].items():
        for label, ci in [("norm_effect", rec["norm_effect_ci95"]),
                          ("clip", rec["sensitivity"]["clip_-1_1_ci95"]),
                          ("floor0", rec["sensitivity"]["floor_0_ci95"])]:
            if ci[0] > 0 or ci[1] < 0:
                any_sig.append(f"{algo}:{label}")
        if rec["sign_test_p"] < 0.05:
            any_sig.append(f"{algo}:sign_test")
    out["any_measure_excludes_zero_or_sig"] = any_sig
    print(f"\nMeasures with CI excluding zero or sign-test p<0.05: "
          f"{any_sig if any_sig else 'NONE'}")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
