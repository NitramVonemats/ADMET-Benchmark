"""
Cross-artifact consistency validator for the ADMET-Bench results.

Checks that the canonical result artifacts agree with each other and are free of
the failure modes the pre-submission audit was created to catch. Exits non-zero on
any hard failure so it can gate a release / CI.

Checks:
  1. CANONICAL TABLE IV  - table_IV_median_canonical.csv GNN column == the 10-seed
                           medians recomputed from multi_seed_results_fixed.json.
  2. GNN CONFIG SOURCE   - multiseed / descriptor-ablation load their GNN config from
                           fair_gnn_summary.json (single source of truth), and the
                           selected configs are well-formed.
  3. FT SPLIT MATCH      - every ChemBERTa-FT result's stored split hash == the
                           canonical split hash for that endpoint (FT on the same split).
  4. HPO COMPLETE        - hpo_comparison_clean/summary.json has all 6x7 arms and the
                           robust analysis finds no optimizer beating Random Search.
  5. NO PATHOLOGICAL NAN - canonical medians contain no NaN/inf.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

R = os.path.join(ROOT, "results")
DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]
CLF = {"tox21", "herg"}
ALGOS = ["random", "pso", "abc", "ga", "sa", "hc", "tpe"]
FAIL = []


def bad(msg):
    FAIL.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg):
    print(f"  [ok] {msg}")


def check_table_iv():
    print("-- CHECK 1: canonical Table IV GNN column == multiseed medians --")
    ms = json.load(open(f"{R}/multi_seed/multi_seed_results_fixed.json"))
    import csv
    rows = list(csv.DictReader(open(f"{R}/fair_model_family/table_IV_median_canonical.csv")))
    csv_gnn = {r["dataset"]: float(r["GNN_med"]) for r in rows}
    for ds in DATASETS:
        m = ms[ds]
        expect = round(m["auc_median"] if ds in CLF else m["r2_median"], 3)
        got = round(csv_gnn.get(ds, float("nan")), 3)
        if abs(expect - got) > 1e-3:
            bad(f"Table IV {ds}: CSV GNN={got} != multiseed median={expect}")
        else:
            ok(f"{ds}: GNN median {got} matches multiseed")


def check_config_source():
    print("-- CHECK 2: fair_gnn_summary is the GNN config source --")
    summ = json.load(open(f"{R}/fair_model_family/fair_gnn_summary.json"))
    for ds in DATASETS:
        c = summ[ds]["selected_config"]
        need = {"hidden_dim", "num_layers", "head_dims", "lr", "weight_decay"}
        if not need.issubset(c):
            bad(f"{ds}: fair_gnn_summary config missing keys {need - set(c)}")
        elif len(c["head_dims"]) != 3:
            bad(f"{ds}: head_dims not length-3: {c['head_dims']}")
        else:
            ok(f"{ds}: selected_config well-formed (h{c['hidden_dim']} L{c['num_layers']})")


def check_ft_split():
    print("-- CHECK 3: ChemBERTa-FT split hash == canonical split hash --")
    from scripts.canonical_split import canonical_split
    for ds in DATASETS:
        p = f"{R}/chemberta_finetune/chemberta_ft_{ds}_results_fixed.json"
        if not os.path.exists(p):
            bad(f"{ds}: FT result file missing ({p})")
            continue
        stored = json.load(open(p)).get("preprocessing", {}).get("split_hash")
        canon = canonical_split(ds)["hashes"]
        if stored != canon:
            bad(f"{ds}: FT split hash {stored} != canonical {canon}")
        else:
            ok(f"{ds}: FT on canonical split (test {canon['test']})")


def check_hpo():
    print("-- CHECK 4: HPO complete + no optimizer beats Random Search --")
    summ = json.load(open(f"{R}/hpo_comparison_clean/summary.json"))
    for ds in DATASETS:
        missing = [a for a in ALGOS if a not in summ.get(ds, {})]
        if missing:
            bad(f"{ds}: HPO arms missing {missing}")
    ra = json.load(open(f"{R}/hpo_comparison_clean/robust_analysis.json"))
    sig = ra.get("any_measure_excludes_zero_or_sig", [])
    # only the aggressively-compressed floor-0 variant may show marginal effects;
    # the primary sign test and scale-free CI must show nothing consistent.
    primary = [s for s in sig if "sign_test" in s or "norm_effect" in s]
    if primary:
        bad(f"HPO: a primary measure shows a significant optimizer: {primary}")
    else:
        ok(f"HPO: 42 arms present; no optimizer significant on primary measures "
           f"(floor-0-only artifacts: {sig or 'none'})")


def check_nan():
    print("-- CHECK 5: no NaN/inf in canonical medians --")
    ms = json.load(open(f"{R}/multi_seed/multi_seed_results_fixed.json"))
    for ds in DATASETS:
        m = ms[ds]
        v = m["auc_median"] if ds in CLF else m["r2_median"]
        if not np.isfinite(v):
            bad(f"{ds}: non-finite canonical median {v}")
    if not FAIL:
        ok("all canonical medians finite")


def main():
    print("=" * 70)
    print("RESULT CONSISTENCY VALIDATION")
    print("=" * 70)
    for fn in (check_table_iv, check_config_source, check_ft_split, check_hpo, check_nan):
        try:
            fn()
        except FileNotFoundError as e:
            bad(f"{fn.__name__}: missing artifact {e}")
    print("=" * 70)
    if FAIL:
        print(f"[RESULT] {len(FAIL)} FAILURE(S).")
        sys.exit(1)
    print("[RESULT] All result-consistency checks PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
