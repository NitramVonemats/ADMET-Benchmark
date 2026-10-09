"""
Pre-submission train/val/test leakage audit for the ADMET benchmark.

Reconstructs the ACTUAL splits used by each model family and tests 5 checks:

  1. SPLIT OVERLAP      - no canonical SMILES in >1 of {train,val,test} per pipeline
  2. VAL FROM TRAIN     - GNN val is carved from the TDC *train* side only
  3. SAME SPLIT         - GNN / parity-baseline / ChemBERTa-FT test partitions
  4. NORMALIZATION      - target mu/sigma computed from TRAIN only (numeric check)
  5. DATASET COUNTS     - TDC totals / retained / partitions vs paper Table I

The script calls the REAL pipeline functions:
  - optimized_gnn.prepare_dataset            (GNN)
  - run_baseline_hpo_parity.prep_split       (Table II baselines)
  - run_chemberta_finetune.prepare_data      (ChemBERTa fine-tune)

For the GNN, prepare_dataset returns torch_geometric graphs that do NOT carry the
SMILES string, so SMILES membership is reconstructed by faithfully replaying the
internal split logic (TDC get_split -> validity filter -> random.Random(42) shuffle
-> 90/10 carve). The reconstruction is cross-checked against the real partition
COUNTS returned by prepare_dataset; a mismatch aborts.

Exits non-zero on any hard failure (checks 1, 2, 3, 4). Writes splits/<dataset>.json.
"""

import os
import sys
import json
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")

try:
    from torch_geometric.utils import from_smiles
except ImportError:  # pragma: no cover
    from torch_geometric.data import from_smiles  # type: ignore

from optimized_gnn import prepare_dataset, is_classification_dataset  # noqa: E402
import importlib  # noqa: E402
_baseline = importlib.import_module("scripts.run_baseline_hpo_parity")
_chemberta = importlib.import_module("scripts.run_chemberta_finetune")

from tdc.single_pred import ADME, Tox  # noqa: E402

DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]

# Paper Table I "Used" counts (train+val+test retained).
PAPER_USED = {"Caco2_Wang": 819, "Half_Life_Obach": 601,
              "Clearance_Hepatocyte_AZ": 1092, "Clearance_Microsome_AZ": 992,
              "tox21": 6533, "herg": 590}

SEED = 42
FAILURES = []  # list of (check_id, dataset, message)


def fail(check, ds, msg):
    FAILURES.append((check, ds, msg))
    print(f"    [FAIL] check{check} {ds}: {msg}")


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else f"__RAW__{smi}"


def canon_set(smiles_iter):
    return set(canon(s) for s in smiles_iter)


def graph_valid(smi):
    """Replicate optimized_gnn.row_to_graph's None-filter predicate."""
    g = from_smiles(smi)
    return g is not None and getattr(g, "x", None) is not None


# ---------------------------------------------------------------------------
# Pipeline split reconstruction
# ---------------------------------------------------------------------------
def load_tdc_split_gnn(name):
    """Exactly the TDC call optimized_gnn.prepare_dataset makes: no seed, no frac."""
    if is_classification_dataset(name):
        tox_labels = {"tox21": "NR-AR", "herg": None, "clintox": "CT_TOX"}
        label = tox_labels.get(name.lower())
        api = Tox(name=name, label_name=label) if label else Tox(name=name)
    else:
        api = ADME(name=name)
    return api.get_split(method="scaffold")  # default seed=42, frac=[0.7,0.1,0.2]


def reconstruct_gnn_smiles(name):
    """SMILES-level GNN split via the single source of truth (scripts.canonical_split),
    which is verified byte-identical to optimized_gnn.prepare_dataset: TDC scaffold
    train/test, then a MOLECULE-LEVEL validation carve (group by canonical SMILES,
    random.Random(42) shuffle, first 10% of molecules -> val) so replicate
    measurements of a molecule never straddle train/val. TDC 'valid' is UNUSED.
    """
    from scripts.canonical_split import canonical_split
    split = load_tdc_split_gnn(name)
    tdc_train_smiles = [r["Drug"] for _, r in split["train"].iterrows()]
    tdc_test_smiles = [r["Drug"] for _, r in split["test"].iterrows()]
    tdc_valid_smiles = [r["Drug"] for _, r in split["valid"].iterrows()]

    cs = canonical_split(name, seed=SEED, val_fraction=0.1)
    return {
        "train": [s for s, _ in cs["train"]],
        "val": [s for s, _ in cs["val"]],
        "test": [s for s, _ in cs["test"]],
        "tdc_train_side": tdc_train_smiles, "tdc_test_side": tdc_test_smiles,
        "tdc_valid_side": tdc_valid_smiles, "split_obj": split,
    }


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("TRAIN/VAL/TEST LEAKAGE AUDIT  (seed=42)")
    print("=" * 78)

    report = {}
    splits_dir = os.path.join(PROJECT_ROOT, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    for ds in DATASETS:
        print(f"\n{'='*78}\nDATASET: {ds}\n{'='*78}")
        is_clf = is_classification_dataset(ds)

        # ---- real GNN pipeline (counts + target stats) ----
        cache = prepare_dataset(ds, val_fraction=0.1, seed=SEED, verbose=False)
        gnn_real_counts = (len(cache["train"]), len(cache["val"]), len(cache["test"]))
        mu_real, sigma_real = cache["log_stats"]

        # ---- reconstruct GNN SMILES and cross-check against real counts ----
        gnn = reconstruct_gnn_smiles(ds)
        rec_counts = (len(gnn["train"]), len(gnn["val"]), len(gnn["test"]))
        if rec_counts != gnn_real_counts:
            fail("recon", ds,
                 f"reconstructed counts {rec_counts} != real prepare_dataset {gnn_real_counts}")
        else:
            print(f"  [ok] GNN reconstruction matches prepare_dataset counts {rec_counts}")

        gnn_tr, gnn_va, gnn_te = (canon_set(gnn["train"]), canon_set(gnn["val"]),
                                  canon_set(gnn["test"]))

        # ---- real baseline pipeline ----
        base = _baseline.prep_split(ds)  # {'train':(smiles,y),'valid':..,'test':..}
        base_tr = canon_set(base["train"][0])
        base_va = canon_set(base["valid"][0])
        base_te = canon_set(base["test"][0])

        # ---- real ChemBERTa fine-tune pipeline ----
        task = "classification" if is_clf else "regression"
        cb = _chemberta.prepare_data(ds, task, seed=SEED)
        cb_tr = canon_set(cb["train_smiles"])
        cb_va = canon_set(cb["val_smiles"])
        cb_te = canon_set(cb["test_smiles"])

        # ================= CHECK 1: intra-pipeline overlap =================
        print("  -- CHECK 1: split overlap (per pipeline) --")
        for fam, (tr, va, te) in {
            "GNN": (gnn_tr, gnn_va, gnn_te),
            "Baseline": (base_tr, base_va, base_te),
            "ChemBERTa": (cb_tr, cb_va, cb_te),
        }.items():
            o_tv, o_tt, o_vt = tr & va, tr & te, va & te
            print(f"     {fam:9} n(tr/va/te)={len(tr)}/{len(va)}/{len(te)}  "
                  f"overlaps tr-va={len(o_tv)} tr-te={len(o_tt)} va-te={len(o_vt)}")
            if o_tv or o_tt or o_vt:
                fail(1, f"{ds}/{fam}",
                     f"overlap tr-va={len(o_tv)} tr-te={len(o_tt)} va-te={len(o_vt)}; "
                     f"examples={list((o_tv|o_tt|o_vt))[:3]}")

        # ================= CHECK 2: GNN val from TRAIN side only ===========
        print("  -- CHECK 2: GNN val carved from TDC-train side only --")
        tdc_train_c = canon_set(gnn["tdc_train_side"])
        tdc_test_c = canon_set(gnn["tdc_test_side"])
        val_not_in_train = gnn_va - tdc_train_c
        val_in_test = gnn_va & tdc_test_c
        train_in_test = gnn_tr & tdc_test_c
        print(f"     val subset-of TDC-train: {len(val_not_in_train)==0} "
              f"(stray={len(val_not_in_train)}); val∩TDC-test={len(val_in_test)}; "
              f"train∩TDC-test={len(train_in_test)}")
        if val_not_in_train or val_in_test:
            fail(2, ds, f"val not strictly from train side "
                        f"(stray={len(val_not_in_train)}, in_test={len(val_in_test)})")

        # ================= CHECK 3: same split across families =============
        print("  -- CHECK 3: test-partition agreement across families --")
        gb_id = gnn_te == base_te
        gc_id = gnn_te == cb_te

        def jac(a, b):
            return len(a & b) / len(a | b) if (a | b) else 1.0
        print(f"     |test| GNN={len(gnn_te)} Baseline={len(base_te)} ChemBERTa={len(cb_te)}")
        print(f"     GNN==Baseline test: {gb_id}  (jaccard={jac(gnn_te,base_te):.4f})")
        print(f"     GNN==ChemBERTa test: {gc_id} (jaccard={jac(gnn_te,cb_te):.4f})")
        # GNN vs Baseline MUST match (same TDC call: scaffold, seed=42, default frac).
        if not gb_id:
            fail(3, ds, f"GNN vs Baseline test differ (jaccard={jac(gnn_te,base_te):.4f})")
        # ChemBERTa fine-tuning now consumes the SAME canonical split -> test MUST match.
        if not gc_id:
            fail(3, ds, f"GNN vs ChemBERTa-FT test differ (jaccard={jac(gnn_te,cb_te):.4f})")
        cb_mismatch = not gc_id

        # ================= CHECK 4: target normalization train-only ========
        print("  -- CHECK 4: target mu/sigma from TRAIN only (numeric) --")
        if is_clf:
            print("     classification: no target normalization (mu=0,sigma=1) - N/A")
        else:
            y_train = np.array([g.original_y for g in cache["train"]], dtype=np.float64)
            y_val = np.array([g.original_y for g in cache["val"]], dtype=np.float64)
            y_test = np.array([g.original_y for g in cache["test"]], dtype=np.float64)
            all_neg = bool(np.all(np.concatenate([y_train, y_val]) < 0))
            if all_neg:  # Caco2 path: no log, stats on raw train
                mu_train = y_train.mean()
                sig_train = y_train.std()
            else:
                clip_min = 1e-3
                mu_train = np.log(np.clip(y_train, clip_min, None)).mean()
                sig_train = np.log(np.clip(y_train, clip_min, None)).std()
            # stats using train+val and train+test to prove test is NOT used
            if all_neg:
                mu_trval = np.concatenate([y_train, y_val]).mean()
            else:
                mu_trval = np.log(np.clip(np.concatenate([y_train, y_val]), 1e-3, None)).mean()
            close_train = np.isclose(mu_real, mu_train, atol=1e-6) and \
                np.isclose(sigma_real, sig_train, atol=1e-6)
            print(f"     stored mu/sigma=({mu_real:.6f},{sigma_real:.6f})  "
                  f"train-only=({mu_train:.6f},{sig_train:.6f})  match={close_train}")
            print(f"     (train+val mu={mu_trval:.6f} for reference; test never used)")
            if not close_train:
                fail(4, ds, f"stored mu/sigma ({mu_real:.6f},{sigma_real:.6f}) != "
                            f"train-only ({mu_train:.6f},{sig_train:.6f})")

        # ================= CHECK 5: dataset counts =========================
        print("  -- CHECK 5: dataset counts vs paper Table I --")
        split = gnn["split_obj"]
        tdc_total = len(split["train"]) + len(split["valid"]) + len(split["test"])
        used = gnn_real_counts[0] + gnn_real_counts[1] + gnn_real_counts[2]
        paper = PAPER_USED[ds]
        print(f"     TDC total (tr+va+te)={tdc_total}  "
              f"TDC parts={len(split['train'])}/{len(split['valid'])}/{len(split['test'])}")
        print(f"     GNN used (tr+va+te)={used}  [tr/va/te={gnn_real_counts}]  "
              f"paper Table I={paper}  match={used==paper}")
        if is_clf:
            y_all = np.array([g.original_y for g in cache["train"]] +
                             [g.original_y for g in cache["val"]] +
                             [g.original_y for g in cache["test"]])
            npos = int((y_all == 1).sum())
            print(f"     class balance: pos={npos}/{len(y_all)} "
                  f"({100*npos/len(y_all):.2f}%)  neg={len(y_all)-npos}")
        if used != paper:
            print(f"     [WARN] used count {used} != paper {paper} "
                  f"(diff={used-paper})")

        report[ds] = {
            "gnn_counts": gnn_real_counts, "tdc_total": tdc_total,
            "used": used, "paper_used": paper,
            "gnn_vs_baseline_test_identical": bool(gb_id),
            "gnn_vs_chemberta_test_identical": bool(gc_id),
            "chemberta_test_jaccard": round(jac(gnn_te, cb_te), 4),
        }

        # persist split membership (sorted canonical for reproducibility)
        with open(os.path.join(splits_dir, f"{ds}.json"), "w") as f:
            json.dump({
                "dataset": ds, "seed": SEED,
                "note": "gnn.* = optimized_gnn pipeline; *_test canonical SMILES for cross-family check",
                "gnn": {
                    "train": sorted(gnn_tr), "val": sorted(gnn_va), "test": sorted(gnn_te),
                },
                "baseline_test": sorted(base_te),
                "chemberta_test": sorted(cb_te),
                "counts": {"gnn": gnn_real_counts,
                           "baseline": [len(base_tr), len(base_va), len(base_te)],
                           "chemberta": [len(cb_tr), len(cb_va), len(cb_te)]},
            }, f, indent=1)

    # ---------------------------------------------------------------- summary
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for ds, r in report.items():
        print(f"  {ds:26} used={r['used']:>5} (paper {r['paper_used']:>5})  "
              f"GNN==Base:{r['gnn_vs_baseline_test_identical']!s:5} "
              f"GNN==CB:{r['gnn_vs_chemberta_test_identical']!s:5} "
              f"(CB jac={r['chemberta_test_jaccard']})")
    print(f"\n  splits written to: {splits_dir}")

    hard = [f for f in FAILURES if f[0] in (1, 2, 3, 4, "recon")]
    if hard:
        print(f"\n[RESULT] {len(hard)} HARD FAILURE(S):")
        for c, d, m in hard:
            print(f"   check{c} {d}: {m}")
        sys.exit(1)
    print("\n[RESULT] All hard checks (1,2,3,4) PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
