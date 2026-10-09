"""
One-command pre-submission audit for ADMET-Bench.

Runs the leakage audit and the result-consistency validator, and statically checks
that the release is internally clean:

  A. LEAKAGE           - scripts/verify_splits.py exits 0 (no train/val/test overlap;
                         GNN / baseline / ChemBERTa-FT share the test partition).
  B. CONSISTENCY       - scripts/validate_results.py exits 0 (artifacts agree).
  C. NO STALE RUNS     - the active canonical-result scripts do not read the deprecated
                         runs/ directory (which held pre-audit, test-selected results).
  D. LOCKFILE          - requirements-lock.txt exists.
  E. SPLITS PERSISTED  - splits/<dataset>.json exists for all six endpoints.
  F. RAW BACKUP        - the pre-audit raw results are preserved (never overwritten).

Exits non-zero if any check fails.
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]

# Scripts that produce the canonical artifacts consumed by the manuscript.
ACTIVE = [
    "scripts/canonical_split.py",
    "scripts/run_fair_gnn_benchmark.py",
    "scripts/run_multi_seed_validation.py",
    "scripts/run_descriptor_ablation.py",
    "scripts/run_hpo_comparison_clean.py",
    "scripts/run_chemberta_finetune.py",
    "scripts/run_baseline_hpo_parity.py",
    "scripts/run_baseline_multiseed.py",
    "scripts/toxicity_extended_metrics.py",
]
FAIL = []


def run(label, rel):
    print(f"\n{'='*70}\n{label}: {rel}\n{'='*70}")
    r = subprocess.run([PY, os.path.join(ROOT, rel)], cwd=ROOT)
    if r.returncode != 0:
        FAIL.append(f"{label} ({rel}) exited {r.returncode}")


def check_no_stale_runs():
    print(f"\n{'='*70}\nC. NO STALE runs/ CONSUMPTION in active scripts\n{'='*70}")
    # Match ACTUAL file access to runs/*.json (open/load/glob/Path), not prose
    # in a docstring that merely mentions the deprecated directory.
    pat = re.compile(r"(open|load|read_text|glob|Path)\s*\([^)]*runs/[^)]*\.json")
    for rel in ACTIVE:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            FAIL.append(f"active script missing: {rel}")
            print(f"  [FAIL] missing {rel}")
            continue
        hits = [ln for ln in open(p, encoding="utf-8").read().splitlines()
                if pat.search(ln)]
        if hits:
            FAIL.append(f"{rel} reads deprecated runs/: {hits[0].strip()[:80]}")
            print(f"  [FAIL] {rel}: {hits[0].strip()[:80]}")
        else:
            print(f"  [ok] {rel}")


def check_lockfile():
    print(f"\n{'='*70}\nD. LOCKFILE\n{'='*70}")
    p = os.path.join(ROOT, "requirements-lock.txt")
    if os.path.exists(p):
        print(f"  [ok] requirements-lock.txt ({sum(1 for _ in open(p))} pins)")
    else:
        FAIL.append("requirements-lock.txt missing")
        print("  [FAIL] requirements-lock.txt missing")


def check_splits():
    print(f"\n{'='*70}\nE. SPLITS PERSISTED\n{'='*70}")
    for ds in DATASETS:
        p = os.path.join(ROOT, "splits", f"{ds}.json")
        if os.path.exists(p):
            print(f"  [ok] splits/{ds}.json")
        else:
            FAIL.append(f"splits/{ds}.json missing")
            print(f"  [FAIL] splits/{ds}.json missing")


def check_backup():
    print(f"\n{'='*70}\nF. RAW PRE-AUDIT BACKUP PRESERVED\n{'='*70}")
    d = os.path.join(ROOT, "results", "_raw_backup_preaudit")
    files = glob.glob(os.path.join(d, "**", "*.*"), recursive=True)
    if files:
        print(f"  [ok] {len(files)} raw backup file(s) under results/_raw_backup_preaudit/")
    elif not os.path.isdir(os.path.join(ROOT, "results")):
        print("  [skip] results/ not present in this checkout (artifacts/ is the released copy)")
    else:
        FAIL.append("raw pre-audit backup missing")
        print("  [FAIL] no raw pre-audit backup found")


def main():
    print("#" * 70)
    print("# ADMET-Bench PRE-SUBMISSION AUDIT")
    print("#" * 70)
    run("A. LEAKAGE", "scripts/verify_splits.py")
    run("B. CONSISTENCY", "scripts/validate_results.py")
    check_no_stale_runs()
    check_lockfile()
    check_splits()
    check_backup()
    print(f"\n{'#'*70}")
    if FAIL:
        print(f"# AUDIT FAILED - {len(FAIL)} issue(s):")
        for f in FAIL:
            print(f"#   - {f}")
        print("#" * 70)
        sys.exit(1)
    print("# AUDIT PASSED - all checks green.")
    print("#" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
