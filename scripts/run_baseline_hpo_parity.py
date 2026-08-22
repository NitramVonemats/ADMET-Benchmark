"""
Fair-HPO parity re-benchmark for the non-GNN baselines (Table IV).

Reviewer concern #5: the GNN received heavy per-endpoint HPO (7 methods x 50 trials)
while the Table IV baselines used a single hardcoded sklearn MLP (256,128) with
defaults. This script gives each baseline a comparable budget: a 50-trial,
validation-selected random search over the MLP head, evaluated once on test.

Design (matches scripts/run_complete_foundation_benchmark.py for consistency):
  - TDC scaffold split, seed=42; tox21 uses label_name='NR-AR'; NaN targets filtered.
  - Regression: StandardScaler on RAW Y (no log); test R2 = r2_score(y_test_orig, inv(pred)).
  - Classification: predict_proba[:,1]; test AUC = roc_auc_score(y_test, proba).
  - Frozen embeddings extracted ONCE per (baseline,dataset) and cached; the untrained
    random projection head is DROPPED (encoder.proj -> Identity) so ChemBERTa/MolCLR
    feed their real pretrained embedding (768-d / 512-d) to the MLP. Morgan-FP feeds
    raw 2048 bits. MolE-FP keeps its (seeded, fixed) placeholder encoder output.
  - Features are standardized (StandardScaler fit on train) uniformly for all baselines.

Outputs (results/baseline_hpo_parity/, gitignored; committed artifact NOT overwritten).
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import importlib.util as _ilu  # noqa: E402
_fpath = os.path.join(PROJECT_ROOT, "archive", "old_code", "adme_gnn", "models", "foundation.py")
_spec = _ilu.spec_from_file_location("foundation_enc", _fpath)
_foundation = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_foundation)
ChemBERTaEncoder = _foundation.ChemBERTaEncoder
MolCLREncoder = _foundation.MolCLREncoder
MolEEncoder = _foundation.MolEEncoder
MorganFingerprintEncoder = _foundation.MorganFingerprintEncoder

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

SEED = 42
N_TRIALS = 50
DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]
BASELINES = ["ChemBERTa", "MolCLR", "Morgan-FP", "MolE-FP"]
HIDDEN_CHOICES = [(256, 128), (512, 256), (384, 128), (256, 128, 64), (128, 64)]
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "baseline_hpo_parity")
CACHE_DIR = os.path.join(OUT_DIR, "emb_cache")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_classification(name):
    return name.lower() in ["tox21", "herg", "clintox"]


def get_dataset(name):
    """Identical to run_complete_foundation_benchmark.get_dataset (seed=42, NR-AR)."""
    if name.lower() in ["tox21", "herg", "clintox"]:
        from tdc.single_pred import Tox
        tox_labels = {"tox21": "NR-AR", "herg": None, "clintox": "CT_TOX"}
        label = tox_labels.get(name.lower())
        data = Tox(name=name, label_name=label) if label else Tox(name=name)
    else:
        from tdc.single_pred import ADME
        data = ADME(name=name)
    return data.get_split(method="scaffold", seed=SEED)


def prep_split(name):
    """Return (train_smiles, y_train, valid_smiles, y_valid, test_smiles, y_test)."""
    split = get_dataset(name)
    out = {}
    for key, df in [("train", split["train"]), ("valid", split["valid"]), ("test", split["test"])]:
        y = df["Y"].values.astype(float)
        mask = ~np.isnan(y)
        out[key] = (df["Drug"].values[mask].tolist(), y[mask])
    return out


# ---------- frozen embedding extraction (pre-projection) ----------
def morgan_bits(smiles_list, n_bits=2048, radius=2):
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        X[i] = np.array(fp, dtype=np.float32)
    return X


def encode_batched(encoder, smiles_list, device, bs=32):
    embs = []
    for i in range(0, len(smiles_list), bs):
        with torch.no_grad():
            e = encoder(smiles_list[i:i + bs], device)
        embs.append(e.detach().cpu().numpy())
    return np.vstack(embs)


def extract_embeddings(baseline, smiles_list, device):
    """Pre-projection frozen representation for a baseline."""
    if baseline == "Morgan-FP":
        return morgan_bits(smiles_list, n_bits=2048, radius=2)
    if baseline == "ChemBERTa":
        enc = ChemBERTaEncoder(model_name="seyonec/ChemBERTa-zinc-base-v1")
        enc.proj = nn.Identity()            # drop untrained projection -> 768-d pretrained
        enc = enc.to(device).eval()
        return encode_batched(enc, smiles_list, device)
    if baseline == "MolCLR":
        # Pass the checkpoint path explicitly: foundation.py is imported from archive/,
        # so its relative default path resolves wrongly and would load RANDOM weights.
        ckpt = os.path.join(PROJECT_ROOT, "external", "MolCLR", "ckpt",
                            "pretrained_gin", "checkpoints", "model.pth")
        assert os.path.exists(ckpt), f"MolCLR checkpoint missing: {ckpt}"
        enc = MolCLREncoder(proj_dim=256, pretrained_path=ckpt)
        enc.proj = nn.Identity()            # drop untrained projection -> 512-d pretrained feat_lin
        enc = enc.to(device).eval()
        return encode_batched(enc, smiles_list, device)
    if baseline == "MolE-FP":
        set_seed(SEED)                      # deterministic placeholder encoder
        enc = MolEEncoder(proj_dim=256).to(device).eval()
        return encode_batched(enc, smiles_list, device)
    raise ValueError(baseline)


def get_cached_embeddings(baseline, dataset, splits, device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{baseline}__{dataset}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["Xtr"], z["Xva"], z["Xte"]
    t0 = time.time()
    Xtr = extract_embeddings(baseline, splits["train"][0], device)
    Xva = extract_embeddings(baseline, splits["valid"][0], device)
    Xte = extract_embeddings(baseline, splits["test"][0], device)
    np.savez_compressed(path, Xtr=Xtr, Xva=Xva, Xte=Xte)
    print(f"      extracted {baseline}/{dataset} {Xtr.shape[1]}-d in {time.time()-t0:.1f}s")
    return Xtr, Xva, Xte


# ---------- 50-trial validation-selected MLP search ----------
def sample_trials(n):
    rng = np.random.RandomState(SEED)
    trials = []
    for _ in range(n):
        h = HIDDEN_CHOICES[rng.randint(len(HIDDEN_CHOICES))]
        lr = float(10 ** rng.uniform(-5, -2))
        alpha = float(10 ** rng.uniform(-6, -2))
        trials.append({"hidden_layer_sizes": h, "learning_rate_init": lr, "alpha": alpha})
    return trials


def run_search(baseline, dataset, splits, device):
    is_clf = is_classification(dataset)
    Xtr, Xva, Xte = get_cached_embeddings(baseline, dataset, splits, device)
    ytr = splits["train"][1]; yva = splits["valid"][1]; yte = splits["test"][1]

    xs = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = xs.transform(Xtr), xs.transform(Xva), xs.transform(Xte)

    if is_clf:
        ytr_f, yva_f, yte_f = ytr.astype(int), yva.astype(int), yte.astype(int)
        ys = None
    else:
        ys = StandardScaler().fit(ytr.reshape(-1, 1))
        ytr_f = ys.transform(ytr.reshape(-1, 1)).flatten()

    best = {"val": -np.inf, "cfg": None}
    for cfg in sample_trials(N_TRIALS):
        try:
            if is_clf:
                mdl = MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED, **cfg)
                mdl.fit(Xtr_s, ytr_f)
                pva = mdl.predict_proba(Xva_s)[:, 1]
                val = roc_auc_score(yva_f, pva)
            else:
                mdl = MLPRegressor(max_iter=500, early_stopping=True, random_state=SEED, **cfg)
                mdl.fit(Xtr_s, ytr_f)
                pva = ys.inverse_transform(mdl.predict(Xva_s).reshape(-1, 1)).flatten()
                val = r2_score(yva, pva)
        except Exception:
            continue
        if val > best["val"]:
            best = {"val": float(val), "cfg": cfg}

    # Retrain best config, evaluate once on test.
    cfg = best["cfg"] or {"hidden_layer_sizes": (256, 128), "learning_rate_init": 1e-3, "alpha": 1e-4}
    res = {"dataset": dataset, "baseline": baseline, "task": "classification" if is_clf else "regression",
           "n_trials": N_TRIALS, "best_cfg": {**cfg, "hidden_layer_sizes": list(cfg["hidden_layer_sizes"])},
           "val_metric": best["val"], "train_size": len(ytr), "test_size": len(yte), "emb_dim": Xtr.shape[1]}
    if is_clf:
        mdl = MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED, **cfg).fit(Xtr_s, ytr_f)
        proba = mdl.predict_proba(Xte_s)[:, 1]
        res["pos_class_index"] = int(list(mdl.classes_).index(1))  # verify [:,1] == P(y=1)
        res["classes"] = [int(c) for c in mdl.classes_]
        res["test_auc"] = float(roc_auc_score(yte_f, proba))
        res["test_f1"] = float(f1_score(yte_f, (proba > 0.5).astype(int), zero_division=0))
        res["test_acc"] = float(accuracy_score(yte_f, (proba > 0.5).astype(int)))
        res["metric"] = res["test_auc"]
    else:
        mdl = MLPRegressor(max_iter=500, early_stopping=True, random_state=SEED, **cfg).fit(Xtr_s, ytr_f)
        pte = ys.inverse_transform(mdl.predict(Xte_s).reshape(-1, 1)).flatten()
        res["test_r2"] = float(r2_score(yte, pte))
        res["metric"] = res["test_r2"]
    return res


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | {N_TRIALS} trials/baseline | seed={SEED}")
    set_seed(SEED)

    all_results = []
    split_cache = {}
    for dataset in DATASETS:
        print(f"\n=== {dataset} ({'clf' if is_classification(dataset) else 'reg'}) ===")
        splits = split_cache.setdefault(dataset, prep_split(dataset))
        print(f"  sizes train/valid/test = {len(splits['train'][0])}/{len(splits['valid'][0])}/{len(splits['test'][0])}")
        for baseline in BASELINES:
            r = run_search(baseline, dataset, splits, device)
            all_results.append(r)
            key = "test_auc" if r["task"] == "classification" else "test_r2"
            print(f"  {baseline:11} -> {key}={r[key]:+.3f} (val={r['val_metric']:+.3f}, cfg={r['best_cfg']['hidden_layer_sizes']})")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(os.path.join(OUT_DIR, "baseline_hpo_parity_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'baseline_hpo_parity_results.json')}")


if __name__ == "__main__":
    main()
