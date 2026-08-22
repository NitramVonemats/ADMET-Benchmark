"""
Single source of truth for the canonical train/validation/test split.

TDC scaffold split (default frac, seed 42) provides train/test; the validation
set is carved from the TDC train side at the MOLECULE level -- all replicate
measurements of a canonical SMILES go to the same partition, so the same molecule
never straddles train/validation. This matches optimized_gnn.prepare_dataset. The
scaffold-split TEST set is used unchanged.

Used by the GNN pipeline (equivalently), the ChemBERTa fine-tuning experiment, and
verify_splits.py so every model family is trained/selected/evaluated on identical
partitions.
"""
import hashlib
import random
import warnings

from rdkit import Chem

warnings.filterwarnings("ignore")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s


def _tdc_split(ds):
    from tdc.single_pred import ADME, Tox
    if ds.lower() == "tox21":
        return Tox(name="tox21", label_name="NR-AR").get_split(method="scaffold")
    if ds.lower() == "herg":
        return Tox(name="herg").get_split(method="scaffold")
    return ADME(name=ds).get_split(method="scaffold")


def _rows(df):
    out = []
    for _, r in df.iterrows():
        if Chem.MolFromSmiles(r["Drug"]) is not None:
            out.append((r["Drug"], float(r["Y"])))
    return out


def _hash(rows):
    return hashlib.md5("|".join(sorted(canon(s) for s, _ in rows)).encode()).hexdigest()[:12]


def canonical_split(ds, seed=42, val_fraction=0.1):
    sp = _tdc_split(ds)
    train_rows, test_rows = _rows(sp["train"]), _rows(sp["test"])
    groups, order = {}, []
    for smi, y in train_rows:
        cs = canon(smi)
        if cs not in groups:
            groups[cs] = []
            order.append(cs)
        groups[cs].append((smi, y))
    mols = list(order)
    random.Random(seed).shuffle(mols)
    n_val = max(1, int(len(mols) * val_fraction))
    if len(mols) - n_val < 1:
        n_val = max(1, len(mols) - 1)
    vset = set(mols[:n_val])
    train = [r for cs in order if cs not in vset for r in groups[cs]]
    val = [r for cs in order if cs in vset for r in groups[cs]]
    return {
        "dataset": ds, "seed": seed, "val_fraction": val_fraction,
        "train": train, "val": val, "test": test_rows,
        "hashes": {"train": _hash(train), "val": _hash(val), "test": _hash(test_rows)},
        "sizes": {"train": len(train), "val": len(val), "test": len(test_rows)},
    }


DATASETS = ["Caco2_Wang", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
            "Clearance_Microsome_AZ", "tox21", "herg"]

if __name__ == "__main__":
    import json, os
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")
    os.makedirs(outdir, exist_ok=True)
    print(f"{'dataset':26} {'tr/va/te':16} train_val_overlap")
    ok = True
    for ds in DATASETS:
        s = canonical_split(ds)
        tr = set(canon(x) for x, _ in s["train"])
        va = set(canon(x) for x, _ in s["val"])
        te = set(canon(x) for x, _ in s["test"])
        tv = len(tr & va); tt = len(tr & te); vt = len(va & te)
        ok = ok and tv == 0 and tt == 0 and vt == 0
        print(f"{ds:26} {s['sizes']['train']}/{s['sizes']['val']}/{s['sizes']['test']:<8} "
              f"train_val={tv} train_test={tt} val_test={vt}")
        json.dump({"dataset": ds, "hashes": s["hashes"], "sizes": s["sizes"],
                   "train_smiles": [x for x, _ in s["train"]],
                   "val_smiles": [x for x, _ in s["val"]],
                   "test_smiles": [x for x, _ in s["test"]]},
                  open(os.path.join(outdir, f"{ds}.json"), "w"))
    print("ALL LEAK-FREE" if ok else "**OVERLAP DETECTED**")
