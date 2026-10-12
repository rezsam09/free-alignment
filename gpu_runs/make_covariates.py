#!/usr/bin/env python
"""make_covariates.py — Pre-registered covariates for CKA residualisation.

Pre-registration specifies exactly two core covariates (config: covariates.core):
  1. ligand_heavy_atom_count   — heavy atoms in the ligand SDF (SDF counts line)
  2. complex_atom_count        — pocket_file_heavy + ligand_heavy
     where pocket_file_heavy = non-H ATOM/HETATM in {pid}_pocket.pdb, excluding waters

Standardisation applied (z-score, fit on train split, applied to all splits).
Output: data/covariates.tsv  — all splits; the docking and bootstrap analyses use the test subset.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT",
    "/home/ec2-user/SageMaker/pdbbind_data")).resolve()
assert ROOT.exists()
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
from src import prereg

prereg.verify_lock(ROOT)
cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
PDB_DIR = ROOT / "PDBbind_2020" / "PDBbind_2020"
SPLIT_COL = cfg["split"]["split_column"]

split = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0)
split.index = split.index.astype(str).str.lower()
pids = list(split.index)
print(f"Complexes in split: {len(pids)}")

ELEM_MISSING = [0, 0]   # [records without an element column, total records]

def count_heavy_pdb(pdb_path: Path) -> int:
    """Count non-H, non-water ATOM/HETATM records."""
    n = 0
    for line in open(pdb_path):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        res = line[17:20].strip()
        if res in ("HOH", "WAT", "TIP", "DOD"):
            continue
        # element column (cols 77-78) if present, else parse atom name
        elem = line[76:78].strip().upper()
        ELEM_MISSING[1] += 1
        if not elem:
            # PQR / stripped files have no element column; atom-name parsing miscounts 4-char H names
            # (e.g. HG21). Counted here and asserted below.
            ELEM_MISSING[0] += 1
            elem = line[12:14].strip().lstrip("0123456789").upper()
        if elem != "H":
            n += 1
    return n

def count_heavy_sdf(sdf_path: Path):
    """Count heavy atoms in ligand SDF using RDKit with explicit-H removal."""
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
        mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
            mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            return None
        n = mol.GetNumHeavyAtoms()
        return n if n > 0 else None
    except Exception:
        return None

rows = []
missing = []
for pid in pids:
    d = PDB_DIR / pid
    pocket = d / f"{pid}_pocket.pdb"
    sdf    = d / f"{pid}_ligand.sdf"
    if not pocket.exists() or not sdf.exists():
        missing.append(pid)
        continue
    lig_heavy = count_heavy_sdf(sdf)
    if lig_heavy is None:
        missing.append(pid)
        continue
    pock_heavy = count_heavy_pdb(pocket)
    rows.append(dict(
        pdb_id=pid,
        ligand_heavy_atom_count=float(lig_heavy),
        complex_atom_count=float(pock_heavy + lig_heavy),
    ))

print(f"  computed: {len(rows)}   missing/skipped: {len(missing)}")
_frac_no_elem = ELEM_MISSING[0] / max(ELEM_MISSING[1], 1)
assert _frac_no_elem < 0.001, (
    f"{100*_frac_no_elem:.1f}% of pocket ATOM/HETATM records have no element column (cols 77-78): "
    "heavy-atom counts would be unreliable (hydrogens with 4-character names counted as heavy). "
    "Use the pdb2pqr/pocket file with elements, or add an element column.")
if missing:
    (ROOT / "data" / "covariates_missing_ids.txt").write_text("\n".join(missing))
    print(f"  WARNING: {len(missing)} complexes dropped (missing files / RDKit parse failure); "
          "downstream scripts require covariates for every core complex and will abort - no imputation")
if missing[:5]:
    print(f"  first missing: {missing[:5]}")

df = pd.DataFrame(rows).set_index("pdb_id")

# z-score: fit on train, apply to all
train_ids = split.loc[split[SPLIT_COL] == "train"].index.tolist()
train_df  = df.loc[df.index.isin(train_ids)]
mu = train_df.mean()
sd = train_df.std().replace(0, 1.0)
df_std = (df - mu) / sd
df_std.index.name = "pdb_id"

out = ROOT / "data" / "covariates.tsv"
df_std.to_csv(out, sep="\t")
print(f"\nSaved: {out}  ({len(df_std)} rows, {len(df_std.columns)} covariates)")
print(f"  mu (train):  {mu.to_dict()}")
print(f"  sd (train):  {sd.to_dict()}")
print(df_std.describe().to_string())
