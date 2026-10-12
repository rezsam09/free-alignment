#!/usr/bin/env python
"""make_reference_kernels.py — Compute Tanimoto and AA-composition kernels for reference CKA.

Tanimoto: Morgan fingerprints (radius=2, 1024 bits per pre-registration) from ligand SDF.
Protein kernel: amino-acid composition of residues within 4 Å of the ligand (cosine similarity).
Both computed only for the test-split core set.

Output:
  data/tanimoto_kernel.npy      — (N, N) float64 Tanimoto similarity matrix
  data/tanimoto_kernel_ids.txt  — one pdb_id per line, in matrix row/col order
  data/aa_comp_kernel.npy       — (N, N) float64 AA-composition cosine similarity matrix
  data/aa_comp_kernel_ids.txt   — same order
"""
import os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: rdkit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ec2-user/SageMaker/pdbbind_data")).resolve()
assert ROOT.exists(), f"ROOT not found: {ROOT}"
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
import pandas as pd
from src import prereg
prereg.verify_lock(ROOT)

cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
EMB_DIR = ROOT / "data" / "embeddings"
PDB_DIR = ROOT / "PDBbind_2020"
# case-insensitive directory lookup (PDBbind dirs are mixed-case, IDs are lowercased)
_pdb_dir_map = {d.name.lower(): d for d in PDB_DIR.iterdir() if d.is_dir()}
print(f"PDBbind: {len(_pdb_dir_map)} directories indexed (case-insensitive lookup)")

# ── build core set ────────────────────────────────────────────────────────────
spl = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0)
spl.index = spl.index.astype(str).str.lower()
split_col = cfg["split"]["split_column"]
assert split_col in spl.columns, f"split column {split_col!r} not in {list(spl.columns)}"
test_ids_from_split = set(pid for pid in spl.index if spl.loc[pid, split_col] == "test")

# Core set must equal the one used by the reference-model analyses: intersection of test ids over primary, anchor and ALL
# untrained inits for all three models (every file required; no silent skipping).
UNTR = ["untrained"] + [f"untrained_init{i}" for i in range(2, 6)]
id_sets = []
for model in ("egnn", "gnina", "distance_gnn"):
    for tag in ["primary", "anchor"] + UNTR:
        idsf = EMB_DIR / f"{model}_{tag}_test_ids.txt"
        assert idsf.exists(), f"missing {idsf}"
        id_sets.append(set(idsf.read_text().splitlines()))

core_ids = sorted(test_ids_from_split.intersection(*id_sets))
n = len(core_ids)
print(f"Core set: {n} complexes")

# ── Tanimoto kernel (1024-bit Morgan, pre-registered) ─────────────────────────
print("\nComputing Tanimoto kernel (radius=2, 1024-bit Morgan FP) ...")
t0 = time.time()
NBITS = 1024   # pre-registered; do NOT change to 2048

fps = []
warn_tan = 0
fp_failed_ids = []
for pid in core_ids:
    _pdb_act = _pdb_dir_map.get(pid)
    fp = None
    if _pdb_act is not None:
        sdf = _pdb_act / f"{_pdb_act.name}_ligand.sdf"
        if sdf.exists():
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=True)
            mol = next((m for m in suppl if m is not None), None)
            if mol is not None:
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=NBITS)
                except Exception:
                    fp = None
    if fp is None:
        warn_tan += 1
        fp_failed_ids.append(pid)
    fps.append(fp)

if warn_tan:
    print(f"  WARNING: {warn_tan} complexes have missing/invalid SDF → NaN rows in kernel")
    print(f"    First 5: {fp_failed_ids[:5]}")
    (ROOT / "data" / "tanimoto_failed_ids.txt").write_text("\n".join(fp_failed_ids))
    print(f"  Saved exclusion list: data/tanimoto_failed_ids.txt")

_valid_fps  = [(vi, fp) for vi, fp in enumerate(fps) if fp is not None]
_vfp_list   = [fp for _, fp in _valid_fps]
tan_mat = np.full((n, n), np.nan, dtype=np.float64)
for _r, (_vi, _fp_i) in enumerate(_valid_fps):
    _sim = DataStructs.BulkTanimotoSimilarity(_fp_i, _vfp_list)
    for _c, (_vj, _) in enumerate(_valid_fps):
        tan_mat[_vi, _vj] = _sim[_c]

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s  mean={np.nanmean(tan_mat):.4f}  max={np.nanmax(tan_mat):.4f}")
np.save(ROOT / "data" / "tanimoto_kernel.npy", tan_mat)
(ROOT / "data" / "tanimoto_kernel_ids.txt").write_text("\n".join(core_ids))
print("  Saved: data/tanimoto_kernel.npy + data/tanimoto_kernel_ids.txt")

# ── AA composition kernel (residues within 4 Å of ligand, pre-registered) ────
print("\nComputing AA-composition-within-4Å kernel ...")
t0 = time.time()

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}
AA_ORDER = sorted(set(AA3_TO_1.values()))   # 20 standard, alphabetical
AA_IDX   = {aa: i for i, aa in enumerate(AA_ORDER)}


def ligand_heavy_coords(sdf_path: Path):
    """Return (N, 3) float64 of heavy-atom coords from ligand SDF. None on failure."""
    if not sdf_path.exists():
        return None
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    return mol.GetConformer(0).GetPositions().astype(np.float64)


def aa_comp_within_4A(pid: str, cutoff: float = 4.0):
    """Return 20-d normalized AA composition of pocket residues within cutoff Å of ligand.

    Uses the PDBbind pocket.pdb (atoms from residues within 10 Å) as input.
    Falls back to full protein.pdb if pocket.pdb absent.
    Returns None if ligand coordinates unavailable.
    """
    _pdb_act = _pdb_dir_map.get(pid)
    if _pdb_act is None:
        return None
    _pdb_act_name = _pdb_act.name
    sdf = _pdb_act / f"{_pdb_act_name}_ligand.sdf"
    lig_coords = ligand_heavy_coords(sdf)
    if lig_coords is None:
        return None

    pock = _pdb_act / f"{_pdb_act_name}_pocket.pdb"
    pdb  = _pdb_act / f"{_pdb_act_name}_protein.pdb"
    pocket_file = pock if pock.exists() else (pdb if pdb.exists() else None)
    if pocket_file is None:
        return None

    # Parse residue atoms from pocket PDB
    residue_atoms: dict = {}   # (res_name, chain, res_seq) -> list of [x, y, z]
    with open(pocket_file) as fh:
        for line in fh:
            if line[:6] not in ("ATOM  ", "HETATM"):
                continue
            res_name = line[17:20].strip()
            if res_name not in AA3_TO_1:
                continue   # skip HETATM non-amino-acid
            chain   = line[21]
            res_seq = line[22:26].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            key = (res_name, chain, res_seq)
            residue_atoms.setdefault(key, []).append([x, y, z])

    if not residue_atoms:
        return None

    # For each residue: check if any atom is within cutoff of any ligand atom
    # Vectorised: (n_res_atoms, 3) vs (n_lig_atoms, 3)
    vec = np.zeros(20, dtype=np.float64)
    for (res_name, _, _), atom_list in residue_atoms.items():
        atom_arr = np.array(atom_list, dtype=np.float64)   # (m, 3)
        diffs = atom_arr[:, None, :] - lig_coords[None, :, :]  # (m, L, 3)
        dists = np.sqrt((diffs ** 2).sum(-1))               # (m, L)
        if dists.min() <= cutoff:
            aa1 = AA3_TO_1[res_name]
            vec[AA_IDX[aa1]] += 1.0   # count residue once

    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


vecs = []
aa_bad = []
for pid in core_ids:
    v = aa_comp_within_4A(pid)
    bad = v is None or v.sum() == 0     # missing files OR no residue within the cutoff
    if bad:
        v = np.zeros(20, dtype=np.float64)
    aa_bad.append(bad)
    vecs.append(v)
aa_bad = np.array(aa_bad)
warn_aa = int(aa_bad.sum())
if warn_aa:
    print(f"  WARNING: {warn_aa} complexes have no valid AA vector -> NaN rows/cols in the AA kernel (like Tanimoto)")
    (ROOT / "data" / "aa_comp_failed_ids.txt").write_text("\n".join(p for p, b in zip(core_ids, aa_bad) if b))

vecs_arr = np.array(vecs, dtype=np.float64)   # (n, 20)

# Cosine similarity: AA_kernel[i, j] = cos(vec_i, vec_j)
norms = np.linalg.norm(vecs_arr, axis=1, keepdims=True)
norms[norms == 0] = 1.0                       # avoid divide-by-zero
vecs_normed = vecs_arr / norms
aa_mat = vecs_normed @ vecs_normed.T          # (n, n), cosine similarity

# Clip to [-1, 1] for numerical stability
aa_mat = np.clip(aa_mat, -1.0, 1.0)
aa_mat[aa_bad, :] = np.nan      # failed complexes are NaN, not an all-zero row that would distort regressions
aa_mat[:, aa_bad] = np.nan

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s  mean={np.nanmean(aa_mat):.4f}  max={np.nanmax(aa_mat):.4f}")
np.save(ROOT / "data" / "aa_comp_kernel.npy", aa_mat)
(ROOT / "data" / "aa_comp_kernel_ids.txt").write_text("\n".join(core_ids))
print("  Saved: data/aa_comp_kernel.npy + data/aa_comp_kernel_ids.txt")

# ── Tanimoto test×train kernel (for leakage analysis) ────────────────────────
print("\nComputing Tanimoto test×train kernel (for run_leakage_r2.py) ...")
t0 = time.time()

train_ids_from_split = set(pid for pid in spl.index if spl.loc[pid, split_col] == "train")
train_sets = []
for model in ("egnn", "gnina", "distance_gnn"):
    idsf = EMB_DIR / f"{model}_primary_train_ids.txt"
    assert idsf.exists(), f"missing {idsf}"
    train_sets.append(set(idsf.read_text().splitlines()))
train_core_ids = sorted(train_ids_from_split.intersection(*train_sets))
n_train = len(train_core_ids)
print(f"  Train core: {n_train} complexes")

train_fps = []
warn_train = 0
for pid in train_core_ids:
    _pdb_act = _pdb_dir_map.get(pid)
    fp = None
    if _pdb_act is not None:
        sdf = _pdb_act / f"{_pdb_act.name}_ligand.sdf"
        if sdf.exists():
            suppl = Chem.SDMolSupplier(str(sdf), removeHs=True)
            mol = next((m for m in suppl if m is not None), None)
            if mol is not None:
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=NBITS)
                except Exception:
                    fp = None
    if fp is None:
        warn_train += 1
    train_fps.append(fp)

if warn_train:
    print(f"  WARNING: {warn_train} train complexes have missing/invalid SDF → NaN cols")

# test×train matrix: rows=test, cols=train; NaN for any failed FP
_valid_tr_fps = [(j, fp) for j, fp in enumerate(train_fps) if fp is not None]
_vtr_list     = [fp for _, fp in _valid_tr_fps]
tan_test_train = np.full((n, n_train), np.nan, dtype=np.float64)
for _vi, _fp_i in _valid_fps:  # _valid_fps from test tanimoto section above
    _sim = DataStructs.BulkTanimotoSimilarity(_fp_i, _vtr_list)
    for _c, (_vj, _) in enumerate(_valid_tr_fps):
        tan_test_train[_vi, _vj] = _sim[_c]

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s  mean={np.nanmean(tan_test_train):.4f}  max={np.nanmax(tan_test_train):.4f}")
np.save(ROOT / "data" / "tanimoto_test_train.npy", tan_test_train)
(ROOT / "data" / "tanimoto_train_ids.txt").write_text("\n".join(train_core_ids))
print("  Saved: data/tanimoto_test_train.npy + data/tanimoto_train_ids.txt")

print("\nDone. Reference kernel files ready for run_residualization_r2.py, run_write_lowsim_ids.py and run_leakage_r2.py")
