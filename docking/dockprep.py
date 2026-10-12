"""dockprep.py -- ONE receptor/ligand prep + docking + RMSD protocol, reused by every complex (methods sec 10)."""
from __future__ import annotations
import re, subprocess, shutil
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from annot import METALS

# metal/cofactor rules under test (the chosen one is frozen in data/step1/metal_rule.json)
RULES = {
    "A_adfr_default":        dict(reinsert_metals=False, adfr_cleanup=None),
    "B_keep_nonstd":         dict(reinsert_metals=False, adfr_cleanup="nphs_lps_waters"),
    "C_reinsert_keep_nonstd": dict(reinsert_metals=True,  adfr_cleanup="nphs_lps_waters"),
}


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return r


def metal_lines(pdb_path):
    return [l for l in open(pdb_path) if l.startswith("HETATM") and
            (l[76:78].strip().upper() or l[12:16].strip().upper()) in METALS]


def prep_receptor(protein_pdb, workdir, rule, adfr_bin, ph=7.4):
    """pdb2pqr (pH) -> [optional metal re-insertion] -> ADFR prepare_receptor. Returns receptor pdbqt path."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    prot_h = workdir / "protonated.pdb"
    r = _run(["pdb2pqr", "--titration-state-method=propka", f"--with-ph={ph}", "--pdb-output", str(prot_h),
              str(protein_pdb), str(workdir / "protonated.pqr")])
    if r.returncode != 0 or not prot_h.exists():
        raise RuntimeError("pdb2pqr failed: " + r.stderr[-800:])
    if rule["reinsert_metals"]:
        keep = [l for l in prot_h.read_text().splitlines() if not l.startswith("END")]
        have = {l[6:11] for l in keep if l.startswith(("ATOM", "HETATM"))}
        keep += [l.rstrip("\n") for l in metal_lines(protein_pdb)]
        prot_h.write_text("\n".join(keep) + "\nEND\n")
    out = workdir / "receptor.pdbqt"
    cmd = [adfr_bin, "-r", str(prot_h), "-o", str(out), "-A", "checkhydrogens"]
    if rule["adfr_cleanup"]:
        cmd += ["-U", rule["adfr_cleanup"]]
    r = _run(cmd)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("ADFR prepare_receptor failed: " + r.stderr[-800:])
    return out


def pdbqt_census(pdbqt_path):
    """residue-level census of a receptor pdbqt. Metals are identified by RESIDUE name (ZN, MG, ...), never by AutoDock atom
    type: the type 'NA' is an N acceptor, not sodium, and would be miscounted as a metal."""
    from annot import METALS
    types, het, metal_atoms = {}, {}, 0
    for l in open(pdbqt_path):
        if l.startswith(("ATOM", "HETATM")):
            t = l[77:79].strip(); types[t] = types.get(t, 0) + 1
            res = l[17:20].strip().upper()
            if res in METALS: metal_atoms += 1
            if l.startswith("HETATM"): het[res] = het.get(res, 0) + 1
    return types, het, metal_atoms


def prep_ligand(sdf_path):
    """Meeko legacy-writer PDBQT string (hydrogens added with coords, as in step 0). Returns (pdbqt, rdkit crystal mol with Hs)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    if mol is None:
        raise RuntimeError("RDKit could not parse ligand sdf")
    molh = Chem.AddHs(mol, addCoords=True)
    setups = MoleculePreparation().prepare(molh)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError("meeko: " + str(err))
    return pdbqt, mol


def box_from_crystal(mol, padding=8.0):
    xyz = mol.GetConformer().GetPositions()[[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]]
    return xyz.mean(0), (xyz.max(0) - xyz.min(0)) + 2 * padding


def dock_one(receptor_pdbqt, lig_pdbqt, center, size, seed, exhaustiveness=8, n_poses=20, cpu=1):
    from vina import Vina
    v = Vina(sf_name="vina", cpu=cpu, seed=int(seed), verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_string(lig_pdbqt)
    v.compute_vina_maps(center=[float(x) for x in center], box_size=[float(x) for x in size])
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    en = np.asarray(v.energies(n_poses=n_poses))
    return dict(energies=en, top_pose_pdbqt=v.poses(n_poses=1))


def mol_from_pdbqt(pdbqt_string, is_dock=None):  # is_dock kept for old callers; Meeko>=0.6 has no such arg
    from meeko import PDBQTMolecule, RDKitMolCreate
    pm = PDBQTMolecule(pdbqt_string, skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pm)
    m = mols[0]
    if m is None:
        raise RuntimeError("could not rebuild RDKit mol from PDBQT (missing SMILES remarks?)")
    return m


def pose_rmsd(pose_mol, crystal_mol):
    """Symmetry-aware heavy-atom RMSD, no realignment (methods sec 10)."""
    a = Chem.RemoveHs(pose_mol); b = Chem.RemoveHs(crystal_mol)
    if a.GetNumHeavyAtoms() != b.GetNumHeavyAtoms():
        raise RuntimeError(f"heavy atom mismatch {a.GetNumHeavyAtoms()} vs {b.GetNumHeavyAtoms()}")
    return float(rdMolAlign.CalcRMS(a, b))
