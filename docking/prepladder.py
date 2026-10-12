"""src/prepladder.py -- Receptor and ligand preparation for Vina docking.

prep_receptor(pdb_path, tmp_dir) -> Path (pdbqt)
prep_ligand(lig_path, tmp_dir)   -> Path (pdbqt)

Receptor pipeline: pdb2pqr → ADFR prepare_receptor → .pdbqt
Ligand pipeline:   Meeko → .pdbqt  (obabel fallback if Meeko fails)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


# ── receptor ──────────────────────────────────────────────────────────────────

def prep_receptor(pdb_path: Path, tmp_dir: Path) -> Path:
    """Prepare receptor PDBQT. Tries pdb2pqr+ADFR; falls back to obabel."""
    pdb_path = Path(pdb_path)
    tmp_dir  = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out = tmp_dir / "receptor.pdbqt"
    if out.exists():
        return out

    # try pdb2pqr + prepare_receptor (ADFR suite)
    if shutil.which("pdb2pqr") and shutil.which("prepare_receptor"):
        pqr = tmp_dir / "receptor.pqr"
        try:
            subprocess.run(
                ["pdb2pqr", "--ff=AMBER", str(pdb_path), str(pqr)],
                check=True, capture_output=True, timeout=120,
            )
            subprocess.run(
                ["prepare_receptor", "-r", str(pqr), "-o", str(out)],
                check=True, capture_output=True, timeout=120,
            )
            return out
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  [prepladder] pdb2pqr/ADFR failed ({e}); trying obabel")

    # fallback: obabel
    if shutil.which("obabel"):
        subprocess.run(
            ["obabel", str(pdb_path), "-O", str(out), "--partialcharge", "gasteiger"],
            check=True, capture_output=True, timeout=120,
        )
        return out

    raise RuntimeError(
        f"Cannot prepare receptor {pdb_path}: neither pdb2pqr+ADFR nor obabel found."
    )


# ── ligand ────────────────────────────────────────────────────────────────────

def prep_ligand(lig_path: Path, tmp_dir: Path) -> Path:
    """Prepare ligand PDBQT via Meeko; falls back to obabel."""
    lig_path = Path(lig_path)
    tmp_dir  = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out = tmp_dir / "ligand.pdbqt"
    if out.exists():
        return out

    # try Meeko
    try:
        from rdkit import Chem
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        if lig_path.suffix.lower() == ".sdf":
            suppl = Chem.SDMolSupplier(str(lig_path), removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        elif lig_path.suffix.lower() == ".mol2":
            mol = Chem.MolFromMol2File(str(lig_path), removeHs=False)
        else:
            mol = None

        if mol is not None:
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(preparator)
            if is_ok:
                out.write_text(pdbqt_str)
                return out
            print(f"  [prepladder] Meeko write failed ({err}); trying obabel")
        else:
            print(f"  [prepladder] RDKit could not read {lig_path.name}; trying obabel")

    except ImportError:
        print("  [prepladder] meeko not installed; trying obabel")
    except Exception as e:
        print(f"  [prepladder] Meeko error ({e}); trying obabel")

    # fallback: obabel
    if shutil.which("obabel"):
        subprocess.run(
            ["obabel", str(lig_path), "-O", str(out),
             "--partialcharge", "gasteiger", "--gen3d"],
            check=True, capture_output=True, timeout=60,
        )
        return out

    raise RuntimeError(
        f"Cannot prepare ligand {lig_path}: Meeko failed and obabel not found."
    )
