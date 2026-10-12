#!/usr/bin/env python
"""make_vina_results.py -- build logs/vina_results.json from the docking notebook's output (STEP2/docking_results.tsv, columns from dockrun.py:
pid, status, stage, err, top1_score, top1_rmsd). top1_rmsd is the RMSD of the TOP-SCORING pose (dockrun: top_pose_pdbqt, energies[0][0]),
which is what config vina.top1_only requires.   Usage: python gpu_runs/make_vina_results.py <docking_results.tsv>
Output record per complex: {top1_rmsd, vina_score, pose_success (<= config vina.pose_success_rmsd_A), error}"""
import json, os, sys
from pathlib import Path
import pandas as pd, yaml
ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
thr = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())["vina"]["pose_success_rmsd_A"]
df = pd.read_csv(sys.argv[1], sep="\t"); assert {"pid", "status", "top1_rmsd", "top1_score"} <= set(df.columns), list(df.columns)
out = {}
for r in df.itertuples():
    ok = r.status == "ok"
    out[str(r.pid).lower()] = dict(top1_rmsd=float(r.top1_rmsd) if ok else None, vina_score=float(r.top1_score) if ok else None,
                                   pose_success=bool(r.top1_rmsd <= thr) if ok else None, error=None if ok else f"{r.stage}: {str(getattr(r, 'err', ''))[:140]}")
(ROOT / "logs" / "vina_results.json").write_text(json.dumps(out, indent=1)); print(len(out), "complexes;", sum(v["error"] is None for v in out.values()), "ok")
