#!/usr/bin/env python
"""dockrun.py -- docking runner (frozen prep ladder + Vina + crystal-referenced RMSD).

  dockrun.py run <ctx.json> --phases main reseed --workers 2 [--limit N]
  dockrun.py one <ctx.json> <phase> <pid> <out.json>        (internal: one complex, one child process)

* Complexes are processed in list-file order; one result file per complex (results/<pid>.json), written atomically.
* Each complex runs in its own child process with a wall-clock timeout. A crash or timeout is an 'infra' attempt, retried once;
  a second failure is final: status 'excluded', stage 'docking_infra'.
* Deterministic failures inside the child (prep ladder exhausted, ligand rebuild mismatch, Vina exception) are final at once:
  status 'excluded', stage 'vina_prep' or 'vina_dock'.
* Resume: complexes whose result has status ok/excluded are skipped. An 'infra' record with attempts=1 gets its second attempt.
* Stop: create the STOP file; no new complex is started, complexes in flight finish, the runner exits.
* Every parameter comes from ctx.json. There are no defaults here.
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FINAL = ("ok", "excluded")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(*a):
    print(now(), *a, flush=True)


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_ids(list_file):
    import csv
    with open(list_file, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        col = "complex_id" if "complex_id" in rd.fieldnames else rd.fieldnames[0]
        return [str(r[col]).strip().lower() for r in rd]


def load_result(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ child: one complex
def run_one(ctx: dict, phase: str, pid: str, out: Path):
    for p in (ctx["root"], str(Path(ctx["root"]) / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PATH"] = ctx["path_prepend"] + os.pathsep + os.environ["PATH"]
    import numpy as np
    from rdkit import Chem
    import dockprep
    import prepladder as PL

    ph = ctx["phases"][phase]
    rec = dict(pid=pid, phase=phase, status="excluded", stage="", err="", err_class="", seed=ph["seed"], receptor_route="", ligand_route="",
               top1_score=None, top1_rmsd=None, energies=None, metals_in_input=None, metal_atoms_in_pdbqt=None,
               ligand_heavy_atoms=None, sec_prep=None, sec_dock=None, vina_version=ctx["vina_version"], cpu=ctx["cpu"],
               finished_at=None)
    idx = json.loads(Path(ctx["index_file"]).read_text())
    wd = Path(ctx["workroot"]) / phase / pid
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)
    rule = dockprep.RULES[ctx["metal_rule"]]
    t0 = time.time()
    try:
        # ---------------- prep (frozen ladder; the same call the Step 1b sweep and round trip used) ----------------
        d = Path(idx[pid])
        prot, sdf, mol2 = d / f"{pid}_protein.pdb", d / f"{pid}_ligand.sdf", d / f"{pid}_ligand.mol2"
        try:
            rec_p, rec["receptor_route"] = PL.prep_receptor_ladder(prot, wd / "rec", rule, ctx["adfr_bin"], ctx["ph"])
            rec["metals_in_input"] = len(dockprep.metal_lines(prot))
            rec["metal_atoms_in_pdbqt"] = int(dockprep.pdbqt_census(rec_p)[2])
        except Exception as e:
            raise _Excl("vina_prep", "REC " + str(e)[:400])
        try:
            lig_pdbqt, crys, rec["ligand_route"] = PL.prep_ligand_ladder(sdf, mol2)
            rebuilt0 = dockprep.mol_from_pdbqt(lig_pdbqt)
            n_crys = Chem.RemoveHs(crys).GetNumHeavyAtoms()
            if Chem.RemoveHs(rebuilt0).GetNumHeavyAtoms() != n_crys:
                raise RuntimeError("rebuild heavy-atom mismatch")
            rec["ligand_heavy_atoms"] = int(n_crys)
        except Exception as e:
            raise _Excl("vina_prep", "LIG " + str(e)[:400])
        rec["sec_prep"] = round(time.time() - t0, 1)

        # ---------------- dock ----------------
        t1 = time.time()
        try:
            cen, size = dockprep.box_from_crystal(crys, ctx["box_padding_A"])
            res = dockprep.dock_one(rec_p, lig_pdbqt, cen, size, ph["seed"], ctx["exhaustiveness"], ctx["n_poses"], cpu=ctx["cpu"])
            top = dockprep.mol_from_pdbqt(res["top_pose_pdbqt"])
            rmsd = float(dockprep.pose_rmsd(top, crys))
            score = float(res["energies"][0][0])
            if not (np.isfinite(rmsd) and np.isfinite(score)):
                raise RuntimeError(f"non-finite result: rmsd={rmsd} score={score}")
        except Exception as e:
            raise _Excl("vina_dock", str(e)[:400])
        rec["sec_dock"] = round(time.time() - t1, 1)
        pose_path = Path(ph["poses_dir"]) / f"{pid}.pdbqt"
        atomic_write(pose_path, res["top_pose_pdbqt"])                 # pose first, result file last: a result implies a pose
        rec.update(status="ok", stage="ok", top1_score=score, top1_rmsd=rmsd, energies=[float(x[0]) for x in res["energies"]])
    except _Excl as e:
        rec.update(status="excluded", stage=e.stage, err=e.msg)
        try:
            rec["err_class"] = PL.classify(e.msg)
        except Exception:
            rec["err_class"] = ""
        rec["sec_prep"] = rec["sec_prep"] if rec["sec_prep"] is not None else round(time.time() - t0, 1)
    except Exception:
        rec.update(status="excluded", stage="vina_dock", err="unexpected: " + traceback.format_exc()[-400:], err_class="other")
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    rec["finished_at"] = now()
    atomic_write(out, json.dumps(rec))
    return 0


class _Excl(Exception):
    def __init__(self, stage, msg):
        super().__init__(msg)
        self.stage, self.msg = stage, msg


# ------------------------------------------------------------------------------------------------ parent: scheduler
def attempt_child(ctx_path, phase, pid, out: Path, timeout_s):
    """One attempt in a fresh child process. Returns (record or None, infra_message)."""
    if out.exists():
        out.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()), "one", str(ctx_path), phase, pid, str(out)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        so, _ = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        p.communicate()
        return None, f"timeout after {timeout_s}s"
    if p.returncode != 0 or not out.exists():
        return None, f"child exit {p.returncode}: " + (so or "")[-300:].replace("\n", " | ")
    rec = load_result(out)
    if rec is None:
        return None, "unreadable child output"
    return rec, ""


def process(ctx, ctx_path, phase, pid):
    ph = ctx["phases"][phase]
    rdir = Path(ph["results_dir"])
    final = rdir / f"{pid}.json"
    if Path(ctx["stop_file"]).exists():
        return None
    prev = load_result(final)
    if prev and prev.get("status") in FINAL:
        return "skip"
    attempts = int(prev.get("attempts", 0)) if prev and prev.get("status") == "infra" else 0
    tmp = rdir / f".{pid}.child.json"
    while attempts < 2:
        attempts += 1
        rec, msg = attempt_child(ctx_path, phase, pid, tmp, ctx["timeout_s"])
        wd = Path(ctx["workroot"]) / phase / pid
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
        if rec is None:
            log(f"[{phase}] {pid} infra attempt {attempts}: {msg[:160]}")
            rec = dict(pid=pid, phase=phase, status="infra", stage="docking_infra", err=msg, err_class="infra", seed=ph["seed"],
                       vina_version=ctx["vina_version"], cpu=ctx["cpu"], finished_at=now())
            if attempts >= 2:
                rec["status"] = "excluded"          # second crash/timeout is final
        rec["attempts"] = attempts
        atomic_write(final, json.dumps(rec))
        if rec["status"] in FINAL:
            if tmp.exists():
                tmp.unlink()
            log(f"[{phase}] {pid} {rec['status']} {rec.get('stage','')} "
                + (f"score {rec['top1_score']:.2f} rmsd {rec['top1_rmsd']:.2f}" if rec["status"] == "ok" else rec.get("err", "")[:100]))
            return rec["status"]
    return "infra"


def run(ctx_path, phases, workers, limit):
    ctx = json.loads(Path(ctx_path).read_text())
    for name in phases:
        ph = ctx["phases"][name]
        ids = read_ids(ph["list_file"])
        Path(ph["results_dir"]).mkdir(parents=True, exist_ok=True)
        Path(ph["poses_dir"]).mkdir(parents=True, exist_ok=True)
        todo = []
        for pid in ids:
            r = load_result(Path(ph["results_dir"]) / f"{pid}.json")
            if not (r and r.get("status") in FINAL):
                todo.append(pid)
        if limit:
            todo = todo[:limit]
        log(f"phase {name}: {len(ids)} in list, {len(todo)} to run, workers {workers}, seed {ph['seed']}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda pid: process(ctx, ctx_path, name, pid), todo))
        if Path(ctx["stop_file"]).exists():
            log("STOP file present: exiting after in-flight complexes")
            return 0
        log(f"phase {name} finished")
    log("all requested phases finished")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("ctx"); r.add_argument("--phases", nargs="+", required=True)
    r.add_argument("--workers", type=int, required=True); r.add_argument("--limit", type=int, default=0)
    o = sub.add_parser("one"); o.add_argument("ctx"); o.add_argument("phase"); o.add_argument("pid"); o.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a.ctx, a.phases, a.workers, a.limit))
    sys.exit(run_one(json.loads(Path(a.ctx).read_text()), a.phase, a.pid, Path(a.out)))
