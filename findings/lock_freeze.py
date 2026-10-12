"""lock_freeze.py -- plan hash-lock, freeze tag, exploratory ledger.
  python findings/lock_freeze.py lock                 hash-lock findings_plan.yaml + pin input files (base lock, split, clusters, covariates,
                                                      the 21 reference embeddings x {train,test} + ids)
  python findings/lock_freeze.py freeze               pin the exact analysis code; allowed ONLY if no result file (findings_out/res_*) exists yet
  python findings/lock_freeze.py freeze --refreeze "reason"   after an unavoidable bug fix: logged, old freeze archived (results must then be re-run)
  python findings/lock_freeze.py verify
  python findings/lock_freeze.py note "text"          append an EXPLORATORY ledger entry
Result-producing scripts call fcommon.require_freeze() and abort unless lock + freeze + unchanged code/plan."""
import sys, json, shutil
import fcommon as fc

def pinned_files(plan):
    files = {"main_config_lock": "config/config.lock.json", "split": plan["split_file"], "clusters": plan["cluster_file"],
             "covariates": plan["covariates_file"]}
    for m in fc.MODELS:
        for t in ["primary", "anchor"] + fc.UNTRAINED_TAGS:
            for sp in (plan["train_split"], plan["eval_split"]):
                for ext in (".npy", "_ids.txt"):
                    rel = f"data/embeddings/{m}_{t}_{sp}{ext}"
                    if (fc.ROOT / rel).exists(): files[f"emb:{m}/{t}/{sp}{ext}"] = rel
    return files

def cmd_lock():
    if fc.LOCK.exists(): raise fc.FindingsError("already locked")
    fc.prereg.verify_lock(fc.ROOT)
    plan = fc.load_plan()
    pins = {}
    for label, rel in pinned_files(plan).items():
        p = fc.ROOT / rel
        if not p.exists():
            if label.startswith("emb:"): continue
            raise fc.FindingsError(f"cannot pin missing file {rel}")
        pins[label] = dict(path=rel, sha256=fc.sha256_file(p))
    need = [f"emb:{m}/{t}/{sp}.npy" for m in fc.MODELS for t in ("primary", "anchor") + tuple(fc.UNTRAINED_TAGS) for sp in (plan["train_split"], plan["eval_split"])]
    miss = [k for k in need if k not in pins]
    if miss: raise fc.FindingsError(f"{len(miss)} reference-model embeddings missing, e.g. {miss[:3]}")
    rec = dict(locked_at=fc.now(), plan_sha256=fc.plan_hash(plan), pinned_files=pins)
    fc.LOCK.write_text(json.dumps(rec, indent=2))
    fc.ledger("lock_freeze", "lock", f"plan {rec['plan_sha256'][:12]} pins={len(pins)}", "planned")
    print(f"locked plan {rec['plan_sha256'][:12]} ({len(pins)} files pinned)")

def results_exist():
    return sorted(p.name for p in fc.OUT.glob("res_*")) if fc.OUT.exists() else []

def cmd_freeze(reason=None):
    rec = fc.require_lock()
    if fc.FREEZE.exists():
        if reason is None: raise fc.FindingsError("already frozen (use --refreeze 'reason' only for an unavoidable bug fix)")
        shutil.copy(fc.FREEZE, fc.FREEZE.with_name(f"freeze.{fc.now().replace(' ', '_').replace(':', '')}.json"))
        fc.ledger("lock_freeze", "refreeze", reason, "exploratory")
    else:
        ex = results_exist()
        if ex: raise fc.FindingsError(f"result files already exist ({ex[:3]}...): the freeze tag must precede every result")
    fr = dict(frozen_at=fc.now(), plan_sha256=rec["plan_sha256"], code_sha256=fc.code_hash(), reason=reason)
    fc.FREEZE.write_text(json.dumps(fr, indent=2))
    fc.ledger("lock_freeze", "freeze", f"code {fr['code_sha256'][:12]}", "planned")
    print(f"frozen: code {fr['code_sha256'][:12]}")

def cmd_verify():
    rec = fc.require_lock(); print("lock OK", rec["plan_sha256"][:12])
    if fc.FREEZE.exists():
        fr = json.loads(fc.FREEZE.read_text())
        print("freeze OK" if fr["code_sha256"] == fc.code_hash() and fr["plan_sha256"] == rec["plan_sha256"] else "FREEZE MISMATCH (code or plan changed)")
    else:
        print("not frozen")

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    if a[0] == "lock": cmd_lock()
    elif a[0] == "freeze":
        cmd_freeze(a[a.index("--refreeze") + 1] if "--refreeze" in a else None)
    elif a[0] == "verify": cmd_verify()
    elif a[0] == "note": fc.ledger("manual", "note", " ".join(a[1:]), "exploratory")
    else: sys.exit(__doc__)
