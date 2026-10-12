"""prereg.py -- tiny helper for the pre-registration discipline in the step-0 plan.

Stages of a config:
  config/config.draft.yaml   editable while you work through step 0
  config/config.yaml         frozen copy, written by lock()
  config/config.lock.json    sha256 of the frozen config + frozen input files

After lock(), the ONLY sanctioned way to change a value is apply_deviation(),
which writes the deviation-log line first and only then changes the config.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

SENTINELS = ("DECIDE", "PENDING")


class LockViolation(RuntimeError):
    pass


# ----------------------------------------------------------------- paths
def find_root(start: Path | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "config").is_dir():
            return cand
    raise FileNotFoundError("No project root found (looked for a 'config/' dir). Run Part 0 first.")


def P(root: Path | None = None) -> dict:
    r = find_root(root)
    return dict(
        root=r,
        draft=r / "config" / "config.draft.yaml",
        frozen=r / "config" / "config.yaml",
        lock=r / "config" / "config.lock.json",
        dev_log=r / "logs" / "deviation_log.md",
        excl=r / "logs" / "exclusions.tsv",
        draft_log=r / "logs" / "draft_changes.log",
    )


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def today() -> str:
    return dt.date.today().isoformat()


# ----------------------------------------------------------------- yaml io
def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _write(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))


def is_locked(root: Path | None = None) -> bool:
    return P(root)["lock"].exists()


def load_config(root: Path | None = None) -> dict:
    p = P(root)
    if p["lock"].exists():
        return _read(p["frozen"])
    print("[prereg] WARNING: config not locked yet -- reading DRAFT.")
    return _read(p["draft"])


# ----------------------------------------------------------------- dotted access
def get(cfg: dict, dotted: str):
    cur = cfg
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def _set(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def pending(cfg: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out += pending(v, key + ".")
        elif v is None or (isinstance(v, str) and v.strip() in SENTINELS):
            out.append(key)
        elif isinstance(v, list) and any(isinstance(x, str) and x.strip() in SENTINELS for x in v):
            out.append(key)
    return out


# ----------------------------------------------------------------- draft editing
def set_draft(dotted: str, value, note: str = "", root: Path | None = None) -> None:
    p = P(root)
    if p["lock"].exists():
        raise LockViolation("Config is locked. Use apply_deviation(...) so the change is logged first.")
    cfg = _read(p["draft"])
    old = None
    try:
        old = get(cfg, dotted)
    except (KeyError, TypeError):
        pass
    _set(cfg, dotted, value)
    _write(p["draft"], cfg)
    with p["draft_log"].open("a") as f:
        f.write(f"{now()}\t{dotted}\t{old!r} -> {value!r}\t{note}\n")


# ----------------------------------------------------------------- hashing / lock
def _canon(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def lock(frozen_files: dict[str, str] | None = None, root: Path | None = None) -> dict:
    """frozen_files: {label: path relative to project root} whose sha256 gets pinned (e.g. the split file)."""
    p = P(root)
    if p["lock"].exists():
        raise LockViolation("Already locked.")
    cfg = _read(p["draft"])
    left = pending(cfg)
    if left:
        raise LockViolation("Cannot lock, unresolved values:\n  - " + "\n  - ".join(left))
    files = {}
    for label, rel in (frozen_files or {}).items():
        fp = p["root"] / rel
        if not fp.exists():
            raise FileNotFoundError(fp)
        files[label] = dict(path=rel, sha256=sha256_file(fp))
    _write(p["frozen"], cfg)
    rec = dict(
        locked_at=now(),
        config_sha256=sha256_text(_canon(cfg)),
        frozen_files=files,
        git_commit=_git_commit(p["root"]),
        revision=0,
        history=[],
    )
    p["lock"].write_text(json.dumps(rec, indent=2))
    with p["dev_log"].open("a") as f:
        f.write(f"\n[{today()}] -- LOCK rev 0 -- config sha256 {rec['config_sha256'][:12]} -- step 0 frozen\n")
    return rec


def verify_lock(root: Path | None = None, quiet: bool = False) -> dict:
    """Call at the top of every downstream notebook. Raises if config or frozen files changed without a logged deviation."""
    p = P(root)
    if not p["lock"].exists():
        raise LockViolation("Not locked. Finish Parts 0-4 first.")
    rec = json.loads(p["lock"].read_text())
    cfg = _read(p["frozen"])
    if sha256_text(_canon(cfg)) != rec["config_sha256"]:
        raise LockViolation("config.yaml no longer matches config.lock.json. Change it only via apply_deviation().")
    for label, info in rec["frozen_files"].items():
        fp = p["root"] / info["path"]
        if not fp.exists() or sha256_file(fp) != info["sha256"]:
            raise LockViolation(f"Frozen file changed or missing: {label} ({info['path']})")
    if not quiet:
        print(f"[prereg] lock OK (rev {rec['revision']}, config {rec['config_sha256'][:12]})")
    return rec


# ----------------------------------------------------------------- deviations
def apply_deviation(dotted: str, new_value, reason: str, root: Path | None = None) -> None:
    """Log FIRST, then change. Bumps lock revision and keeps the old hash in history."""
    p = P(root)
    if not reason or len(reason.strip()) < 10:
        raise ValueError("Give a real reason (>= 10 chars).")
    rec = verify_lock(root, quiet=True)
    cfg = _read(p["frozen"])
    old = get(cfg, dotted)
    with p["dev_log"].open("a") as f:
        f.write(f"[{today()}] -- changed {dotted} from {old!r} to {new_value!r} -- {reason}\n")
    new_cfg = copy.deepcopy(cfg)
    _set(new_cfg, dotted, new_value)
    _write(p["frozen"], new_cfg)
    rec["history"].append(dict(revision=rec["revision"], config_sha256=rec["config_sha256"], superseded_at=now()))
    rec["revision"] += 1
    rec["config_sha256"] = sha256_text(_canon(new_cfg))
    rec["git_commit"] = _git_commit(p["root"])
    p["lock"].write_text(json.dumps(rec, indent=2))


def log_deviation_note(text: str, root: Path | None = None) -> None:
    """Free-text entry (e.g. a decision made outside the config). Same format discipline: [date] -- what -- why."""
    p = P(root)
    with p["dev_log"].open("a") as f:
        f.write(f"[{today()}] -- {text}\n")


# ----------------------------------------------------------------- exclusions
def log_exclusion(complex_id: str, stage: str, reason: str, root: Path | None = None) -> None:
    p = P(root)
    new = not p["excl"].exists()
    with p["excl"].open("a") as f:
        if new:
            f.write("complex_id\tstage\treason\tlogged_at\n")
        f.write(f"{complex_id}\t{stage}\t{reason}\t{now()}\n")


def log_exclusions(rows, stage: str, root: Path | None = None) -> int:
    """rows: iterable of (complex_id, reason). Bulk version of log_exclusion."""
    p = P(root)
    new = not p["excl"].exists()
    n = 0
    with p["excl"].open("a") as f:
        if new:
            f.write("complex_id\tstage\treason\tlogged_at\n")
        t = now()
        for cid, reason in rows:
            f.write(f"{cid}\t{stage}\t{reason}\t{t}\n")
            n += 1
    return n
