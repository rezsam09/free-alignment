"""fcommon.py -- shared helpers for the Findings-paper scripts.
Drop the `findings/` folder into the project root (next to src/, config/, data/). Scripts take PROJECT_ROOT from the env.
Statistics (debiased CKA, exact weighted cluster bootstrap, kNN, Procrustes, PCA) are REUSED from src/alignlib.py (the
library code); nothing is re-implemented here except fast pairwise CKA for the sweep (verified against alignlib in tests).
"""
from __future__ import annotations
import os, sys, json, hashlib, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
assert ROOT.exists(), f"PROJECT_ROOT not found: {ROOT}"
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "findings"), str(ROOT)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))  # joblib workers
for _p in (str(ROOT / "findings"), str(ROOT)):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0, _p)
from src import alignlib as al
from src import prereg

FDIR = ROOT / "findings"
PLAN = FDIR / "plan" / "findings_plan.yaml"
LOCK = FDIR / "plan" / "findings_plan.lock.json"
FREEZE = FDIR / "plan" / "freeze.json"
OUT = ROOT / "findings_out"
LEDGER = OUT / "ledger.tsv"
EMB = ROOT / "data" / "embeddings"
MODELS = ("egnn", "distance_gnn", "gnina")
GRAPH_MODELS = ("egnn", "distance_gnn")
UNTRAINED_TAGS = ["untrained"] + [f"untrained_init{i}" for i in range(2, 6)]
PAIRS = [("egnn", "gnina"), ("egnn", "distance_gnn"), ("gnina", "distance_gnn")]


class FindingsError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- hashing / plan
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()

def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)

def plan_hash(plan: dict) -> str:
    return hashlib.sha256(_canon(plan).encode()).hexdigest()

def load_plan() -> dict:
    plan = yaml.safe_load(PLAN.read_text())
    bad = [k for k, v in _flatten(plan) if v is None and k not in NULL_OK]
    if bad:
        raise FindingsError("plan has unset (null) values: " + ", ".join(bad))
    return plan

NULL_OK = {"sweep.n_complexes"}

def _flatten(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, pre + k + ".")
        else:
            yield pre + k, v

def code_hash() -> str:
    """Hash of every findings/*.py (tests excluded): freezing pins the exact analysis code."""
    h = hashlib.sha256()
    for p in sorted(FDIR.glob("*.py")):
        h.update(p.name.encode()); h.update(sha256_file(p).encode())
    return h.hexdigest()

def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------- lock / freeze / ledger
def require_lock() -> dict:
    """The Findings plan must be hash-locked, unchanged, and every pinned input file unchanged. The reference-model configuration lock must also verify."""
    prereg.verify_lock(ROOT, quiet=True)
    if not LOCK.exists():
        raise FindingsError("Findings plan not locked. Run: python findings/lock_freeze.py lock")
    rec = json.loads(LOCK.read_text())
    plan = yaml.safe_load(PLAN.read_text())
    if plan_hash(plan) != rec["plan_sha256"]:
        raise FindingsError("findings_plan.yaml changed after locking. Revert it (a new plan = a new, logged revision).")
    for label, info in rec["pinned_files"].items():
        fp = ROOT / info["path"]
        if not fp.exists() or sha256_file(fp) != info["sha256"]:
            raise FindingsError(f"pinned file changed or missing: {label} ({info['path']})")
    return rec

def require_freeze(script: str) -> dict:
    """Result-producing scripts refuse to run until the freeze tag exists AND the code/plan are byte-identical to the frozen ones."""
    rec = require_lock()
    if not FREEZE.exists():
        raise FindingsError("No freeze tag. Finish code testing on the synthetic fixture, then run: python findings/lock_freeze.py freeze")
    fr = json.loads(FREEZE.read_text())
    if fr["plan_sha256"] != rec["plan_sha256"]:
        raise FindingsError("freeze tag was made for a different plan")
    if fr["code_sha256"] != code_hash():
        raise FindingsError("findings/*.py changed after the freeze tag. Analyses must run on frozen code. "
                            "If a bug fix is unavoidable: log it with `lock_freeze.py note`, re-freeze (`freeze --refreeze`), and report it.")
    ledger(script, "run_start", f"plan {rec['plan_sha256'][:10]} code {fr['code_sha256'][:10]}", "planned")
    return fr

def ledger(script: str, kind: str, text: str, status: str = "planned"):
    """Append-only ledger. status: planned (in the locked plan) | exploratory (anything else). Separate from the deviation log."""
    OUT.mkdir(exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a") as f:
        if new: f.write("time\tscript\tkind\tstatus\ttext\n")
        f.write(f"{now()}\t{script}\t{kind}\t{status}\t{text.replace(chr(9), ' ').replace(chr(10), ' ')}\n")


# ----------------------------------------------------------------------------- ids / split / core set
def split_df(plan) -> pd.DataFrame:
    df = pd.read_csv(ROOT / plan["split_file"], sep="\t", index_col=0)
    df.index = df.index.astype(str).str.lower()
    assert plan["split_column"] in df.columns and plan["affinity_column"] in df.columns, list(df.columns)
    return df

def load_emb(model, tag, split):
    npy, idf = EMB / f"{model}_{tag}_{split}.npy", EMB / f"{model}_{tag}_{split}_ids.txt"
    if not npy.exists():
        return None, None
    assert idf.exists(), f"missing ids file {idf}"
    E = np.load(npy).astype(np.float64)
    ids = [s.lower() for s in idf.read_text().split()]
    assert len(ids) == E.shape[0], f"{npy.name}: {len(ids)} ids vs {E.shape[0]} rows"
    assert len(set(ids)) == len(ids), f"{idf.name}: duplicate ids"
    return E, ids

def core_ids(plan) -> list:
    """Core set = intersection of the ids of primary+anchor test embeddings of all 3 models (same rule as the reference-model pipeline)."""
    sets = []
    for m in MODELS:
        for t in ("primary", "anchor"):
            E, ids = load_emb(m, t, plan["eval_split"])
            if E is None: raise FindingsError(f"missing embeddings {m}/{t}/{plan['eval_split']}")
            sets.append(set(ids))
    core = sorted(set.intersection(*sets))
    sp = split_df(plan)
    bad = [p for p in core if p not in sp.index or sp.loc[p, plan["split_column"]] != plan["eval_split"]]
    if bad: raise FindingsError(f"{len(bad)} core ids are not in the {plan['eval_split']} split (e.g. {bad[:3]})")
    return core

def pick(E, ids, want, name=""):
    return al.strict_subset(E, ids, want, name=name)

_REP = {}
def rep_test(model, tag, plan, core, k=None, cache=True):
    """Test-split embedding of (model, tag) on `core`: z-score + PCA(k) with TRAIN-ONLY statistics (alignlib.Rep).
    The CKA analyses use PCA-64 on train-centered data exactly as in the reference-model pipeline (alignlib.pca_train_fit)."""
    k = k or plan["pca_dim"]
    key = (model, tag, k, tuple(core[:2]), len(core))
    if key in _REP: return _REP[key]
    Etr, itr = load_emb(model, tag, plan["train_split"])
    if Etr is None: raise FindingsError(f"{model}/{tag}: train embeddings required for train-fit statistics")
    Ete, ite = load_emb(model, tag, plan["eval_split"])
    if Ete is None: raise FindingsError(f"{model}/{tag}: missing {plan['eval_split']} embeddings")
    Ete = pick(Ete, ite, core, f"{model}/{tag}")
    r = al.Rep(k=min(k, Etr.shape[0] - 1, Etr.shape[1])).fit(Etr)
    _REP[key] = r.transform(Ete)
    return _REP[key]

def cka_pca(model, tag, plan, core, k=None):
    """Centered PCA-k (no z-score) -- the representation used for CKA in the reference-model pipeline (_tier1_common.projected)."""
    k = k or plan["pca_dim"]
    key = ("cka", model, tag, k, len(core))
    if key in _REP: return _REP[key]
    Etr, _ = load_emb(model, tag, plan["train_split"])
    if Etr is None: raise FindingsError(f"{model}/{tag}: train embeddings required")
    Ete, ite = load_emb(model, tag, plan["eval_split"])
    mu, pca = al.pca_train_fit(Etr, k, name=f"{model}/{tag}")
    _REP[key] = pca.transform(pick(Ete, ite, core, f"{model}/{tag}") - mu)
    return _REP[key]

def clusters(plan, ids):
    p = ROOT / plan["cluster_file"]
    lab = al.load_cluster_labels(p, ids, plan["cluster_col"])
    frac = float((lab < 0).mean())
    if frac > plan["max_unclustered_frac"]:
        raise FindingsError(f"{frac:.1%} of complexes have no cluster label (> {plan['max_unclustered_frac']:.0%}); regenerate clusters")
    return al.fill_singleton_groups(lab)


# ----------------------------------------------------------------------------- fast pairwise debiased CKA for the sweep
class GramStats:
    """Zero-diagonal centered-feature Gram pieces so many pairwise debiased CKAs cost O(n^2) each (vs O(n^2 d) from scratch).
    Equals al.cka_feature up to float error (tested)."""
    def __init__(self, X):
        X = np.asarray(X, np.float64); X = X - X.mean(0)
        K = X @ X.T; np.fill_diagonal(K, 0.0)
        self.K, self.s, self.n = K, K.sum(1), X.shape[0]
        self.self_h = self._h(self)
    @staticmethod
    def _h(a, b=None):
        b = a if b is None else b
        n = a.n
        t1 = float((a.K * b.K).sum()); t2 = a.s.sum() * b.s.sum() / ((n - 1) * (n - 2)); t3 = 2.0 * float(a.s @ b.s) / (n - 2)
        return (t1 + t2 - t3) / (n * (n - 3))
    def cka(self, other):
        d = self.self_h * other.self_h
        return float(self._h(self, other) / np.sqrt(d)) if d > 0 else float("nan")


# ----------------------------------------------------------------------------- misc stats
def zscore_train(Xtr, Xte, tol=1e-8):
    mu, sd = Xtr.mean(0), Xtr.std(0); keep = sd > tol
    return (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]

def write_tsv(df: pd.DataFrame, name: str):
    OUT.mkdir(exist_ok=True)
    p = OUT / name; df.to_csv(p, sep="\t", index=False); return p

def write_json(obj, name: str):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

def ci(vals, level=0.95):
    """Percentile CI over the finite draws; (nan, nan) when no draw is defined (e.g. n_AB with ceiling - floor < 0.05 in every draw)."""
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if v.size == 0: return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))

def stable_int(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)
