"""sweeplib.py -- config grid, embedding cache and pair metrics shared by a1_floor_sweep.py and a4_ablation.py."""
import itertools, time
import numpy as np
import fcommon as fc
import randnets as rn
from build_graphs import Graphs
from sklearn.decomposition import PCA
al = fc.al

def make_configs(plan):
    sw = plan["sweep"]; base = dict(sw["base"])
    assert set(base) == set(rn.CFG_KEYS), f"sweep.base must have exactly {rn.CFG_KEYS}"
    cfgs = [dict(base)]
    for f, vals in sw["vary"].items():
        assert f in base, f"unknown factor {f}"
        for v in vals:
            c = dict(base); c[f] = v; cfgs.append(c)
    levels = {f: sorted({base[f], *sw["vary"].get(f, [])}, key=str) for f in base}
    rng = np.random.default_rng(plan["seed"] + 101)
    seen = {rn.cfg_id(c) for c in cfgs}
    tries = 0
    n_target = len(cfgs) + sw["n_random"]
    total = int(np.prod([len(v) for v in levels.values()]))
    n_target = min(n_target, total)
    while len(cfgs) < n_target and tries < 100000:
        tries += 1
        c = {f: levels[f][int(rng.integers(len(levels[f])))] for f in base}
        if rn.cfg_id(c) in seen: continue
        seen.add(rn.cfg_id(c)); cfgs.append(c)
    for c in sw["extra_configs"]:
        c = {**base, **c}
        assert set(c) == set(rn.CFG_KEYS)
        if rn.cfg_id(c) not in seen: seen.add(rn.cfg_id(c)); cfgs.append(c)
    for c in cfgs:
        assert c["readout"] in ("mean", "sum", "max") and c["aggr"] in ("sum", "mean") and c["depth"] >= 1 and c["width"] >= 2 and c["n_rbf"] >= 2
    return cfgs

def sweep_ids(plan, core):
    n = plan["sweep"]["n_complexes"]
    if n is None or n >= len(core): return list(core)
    if n < 50: raise fc.FindingsError("sweep.n_complexes < 50 is too small for debiased CKA")
    rng = np.random.default_rng(plan["seed"] + 7)
    return [core[i] for i in sorted(rng.choice(len(core), n, replace=False))]

def load_items(g: Graphs, ids):
    return [g.get(p) for p in ids]

def emb_path(plan, cfg, arch, seed):
    d = fc.OUT / "sweep_cache" / fc.plan_hash(plan)[:10]; d.mkdir(parents=True, exist_ok=True)
    return d / f"{rn.cfg_id(cfg, arch)}__{arch}__s{seed}.npy"

def get_embedding(plan, cfg, arch, seed, items, n_elem):
    p = emb_path(plan, cfg, arch, seed)
    if p.exists(): return np.load(p)
    E = rn.embed_all(items, cfg, arch, seed, n_elem, plan["seed"], np.dtype(plan["sweep"]["dtype"]).type)
    np.save(p.with_suffix(".tmp.npy"), E); p.with_suffix(".tmp.npy").replace(p)
    return E

def prep(E, k):
    """Centre; if width > k: PCA-k fit on the evaluation set itself (label-free, declared deviation from train-fit PCA)."""
    E = np.asarray(E, np.float64); E = E - E.mean(0)
    if E.shape[1] <= k: return E
    return PCA(n_components=min(k, E.shape[0] - 1), random_state=0).fit_transform(E)

def knn_idx(X, k):
    return al._knn_idx(X, k, center=True)

def mutual(nx, ny, k):
    return float((nx[:, :, None] == ny[:, None, :]).any(2).sum(1).mean() / k)

class Rep:
    """Everything needed from one embedding for pair metrics (computed once)."""
    def __init__(self, E, k_pca, k_nn, Z=None, deg=1):
        self.X = prep(E, k_pca); self.G = fc.GramStats(self.X); self.nn = knn_idx(self.X, k_nn)
        if Z is not None:
            self.Xr = prep(al.ols_resid(np.asarray(E, np.float64), al.design_matrix(Z, deg)), k_pca); self.Gr = fc.GramStats(self.Xr)

def pair_metrics(a: Rep, b: Rep, k_nn):
    d = dict(cka=a.G.cka(b.G), knn=mutual(a.nn, b.nn, k_nn), procrustes=al.procrustes_angular(a.X, b.X))
    if hasattr(a, "Gr") and hasattr(b, "Gr"): d["cka_resid"] = a.Gr.cka(b.Gr)
    else: d["cka_resid"] = float("nan")
    return d
