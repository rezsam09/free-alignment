"""randnets.py -- forward-only, randomly initialised EGNN-family and Distance-GNN encoders in pure numpy (no torch, no training).
  node input : one-hot element (+ "other") and a ligand flag -> linear to `width`
  edges      : all atom pairs within `radius` (fixed from the INPUT coordinates; self loops excluded)
  message    : m_ij = silu(W2 silu(Wi h_i + Wj h_j + We e_ij))
  node update: h_i <- h_i + Wh2 silu(Wh1 [h_i, agg_i]),  agg_i = sum_j m_ij  (aggr=sum)  or mean_j m_ij (aggr=mean)
  egnn          e_ij = d_ij^2 / radius^2 (1 feature, recomputed from coordinates that are updated each layer:
                x_i <- x_i + mean_j (x_i - x_j) * tanh(w_x . m_ij));   -> E(3)-invariant node features
  distance_gnn  e_ij = Gaussian RBF(d_ij) with n_rbf centres on [0, radius]; coordinates fixed.
  readout       mean / sum / max over ALL atoms of the complex  -> embedding of size `width` (the pooled vector).
Weights: standard-normal draws * gain / sqrt(fan_in), zero biases. Draws are keyed by (base seed, arch, width, depth, seed, tensor, layer), so configs that
differ only in gain / radius / readout / aggr / n_rbf-prefix share the same underlying draws (common random numbers).
These are RE-IMPLEMENTATIONS of the architecture family, not the trained models' code: state this in the paper.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import fcommon as fc

ARCHS = ("egnn", "distance_gnn")
CFG_KEYS = ("width", "depth", "radius", "gain", "readout", "n_rbf", "aggr")

def cfg_id(cfg, arch=None):
    """Canonical id. For egnn n_rbf is irrelevant and is dropped (so the embedding cache is shared)."""
    c = dict(cfg)
    if arch == "egnn": c["n_rbf"] = 0
    return "w{width}_d{depth}_r{radius:g}_g{gain:g}_{readout}_k{n_rbf}_{aggr}".format(**c)

def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))

def _randn(base, arch, cfg, seed, name, layer, shape):
    r = np.random.default_rng(fc.stable_int(base, arch, cfg["width"], cfg["depth"], seed, name, layer))
    return r.standard_normal(shape)

def make_params(cfg, arch, seed, in_dim, base_seed, dtype=np.float32):
    w, g, L = cfg["width"], cfg["gain"], cfg["depth"]
    ef = 1 if arch == "egnn" else cfg["n_rbf"]
    P = dict(W_in=(_randn(base_seed, arch, cfg, seed, "in", -1, (in_dim, w)) * g / np.sqrt(in_dim)).astype(dtype), layers=[])
    for l in range(L):
        def mk(name, shape, fan): return (_randn(base_seed, arch, cfg, seed, name, l, shape) * g / np.sqrt(fan)).astype(dtype)
        P["layers"].append(dict(Wi=mk("Wi", (w, w), 3 * w), Wj=mk("Wj", (w, w), 3 * w), We=mk("We", (ef, w), 3 * w), W2=mk("W2", (w, w), w),
                                Wh1=mk("Wh1", (2 * w, w), 2 * w), Wh2=mk("Wh2", (w, w), w), wx=mk("wx", (w,), w)))
    return P

def edges_for(xyz, radius):
    """Directed edges (src, dst), both directions of every pair within radius, no self loops."""
    pr = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    if len(pr) == 0: return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate([pr[:, 0], pr[:, 1]]).astype(np.int64), np.concatenate([pr[:, 1], pr[:, 0]]).astype(np.int64)

class Batch:
    """Concatenated complexes. xyz (A,3) float64; feat (A,in_dim); src,dst (E,) global indices; off (n+1,) atom offsets."""
    def __init__(self, items, radius, n_elem):
        xs, fs, S, D, off, a0 = [], [], [], [], [0], 0
        for xyz, el, lig in items:
            s, d = edges_for(xyz, radius)
            S.append(s + a0); D.append(d + a0); a0 += len(el); off.append(a0); xs.append(xyz)
            f = np.zeros((len(el), n_elem + 1)); f[np.arange(len(el)), el] = 1.0; f[:, n_elem] = lig; fs.append(f)
        self.xyz = np.vstack(xs).astype(np.float64); self.feat = np.vstack(fs)
        self.src, self.dst, self.off = np.concatenate(S), np.concatenate(D), np.array(off)

def forward(P, cfg, arch, B: Batch, dtype=np.float32):
    with np.errstate(over="ignore", invalid="ignore"):      # overflow is reported below as FindingsError
        return _forward(P, cfg, arch, B, dtype)

def _forward(P, cfg, arch, B: Batch, dtype):
    A = B.xyz.shape[0]; E = len(B.src); radius = cfg["radius"]
    h = (B.feat.astype(dtype) @ P["W_in"])
    x = B.xyz.copy()
    S = sparse.csr_matrix((np.ones(E, dtype), (B.dst, np.arange(E))), shape=(A, E))
    deg = np.maximum(np.asarray(S.sum(1)).ravel(), 1.0).astype(dtype)
    if arch == "distance_gnn":
        cen = np.linspace(0.0, radius, cfg["n_rbf"]); gam = 0.5 / ((cen[1] - cen[0]) ** 2 if cfg["n_rbf"] > 1 else radius ** 2)
        d = np.linalg.norm(x[B.src] - x[B.dst], axis=1)
        ef_fixed = np.exp(-gam * (d[:, None] - cen[None, :]) ** 2).astype(dtype)
    for Lp in P["layers"]:
        if arch == "egnn":
            diff = x[B.dst] - x[B.src]; ef = ((diff ** 2).sum(1) / radius ** 2)[:, None].astype(dtype)
        else:
            ef = ef_fixed
        pre = (h @ Lp["Wi"])[B.dst] + (h @ Lp["Wj"])[B.src] + ef @ Lp["We"]
        m = silu(silu(pre) @ Lp["W2"])
        agg = S @ m
        if cfg["aggr"] == "mean": agg = agg / deg[:, None]
        h = h + silu(np.concatenate([h, agg], 1) @ Lp["Wh1"]) @ Lp["Wh2"]
        if arch == "egnn":
            w = np.tanh(m @ Lp["wx"]).astype(np.float64)
            upd = sparse.csr_matrix((np.ones(E), (B.dst, np.arange(E))), shape=(A, E)) @ (diff * w[:, None])
            x = x + upd / deg.astype(np.float64)[:, None]
    off = B.off[:-1]
    if cfg["readout"] == "max": out = np.maximum.reduceat(h, off, axis=0)
    else:
        out = np.add.reduceat(h, off, axis=0)
        if cfg["readout"] == "mean": out = out / np.diff(B.off)[:, None].astype(dtype)
    if not np.isfinite(out).all(): raise fc.FindingsError(f"non-finite embedding for {cfg_id(cfg, arch)} (reduce gain/depth)")
    return out.astype(np.float64)

def batches(items, radius, n_elem, max_edges=120000):
    """Yield (start, stop, Batch) with an edge budget; embeddings do not depend on how complexes are batched."""
    cur, start, ne = [], 0, 0
    for i, it in enumerate(items):
        e_est = len(it[0]) * 40
        if cur and ne + e_est > max_edges:
            yield start, i, Batch(cur, radius, n_elem); cur, start, ne = [], i, 0
        cur.append(it); ne += e_est
    if cur: yield start, len(items), Batch(cur, radius, n_elem)

def embed_all(items, cfg, arch, seed, n_elem, base_seed, dtype=np.float32):
    P = make_params(cfg, arch, seed, n_elem + 1, base_seed, dtype)
    out = np.zeros((len(items), cfg["width"]))
    for a, b, B in batches(items, cfg["radius"], n_elem):
        out[a:b] = forward(P, cfg, arch, B, dtype)
    return out
