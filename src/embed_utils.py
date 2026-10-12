"""embed_utils.py -- model-agnostic embedding-extraction helpers + reference EGNN / distance GNN.

The 'embedding' of a complex is defined here as the INPUT to a named readout layer
(i.e. the pooled complex-level vector just before the prediction head).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------ extraction
class PreHeadCapture:
    """Context manager that stores the tensor fed INTO the submodule called `layer_name`."""

    def __init__(self, model: nn.Module, layer_name: str):
        mods = dict(model.named_modules())
        if layer_name not in mods:
            raise KeyError(f"{layer_name!r} not in model. Available Linear layers: "
                           f"{[n for n, m in mods.items() if isinstance(m, nn.Linear)]}")
        self.layer_name = layer_name
        self.value = None
        self._h = mods[layer_name].register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.value = args[0].detach()

    def close(self):
        self._h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@torch.no_grad()
def trace_linear_layers(model: nn.Module, *args, **kwargs):
    """Run one forward pass and list every nn.Linear in execution order with its input shape."""
    order, hooks = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            def make(n):
                def hook(mod, inp, out):
                    order.append((n, mod.in_features, mod.out_features, tuple(inp[0].shape)))
                return hook
            hooks.append(m.register_forward_hook(make(name)))
    was_training = model.training
    model.eval()
    try:
        out = model(*args, **kwargs)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return order, out


@torch.no_grad()
def find_shared_input_heads(model: nn.Module, *args, **kwargs):
    """Group nn.Linear layers that received the *identical* input tensor in one forward pass.
    Parallel heads (e.g. a pose head and an affinity head) share the pooled feature vector: that vector
    is the natural embedding. Returns list of (input_shape, [layer names])."""
    seen, hooks = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            def make(n):
                def hook(mod, inp, out):
                    seen.append((n, inp[0].detach().clone()))
                return hook
            hooks.append(m.register_forward_hook(make(name)))
    was_training = model.training
    model.eval()
    try:
        model(*args, **kwargs)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    groups = []
    for n, t in seen:
        for g in groups:
            if g["t"].shape == t.shape and torch.equal(g["t"], t):
                g["names"].append(n)
                break
        else:
            groups.append(dict(t=t, names=[n]))
    return [(tuple(g["t"].shape), g["names"]) for g in groups if len(g["names"]) > 1]


# ------------------------------------------------------------------ sanity report (numpy only)
def embedding_sanity(E, covariates=None) -> dict:
    """Quick health check of an (N, D) embedding matrix. covariates: optional (N, k) array."""
    E = np.asarray(E, dtype=np.float64)
    N, D = E.shape
    out = dict(N=N, D=D, n_nonfinite=int((~np.isfinite(E)).sum()))
    out["dead_dim_frac"] = float((E.std(0) < 1e-8).mean())
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    ev = S ** 2
    tot = ev.sum()
    out["participation_ratio"] = float(tot ** 2 / (ev ** 2).sum()) if tot > 0 else 0.0
    out["top1_var_frac"] = float(ev[0] / tot) if tot > 0 else float("nan")
    out["top5_var_frac"] = float(ev[:5].sum() / tot) if tot > 0 else float("nan")
    if covariates is not None:
        Z = np.asarray(covariates, dtype=np.float64)
        Z = (Z - Z.mean(0)) / Z.std(0)
        Z = np.c_[np.ones(N), Z]
        beta, *_ = np.linalg.lstsq(Z, Ec, rcond=None)
        resid = Ec - Z @ beta
        out["frac_var_explained_by_covariates"] = float(1 - (resid ** 2).sum() / (Ec ** 2).sum())
        k = min(5, len(S))
        pcs = U[:, :k] * S[:k]
        b2, *_ = np.linalg.lstsq(Z, pcs, rcond=None)
        r = pcs - Z @ b2
        out["pc_R2_on_covariates"] = [float(1 - (r[:, i] ** 2).sum() / (pcs[:, i] ** 2).sum()) for i in range(k)]
    return out


def orientation_sensitivity(E_rot) -> dict:
    """E_rot: (N complexes, R orientations, D). Compares variation of one complex across orientations
    (within) with variation across complexes (between). The larger within/between is, the more the embedding
    depends on grid orientation. There is no established cutoff: pick one, and lock it in the config, before looking."""
    E = np.asarray(E_rot, dtype=np.float64)
    within = E.var(axis=1, ddof=1).sum(axis=1).mean()
    between = E.mean(axis=1).var(axis=0, ddof=1).sum()
    return dict(within=float(within), between=float(between), within_over_between=float(within / between))


# ------------------------------------------------------------------ geometry helpers
def random_rotation(seed: int | None = None, dtype=torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    A = torch.randn(3, 3, generator=g, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def radius_edges(x: torch.Tensor, batch: torch.Tensor, cutoff: float = 5.0) -> torch.Tensor:
    d = torch.cdist(x, x)
    same = batch[:, None] == batch[None, :]
    eye = torch.eye(len(x), dtype=torch.bool, device=x.device)
    return ((d < cutoff) & same & ~eye).nonzero().t().contiguous()  # (2, E): row=i receives from col=j


def mean_pool(h: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    s = h.new_zeros(n_graphs, h.size(1)).index_add_(0, batch, h)
    c = h.new_zeros(n_graphs, 1).index_add_(0, batch, h.new_ones(h.size(0), 1))
    return s / c.clamp(min=1)


# ------------------------------------------------------------------ REFERENCE models (replace with yours)
class EGNNLayer(nn.Module):
    """E(n)-equivariant layer in the style of Satorras et al. (2021), sum-aggregated messages, mean coordinate update."""

    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False))
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, h, x, edge_index):
        row, col = edge_index
        rel = x[row] - x[col]
        d2 = (rel ** 2).sum(-1, keepdim=True)
        m = self.edge_mlp(torch.cat([h[row], h[col], d2], dim=-1))
        w = self.coord_mlp(m)
        dx = torch.zeros_like(x).index_add_(0, row, rel * w)
        cnt = x.new_zeros(x.size(0), 1).index_add_(0, row, torch.ones_like(w))
        x = x + dx / cnt.clamp(min=1)
        agg = torch.zeros_like(h).index_add_(0, row, m)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h, x


class EGNNAffinity(nn.Module):
    def __init__(self, n_types: int = 16, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(n_types, hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, z, x, edge_index, batch, n_graphs):
        h = self.embed(z)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return self.readout(mean_pool(h, batch, n_graphs)).squeeze(-1)


class DistanceGNN(nn.Module):
    """Message passing that sees only interatomic distances (Gaussian RBF) and atom types."""

    def __init__(self, n_types: int = 16, hidden: int = 64, n_layers: int = 3, n_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.embed = nn.Embedding(n_types, hidden)
        self.register_buffer("centers", torch.linspace(0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(hidden + n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(n_layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, z, x, edge_index, batch, n_graphs):
        row, col = edge_index
        d = (x[row] - x[col]).norm(dim=-1, keepdim=True)
        rbf = torch.exp(-self.gamma * (d - self.centers) ** 2)
        h = self.embed(z)
        for msg, upd in zip(self.msg, self.upd):
            m = msg(torch.cat([h[col], rbf], dim=-1))
            agg = torch.zeros_like(h).index_add_(0, row, m)
            h = h + upd(torch.cat([h, agg], dim=-1))
        return self.readout(mean_pool(h, batch, n_graphs)).squeeze(-1)


# ------------------------------------------------------------------ diagnostic-only alignment metric
def mutual_knn_alignment(X, Y, k: int = 10) -> float:
    """Mutual k-NN alignment (Huh et al.-style): fraction of each point's k nearest neighbours in X
    that are also among its k nearest neighbours in Y, averaged over points and symmetrized.
    Companion to CKA, never a substitute: linear CKA is known to sometimes disagree with other
    alignment metrics, so this is a second, differently-biased lens. Diagnostic only -- never
    confirmatory, never correction-adjusted (see diagnostics.mutual_knn in the config)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    k = min(k, n - 1)

    def knn_sets(Z):
        d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        idx = np.argsort(d, axis=1)[:, :k]
        return [set(row.tolist()) for row in idx]

    Nx, Ny = knn_sets(X), knn_sets(Y)
    overlaps = [len(Nx[i] & Ny[i]) / k for i in range(n)]
    return float(np.mean(overlaps))
