# alignlib.py -- representation-similarity functions used by the analyses in this repository
# (debiased linear CKA, weighted cluster-bootstrap CKA, mutual kNN, Procrustes angle, PCA and residualisation helpers).
from __future__ import annotations
import numpy as np
from scipy import stats

def gram(X):
    Xc = X - X.mean(0, keepdims=True)
    return Xc @ Xc.T

def center_gram(K):
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()

def hsic_u(K, L, mask=None):
    """Unbiased HSIC (Song et al. 2012) as used for debiased CKA (Murphy et al. 2024). Diagonal is zeroed.
    The old `mask` argument is removed: zeroing duplicate pairs while keeping the n(n-3) normalisation was
    inexact. Use cka_feature_w / boot_cka_feature (exact weighted estimator) for bootstrap samples."""
    if mask is not None:
        raise ValueError("mask is no longer supported; use alignlib.cka_feature_w / boot_cka_feature")
    n = K.shape[0]
    K0, L0 = K.copy(), L.copy()
    np.fill_diagonal(K0, 0.0); np.fill_diagonal(L0, 0.0)
    sK, sL = K0.sum(1), L0.sum(1)
    t1 = (K0 * L0).sum()
    t2 = sK.sum() * sL.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 * (sK @ sL) / (n - 2)
    return float((t1 + t2 - t3) / (n * (n - 3)))

def cka_debiased(K, L):
    hkl, hkk, hll = hsic_u(K, L), hsic_u(K, K), hsic_u(L, L)
    d = hkk * hll
    return float(hkl / np.sqrt(d)) if d > 0 else float("nan")

def _hsic_u_feat(XA, XB):
    """Debiased HSIC from already-centered (N,DA)/(N,DB) matrices. O(N·DA·DB)."""
    n = XA.shape[0]
    normA2 = (XA ** 2).sum(1)
    normB2 = (XB ** 2).sum(1)
    sA = XA @ XA.sum(0) - normA2   # K0 row-sums (diagonal removed)
    sB = XB @ XB.sum(0) - normB2
    C = XA.T @ XB                   # (DA, DB)
    t1 = float((C * C).sum()) - float(normA2 @ normB2)
    t2 = float(sA.sum()) * float(sB.sum()) / ((n - 1) * (n - 2))
    t3 = 2.0 * float(sA @ sB) / (n - 2)
    return float((t1 + t2 - t3) / (n * (n - 3)))

def cka_feature(XA, XB):
    """Debiased CKA from feature matrices. Exactly equals cka_debiased(gram(XA), gram(XB))."""
    XA = (XA - XA.mean(0)).astype(np.float64)
    XB = (XB - XB.mean(0)).astype(np.float64)
    hkl = _hsic_u_feat(XA, XB)
    hkk = _hsic_u_feat(XA, XA)
    hll = _hsic_u_feat(XB, XB)
    d = hkk * hll
    return float(hkl / np.sqrt(d)) if d > 0 else float("nan")

def _hsic_u_w(XA, XB, w):
    """Exact debiased HSIC of a MULTISET: complex i appears w[i] times. Sums run over DISTINCT complexes
    (copies of the same complex never pair with each other) and every term is normalised by the matching
    weighted count of distinct pairs / triples / quadruples (so w = 1 reproduces Song et al. exactly):
        HSIC = N2/P2 + N4/P4 - 2 N3/P3
    N2 = sum_{i!=j} w_i w_j K_ij L_ij;  N3 = sum_{i,j,q distinct} w_i w_j w_q K_ij L_iq;
    N4 = sum_{i,j,q,r distinct} w_i w_j w_q w_r K_ij L_qr  (inclusion-exclusion below)."""
    n = float(w.sum()); w2 = w * w
    S2 = float(w2.sum()); S3 = float((w2 * w).sum()); S4 = float((w2 * w2).sum())
    P2 = n * n - S2
    P3 = n ** 3 - 3 * n * S2 + 2 * S3
    P4 = n ** 4 - 6 * n * n * S2 + 3 * S2 * S2 + 8 * n * S3 - 6 * S4
    if min(P2, P3, P4) <= 1e-9:          # fewer than 4 distinct complexes in the draw
        return float("nan")
    nA = (XA ** 2).sum(1); nB = (XB ** 2).sum(1)
    sA = XA @ (w @ XA) - w * nA          # sum_{j != i} w_j K_ij
    sB = XB @ (w @ XB) - w * nB
    C1 = (XA * w[:, None]).T @ XB
    C2 = (XA * w2[:, None]).T @ XB
    N2 = float((C1 * C1).sum()) - float((w2 * nA * nB).sum())
    B22 = float((C2 * C2).sum()) - float((w2 * w2 * nA * nB).sum())      # sum_{i!=j} w_i^2 w_j^2 K L
    B12 = float((C1 * C2).sum()) - float((w * w2 * nA * nB).sum())       # sum_{i!=j} w_i w_j^2 K L
    N4 = float(w @ sA) * float(w @ sB) - 4.0 * float((w2 * sA) @ sB) + 2.0 * B22
    N3 = float((w * sA) @ sB) - B12
    return float(N2 / P2 + N4 / P4 - 2.0 * N3 / P3)

def cka_feature_w(XA, XB, w):
    """Debiased CKA of the multiset with multiplicities w (len N, >=0); exact distinct-complex normalisation."""
    w = np.asarray(w, dtype=np.float64)
    XA = np.asarray(XA, dtype=np.float64); XB = np.asarray(XB, dtype=np.float64)
    n = w.sum()
    XA = XA - (w @ XA) / n
    XB = XB - (w @ XB) / n
    hkl, hkk, hll = _hsic_u_w(XA, XB, w), _hsic_u_w(XA, XA, w), _hsic_u_w(XB, XB, w)
    d = hkk * hll
    return float(hkl / np.sqrt(d)) if d > 0 else float("nan")

def fill_singleton_groups(groups):
    """Cluster labels with every unclustered complex (label < 0) turned into its own singleton cluster,
    so unclustered complexes stay in the bootstrap instead of being silently dropped."""
    g = np.asarray(groups).astype(np.int64).copy()
    neg = g < 0
    if neg.any():
        start = int(g.max()) + 1 if (~neg).any() else 0
        g[neg] = start + np.arange(int(neg.sum()))
    return g

def boot_weights(n, rng, groups=None):
    """Multiplicities for one bootstrap draw. groups=None: resample complexes; else resample clusters
    (groups must already be >=0, see fill_singleton_groups)."""
    if groups is None:
        return np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
    labs, inv = np.unique(groups, return_inverse=True)
    k = len(labs)
    counts = np.bincount(rng.integers(0, k, k), minlength=k)
    return counts[inv].astype(np.float64)

def strict_subset(E, ids, core_ids, name=""):
    """Rows of E in core_ids order. Fails loudly on length mismatch, duplicate ids or missing ids."""
    ids = list(ids)
    assert len(ids) == E.shape[0], f"{name}: {len(ids)} ids vs {E.shape[0]} rows"
    assert len(set(ids)) == len(ids), f"{name}: duplicate ids"
    rev = {p: i for i, p in enumerate(ids)}
    missing = [p for p in core_ids if p not in rev]
    assert not missing, f"{name}: {len(missing)} core ids missing (e.g. {missing[:3]})"
    return E[[rev[p] for p in core_ids]]

def pca_train_fit(E_tr, k=64, name=""):
    """Per-(model,tag) PCA fit on TRAIN embeddings only. No fallback: missing train embeddings abort."""
    if E_tr is None:
        raise RuntimeError(f"{name}: train embeddings required for train-fit PCA")
    kk = min(k, E_tr.shape[0] - 1, E_tr.shape[1])
    mu = E_tr.mean(0)
    return mu, PCA(n_components=kk, random_state=0).fit(E_tr - mu)

def _knn_idx(X, k, center=False):
    Z = X - X.mean(0, keepdims=True) if center else X
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]

def mutual_knn(X, Y, k=10, center=False):
    """Per-complex a_i = |kNN_X(i) & kNN_Y(i)| / k (cosine). Returns (a_i, mean)."""
    n = X.shape[0]
    nx, ny = _knn_idx(X, k, center), _knn_idx(Y, k, center)
    A = np.zeros((n, n), bool); A[np.arange(n)[:, None], nx] = True
    a = A[np.arange(n)[:, None], ny].sum(1) / k
    return a, float(a.mean())

def proj_basis(Z, tol=1e-8):
    """Orthonormal basis of the column space of mean-centered Z (rank-safe)."""
    Zc = Z - Z.mean(0, keepdims=True)
    U, s, _ = np.linalg.svd(Zc, full_matrices=False)
    r = int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0
    return U[:, :r]

def resid_gram(K, U):
    """K_res = P K P with P = I - U U^T (K must be the Gram of mean-centered features)."""
    KU = K @ U
    return K - U @ KU.T - KU @ U.T + U @ (U.T @ KU) @ U.T

def r2_on_basis(K, U):
    """Share of embedding variance explained by span(U), from the Gram matrix."""
    Kc = center_gram(K)
    return float(1 - np.trace(resid_gram(Kc, U)) / np.trace(Kc))

def procrustes_angular(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    nuc = np.linalg.svd(Xc.T @ Yc, compute_uv=False).sum()
    return float(np.arccos(np.clip(nuc / (np.linalg.norm(Xc) * np.linalg.norm(Yc)), -1, 1)))

class Rep:
    """z-score per dim (train stats, dead dims dropped) then PCA(k) fit on train only."""
    def __init__(self, k=64, tol=1e-8, seed=0):
        self.k, self.tol, self.seed = k, tol, seed
    def fit(self, Xtr):
        self.mu = Xtr.mean(0); sd = Xtr.std(0)
        self.keep = sd > self.tol; self.sd = sd[self.keep]; self.n_dead = int((~self.keep).sum())
        self.pca = PCA(n_components=self.k, random_state=self.seed).fit((Xtr[:, self.keep] - self.mu[self.keep]) / self.sd)
        return self
    def transform(self, X):
        return self.pca.transform((X[:, self.keep] - self.mu[self.keep]) / self.sd)

def load_cluster_labels(path, ids, col):
    """Cluster label per id from a TSV (index = complex id). The column is NAMED, never guessed: a missing column
    aborts (clusters.tsv may carry several identity levels). Ids absent from the file get -1 (-> singleton clusters)."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", index_col=0)
    assert col in df.columns, f"cluster column {col!r} not in {path} (columns: {list(df.columns)}); set CLUSTER_COL"
    lab = {str(k).lower(): int(v) for k, v in df[col].items()}
    return np.array([lab.get(str(p).lower(), -1) for p in ids], dtype=int)

def design_matrix(cov, degree=1):
    """[1, cov, cov^2, ...] with cov (n, p)."""
    cov = np.asarray(cov, float)
    return np.column_stack([np.ones(len(cov))] + [cov[:, j] ** d for d in range(1, degree + 1) for j in range(cov.shape[1])])

def ols_resid(X, Z):
    """Residual of X (n, D) after OLS on design Z (n, p) (Z must contain the intercept)."""
    return X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]

def resid_then_pca(E_tr, Z_tr, E_te, Z_te, k=64, name=""):
    """Residualize the RAW embeddings on the covariates BEFORE dimensionality reduction. Each split is residualized
    by OLS fit within that split (no cross-split information), the PCA basis is fit on the TRAIN residuals only, and
    the TEST residuals are projected on it. Returns (n_test, k)."""
    dt = np.float32 if E_tr.shape[1] > 4096 else np.float64      # GNINA: 7393 x 27648 float64 is 1.6 GB
    Rtr = ols_resid(np.asarray(E_tr, dt), np.asarray(Z_tr, dt))
    mu, pca = pca_train_fit(Rtr, k, name=name)
    Rte = ols_resid(E_te.astype(np.float64), Z_te)
    return pca.transform(Rte - mu)
