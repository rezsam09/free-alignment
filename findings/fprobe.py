"""fprobe.py -- linear / MLP probes with train-only input statistics, grouped (protein-cluster) cross-validation, weighted metrics for cluster bootstrap.
Ridge: closed form via SVD, penalty chosen by NESTED grouped CV inside each outer training fold (so the outer held-out fold never informs the penalty).
Metrics: continuous -> pooled out-of-fold R2; binary -> pooled out-of-fold AUROC (ties = 0.5). Both accept bootstrap multiplicities w."""
import numpy as np
from sklearn.model_selection import GroupKFold

def r2_w(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    ym = (w * y).sum() / w.sum(); den = (w * (y - ym) ** 2).sum()
    return float("nan") if den <= 0 else float(1.0 - (w * (y - p) ** 2).sum() / den)

def auc_prep(p):
    u, inv = np.unique(p, return_inverse=True)
    return inv, len(u)

def auc_w(y, inv, nu, w=None):
    w = np.ones(len(y)) if w is None else w
    wp = np.bincount(inv, weights=w * y, minlength=nu); wn = np.bincount(inv, weights=w * (1 - y), minlength=nu)
    Wp, Wn = wp.sum(), wn.sum()
    if Wp <= 0 or Wn <= 0: return float("nan")
    cumn = np.cumsum(wn) - wn
    return float((wp * (cumn + 0.5 * wn)).sum() / (Wp * Wn))

class Scorer:
    """metric(p, w) for one target. typ: continuous | binary."""
    def __init__(self, y, typ): self.y, self.typ = np.asarray(y, float), typ
    def prep(self, p): return auc_prep(p) if self.typ == "binary" else None
    def __call__(self, p, w=None, aux=None):
        if self.typ == "binary":
            inv, nu = aux if aux is not None else auc_prep(p)
            return auc_w(self.y, inv, nu, w)
        return r2_w(self.y, p, w)

def ridge_solver(Xtr, ytr):
    mu, ym = Xtr.mean(0), ytr.mean()
    U, s, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    Uty = U.T @ (ytr - ym)
    def predict(X, alpha): return (X - mu) @ (Vt.T @ ((s / (s ** 2 + alpha)) * Uty)) + ym
    return predict

def choose_alpha(X, y, groups, alphas, inner):
    gk = GroupKFold(n_splits=min(inner, len(np.unique(groups))))
    sse = np.zeros(len(alphas))
    for tr, va in gk.split(X, y, groups):
        f = ridge_solver(X[tr], y[tr])
        for k, a in enumerate(alphas): sse[k] += ((y[va] - f(X[va], a)) ** 2).sum()
    return alphas[int(np.argmin(sse))]

def folds_for(groups, k):
    ng = len(np.unique(groups))
    if ng < k: raise ValueError(f"only {ng} groups for {k}-fold grouped CV")
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))

def oof_ridge(X, y, groups, alphas, folds, inner, fold_list=None):
    fold_list = fold_list or folds_for(groups, folds)
    P = np.full(len(y), np.nan)
    for tr, te in fold_list:
        a = choose_alpha(X[tr], y[tr], groups[tr], alphas, inner)
        P[te] = ridge_solver(X[tr], y[tr])(X[te], a)
    assert np.isfinite(P).all()
    return P

def oof_mlp(X, y, fold_list, hidden, max_iter, alpha, seed):
    from sklearn.neural_network import MLPRegressor
    import warnings
    P = np.full(len(y), np.nan)
    for tr, te in fold_list:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8; ym, ys = y[tr].mean(), y[tr].std() + 1e-8
        m = MLPRegressor(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=max_iter, random_state=seed, early_stopping=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # ConvergenceWarning at the fixed iteration budget is expected and documented
            m.fit((X[tr] - mu) / sd, (y[tr] - ym) / ys)
        P[te] = m.predict((X[te] - mu) / sd) * ys + ym
    assert np.isfinite(P).all()
    return P

def control_target(X, typ, prevalence, k, seed):
    """Random-label control task (Hewitt & Liang style): k-means cells of the probe INPUT get i.i.d. random values; held-out complexes share cells with
    training complexes, so the score measures how well the probe can fit an arbitrary function of the representation (capacity), not information."""
    from sklearn.cluster import KMeans
    k = min(k, max(2, X.shape[0] // 20))
    cell = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(X)
    r = np.random.default_rng(seed)
    v = r.standard_normal(k) if typ == "continuous" else (r.random(k) < prevalence).astype(float)
    return v[cell]

def shuffled_null(X, y, groups, scorer, alphas, folds, inner, fold_list, n_perm, seed):
    r = np.random.default_rng(seed); out = []
    for _ in range(n_perm):
        sc = Scorer(r.permutation(y), scorer.typ)
        out.append(sc(oof_ridge(X, sc.y, groups, alphas, folds, inner, fold_list)))
    return np.array(out)
