"""input_kernels.py (A2) -- label-free input-geometry feature maps for every parsed complex (train + test).
size (2 covariates: ligand heavy atoms, complex atoms), composition (element fractions: whole complex + ligand),
dist_hist (normalised histogram of all intra-complex pair distances), degree (normalised histogram of per-atom neighbour counts).
All maps are z-scored per feature with TRAIN-split statistics (features constant in train are dropped). Linear CKA on a map = kernel CKA.
Output: findings_out/kernels.npz.  Also checks the size map against data/covariates.tsv (rank correlation > 0.99, else abort)."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
import fcommon as fc
from build_graphs import Graphs

def raw_features(g: Graphs, plan, pid):
    xyz, el, lig = g.get(pid)
    n = len(el); ne = g.n_elem; kp = plan["kernels"]
    size = [float(lig.sum()), float(n)]
    comp = np.concatenate([np.bincount(el, minlength=ne) / n, np.bincount(el[lig], minlength=ne) / max(lig.sum(), 1)])
    tree = cKDTree(xyz)
    dh = kp["distance_hist"]
    pairs = tree.sparse_distance_matrix(tree, dh["max_dist"], output_type="ndarray")
    d = pairs["v"][pairs["i"] < pairs["j"]]
    hist = np.histogram(d, bins=dh["n_bins"], range=(0, dh["max_dist"]))[0].astype(float)
    hist = hist / max(hist.sum(), 1)
    dg = kp["degree"]
    cnt = np.array([len(x) - 1 for x in tree.query_ball_point(xyz, dg["radius"])])
    deg = np.histogram(np.minimum(cnt, dg["max_degree"]), bins=dg["n_bins"], range=(0, dg["max_degree"] + 1e-9))[0].astype(float)
    deg = deg / n
    return size, comp, hist, deg

def main():
    plan = fc.load_plan(); fc.require_lock()
    g = Graphs()
    F = {k: [] for k in ("size", "composition", "dist_hist", "degree")}
    for pid in g.ids:
        for k, v in zip(F, raw_features(g, plan, pid)): F[k].append(v)
    F = {k: np.array(v, float) for k, v in F.items()}
    tr = np.array([s == plan["train_split"] for s in g.split])
    if tr.sum() < 10: raise fc.FindingsError("too few train graphs for train-fit standardisation")
    out = {}
    for k, X in F.items():
        mu, sd = X[tr].mean(0), X[tr].std(0); keep = sd > 1e-9
        out[k] = (X[:, keep] - mu[keep]) / sd[keep]
    cov = pd.read_csv(fc.ROOT / plan["covariates_file"], sep="\t", index_col=0); cov.index = cov.index.astype(str).str.lower()
    common = [p for p in g.ids if p in cov.index]
    if len(common) < 0.9 * len(g.ids): raise fc.FindingsError("covariates.tsv covers < 90 % of parsed complexes")
    ii = [g.index[p] for p in common]
    for j, col in enumerate(("ligand_heavy_atom_count", "complex_atom_count")):
        r = stats.spearmanr(F["size"][ii, j], cov.loc[common, col].values)[0]
        if not r > 0.99: raise fc.FindingsError(f"graph-derived {col} disagrees with covariates.tsv (Spearman {r:.3f})")
    np.savez_compressed(fc.OUT / "kernels.npz", ids=np.array(g.ids), split=g.split, **out)
    fc.ledger("input_kernels", "kernels", " ".join(f"{k}:{v.shape[1]}d" for k, v in out.items()), "planned")
    print({k: v.shape for k, v in out.items()})

def load_kernels(ids):
    z = np.load(fc.OUT / "kernels.npz")
    idx = {p: i for i, p in enumerate(z["ids"].tolist())}
    miss = [p for p in ids if p not in idx]
    if miss: raise fc.FindingsError(f"{len(miss)} ids missing from kernels.npz (e.g. {miss[:3]})")
    rows = [idx[p] for p in ids]
    return {k: z[k][rows] for k in ("size", "composition", "dist_hist", "degree")}

if __name__ == "__main__":
    main()
