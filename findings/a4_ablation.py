"""a4_ablation.py (A4) -- which input ablation destroys the random-network floor?
Conditions per config: none | shuffle_types (element labels permuted among all atoms of a complex, coordinates kept, ligand flag stays with the position)
| shuffle_types_within_role (permuted within ligand and within pocket separately) | random_coords (coordinates replaced by an isotropic Gaussian cloud with the same
RMS radius and centroid; types kept; radius graph rebuilt).  The SAME ablated inputs feed every network of a condition.
Outputs: res_ablation_pairs.tsv (all pairs), res_ablation_boot.tsv (cluster-bootstrap CI of the mean MATCHED cross-architecture CKA per condition and of
its change vs 'none', one set of cluster multiplicities per draw shared by all conditions = paired)."""
import numpy as np, pandas as pd, itertools
import fcommon as fc, randnets as rn, sweeplib as sl
from build_graphs import Graphs
from input_kernels import load_kernels
al = fc.al

def ablate(items, ids, kind, seed):
    out = []
    for (xyz, el, lig), pid in zip(items, ids):
        r = np.random.default_rng(fc.stable_int(seed, "abl", kind, pid))
        if kind == "none": out.append((xyz, el, lig))
        elif kind == "shuffle_types": out.append((xyz, el[r.permutation(len(el))], lig))
        elif kind == "shuffle_types_within_role":
            e2 = el.copy()
            for m in (lig, ~lig):
                ix = np.where(m)[0]; e2[ix] = el[ix[r.permutation(len(ix))]]
            out.append((xyz, e2, lig))
        elif kind == "random_coords":
            c = xyz.mean(0); rms = np.sqrt(((xyz - c) ** 2).sum(1).mean())
            out.append((c + r.standard_normal(xyz.shape) * rms / np.sqrt(3.0), el, lig))
        else: raise ValueError(kind)
    return out

def main():
    plan = fc.load_plan(); fc.require_freeze("a4_ablation")
    core = fc.core_ids(plan); ids = sl.sweep_ids(plan, core)
    g = Graphs(); g.rows(ids); items = sl.load_items(g, ids); K = load_kernels(ids)
    cfgs = {rn.cfg_id(c): c for c in sl.make_configs(plan)}
    want = [rn.cfg_id(dict(plan["sweep"]["base"])) if x == "base" else x for x in plan["ablation"]["configs"]]
    miss = [x for x in want if x not in cfgs]
    if miss: raise fc.FindingsError(f"ablation configs not in the grid: {miss}")
    sw = plan["sweep"]; S = sw["n_seeds"]; kinds = ["none"] + plan["ablation"]["kinds"]
    groups = fc.clusters(plan, ids); rng = np.random.default_rng(plan["seed"] + 4)
    W = [al.boot_weights(len(ids), rng, groups) for _ in range(plan["ablation"]["n_boot"])]
    prow, brow = [], []
    for cid in want:
        c = cfgs[cid]; X = {}
        for kind in kinds:
            it = ablate(items, ids, kind, plan["seed"])
            R = {a: [] for a in rn.ARCHS}
            for a in rn.ARCHS:
                for s in range(S):
                    E = rn.embed_all(it, c, a, s, g.n_elem, plan["seed"], np.dtype(sw["dtype"]).type)
                    R[a].append(sl.Rep(E, sw["pca_k"], sw["knn_k"], K["size"], sw["resid_degree"]))
            X[kind] = R
            for a in rn.ARCHS:
                for i, j in itertools.combinations(range(S), 2):
                    prow.append(dict(cfg_id=cid, condition=kind, kind=f"within_{a}", i=i, j=j, matched=False, **sl.pair_metrics(R[a][i], R[a][j], sw["knn_k"])))
            for i in range(S):
                for j in range(S):
                    prow.append(dict(cfg_id=cid, condition=kind, kind="cross", i=i, j=j, matched=(i == j), **sl.pair_metrics(R["egnn"][i], R["distance_gnn"][j], sw["knn_k"])))
        boot = {k: np.array([np.nanmean([al.cka_feature_w(X[k]["egnn"][i].X, X[k]["distance_gnn"][i].X, w) for i in range(S)]) for w in W]) for k in kinds}
        for k in kinds:
            obs = np.mean([X[k]["egnn"][i].G.cka(X[k]["distance_gnn"][i].G) for i in range(S)])
            lo, hi = fc.ci(boot[k], plan["ci_level"]); d = boot[k] - boot["none"]
            dlo, dhi = fc.ci(d, plan["ci_level"]) if k != "none" else (np.nan, np.nan)
            brow.append(dict(cfg_id=cid, condition=k, matched_cross_cka=obs, ci_lo=lo, ci_hi=hi,
                             delta_vs_none=obs - np.mean([X["none"]["egnn"][i].G.cka(X["none"]["distance_gnn"][i].G) for i in range(S)]) if k != "none" else 0.0,
                             delta_ci_lo=dlo, delta_ci_hi=dhi))
    fc.write_tsv(pd.DataFrame(prow), "res_ablation_pairs.tsv"); fc.write_tsv(pd.DataFrame(brow), "res_ablation_boot.tsv")
    fc.ledger("a4_ablation", "done", f"{len(want)} configs, kinds={kinds}")
    print(pd.DataFrame(brow).to_string())

if __name__ == "__main__":
    main()
