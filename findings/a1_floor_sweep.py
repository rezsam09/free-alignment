"""a1_floor_sweep.py (A1 + A2 + A5-on-sweep) -- random-network grid.
For every config x {egnn, distance_gnn} x seed: embed the sweep complexes (cached, resumable), then record for
  within-egnn pairs, within-distance_gnn pairs and all 25 cross pairs (matched = same seed index):
  debiased CKA, mutual kNN (centered cosine), Procrustes angle, and CKA after size-residualisation (A5).
Also CKA(kernel, network) for every input-geometry kernel (A2) and CKA(kernel, kernel).
Usage:  python findings/a1_floor_sweep.py --estimate     (times one network, projects total cost; needs lock only)
        python findings/a1_floor_sweep.py                (needs lock + freeze)"""
import sys, time, itertools
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import fcommon as fc, randnets as rn, sweeplib as sl
from build_graphs import Graphs
from input_kernels import load_kernels

def setup(plan):
    core = fc.core_ids(plan); ids = sl.sweep_ids(plan, core)
    g = Graphs(); rows = g.rows(ids); items = sl.load_items(g, ids)
    K = load_kernels(ids)
    return ids, g, items, K

def main():
    plan = fc.load_plan()
    est = "--estimate" in sys.argv
    if est: fc.require_lock()
    else: fc.require_freeze("a1_floor_sweep")
    ids, g, items, K = setup(plan)
    cfgs = sl.make_configs(plan); sw = plan["sweep"]; S = sw["n_seeds"]
    n_elem = g.n_elem
    if est:
        t = time.time(); rn.embed_all(items, cfgs[0], "egnn", 0, n_elem, plan["seed"], np.dtype(sw["dtype"]).type); dt = time.time() - t
        jobs = len({(a, rn.cfg_id(c, a), s) for c in cfgs for a in rn.ARCHS for s in range(S)})
        print(f"{len(items)} complexes, {len(cfgs)} configs, {jobs} unique networks; base net {dt:.1f}s -> ~{jobs * dt / 3600 / max(sw['n_jobs'], 1):.1f} h on {sw['n_jobs']} workers (deeper/wider configs cost more)")
        return
    fc.write_tsv(pd.DataFrame([{**c, "cfg_id": rn.cfg_id(c)} for c in cfgs]), "res_sweep_configs.tsv")
    jobs, seen = [], set()
    for c in cfgs:
        for a in rn.ARCHS:
            for s in range(S):
                key = (a, rn.cfg_id(c, a), s)
                if key not in seen: seen.add(key); jobs.append((c, a, s))
    Parallel(n_jobs=sw["n_jobs"], verbose=5)(delayed(sl.get_embedding)(plan, c, a, s, items, n_elem) for c, a, s in jobs)
    kn = {k: fc.GramStats(v) for k, v in K.items()}
    names = list(kn)
    kk = [dict(k1=a, k2=b, cka=kn[a].cka(kn[b])) for a, b in itertools.combinations(names, 2)]
    fc.write_tsv(pd.DataFrame(kk), "res_kernel_kernel.tsv")
    Z = K["size"]; kp, kn_ = sw["pca_k"], sw["knn_k"]
    pair_rows, ker_rows = [], []
    for c in cfgs:
        cid = rn.cfg_id(c)
        R = {a: [sl.Rep(sl.get_embedding(plan, c, a, s, items, n_elem), kp, kn_, Z, sw["resid_degree"]) for s in range(S)] for a in rn.ARCHS}
        for a in rn.ARCHS:
            for s in range(S):
                for kname, G in kn.items():
                    ker_rows.append(dict(cfg_id=cid, arch=a, seed=s, kernel=kname, cka=G.cka(R[a][s].G)))
            for i, j in itertools.combinations(range(S), 2):
                pair_rows.append(dict(cfg_id=cid, kind=f"within_{a}", i=i, j=j, matched=False, **sl.pair_metrics(R[a][i], R[a][j], kn_)))
        for i in range(S):
            for j in range(S):
                pair_rows.append(dict(cfg_id=cid, kind="cross", i=i, j=j, matched=(i == j), **sl.pair_metrics(R["egnn"][i], R["distance_gnn"][j], kn_)))
    P = pd.DataFrame(pair_rows)
    if not np.isfinite(P["cka"]).all(): fc.ledger("a1", "warn", f"{(~np.isfinite(P['cka'])).sum()} non-finite CKA pairs (degenerate embeddings)")
    fc.write_tsv(P, "res_sweep_pairs.tsv"); fc.write_tsv(pd.DataFrame(ker_rows), "res_sweep_kernel_cka.tsv")
    fc.write_tsv(pd.DataFrame({"id": ids}), "res_sweep_ids.tsv")
    summ = P.groupby(["cfg_id", "kind"])[["cka", "knn", "procrustes", "cka_resid"]].agg(["mean", "std"]).reset_index()
    summ.columns = ["_".join(x).strip("_") for x in summ.columns]
    fc.write_tsv(summ, "res_sweep_summary.tsv")
    fc.ledger("a1_floor_sweep", "done", f"{len(cfgs)} configs, {len(jobs)} networks, n={len(ids)}")
    print(summ[summ.kind == "cross"][["cfg_id", "cka_mean", "cka_resid_mean"]].head(10).to_string())

if __name__ == "__main__":
    main()
