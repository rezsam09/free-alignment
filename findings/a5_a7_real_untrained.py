"""a5_a7_real_untrained.py -- the EXISTING untrained embeddings (3 models x 5 inits), reference-model preprocessing (train-fit PCA-64, centred; CKA debiased).
 A7  matched (init i vs init i) vs mismatched (i != j) 5x5 pairings per model pair, cluster-bootstrap CI of the difference.
 A5  size-residualised floor: raw embeddings residualised on the 2 covariates (OLS within each split) BEFORE train-fit PCA-64 (alignlib.resid_then_pca); floor raw vs residualised.
 A6  GNINA: its 5 untrained inits (+ all models) vs every input-geometry kernel.
 also: within-architecture random-vs-random CKA (10 pairs per model)."""
import itertools
import numpy as np, pandas as pd
import fcommon as fc
from input_kernels import load_kernels
al = fc.al

def covariate_matrix(plan, ids):
    cov = pd.read_csv(fc.ROOT / plan["covariates_file"], sep="\t", index_col=0); cov.index = cov.index.astype(str).str.lower()
    miss = [p for p in ids if p not in cov.index]
    if miss: raise fc.FindingsError(f"{len(miss)} ids lack covariates (e.g. {miss[:3]})")
    return cov.loc[ids, ["ligand_heavy_atom_count", "complex_atom_count"]].values.astype(float)

def resid_rep(model, tag, plan, core, deg, pool=None):
    Etr, itr = fc.load_emb(model, tag, plan["train_split"]); Ete, ite = fc.load_emb(model, tag, plan["eval_split"])
    if pool: Etr, Ete = (E.reshape(len(E), pool, -1).mean(2) for E in (Etr, Ete))
    Ete = fc.pick(Ete, ite, core, f"{model}/{tag}")
    Ztr = al.design_matrix(covariate_matrix(plan, itr), deg); Zte = al.design_matrix(covariate_matrix(plan, core), deg)
    return al.resid_then_pca(Etr, Ztr, Ete, Zte, plan["pca_dim"], name=f"{model}/{tag}")

def pca_rep(model, tag, plan, core, pool=None):
    if not pool: return fc.cka_pca(model, tag, plan, core)
    Etr, _ = fc.load_emb(model, tag, plan["train_split"]); Ete, ite = fc.load_emb(model, tag, plan["eval_split"])
    Etr, Ete = (E.reshape(len(E), pool, -1).mean(2) for E in (Etr, Ete))
    mu, pca = al.pca_train_fit(Etr, plan["pca_dim"], name=f"{model}/{tag}/pooled")
    return pca.transform(fc.pick(Ete, ite, core, tag) - mu)

def pairing_stats(RA, RB, w=None):
    n = len(RA); M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = al.cka_feature(RA[i], RB[j]) if w is None else al.cka_feature_w(RA[i], RB[j], w)
    d = np.diag(M); off = M[~np.eye(n, dtype=bool)]
    return M, float(d.mean()), float(off.mean())

def main():
    plan = fc.load_plan(); fc.require_freeze("a5_a7_real_untrained")
    core = fc.core_ids(plan); ru = plan["real_untrained"]; nb = plan["n_boot"]
    tags = fc.UNTRAINED_TAGS[:ru["n_init"]]
    groups = fc.clusters(plan, core); rng = np.random.default_rng(plan["seed"] + 5)
    W = [al.boot_weights(len(core), rng, groups) for _ in range(nb)]
    R = {m: [pca_rep(m, t, plan, core) for t in tags] for m in fc.MODELS}
    Rr = {m: [resid_rep(m, t, plan, core, ru["resid_degree"]) for t in tags] for m in fc.MODELS}
    pair_rows, mat_rows, within = [], [], []
    for mA, mB in fc.PAIRS:
        M, fm_, fo = pairing_stats(R[mA], R[mB]); Mr, fmr, for_ = pairing_stats(Rr[mA], Rr[mB])
        for i in range(len(tags)):
            for j in range(len(tags)): mat_rows.append(dict(pair=f"{mA}_x_{mB}", i=i, j=j, cka_raw=M[i, j], cka_resid=Mr[i, j]))
        bm, bmm, bmr, bdr = [], [], [], []
        for w in W:
            _, a, b = pairing_stats(R[mA], R[mB], w); _, ar, _b = pairing_stats(Rr[mA], Rr[mB], w)
            bm.append(a); bmm.append(a - b); bmr.append(ar); bdr.append(ar - a)
        lo, hi = fc.ci(np.array(bm), plan["ci_level"]); dl, dh = fc.ci(np.array(bmm), plan["ci_level"])
        rl, rh = fc.ci(np.array(bmr), plan["ci_level"]); xl, xh = fc.ci(np.array(bdr), plan["ci_level"])
        pair_rows.append(dict(pair=f"{mA}_x_{mB}", matched_floor=fm_, matched_lo=lo, matched_hi=hi, mismatched=fo, matched_minus_mismatched=fm_ - fo, mm_lo=dl, mm_hi=dh,
                              floor_resid=fmr, floor_resid_lo=rl, floor_resid_hi=rh, resid_minus_raw=fmr - fm_, rr_lo=xl, rr_hi=xh))
    for m in fc.MODELS:
        v = [al.cka_feature(R[m][i], R[m][j]) for i, j in itertools.combinations(range(len(tags)), 2)]
        vr = [al.cka_feature(Rr[m][i], Rr[m][j]) for i, j in itertools.combinations(range(len(tags)), 2)]
        within.append(dict(model=m, within_arch_random_cka=np.mean(v), sd=np.std(v), within_arch_random_cka_resid=np.mean(vr)))
    fc.write_tsv(pd.DataFrame(pair_rows), "res_real_floor_pairing.tsv"); fc.write_tsv(pd.DataFrame(mat_rows), "res_real_matrix.tsv"); fc.write_tsv(pd.DataFrame(within), "res_real_within.tsv")
    # A6
    K = load_kernels(core); ker = []
    reps = {(m, t): R[m][i] for m in fc.MODELS for i, t in enumerate(tags)}
    for (m, t), X in reps.items():
        for k, Fk in K.items(): ker.append(dict(model=m, tag=t, kernel=k, cka=al.cka_feature(Fk, X)))
    fc.write_tsv(pd.DataFrame(ker), "res_real_kernel_cka.tsv")
    fc.ledger("a5_a7_real_untrained", "done", f"n={len(core)}")
    print(pd.DataFrame(pair_rows).T.to_string())

if __name__ == "__main__":
    main()
