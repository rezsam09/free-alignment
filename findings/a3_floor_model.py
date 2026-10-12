"""a3_floor_model.py (A3) -- is the random-network floor predicted by input-kernel CKAs?  One row per sweep config.
Response (per plan): cross = mean CKA over the S x S cross-architecture pairs; within_egnn / within_distance_gnn = mean over seed pairs.
Predictors: kernel = mean over the relevant arch(s) & seeds of CKA(kernel map, network); config = log2 width, depth, radius, log2 gain, n_rbf, readout dummies, aggr dummy.
Models (fixed in the plan): M0 intercept | M1 config | M2 kernels | M3 kernels+config | one model per single kernel.  Ridge on standardised predictors with a FIXED alpha.
Validation holds out whole configurations: repeated K-fold over configs (+ leave-one-factor-level-out = extrapolation).  Reported: pooled held-out R2, the PRE-SPECIFIED
verdicts  kernels_predict_floor (R2(M2) >= success_r2)  and  kernels_add_beyond_config (R2(M3)-R2(M1) >= increment_r2), config-bootstrap CIs."""
import numpy as np, pandas as pd
import fcommon as fc, randnets as rn, sweeplib as sl
al = fc.al

def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    keep = Xtr.std(0) > 1e-12
    if keep.sum() == 0: return np.full(len(Xte), ytr.mean())
    mu, sd = Xtr[:, keep].mean(0), Xtr[:, keep].std(0)
    A, B = (Xtr[:, keep] - mu) / sd, (Xte[:, keep] - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym

def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def cv_predict(X, y, alpha, folds, repeats, seed):
    n = len(y); P = np.zeros((repeats, n))
    for r in range(repeats):
        perm = np.random.default_rng(seed + r).permutation(n)
        for f in range(folds):
            te = perm[f::folds]; tr = np.setdiff1d(np.arange(n), te)
            P[r, te] = ridge_fit_predict(X[tr], y[tr], X[te], alpha)
    return P

def lofo_predict(X, y, cfgs, factors, base, alpha):
    """Leave-one-factor-level-out: hold out every config whose factor f equals a non-base level v. Returns (idx, pred) pooled."""
    idx, pred = [], []
    for f in factors:
        for v in sorted({c[f] for c in cfgs}, key=str):
            if v == base[f]: continue
            te = np.array([i for i, c in enumerate(cfgs) if c[f] == v]); tr = np.setdiff1d(np.arange(len(y)), te)
            if len(tr) < 8: continue
            idx += list(te); pred += list(ridge_fit_predict(X[tr], y[tr], X[te], alpha))
    return np.array(idx), np.array(pred)

def config_features(cfgs):
    return np.array([[np.log2(c["width"]), c["depth"], c["radius"], np.log2(c["gain"]), c["n_rbf"], c["readout"] == "sum", c["readout"] == "max", c["aggr"] == "mean"] for c in cfgs], float)

def main():
    plan = fc.load_plan(); fc.require_freeze("a3_floor_model")
    fm = plan["floor_model"]
    C = pd.read_csv(fc.OUT / "res_sweep_configs.tsv", sep="\t"); P = pd.read_csv(fc.OUT / "res_sweep_pairs.tsv", sep="\t"); KC = pd.read_csv(fc.OUT / "res_sweep_kernel_cka.tsv", sep="\t")
    if len(C) < fm["min_configs"]: raise fc.FindingsError(f"only {len(C)} configs (< {fm['min_configs']}): enlarge sweep.n_random before freezing")
    cfgs = C[list(rn.CFG_KEYS)].to_dict("records"); ids = C["cfg_id"].tolist()
    base = plan["sweep"]["base"]; factors = list(plan["sweep"]["vary"])
    knames = sorted(KC.kernel.unique())
    Xc = config_features(cfgs)
    rows, verdicts = [], []
    for resp in fm["responses"]:
        sub = P[P.kind == resp]
        y = sub.groupby("cfg_id").cka.mean().reindex(ids).values
        if not np.isfinite(y).all(): raise fc.FindingsError(f"non-finite response for {resp} (degenerate configs)")
        archs = rn.ARCHS if resp == "cross" else (resp.replace("within_", ""),)
        kk = KC[KC.arch.isin(archs)].groupby(["cfg_id", "kernel"]).cka.mean().unstack().reindex(ids)[knames]
        if not np.isfinite(kk.values).all(): raise fc.FindingsError("non-finite kernel CKA")
        Xk = kk.values
        models = {"M0_intercept": np.zeros((len(y), 1)), "M1_config": Xc, "M2_kernels": Xk, "M3_kernels_config": np.hstack([Xk, Xc])}
        for j, k in enumerate(knames): models[f"K_{k}"] = Xk[:, [j]]
        oof, res = {}, {}
        for name, X in models.items():
            Pm = cv_predict(X, y, fm["alpha"], fm["cv_folds"], fm["cv_repeats"], plan["seed"])
            oof[name] = Pm.mean(0)
            rr = np.array([r2(y, p) for p in Pm])
            ii, pp = lofo_predict(X, y, cfgs, factors, base, fm["alpha"])
            rows.append(dict(response=resp, model=name, r2_cv_mean=rr.mean(), r2_cv_sd=rr.std(), r2_of_mean_pred=r2(y, oof[name]),
                             r2_lofo=r2(y[ii], pp) if len(ii) > 3 else np.nan, n_lofo=len(ii)))
        rng = np.random.default_rng(plan["seed"] + 3); n = len(y)
        b2, bd = [], []
        for _ in range(fm["n_boot"]):
            s = rng.integers(0, n, n)
            if y[s].std() == 0: continue
            b2.append(r2(y[s], oof["M2_kernels"][s])); bd.append(r2(y[s], oof["M3_kernels_config"][s]) - r2(y[s], oof["M1_config"][s]))
        lo2, hi2 = fc.ci(np.array(b2), plan["ci_level"]); lod, hid = fc.ci(np.array(bd), plan["ci_level"])
        R = {r["model"]: r for r in rows if r["response"] == resp}
        verdicts.append(dict(response=resp, n_configs=n, y_mean=y.mean(), y_sd=y.std(),
                             r2_M2=R["M2_kernels"]["r2_cv_mean"], r2_M2_ci_lo=lo2, r2_M2_ci_hi=hi2,
                             kernels_predict_floor=bool(R["M2_kernels"]["r2_cv_mean"] >= fm["success_r2"]),
                             r2_M3_minus_M1=R["M3_kernels_config"]["r2_cv_mean"] - R["M1_config"]["r2_cv_mean"], inc_ci_lo=lod, inc_ci_hi=hid,
                             kernels_add_beyond_config=bool(R["M3_kernels_config"]["r2_cv_mean"] - R["M1_config"]["r2_cv_mean"] >= fm["increment_r2"]),
                             best_single_kernel=max([k for k in knames], key=lambda k: R[f"K_{k}"]["r2_cv_mean"])))
        sp = pd.DataFrame({k: [pd.Series(Xk[:, j]).corr(pd.Series(y), method="spearman")] for j, k in enumerate(knames)})
        sp.insert(0, "response", resp); rows_sp = sp
        fc.write_tsv(rows_sp, f"res_floor_spearman_{resp}.tsv")
    fc.write_tsv(pd.DataFrame(rows), "res_floor_model_r2.tsv"); fc.write_tsv(pd.DataFrame(verdicts), "res_floor_model_verdict.tsv")
    fc.ledger("a3_floor_model", "done", f"{len(C)} configs")
    print(pd.DataFrame(verdicts).T.to_string())

if __name__ == "__main__":
    main()
