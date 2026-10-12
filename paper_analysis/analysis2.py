"""Additional robustness analyses computed from the stored sweep results (pair level and setting level).
Writes numbers2.json and tab_robust.tex. Run: python3 -I analysis2.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

import os
HERE = Path(__file__).resolve().parent
F = Path(os.environ.get("FINDINGS_OUT", HERE.parent / "findings_out")); OUT = Path(os.environ.get("PAPER_OUT", HERE))
rd = lambda n: pd.read_csv(F / n, sep="\t")
P, C, S, SK = rd("res_sweep_pairs.tsv"), rd("res_sweep_configs.tsv"), rd("res_sweep_summary.tsv"), rd("res_sweep_kernel_cka.tsv")
ids = C.cfg_id.tolist()
N = {}

# ---------------------------------------------------------------- seed versus setting variance (cross pairs)
cp = P[P.kind == "cross"].copy()
gm = cp.cka.mean(); sst = ((cp.cka - gm) ** 2).sum()
ssb = cp.groupby("cfg_id").cka.apply(lambda s: len(s) * (s.mean() - gm) ** 2).sum()
N["eta2_setting"] = float(ssb / sst)
cr = S[S.kind == "cross"].set_index("cfg_id").loc[ids]
N["rho_mean_sd"] = float(stats.spearmanr(cr.cka_mean, cr.cka_std)[0])
N["sd_range"] = [float(cr.cka_std.min()), float(cr.cka_std.max())]
# split-half reliability of the setting ranking
m = cp[cp.matched].groupby("cfg_id").cka.mean().loc[ids]; mm = cp[~cp.matched].groupby("cfg_id").cka.mean().loc[ids]
N["split_half_spearman"] = float(stats.spearmanr(m, mm)[0]); N["split_half_pearson"] = float(np.corrcoef(m, mm)[0, 1])
N["split_half_n"] = [int(cp.matched.sum() / len(ids)), int((~cp.matched).sum() / len(ids))]
# seed-index split: pairs with (i+j) even versus odd
cp["par"] = (cp.i + cp.j) % 2
a = cp[cp.par == 0].groupby("cfg_id").cka.mean().loc[ids]; b = cp[cp.par == 1].groupby("cfg_id").cka.mean().loc[ids]
N["parity_split_spearman"] = float(stats.spearmanr(a, b)[0])

# ---------------------------------------------------------------- regressions
def ridge(Xtr, ytr, Xte, alpha):
    keep = Xtr.std(0) > 1e-12
    if keep.sum() == 0: return np.full(len(Xte), ytr.mean())
    mu, sd = Xtr[:, keep].mean(0), Xtr[:, keep].std(0)
    A, B = (Xtr[:, keep] - mu) / sd, (Xte[:, keep] - mu) / sd
    ym = ytr.mean(); w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym
r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def cv(X, y, alpha=1.0, folds=5, reps=20, seed=20261010):
    n = len(y); out = []
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(n); Pm = np.zeros(n)
        for f in range(folds):
            te = perm[f::folds]; tr = np.setdiff1d(np.arange(n), te); Pm[te] = ridge(X[tr], y[tr], X[te], alpha)
        out.append(r2(y, Pm))
    return float(np.mean(out))
cfgs = C.to_dict("records")
Xc = np.array([[np.log2(c["width"]), c["depth"], c["radius"], np.log2(c["gain"]), c["n_rbf"], c["readout"] == "sum", c["readout"] == "max", c["aggr"] == "mean"] for c in cfgs], float)
fn = ["width", "depth", "radius", "gain", "radial functions", "readout (sum)", "readout (max)", "aggregation (mean)"]
y = cr.cka_mean.values
# standardised coefficients, full fit
mu, sd = Xc.mean(0), Xc.std(0); A = (Xc - mu) / sd
w = np.linalg.solve(A.T @ A + np.eye(8), A.T @ (y - y.mean()))
N["std_coef"] = dict(zip(fn, map(float, w)))
# leave-one-factor-out
groups = {"width": [0], "depth": [1], "radius": [2], "gain": [3], "radial functions": [4], "readout": [5, 6], "aggregation": [7]}
full = cv(Xc, y); N["cv_full"] = full
N["drop_factor"] = {g: float(full - cv(np.delete(Xc, ix, 1), y)) for g, ix in groups.items()}
N["only_factor"] = {g: float(cv(Xc[:, ix], y)) for g, ix in groups.items()}
# sensitivity: drop the three width-128 settings (PCA-reduced)
k24 = np.array([c["width"] != 128 for c in cfgs]); X24, y24 = Xc[k24], y[k24]
f24 = cv(X24, y24)
N["no_w128"] = {"n": int(k24.sum()), "cv_full": f24,
                "drop_readout": float(f24 - cv(np.delete(X24, [5, 6], 1), y24)),
                "readout_alone": float(cv(X24[:, [5, 6]], y24)),
                "median_cka": float(np.median(y24)), "min": float(y24.min()), "max": float(y24.max())}
# permutation test of hyperparameter model
rng = np.random.default_rng(1); obs = cv(Xc, y, reps=1)
null = [cv(Xc, rng.permutation(y), reps=1, seed=int(rng.integers(1e6))) for _ in range(500)]
N["perm_p_M1"] = float((np.sum(np.array(null) >= obs) + 1) / 501); N["perm_obs_single_rep"] = float(obs); N["perm_null_q95"] = float(np.quantile(null, .95))

# kernel predictors (mean of both archs and seeds)
kk = SK.groupby(["cfg_id", "kernel"]).cka.mean().unstack().loc[ids][["composition", "degree", "dist_hist", "size"]].values
resp = {"CKA": cr.cka_mean.values, "$k$NN overlap": cr.knn_mean.values, "Procrustes angle": cr.procrustes_mean.values, "CKA, size-residualised": cr.cka_resid_mean.values}
rows = {}
for k, yy in resp.items():
    rows[k] = dict(config=cv(Xc, yy), kernels=cv(kk, yy), both=cv(np.hstack([kk, Xc]), yy))
N["resp_models"] = rows
# alpha sensitivity and logit response
N["alpha_sens"] = {str(a_): dict(config=cv(Xc, y, a_), kernels=cv(kk, y, a_)) for a_ in (0.1, 1.0, 10.0)}
yl = np.log(y / (1 - y))
N["logit"] = dict(config=cv(Xc, yl), kernels=cv(kk, yl), both=cv(np.hstack([kk, Xc]), yl))
kn_ = ["composition", "degree", "dist_hist", "size"]
N["logit_single"] = {k_: cv(kk[:, [j]], yl) for j, k_ in enumerate(kn_)}
N["knn_single"] = {k_: cv(kk[:, [j]], cr.knn_mean.values) for j, k_ in enumerate(kn_)}
N["raw_single"] = {k_: cv(kk[:, [j]], y) for j, k_ in enumerate(kn_)}
N["rho_knn_kernel"] = {k_: float(stats.spearmanr(kk[:, j], cr.knn_mean.values)[0]) for j, k_ in enumerate(kn_)}
# readout-excluded kernel and config fits on logit
N["logit_no_readout_config"] = cv(np.delete(Xc, [5, 6], 1), yl)
# excluding the ten random combinations (one-factor-at-a-time settings only, n=17)
base = dict(width=64, depth=3, radius=5.0, gain=1.0, readout="mean", n_rbf=16, aggr="sum")
isofat = np.array([sum(c[k] != base[k] for k in base) <= 1 for c in cfgs])
N["n_ofat"] = int(isofat.sum())
N["ofat_only"] = dict(config=cv(Xc[isofat], y[isofat]), kernels=cv(kk[isofat], y[isofat]))
N["random_only"] = dict(n=int((~isofat).sum()), config=cv(Xc[~isofat], y[~isofat]), kernels=cv(kk[~isofat], y[~isofat]))
# variance of the response among random combinations versus among one-factor settings
N["y_sd_ofat"] = float(y[isofat].std()); N["y_sd_random"] = float(y[~isofat].std())

json.dump(N, open(OUT / "numbers2.json", "w"), indent=1)
t = ["\\begin{tabular}{lccc}", "\\toprule", "Response & Hyperparameters & Four kernels & Both \\\\", "\\midrule"]
for k, v in rows.items(): t.append(f"{k} & {v['config']:.2f} & {v['kernels']:.2f} & {v['both']:.2f} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_robust.tex").write_text("\n".join(t))
t = ["\\begin{tabular}{lcc}", "\\toprule", "Factor & Held-out $R^2$ drop when removed & $R^2$ with factor alone \\\\", "\\midrule"]
for g in groups: t.append(f"{g.capitalize()} & {N['drop_factor'][g]:+.3f} & {N['only_factor'][g]:.3f} \\\\")
t += ["\\midrule", f"All eight predictors & \\multicolumn{{2}}{{c}}{{{full:.3f}}} \\\\", "\\bottomrule", "\\end{tabular}"]
(OUT / "tab_factor.tex").write_text("\n".join(t))
print(json.dumps(N, indent=1))

# ================================================================ additions: intervals for the local-overlap comparison, leverage, reference-model rank and ceilings
def oof(X, y, alpha=1.0, folds=5, reps=20, seed=20261010):
    n = len(y); acc = np.zeros(n)
    for r in range(reps):
        perm = np.random.default_rng(seed + r).permutation(n); Pm = np.zeros(n)
        for f in range(folds):
            te = perm[f::folds]; tr = np.setdiff1d(np.arange(n), te); Pm[te] = ridge(X[tr], y[tr], X[te], alpha)
        acc += Pm
    return acc / reps
def boot_diff(Xa, Xb, yy, nb=1000, seed=20261013):
    pa, pb = oof(Xa, yy), oof(Xb, yy); rng = np.random.default_rng(seed); n = len(yy); da, db, dd = [], [], []
    for _ in range(nb):
        s = rng.integers(0, n, n)
        if yy[s].std() == 0: continue
        a_, b_ = r2(yy[s], pa[s]), r2(yy[s], pb[s]); da.append(a_); db.append(b_); dd.append(b_ - a_)
    q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return dict(r2_a=r2(yy, pa), r2_b=r2(yy, pb), ci_a=q(da), ci_b=q(db), diff=r2(yy, pb) - r2(yy, pa), ci_diff=q(dd))
yk = cr.knn_mean.values
N["knn_boot"] = boot_diff(Xc, kk, yk)            # a = hyperparameters, b = kernels
N["knn_boot_both_vs_config"] = boot_diff(Xc, np.hstack([kk, Xc]), yk)
N["logit_boot"] = boot_diff(Xc, kk, yl)
# leverage of the sum-readout settings on the size kernel
szk = SK[SK.kernel == "size"].groupby("cfg_id").cka.mean().loc[ids]
rdo = np.array([c["readout"] for c in cfgs])
N["n_readout"] = {r_: int((rdo == r_).sum()) for r_ in ("mean", "sum", "max")}
N["size_kernel_cka_by_readout"] = {r_: [float(szk.values[rdo == r_].min()), float(np.median(szk.values[rdo == r_])), float(szk.values[rdo == r_].max())] for r_ in ("mean", "sum", "max")}
N["rho_size_kernel_excl_sum"] = float(stats.spearmanr(szk.values[rdo != "sum"], y[rdo != "sum"])[0])
N["logit_size_excl_sum"] = float(cv(kk[rdo != "sum"][:, [3]], yl[rdo != "sum"]))
N["max_readout_cka"] = [float(v) for v in y[rdo == "max"]]
json.dump(N, open(OUT / "numbers2.json", "w"), indent=1)

REF = Path(os.environ.get("REFERENCE_LOGS", HERE.parent / "reference_logs"))
if (REF / "h1_results.json").exists():
    h1 = json.load(open(REF / "h1_results.json")); eh = json.load(open(REF / "embedding_health_2a.json"))
    ce = {m: h1[f"{m}_ceiling"] for m in ("egnn", "distance_gnn", "gnina")}
    N["ceilings"] = {m: [c["cka"], c["ci_lo"], c["ci_hi"]] for m, c in ce.items()}
    nm = dict(egnn="EGNN", distance_gnn="Distance-GNN", gnina="GNINA-type CNN")
    t = ["\\begin{tabular}{lccc}", "\\toprule", "Model & Same-architecture CKA (two training seeds) & 95\\% CI & Mutual $k$NN \\\\", "\\midrule"]
    for m in ce: t.append(f"{nm[m]} & {ce[m]['cka']:.3f} & [{ce[m]['ci_lo']:.3f}, {ce[m]['ci_hi']:.3f}] & {ce[m]['mknn_mean']:.3f} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]; (OUT / "tab_ceiling.tex").write_text("\n".join(t))
    t = ["\\begin{tabular}{llccc}", "\\toprule", "Model & Weights & Participation ratio & Top-component variance share & Variance explained by size \\\\", "\\midrule"]
    for k, v in eh.items():
        m, w, _ = k.split("/")
        t.append(f"{nm[m]} & {w} & {v['participation_ratio']:.2f} & {v['top1_var_frac']:.2f} & {v['frac_var_explained_by_covariates']:.2f} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]; (OUT / "tab_rank.tex").write_text("\n".join(t))
    json.dump(N, open(OUT / "numbers2.json", "w"), indent=1)
for k in ("knn_boot", "knn_boot_both_vs_config", "logit_boot", "n_readout", "size_kernel_cka_by_readout", "rho_size_kernel_excl_sum", "logit_size_excl_sum", "max_readout_cka", "ceilings"):
    print(k, json.dumps(N.get(k)))

# ---------------------------------------------------------------- trained reference pairs over the four seed combinations
if (REF / "h1_results.json").exists():
    h1 = json.load(open(REF / "h1_results.json")); ms = h1["multi_seed_h1"]
    names = {"egnn_x_distance_gnn": "EGNN $\\times$ Distance-GNN", "egnn_x_gnina": "EGNN $\\times$ GNINA-type CNN", "gnina_x_distance_gnn": "GNINA-type CNN $\\times$ Distance-GNN"}
    t = ["\\begin{tabular}{lccccc}", "\\toprule", "Pair & P$\\times$P (95\\% CI) & P$\\times$A & A$\\times$P & A$\\times$A & Untrained baseline \\\\", "\\midrule"]
    seeds = {}
    for k, nm_ in names.items():
        p = h1[k]; vals = [ms[c][k]["cka"] for c in ("primary_anchor", "anchor_primary", "anchor_anchor")]
        t.append(f"{nm_} & {p['cka']:.3f} [{p['ci_lo']:.3f}, {p['ci_hi']:.3f}] & " + " & ".join(f"{v:.3f}" for v in vals) + f" & {p['floor']:.3f} \\\\")
        seeds[k] = [p["cka"]] + vals
    t += ["\\bottomrule", "\\end{tabular}"]; (OUT / "tab_trained.tex").write_text("\n".join(t))
    N["trained_seed_combinations"] = seeds
    json.dump(N, open(OUT / "numbers2.json", "w"), indent=1)
    print("trained seeds", json.dumps(seeds))
