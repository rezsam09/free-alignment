"""Recompute every number, table and figure of the paper from the raw result TSV/JSON files.
Run:  python3 -I analysis.py   (writes numbers.json, tab_*.tex, fig_*.pdf in this directory)"""
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = Path(__file__).resolve().parent
F = Path(os.environ.get("FINDINGS_OUT", HERE.parent / "findings_out"))
LOGS = Path(os.environ.get("REFERENCE_LOGS", HERE.parent / "reference_logs"))  # JSON logs of the trained reference models (see README)
OUT = Path(os.environ.get("PAPER_OUT", HERE / "generated")); OUT.mkdir(exist_ok=True)
rd = lambda n: pd.read_csv(F / n, sep="\t")
N = {}

# ---------------------------------------------------------------- sweep
S, C, P = rd("res_sweep_summary.tsv"), rd("res_sweep_configs.tsv"), rd("res_sweep_pairs.tsv")
cr = S[S.kind == "cross"].set_index("cfg_id")
W = S.pivot(index="cfg_id", columns="kind", values="cka_mean")
wavg = (W.within_egnn + W.within_distance_gnn) / 2
TR_CKA = json.load(open(LOGS / "h1_results.json"))
tr = TR_CKA["egnn_x_distance_gnn"]
N["trained_cka"] = tr["cka"]; N["trained_ci"] = [tr["ci_lo"], tr["ci_hi"]]
N["real_floor"] = tr["floor"]; N["real_floor_vals"] = tr["floor_vals"]
N["ceil_egnn"] = TR_CKA["egnn_ceiling"]["cka"]; N["ceil_dgnn"] = TR_CKA["distance_gnn_ceiling"]["cka"]
N["trained_gnina_egnn"] = [TR_CKA["egnn_x_gnina"][k] for k in ("cka", "ci_lo", "ci_hi", "floor")]
N["trained_gnina_dgnn"] = [TR_CKA["gnina_x_distance_gnn"][k] for k in ("cka", "ci_lo", "ci_hi", "floor")]
y = cr.cka_mean
N["sweep"] = dict(n_cfg=len(cr))
N["sweep"].update(median=y.median(), q1=y.quantile(.25), q3=y.quantile(.75), mn=y.min(), mx=y.max(), mean=y.mean(),
                  n_gt08=int((y > .8).sum()), n_gt09=int((y > .9).sum()), n_gt05=int((y > .5).sum()), n_gt_trained=int((y > tr["cka"]).sum()),
                  base=cr.loc["w64_d3_r5_g1_mean_k16_sum", "cka_mean"], base_sd=cr.loc["w64_d3_r5_g1_mean_k16_sum", "cka_std"],
                  knn_min=cr.knn_mean.min(), knn_max=cr.knn_mean.max(), knn_med=cr.knn_mean.median(),
                  rho_proc=stats.spearmanr(y, cr.procrustes_mean)[0], rho_knn=stats.spearmanr(y, cr.knn_mean)[0],
                  within_cross_signed=(y - wavg).mean(), within_cross_abs=(y - wavg).abs().mean(), within_cross_r=np.corrcoef(y, wavg)[0, 1],
                  within_egnn_mean=W.within_egnn.mean(), within_dgnn_mean=W.within_distance_gnn.mean(), cross_mean=y.mean(),
                  resid_mean_diff=(cr.cka_resid_mean - y).mean(), resid_min=(cr.cka_resid_mean - y).min(), resid_max=(cr.cka_resid_mean - y).max())
cp = P[P.kind == "cross"]
N["sweep"].update(n_cross_pairs=len(cp), n_pairs_gt_trained=int((cp.cka > tr["cka"]).sum()), matched=cp[cp.matched].cka.mean(), mismatched=cp[~cp.matched].cka.mean())
N["sweep"]["n_networks"] = 260
sel = cr[(cr.cka_mean >= .85) & (cr.cka_mean <= .95)]
N["sweep"].update(n_85_95=len(sel), knn_85_95=[sel.knn_mean.min(), sel.knn_mean.max()])
N["sweep"]["trained_knn"] = TR_CKA["egnn_x_distance_gnn"]["mknn_mean"]
N["sweep"]["n_knn_gt_trained"] = int((cr.knn_mean > TR_CKA["egnn_x_distance_gnn"]["mknn_mean"]).sum())

# one-factor-at-a-time
base = dict(width=64, depth=3, radius=5.0, gain=1.0, readout="mean", n_rbf=16, aggr="sum")
Ci = C.set_index("cfg_id")
ofat = []
for cid, r in Ci.iterrows():
    d = [k for k in base if r[k] != base[k]]
    if len(d) <= 1:
        ofat.append((d[0] if d else "base", r[d[0]] if d else "", cid))
order = ["base", "width", "depth", "radius", "gain", "readout", "n_rbf", "aggr"]
names = dict(base="Base", width="Width", depth="Depth", radius="Radius (\\AA)", gain="Weight gain", readout="Readout", n_rbf="Radial functions", aggr="Message aggregation")
def _k(v):
    try: return (0, float(v))
    except (TypeError, ValueError): return (1, str(v))
ofat.sort(key=lambda t: (order.index(t[0]), _k(t[1])))
N["ofat"] = {f"{f}={l}": dict(cka=cr.loc[c, "cka_mean"], knn=cr.loc[c, "knn_mean"], proc=cr.loc[c, "procrustes_mean"]) for f, l, c in ofat}
lines = ["\\begin{tabular}{llcccc}", "\\toprule", "Factor & Level & CKA & s.d. & $k$NN & Procrustes \\\\", "\\midrule"]
prev = None
for f, l, c in ofat:
    if prev is not None and f != prev: lines.append("\\addlinespace")
    lab = names[f] if f != prev else ""
    lev = "w64, d3, r5, g1, mean readout, 16 RBF, sum" if f == "base" else (f"{l:g}" if not isinstance(l, str) else l)
    lines.append(f"{lab} & {lev} & {cr.loc[c,'cka_mean']:.3f} & {cr.loc[c,'cka_std']:.3f} & {cr.loc[c,'knn_mean']:.3f} & {cr.loc[c,'procrustes_mean']:.3f} \\\\")
    prev = f
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_ofat.tex").write_text("\n".join(lines))

# all 27
SK = rd("res_sweep_kernel_cka.tsv")
skm = SK.groupby(["cfg_id", "kernel"]).cka.mean().unstack()
rows = []
for cid in cr.sort_values("cka_mean", ascending=False).index:
    rows.append(f"\\texttt{{{cid.replace('_','\\_')}}} & {cr.loc[cid,'cka_mean']:.3f} & {cr.loc[cid,'cka_std']:.3f} & {wavg[cid]:.3f} & {cr.loc[cid,'cka_resid_mean']:.3f} & {cr.loc[cid,'knn_mean']:.3f} & {cr.loc[cid,'procrustes_mean']:.3f} & {skm.loc[cid,'size']:.3f} & {skm.loc[cid,'degree']:.3f} \\\\")
hdr = "Setting & Cross & s.d. & Within & Size-resid. & $k$NN & Procr. & Size k. & Degree k."
(OUT / "tab_all.tex").write_text("{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\begin{longtable}{lcccccccc}\n\\caption{All 27 random-network settings sorted by cross-architecture CKA. Cross: EGNN $\\times$ Distance-GNN CKA, mean and s.d. over 25 pairs. Within: mean of the two same-architecture means (10 pairs each). Size-resid.: cross CKA after linear removal of ligand and complex atom counts. $k$NN: mutual $k$-nearest-neighbour overlap. Procr.: Procrustes angle in radians. Size k. and Degree k.: mean CKA between the networks and the size and neighbour-count kernels. $n=1{,}200$ complexes.}\\label{tab:all}\\\\\n\\toprule\n" + hdr + " \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n" + hdr + " \\\\\n\\midrule\n\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}}\n")

# ---------------------------------------------------------------- A3 floor model
R2 = rd("res_floor_model_r2.tsv"); V = rd("res_floor_model_verdict.tsv"); SPc = rd("res_floor_spearman_cross.tsv")
a3 = {}
for resp in ("cross", "within_egnn", "within_distance_gnn"):
    a3[resp] = {r.model: dict(cv=r.r2_cv_mean, sd=r.r2_cv_sd, lofo=r.r2_lofo) for r in R2[R2.response == resp].itertuples()}
N["a3"] = a3
N["a3_verdict"] = V.set_index("response").to_dict("index")
N["a3_spearman"] = SPc.iloc[0].drop("response").to_dict()
kk = rd("res_kernel_kernel.tsv"); N["kernel_kernel"] = {f"{r.k1}-{r.k2}": r.cka for r in kk.itertuples()}
lab = dict(M0_intercept="Intercept only", M1_config="Hyperparameters", M2_kernels="Four input kernels", M3_kernels_config="Kernels + hyperparameters",
           K_size="Size kernel", K_composition="Composition kernel", K_dist_hist="Distance-histogram kernel", K_degree="Neighbour-count kernel")
t = ["\\begin{tabular}{lccc}", "\\toprule", "Predictors & Cross & Within EGNN & Within Distance-GNN \\\\", "\\midrule"]
for m in lab:
    t.append(f"{lab[m]} & " + " & ".join(f"{a3[r][m]['cv']:.2f}" for r in ("cross", "within_egnn", "within_distance_gnn")) + " \\\\")
t += ["\\midrule", "Hyperparameters, leave-one-level-out & " + " & ".join(f"{a3[r]['M1_config']['lofo']:.2f}" for r in ("cross", "within_egnn", "within_distance_gnn")) + " \\\\",
      "Four input kernels, leave-one-level-out & " + " & ".join(f"{a3[r]['M2_kernels']['lofo']:.2f}" for r in ("cross", "within_egnn", "within_distance_gnn")) + " \\\\", "\\bottomrule", "\\end{tabular}"]
(OUT / "tab_a3.tex").write_text("\n".join(t))

# ---------------------------------------------------------------- A4 ablation
A = rd("res_ablation_boot.tsv"); N["ablation"] = A.set_index("condition").drop(columns="cfg_id").to_dict("index")
Ap = rd("res_ablation_pairs.tsv")
N["ablation_pairs_cross_mean"] = Ap[Ap.kind == "cross"].groupby("condition").cka.mean().to_dict()
N["ablation_pairs_knn_cross_mean"] = Ap[Ap.kind == "cross"].groupby("condition").knn.mean().to_dict()
N["ablation_within"] = Ap[Ap.kind.str.startswith("within")].groupby(["condition", "kind"]).cka.mean().unstack().to_dict("index")

# ---------------------------------------------------------------- real untrained
RP = rd("res_real_floor_pairing.tsv").set_index("pair"); N["real_pairing"] = RP.to_dict("index")
RW = rd("res_real_within.tsv").set_index("model"); N["real_within"] = RW.to_dict("index")
RK = rd("res_real_kernel_cka.tsv"); rk = RK.groupby(["model", "kernel"]).cka.agg(["mean", "std", "min", "max"])
N["real_kernel"] = {f"{m}/{k}": v for (m, k), v in rk.to_dict("index").items()}
N["sweep_base_kernel"] = SK[SK.cfg_id == "w64_d3_r5_g1_mean_k16_sum"].groupby("kernel").cka.mean().to_dict()
N["sweep_kernel_mean_all"] = SK.groupby("kernel").cka.mean().to_dict()

# ---------------------------------------------------------------- probes
G = rd("res_probe_gain.tsv"); M = rd("res_probe_metrics.tsv"); D = rd("res_probe_dims.tsv"); ML = rd("res_probe_mlp.tsv")
N["probe_verdicts"] = {rep: G[G.replicate == rep].verdict.value_counts().to_dict() for rep in ("primary", "anchor")}
N["probe_cells"] = int(len(G[G.replicate == "primary"]))
gcells = {}
for r in G.itertuples():
    gcells[f"{r.model}/{r.target}/{r.replicate}"] = dict(trained=r.trained, untrained=r.untrained_mean, kernel=r.best_kernel_metric, ref=r.reference, ref_is=r.reference_is, gain=r.gain, lo=r.gain_lo, hi=r.gain_hi, v=r.verdict)
N["probe_gain"] = gcells
untr = M[M.kind == "untrained"].copy(); untr["model"] = untr.input.str.split("/").str[0]
N["probe_untrained_mean"] = untr.groupby(["target", "model"]).metric.mean().unstack().to_dict("index")
N["probe_kernel"] = M[M.kind == "kernel"].pivot(index="target", columns="input", values="metric").to_dict("index")
N["probe_selectivity"] = M[M.input.isin(["egnn/primary", "egnn/anchor"])].set_index(["target", "input"]).selectivity.unstack().to_dict("index")
# dimension check
dd = D[(D.target == "affinity")].pivot_table(index=["model", "tag"], columns="dim_setting", values="metric")
N["probe_dims_affinity"] = {f"{a}/{b}": v for (a, b), v in dd.to_dict("index").items()}
N["probe_dim_max_abs_dev"] = float((D.assign(g=D.model + D.tag + D.target).groupby("g").metric.agg(lambda s: s.max() - s.min())).max())
dtmp = D.assign(g=D.model + "/" + D.tag + "/" + D.target); sp_ = dtmp[dtmp.dim_setting.isin(["32", "64", "128", "full"])].groupby("g").metric.agg(lambda s: s.max() - s.min())
trn = sp_[[k for k in sp_.index if "/primary/" in k]]
N["probe_dim_dev_primary_max"] = float(trn.max()); N["probe_dim_dev_primary_argmax"] = trn.idxmax()
# MLP
mlp = ML.copy(); mlp["better_mlp"] = mlp.metric_mlp > mlp.metric_ridge
N["mlp_better_frac"] = float(mlp.better_mlp.mean()); N["mlp_n"] = int(len(mlp))
N["mlp_affinity_egnn_primary"] = ML[(ML.input == "egnn/primary") & (ML.target == "affinity")][["metric_mlp", "metric_ridge"]].iloc[0].to_dict()


# PCA-32 truncation check for the one probe gain (graph networks: 128 and "full" equal 64 because embeddings are 64-d)
gd = D[D.model.isin(["egnn", "distance_gnn"])].pivot_table(index=["model", "tag", "target"], columns="dim_setting", values="metric")
N["pca_check"] = dict(
    max_abs_dev_32_vs_64=float((gd["32"] - gd["64"]).abs().max()), median_abs_dev_32_vs_64=float((gd["32"] - gd["64"]).abs().median()), n=int(len(gd)),
    max_abs_dev_128_full=float(max((gd["128"] - gd["64"]).abs().max(), (gd["full"] - gd["64"]).abs().max())),
    egnn_contacts_trained_32=float(gd.loc[("egnn", "primary", "contacts")]["32"]), egnn_contacts_trained_64=float(gd.loc[("egnn", "primary", "contacts")]["64"]),
    egnn_contacts_untr_mean_32=float(np.mean([gd.loc[("egnn", t, "contacts")]["32"] for t in ["untrained", "untrained_init2", "untrained_init3", "untrained_init4", "untrained_init5"]])),
    egnn_contacts_untr_mean_64=float(np.mean([gd.loc[("egnn", t, "contacts")]["64"] for t in ["untrained", "untrained_init2", "untrained_init3", "untrained_init4", "untrained_init5"]])),
    egnn_affinity_trained_32=float(gd.loc[("egnn", "primary", "affinity")]["32"]), egnn_affinity_trained_64=float(gd.loc[("egnn", "primary", "affinity")]["64"]))

# ---------------------------------------------------------------- targets
T = pd.read_csv(F / "targets.tsv", sep="\t", index_col=0)
N["targets"] = dict(n=len(T), spearman_affinity=T.corr(method="spearman")["affinity"].to_dict(), pearson_affinity=T.corr()["affinity"].to_dict(),
                    heavy_contacts=T[["ligand_heavy_atoms", "contacts"]].corr().iloc[0, 1],
                    mean=T.mean().to_dict(), sd=T.std().to_dict(), mn=T.min().to_dict(), mx=T.max().to_dict())
# partial correlation affinity-contacts given ligand heavy atoms (rank based)
from scipy.stats import rankdata
ra, rc, rh = (rankdata(T[c]) for c in ("affinity", "contacts", "ligand_heavy_atoms"))
def pc(a, b, z):
    rab, raz, rbz = np.corrcoef(a, b)[0, 1], np.corrcoef(a, z)[0, 1], np.corrcoef(b, z)[0, 1]
    return (rab - raz * rbz) / np.sqrt((1 - raz**2) * (1 - rbz**2))
N["targets"]["partial_spearman_aff_contacts_given_heavy"] = pc(ra, rc, rh)

# ---------------------------------------------------------------- output agreement + performance + health
OA = rd("res_output_agreement.tsv"); N["agree"] = {f"{r.pair}/{r.statistic}": [r.value, r.ci_lo, r.ci_hi] for r in OA.itertuples()}
OP = rd("res_output_perf.tsv"); N["perf"] = OP.set_index("model").to_dict("index")
pt = json.load(open(LOGS / "performance_table.json")); N["size_only_pearson"] = pt["rows"]["size_only_ols"]["pearson"]["value"]
N["size_only_r2"] = pt["rows"]["size_only_ols"]["r2"]["value"]
eh = json.load(open(LOGS / "embedding_health_2a.json")); N["participation"] = {k: v["participation_ratio"] for k, v in eh.items()}
N["covar_var"] = {k: v["frac_var_explained_by_covariates"] for k, v in eh.items()}
N["trained_other_pairs"] = {k: [TR_CKA[k][x] for x in ("cka", "ci_lo", "ci_hi", "floor", "mknn_mean")] for k in ("egnn_x_gnina", "egnn_x_distance_gnn", "gnina_x_distance_gnn")}
N["gnina_ceiling"] = TR_CKA["gnina_ceiling"]["cka"]

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(conv(N), open(OUT / "numbers.json", "w"), indent=1)

# ---------------------------------------------------------------- probe table (primary replicate)
tg = ["affinity", "contacts", "ligand_heavy_atoms", "complex_atoms", "pocket_volume_proxy", "buried_fraction", "pocket_polar_frac", "rotatable_bonds", "ligand_charge"]
tn = dict(affinity="Affinity ($-\\log$ units)", contacts="Contacts", ligand_heavy_atoms="Ligand heavy atoms", complex_atoms="Complex atoms", pocket_volume_proxy="Pocket volume proxy",
          buried_fraction="Buried ligand fraction", pocket_polar_frac="Pocket polar fraction", rotatable_bonds="Rotatable bonds", ligand_charge="Ligand charge")
sym = dict(gain="gain", no_gain="none", inconclusive="inconcl.")
def fmt(r): return f"{r.gain:+.3f} [{r.gain_lo:+.3f}, {r.gain_hi:+.3f}] {sym[r.verdict]}"
t = ["{\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccc}", "\\toprule", "Target & EGNN & Distance-GNN & GNINA \\\\", "\\midrule"]
for tgt in tg:
    cells = []
    for m in ("egnn", "distance_gnn", "gnina"):
        r = G[(G.model == m) & (G.target == tgt) & (G.replicate == "primary")].iloc[0]; cells.append(fmt(r))
    t.append(f"{tn[tgt]} & " + " & ".join(cells) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}}"]
(OUT / "tab_probe.tex").write_text("\n".join(t))
# anchor replicate
t = ["{\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccc}", "\\toprule", "Target & EGNN & Distance-GNN & GNINA \\\\", "\\midrule"]
for tgt in tg:
    cells = []
    for m in ("egnn", "distance_gnn", "gnina"):
        r = G[(G.model == m) & (G.target == tgt) & (G.replicate == "anchor")].iloc[0]; cells.append(fmt(r))
    t.append(f"{tn[tgt]} & " + " & ".join(cells) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}}"]
(OUT / "tab_probe_anchor.tex").write_text("\n".join(t))
# probe values for the three informative targets
t = ["\\begin{tabular}{llccccc}", "\\toprule", "Target & Model & Trained & Untrained (mean of 5) & Best input kernel & Gain & 95\\% CI \\\\", "\\midrule"]
for tgt in ("affinity", "contacts", "rotatable_bonds"):
    for m in ("egnn", "distance_gnn", "gnina"):
        r = G[(G.model == m) & (G.target == tgt) & (G.replicate == "primary")].iloc[0]
        t.append(f"{tn[tgt] if m=='egnn' else ''} & {dict(egnn='EGNN',distance_gnn='Distance-GNN',gnina='GNINA')[m]} & {r.trained:.3f} & {r.untrained_mean:.3f} & {r.best_kernel_metric:.3f} & {r.gain:+.3f} & [{r.gain_lo:+.3f}, {r.gain_hi:+.3f}] \\\\")
    t.append("\\addlinespace")
t = t[:-1] + ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_probe_values.tex").write_text("\n".join(t))

# descriptor table
t = ["\\begin{tabular}{lccc}", "\\toprule", "Descriptor & Mean (s.d.) & Range & Spearman with $-\\log$ affinity \\\\", "\\midrule"]
for c in ["ligand_heavy_atoms", "complex_atoms", "contacts", "rotatable_bonds", "pocket_volume_proxy", "buried_fraction", "pocket_polar_frac", "ligand_charge"]:
    fm = ".3f" if c in ("buried_fraction", "pocket_polar_frac") else ".1f"
    t.append(f"{tn[c]} & {T[c].mean():{fm}} ({T[c].std():{fm}}) & {T[c].min():{fm}} to {T[c].max():{fm}} & {T.corr(method='spearman')['affinity'][c]:+.2f} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_desc.tex").write_text("\n".join(t))


# compact main-text table
def gv(c): return f"{cr.loc[c,'cka_mean']:.3f} ({cr.loc[c,'cka_std']:.3f})"
L = [("Base (w64, d3, r5, g1, mean, sum)", "w64_d3_r5_g1_mean_k16_sum"), ("Width 32", "w32_d3_r5_g1_mean_k16_sum"), ("Width 128", "w128_d3_r5_g1_mean_k16_sum"),
     ("Depth 1", "w64_d1_r5_g1_mean_k16_sum"), ("Depth 6", "w64_d6_r5_g1_mean_k16_sum"), ("Radius 4 \\AA", "w64_d3_r4_g1_mean_k16_sum"), ("Radius 8 \\AA", "w64_d3_r8_g1_mean_k16_sum")]
R_ = [("Gain 0.5", "w64_d3_r5_g0.5_mean_k16_sum"), ("Gain 2", "w64_d3_r5_g2_mean_k16_sum"), ("Max readout", "w64_d3_r5_g1_max_k16_sum"), ("Sum readout", "w64_d3_r5_g1_sum_k16_sum"),
      ("Mean aggregation", "w64_d3_r5_g1_mean_k16_mean")]
t = ["\\begin{tabular}{lclc}", "\\toprule", "Setting & CKA & Setting & CKA \\\\", "\\midrule"]
for i in range(len(L)):
    a = f"{L[i][0]} & {gv(L[i][1])}"
    b = (f"{R_[i][0]} & {gv(R_[i][1])}") if i < len(R_) else ("Median of 27 settings & " + f"{y.median():.3f}" if i == 5 else "Range of 27 settings & " + f"{y.min():.3f} to {y.max():.3f}")
    t.append(a + " & " + b + " \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_main.tex").write_text("\n".join(t))

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": .6})
# sweep
order_ = cr.sort_values("cka_mean").index
fig, ax = plt.subplots(figsize=(5.4, 4.6))
yy = np.arange(len(order_))
ax.barh(yy, cr.loc[order_, "cka_mean"], xerr=cr.loc[order_, "cka_std"], color="#7a9cc6", height=.7, error_kw=dict(lw=.6, capsize=1.2))
ax.scatter(wavg[order_], yy, marker="D", s=9, color="#222", zorder=3, label="same architecture (mean of the two)")
ax.set_yticks(yy); ax.set_yticklabels([c.replace("_", " ") for c in order_], fontsize=5)
ax.set_xlim(0, 1.02); ax.set_xlabel("debiased linear CKA"); ax.legend(loc="lower right", fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_sweep.pdf"); plt.close(fig)
# metrics scatter
fig, axs = plt.subplots(1, 3, figsize=(5.6, 1.9))
for ax, (col, lab_) in zip(axs, [("procrustes_mean", "Procrustes angle (rad)"), ("knn_mean", "mutual $k$NN overlap"), (None, "CKA to size kernel")]):
    xv = cr[col] if col else skm.loc[cr.index, "size"]
    ax.scatter(xv, cr.cka_mean, s=9, color="#34699a"); ax.set_xlabel(lab_); ax.set_ylabel("cross CKA" if ax is axs[0] else "")
    ax.set_title(f"Spearman {stats.spearmanr(xv, cr.cka_mean)[0]:+.2f}", fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "fig_metrics.pdf"); plt.close(fig)
# ablation
cond = ["none", "shuffle_types", "shuffle_types_within_role", "random_coords"]
cl = ["intact input", "element types\nshuffled", "types shuffled\nwithin role", "coordinates\nrandomised"]
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4))
Aa = A.set_index("condition")
axs[0].bar(range(4), [Aa.loc[c, "matched_cross_cka"] for c in cond], yerr=[[Aa.loc[c, "matched_cross_cka"] - Aa.loc[c, "ci_lo"] for c in cond], [Aa.loc[c, "ci_hi"] - Aa.loc[c, "matched_cross_cka"] for c in cond]], color=["#7a9cc6", "#bbb", "#bbb", "#d98e4a"], capsize=2, error_kw=dict(lw=.6))
axs[0].set_xticks(range(4)); axs[0].set_xticklabels(cl, fontsize=5.5, rotation=20, ha="right"); axs[0].set_ylim(.7, 1.0); axs[0].set_ylabel("matched cross CKA")
d_ = cond[1:]
axs[1].bar(range(3), [Aa.loc[c, "delta_vs_none"] for c in d_], yerr=[[Aa.loc[c, "delta_vs_none"] - Aa.loc[c, "delta_ci_lo"] for c in d_], [Aa.loc[c, "delta_ci_hi"] - Aa.loc[c, "delta_vs_none"] for c in d_]], color=["#bbb", "#bbb", "#d98e4a"], capsize=2, error_kw=dict(lw=.6))
axs[1].axhline(0, color="k", lw=.5); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(cl[1:], fontsize=5.5, rotation=20, ha="right"); axs[1].set_ylabel("change vs. intact input")
fig.tight_layout(); fig.savefig(OUT / "fig_ablation.pdf"); plt.close(fig)
# kernel
fig, ax = plt.subplots(figsize=(4.4, 1.9))
ks = ["size", "composition", "dist_hist", "degree"]; kl = ["size", "composition", "distance hist.", "neighbour-count hist."]
for i, (m, col) in enumerate([("egnn", "#34699a"), ("distance_gnn", "#d98e4a"), ("gnina", "#6b8e23")]):
    ax.bar(np.arange(4) + (i - 1) * .26, [rk.loc[(m, k), "mean"] for k in ks], .26, yerr=[rk.loc[(m, k), "std"] for k in ks], color=col, capsize=1.5, error_kw=dict(lw=.5), label=dict(egnn="EGNN", distance_gnn="Distance-GNN", gnina="GNINA")[m])
ax.set_xticks(range(4)); ax.set_xticklabels(kl, fontsize=6); ax.set_ylabel("CKA to input kernel"); ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_kernel.pdf"); plt.close(fig)
# probe heatmap
models = ["egnn", "distance_gnn", "gnina"]; ml = ["EGNN", "Distance-GNN", "GNINA"]
H = np.array([[G[(G.model == m) & (G.target == tgt) & (G.replicate == "primary")].gain.iloc[0] for m in models] for tgt in tg])
Vv = [[G[(G.model == m) & (G.target == tgt) & (G.replicate == "primary")].verdict.iloc[0] for m in models] for tgt in tg]
fig, ax = plt.subplots(figsize=(3.6, 2.9))
im = ax.imshow(np.clip(H, -0.4, 0.2), cmap="RdBu", vmin=-.4, vmax=.4, aspect="auto")
for i in range(len(tg)):
    for j in range(3):
        ax.text(j, i, f"{H[i,j]:+.2f}" + ("*" if Vv[i][j] == "gain" else ""), ha="center", va="center", fontsize=6, color=("white" if abs(np.clip(H[i,j], -0.4, 0.2)) > 0.22 else "black"))
ax.set_xticks(range(3)); ax.set_xticklabels(ml, fontsize=6.5); ax.set_yticks(range(len(tg))); ax.set_yticklabels([tn[x].split(" (")[0].replace("\\", "") for x in tg], fontsize=6)
fig.colorbar(im, ax=ax, shrink=.8, label="trained minus reference (clipped)")
fig.tight_layout(); fig.savefig(OUT / "fig_probe.pdf"); plt.close(fig)

# decodability before training: untrained mean (sd over inits), trained, all-kernel reference
dt = ["affinity", "contacts", "ligand_heavy_atoms", "complex_atoms", "rotatable_bonds", "buried_fraction", "pocket_polar_frac", "pocket_volume_proxy", "ligand_charge"]
Mx = M.copy(); Mx["model"] = Mx.input.str.split("/").str[0]; Mx["tag"] = Mx.input.str.split("/").str[1]
rows_ = []
for tgt in dt:
    r = [tn[tgt]]
    for m in ("egnn", "distance_gnn", "gnina"):
        u = Mx[(Mx.target == tgt) & (Mx.model == m) & Mx.tag.str.startswith("untrained")].metric
        trn_ = Mx[(Mx.target == tgt) & (Mx.input == f"{m}/primary")].metric.iloc[0]
        r += [f"{u.mean():.2f} ({u.std():.2f})", f"{trn_:.2f}"]
    r.append(f"{Mx[(Mx.target == tgt) & (Mx.input == 'kernel/all_kernels')].metric.iloc[0]:.2f}")
    rows_.append(" & ".join(r) + " \\\\")
t = ["{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{lccccccc}", "\\toprule", " & \\multicolumn{2}{c}{EGNN} & \\multicolumn{2}{c}{Distance-GNN} & \\multicolumn{2}{c}{GNINA-type CNN} & Kernels \\\\", "Target & Untrained & Trained & Untrained & Trained & Untrained & Trained & all four \\\\", "\\midrule"] + rows_ + ["\\bottomrule", "\\end{tabular}}"]
(OUT / "tab_decode.tex").write_text("\n".join(t))
N["decode"] = {}
fig, ax = plt.subplots(figsize=(6.0, 2.5))
tg5 = dt
xx = np.arange(len(tg5)); wd = .2
for k_, (lab_, col_) in enumerate([("EGNN untrained", "#a9c2de"), ("EGNN trained", "#34699a"), ("Distance-GNN untrained", "#f0c9a0"), ("Distance-GNN trained", "#d98e4a")]):
    m = "egnn" if k_ < 2 else "distance_gnn"
    vals, errs = [], []
    for tgt in tg5:
        if k_ % 2 == 0:
            u = Mx[(Mx.target == tgt) & (Mx.model == m) & Mx.tag.str.startswith("untrained")].metric; vals.append(u.mean()); errs.append(u.std())
        else:
            vals.append(Mx[(Mx.target == tgt) & (Mx.input == f"{m}/primary")].metric.iloc[0]); errs.append(0)
    ax.bar(xx + (k_ - 1.5) * wd, vals, wd, yerr=errs, color=col_, label=lab_, capsize=1.2, error_kw=dict(lw=.5))
ax.set_xticks(xx); ax.set_xticklabels([tn[t_].split(" (")[0] for t_ in tg5], fontsize=5.5, rotation=20, ha="right")
ax.set_ylabel("probe $R^2$"); ax.set_ylim(-.1, 1.05); ax.legend(fontsize=5.5, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig_decode.pdf"); plt.close(fig)

print("done")
