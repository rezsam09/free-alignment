"""b_probe_panel.py (B1-B5) -- what is linearly decodable from trained vs untrained embeddings vs input-geometry kernels.
Inputs : 3 models x {primary, anchor, 5 untrained} (z-score + PCA-64 fit on TRAIN only) and kernel maps size / composition / dist_hist / degree / all_kernels.
Probe  : ridge, nested grouped CV by protein cluster on the evaluation split, pooled out-of-fold R2 (continuous) or AUROC (binary).
Key quantity (B4): training gain = metric(trained) - max( mean over 5 untrained inits , best kernel ), per bootstrap draw (max over kernels per draw = conservative).
Reading rule (pre-specified margins in plan): gain if CI_low > margin ; no_gain if CI_high < margin ; else inconclusive.
Controls (B5): shuffled-target null (n_perm), random-label control task + selectivity, PCA-dimension check, small MLP probe."""
import numpy as np, pandas as pd
import fcommon as fc, fprobe as fp
from input_kernels import load_kernels
from b0_make_targets import load_targets
al = fc.al

def full_rep(model, tag, plan, core):
    Etr, _ = fc.load_emb(model, tag, plan["train_split"])
    if Etr.shape[1] > plan["probe"]["full_max_dim"]: return None
    Ete, ite = fc.load_emb(model, tag, plan["eval_split"])
    return fc.zscore_train(Etr, fc.pick(Ete, ite, core, f"{model}/{tag}"))[1]

def model_inputs(plan, core, k=None):
    return {f"{m}/{t}": fc.rep_test(m, t, plan, core, k=k) for m in fc.MODELS for t in ["primary", "anchor"] + fc.UNTRAINED_TAGS}

def kernel_inputs(core):
    K = load_kernels(core); K["all_kernels"] = np.hstack([K["size"], K["composition"], K["dist_hist"], K["degree"]])
    return {f"kernel/{k}": v for k, v in K.items()}

def verdict(lo, hi, margin):
    if not np.isfinite(lo): return "undefined"
    return "gain" if lo > margin else ("no_gain" if hi < margin else "inconclusive")

def main():
    plan = fc.load_plan(); fc.require_freeze("b_probe_panel")
    pr = plan["probe"]; core = fc.core_ids(plan); T, meta = load_targets()
    miss = [p for p in core if p not in T.index]
    if miss: raise fc.FindingsError(f"{len(miss)} core ids lack target rows (e.g. {miss[:3]}); rerun b0_make_targets.py")
    T = T.loc[core]; groups_all = fc.clusters(plan, core)
    INP = {**model_inputs(plan, core), **kernel_inputs(core)}
    kernel_names = [n for n in INP if n.startswith("kernel/")]
    cells = {}
    rows, gains, dims_rows, mlp_rows = [], [], [], []
    rng_master = np.random.default_rng(plan["seed"] + 11)
    for target, tm in meta.items():
        typ = tm["type"]; mask = T[target].notna().values
        if mask.sum() < pr["min_n"]:
            fc.ledger("b_probe_panel", "skip", f"{target}: n={mask.sum()} < min_n", "planned"); continue
        y = T[target].values[mask].astype(float); g = groups_all[mask]; n = len(y)
        if typ == "binary" and (y.sum() < 20 or (1 - y).sum() < 20):
            fc.ledger("b_probe_panel", "skip", f"{target}: fewer than 20 positives or negatives", "planned"); continue
        sc = fp.Scorer(y, typ); fl = fp.folds_for(g, pr["folds"])
        P, aux, M0 = {}, {}, {}
        seed_t = plan["seed"] + fc.stable_int(target) % 100000
        W = [al.boot_weights(n, np.random.default_rng(seed_t + b), g) for b in range(plan["n_boot"])]
        for name, X in INP.items():
            Xs = X[mask]; P[name] = fp.oof_ridge(Xs, y, g, pr["alphas"], pr["folds"], pr["inner_folds"], fl)
            aux[name] = sc.prep(P[name]); M0[name] = sc(P[name], None, aux[name])
        MB = {name: np.array([sc(P[name], w, aux[name]) for w in W]) for name in INP}
        for name, X in INP.items():
            Xs = X[mask]; lo, hi = fc.ci(MB[name], plan["ci_level"])
            null = fp.shuffled_null(Xs, y, g, sc, pr["alphas"], pr["folds"], pr["inner_folds"], fl, pr["n_perm"], seed_t + 1)
            if name not in cells:
                from sklearn.cluster import KMeans
                kk = min(pr["control_k"], max(2, X.shape[0] // 20)); cells[name] = KMeans(n_clusters=kk, n_init=3, random_state=plan["seed"]).fit_predict(X)
            rr = np.random.default_rng(fc.stable_int(plan["seed"], name, target))
            vals = rr.standard_normal(cells[name].max() + 1) if typ == "continuous" else (rr.random(cells[name].max() + 1) < y.mean()).astype(float)
            yc = vals[cells[name]][mask]; scc = fp.Scorer(yc, typ)
            ctrl = scc(fp.oof_ridge(Xs, yc, g, pr["alphas"], pr["folds"], pr["inner_folds"], fl))
            kind = "kernel" if name.startswith("kernel/") else ("trained" if name.split("/")[1] in ("primary", "anchor") else "untrained")
            rows.append(dict(input=name, kind=kind, target=target, type=typ, n=n, dim=Xs.shape[1], metric=M0[name], ci_lo=lo, ci_hi=hi, null_mean=np.nanmean(null),
                             null_p95=np.nanquantile(null, 0.95), above_null=bool(M0[name] > np.nanquantile(null, 0.95)), control_metric=ctrl, selectivity=M0[name] - ctrl))
        for m in fc.MODELS:
            unt = [f"{m}/{t}" for t in fc.UNTRAINED_TAGS]
            for tr_tag in ("primary", "anchor"):
                tr = f"{m}/{tr_tag}"
                ub = np.mean([MB[u] for u in unt], 0); kb = np.nanmax([MB[k] for k in kernel_names], 0)
                gb = MB[tr] - np.fmax(ub, kb)
                u0 = np.mean([M0[u] for u in unt]); kn = max(kernel_names, key=lambda k: M0[k]); ref = max(u0, M0[kn]); lo, hi = fc.ci(gb, plan["ci_level"])
                mg = plan["probe"]["margin"][typ]
                gains.append(dict(model=m, replicate=tr_tag, target=target, type=typ, n=n, trained=M0[tr], untrained_mean=u0, best_kernel=kn, best_kernel_metric=M0[kn],
                                  reference=ref, reference_is=("untrained" if u0 >= M0[kn] else "kernel"), gain=M0[tr] - ref, gain_lo=lo, gain_hi=hi, margin=mg, verdict=verdict(lo, hi, mg)))
        # dimension check (point estimates)
        for m in fc.MODELS:
            for tag in ["primary"] + fc.UNTRAINED_TAGS:
                for d in pr["dims"]:
                    X = full_rep(m, tag, plan, core) if d == "full" else fc.rep_test(m, tag, plan, core, k=d)
                    if X is None: continue
                    Pd = fp.oof_ridge(X[mask], y, g, pr["alphas"], pr["folds"], pr["inner_folds"], fl)
                    dims_rows.append(dict(model=m, tag=tag, target=target, dim_setting=str(d), eff_dim=X.shape[1], metric=sc(Pd)))
        # MLP sensitivity (fixed hyper-parameters, no tuning)
        for name in [f"{m}/{t}" for m in fc.MODELS for t in ["primary"] + fc.UNTRAINED_TAGS] + ["kernel/all_kernels"]:
            Pm = fp.oof_mlp(INP[name][mask], y, fl, pr["mlp"]["hidden"], pr["mlp"]["max_iter"], pr["mlp"]["alpha"], pr["mlp"]["seed"])
            mlp_rows.append(dict(input=name, target=target, metric_mlp=sc(Pm), metric_ridge=M0[name]))
    fc.write_tsv(pd.DataFrame(rows), "res_probe_metrics.tsv"); fc.write_tsv(pd.DataFrame(gains), "res_probe_gain.tsv")
    fc.write_tsv(pd.DataFrame(dims_rows), "res_probe_dims.tsv"); fc.write_tsv(pd.DataFrame(mlp_rows), "res_probe_mlp.tsv")
    fc.ledger("b_probe_panel", "done", f"{len(set(r['target'] for r in rows))} targets")
    G = pd.DataFrame(gains); print(G[G.replicate == "primary"][["model", "target", "trained", "reference", "reference_is", "gain", "gain_lo", "gain_hi", "verdict"]].to_string())

if __name__ == "__main__":
    main()
