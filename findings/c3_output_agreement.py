"""c3_output_agreement.py (C3) -- do the three models agree in their OUTPUTS?  Uses existing {model}_primary_test_pred.npy (rows = {model}_primary_test_ids.txt).
Per pair: Pearson / Spearman of predictions, Pearson of residuals (y - pred), residual Pearson after OLS-removing the size covariates,
and the partial correlation of predictions given the measured affinity.  Cluster-bootstrap CIs (weighted Pearson; the size-adjusted residuals are fitted once, not refitted per draw).
Also per-model test r / RMSE / R2 for context."""
import numpy as np, pandas as pd
import fcommon as fc
from a5_a7_real_untrained import covariate_matrix
al = fc.al

def wcorr(a, b, w):
    sw = w.sum(); ma, mb = (w * a).sum() / sw, (w * b).sum() / sw
    c = (w * (a - ma) * (b - mb)).sum(); d = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())
    return float(c / d) if d > 0 else float("nan")

def partial(a, b, z, w):
    rab, raz, rbz = wcorr(a, b, w), wcorr(a, z, w), wcorr(b, z, w)
    return float((rab - raz * rbz) / np.sqrt((1 - raz ** 2) * (1 - rbz ** 2)))

def main():
    plan = fc.load_plan(); fc.require_freeze("c3_output_agreement")
    core = fc.core_ids(plan); sp = fc.split_df(plan); y = sp.loc[core, plan["affinity_column"]].values.astype(float)
    pred = {}
    for m in fc.MODELS:
        p = np.load(fc.EMB / plan["agreement"]["pred_pattern"].format(model=m)).ravel()
        ids = [s.lower() for s in (fc.EMB / f"{m}_primary_test_ids.txt").read_text().split()]
        if len(ids) != len(p): raise fc.FindingsError(f"{m}: {len(p)} predictions vs {len(ids)} ids")
        pred[m] = fc.pick(p[:, None], ids, core, f"{m}/pred")[:, 0]
        if not np.isfinite(pred[m]).all(): raise fc.FindingsError(f"{m}: non-finite predictions")
    Z = al.design_matrix(covariate_matrix(plan, core), plan["agreement"]["resid_degree"])
    res = {m: y - pred[m] for m in fc.MODELS}; res_adj = {m: al.ols_resid(res[m][:, None], Z)[:, 0] for m in fc.MODELS}
    groups = fc.clusters(plan, core); rng = np.random.default_rng(plan["seed"] + 31)
    W = [al.boot_weights(len(core), rng, groups) for _ in range(plan["n_boot"])]
    one = np.ones(len(core)); rows = []
    for a, b in fc.PAIRS:
        stats_ = {"pred_pearson": lambda w: wcorr(pred[a], pred[b], w), "resid_pearson": lambda w: wcorr(res[a], res[b], w),
                  "resid_pearson_size_adjusted": lambda w: wcorr(res_adj[a], res_adj[b], w), "pred_partial_given_y": lambda w: partial(pred[a], pred[b], y, w)}
        for name, f in stats_.items():
            v = f(one); bs = np.array([f(w) for w in W]); lo, hi = fc.ci(bs, plan["ci_level"]); rows.append(dict(pair=f"{a}_x_{b}", statistic=name, value=v, ci_lo=lo, ci_hi=hi))
        from scipy import stats as st
        rows.append(dict(pair=f"{a}_x_{b}", statistic="pred_spearman", value=float(st.spearmanr(pred[a], pred[b])[0]), ci_lo=np.nan, ci_hi=np.nan))
    fc.write_tsv(pd.DataFrame(rows), "res_output_agreement.tsv")
    perf = [dict(model=m, test_r=float(np.corrcoef(y, pred[m])[0, 1]), rmse=float(np.sqrt(((y - pred[m]) ** 2).mean())), r2=float(1 - ((y - pred[m]) ** 2).sum() / ((y - y.mean()) ** 2).sum())) for m in fc.MODELS]
    fc.write_tsv(pd.DataFrame(perf), "res_output_perf.tsv")
    fc.ledger("c3_output_agreement", "done", f"n={len(core)}")
    print(pd.DataFrame(rows).to_string())

if __name__ == "__main__":
    main()
