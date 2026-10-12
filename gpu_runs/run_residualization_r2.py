#!/usr/bin/env python
"""run_residualization_r2.py -- reference-kernel and covariate residualization of CKA, low-sim n_AB.

Fixes vs previous version:
  * strict id alignment everywhere (no `if pid in rev` shortening); missing ids abort
  * per-tag train-fit PCA-64, hard error if train embeddings are missing (no test-fit fallback)
  * reference-kernel rows that are NaN OR zero-diagonal (failed AA-composition vector) are dropped, not regressed on
  * covariates must exist for all core complexes (no median imputation)
  * low-sim n_AB uses the same matched-floor / min-ceiling calibration as run_leakage_r2.py (all 5 inits required)
Reference-kernel residualization is now done in FEATURE space: the (centered) reference kernel is
eigendecomposed, its leading components (REF_VAR_FRAC of its variance, at most REF_MAX_COMP) form an orthonormal
basis U, and each embedding's Gram becomes P K P with P = I - U U^T (alignlib.resid_gram) - i.e. the reference
structure is partialled out of the features, linear in the reference's kernel feature map. The old entry-wise
regression of kernel entries on kernel entries is gone. Also reported: share of each embedding's variance that
lies in span(U) (alignlib.r2_on_basis).
Covariate residualization is reported twice: post-PCA (residualize the PCA-64 projections) and pre-PCA (residualize the raw
embeddings, then PCA-64 fit on train residuals - 'pre_pca' key), because top-64 PCs of GNINA may be mostly size.
Caveat: covariate residualization is linear only; nonlinear size dependence is covered by run_h1_size_resid.py (poly2).
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ec2-user/SageMaker/pdbbind_data")).resolve()
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "gpu_runs")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import prereg
import src.alignlib as al
prereg.verify_lock(ROOT)

LOG = ROOT / "logs"
EMB = ROOT / "data" / "embeddings"
PCA_DIM = 64
MIN_DEN = 0.05
MODELS = ("egnn", "distance_gnn", "gnina")
UNTRAINED_TAGS = ["untrained"] + [f"untrained_init{i}" for i in range(2, 6)]
CROSS_PAIRS = [
    ("egnn", "gnina", "egnn_x_gnina"),
    ("egnn", "distance_gnn", "egnn_x_distance_gnn"),
    ("gnina", "distance_gnn", "gnina_x_distance_gnn"),
]
REQUIRED = [(m, t) for m in MODELS for t in ("primary", "anchor")]


def load_emb(model, tag, sp="test"):
    npy = EMB / f"{model}_{tag}_{sp}.npy"
    idsf = EMB / f"{model}_{tag}_{sp}_ids.txt"
    if not npy.exists():
        return None, []
    assert idsf.exists(), f"missing {idsf}"
    return np.load(npy).astype(np.float64), idsf.read_text().splitlines()


REF_VAR_FRAC = 0.95   # fraction of the centered reference kernel's variance kept in the basis
REF_MAX_COMP = 100


def ref_basis(K_ref):
    """Orthonormal basis (n x r) of the leading feature-space directions of a PSD reference kernel.
    K_ref is centered first, so the basis is orthogonal to the constant vector."""
    Kc = al.center_gram(0.5 * (K_ref + K_ref.T))
    w, V = np.linalg.eigh(Kc)
    w, V = w[::-1], V[:, ::-1]
    pos = w > 1e-10 * max(w[0], 1e-300)
    w, V = w[pos], V[:, pos]
    r = int(min(np.searchsorted(np.cumsum(w) / w.sum(), REF_VAR_FRAC) + 1, REF_MAX_COMP, len(w)))
    return V[:, :r], r, float(w[:r].sum() / w.sum())


def ols_resid_mat(X, Z):
    Z = np.column_stack([np.ones(len(Z)), Z])
    return X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]


# ── core set: primary, anchor and all untrained inits must exist ─────────────
keys = REQUIRED + [(m, t) for m in MODELS for t in UNTRAINED_TAGS]
RAW = {}
for m, t in keys:
    E, ids = load_emb(m, t)
    assert E is not None, f"Missing test embeddings {m}/{t}"
    RAW[(m, t)] = (E, ids)
core_ids = sorted(set.intersection(*[set(v[1]) for v in RAW.values()]))
n = len(core_ids)
print(f"Core set: {n}")

P = {}
for (m, t), (E, ids) in RAW.items():
    E_tr, _ = load_emb(m, t, "train")
    mu, pca = al.pca_train_fit(E_tr, PCA_DIM, name=f"{m}/{t}")
    P[(m, t)] = pca.transform(al.strict_subset(E, ids, core_ids, name=f"{m}/{t}") - mu)
Kraw = {k: al.gram(P[k]) for k in REQUIRED}

results = {"reference_kernel": {}, "covariate_residualization": {}, "zero_aa_flag": {}}

# ── reference-kernel residualization ─────────────────────────────────────────
for kname, kfile, idfile in [
    ("tanimoto", ROOT / "data/tanimoto_kernel.npy", ROOT / "data/tanimoto_kernel_ids.txt"),
    ("aa_comp", ROOT / "data/aa_comp_kernel.npy", ROOT / "data/aa_comp_kernel_ids.txt"),
]:
    if not kfile.exists():
        print(f"[SKIP] {kname} kernel not found")
        continue
    K_full = np.load(kfile)
    ref_ids = idfile.read_text().splitlines()
    assert K_full.shape[0] == len(ref_ids), f"{kname}: kernel/ids size mismatch"
    rev = {p: i for i, p in enumerate(ref_ids)}
    miss = [p for p in core_ids if p not in rev]
    assert not miss, f"{kname}: {len(miss)} core ids absent from reference kernel (e.g. {miss[:3]})"
    ci = [rev[p] for p in core_ids]
    K_ref = K_full[np.ix_(ci, ci)]

    diag = K_ref.diagonal()
    bad = np.isnan(diag) | (diag == 0) | np.all(np.isnan(K_ref), axis=1)   # NaN rows AND zero AA vectors
    valid = np.flatnonzero(~bad)
    if kname == "aa_comp":
        results["zero_aa_flag"] = {"n_zero_in_core": int(bad.sum()), "n_core": n,
                                   "fraction": round(float(bad.mean()), 4), "action": "dropped"}
    n_drop = int(bad.sum())
    print(f"\n--- {kname}: {len(valid)}/{n} valid ({n_drop} dropped: NaN or zero-diagonal reference rows) ---")
    assert not np.isnan(K_ref[np.ix_(valid, valid)]).any(), f"{kname}: NaN left after dropping bad rows"
    K_ref_v = K_ref[np.ix_(valid, valid)]
    U, r_comp, ref_var = ref_basis(K_ref_v)
    print(f"  reference basis: {r_comp} components, {100*ref_var:.1f}% of reference variance")
    rec = {}
    for mA, mB, key in CROSS_PAIRS:
        # Grams recomputed on the valid subset from the features (re-centered on that subset)
        XA, XB = P[(mA, "primary")][valid], P[(mB, "primary")][valid]
        KA, KB = al.gram(XA), al.gram(XB)
        raw = al.cka_debiased(KA, KB)
        res = al.cka_debiased(al.resid_gram(KA, U), al.resid_gram(KB, U))
        r2A, r2B = al.r2_on_basis(KA, U), al.r2_on_basis(KB, U)
        print(f"  {key}: raw={raw:.4f} resid={res:.4f} delta={raw-res:.4f}  "
              f"(ref explains {100*r2A:.1f}% / {100*r2B:.1f}% of embedding variance)")
        rec[key] = dict(cka_raw=round(raw, 4), cka_resid=round(res, 4), delta_cka=round(raw - res, 4),
                        n_valid=int(len(valid)), n_dropped=n_drop, n_ref_components=r_comp,
                        ref_variance_kept=round(ref_var, 4),
                        ref_r2_embedding_A=round(r2A, 4), ref_r2_embedding_B=round(r2B, 4),
                        method="feature_space_projection_off_reference_kernel_basis")
    results["reference_kernel"][kname] = rec

# ── covariate residualization (linear, test-fit OLS) ─────────────────────────
cov_path = ROOT / "data" / "covariates.tsv"
if cov_path.exists():
    cov = pd.read_csv(cov_path, sep="\t", index_col=0)
    cov.index = cov.index.astype(str).str.lower()
    cov_cols = [c for c in ["complex_atom_count", "ligand_heavy_atom_count", "n_complex_atoms",
                            "n_lig_heavy_atoms", "mw", "logp", "hbd", "hba"] if c in cov.columns]
    if not cov_cols:
        results["covariate_residualization"]["status"] = "skipped: no recognised columns"
    else:
        miss = [p for p in core_ids if p not in cov.index]
        assert not miss, f"{len(miss)} core complexes lack covariates (e.g. {miss[:3]}); no imputation"
        arr = cov.loc[core_ids, cov_cols].to_numpy(dtype=float)
        assert np.isfinite(arr).all(), "non-finite covariates among core complexes"
        sd = arr.std(0); sd[sd == 0] = 1
        cs = (arr - arr.mean(0)) / sd
        vif = {}
        for i, c in enumerate(cov_cols):
            if len(cov_cols) > 1:
                Z = np.column_stack([np.ones(n), np.delete(cs, i, axis=1)])
                pred = Z @ np.linalg.lstsq(Z, cs[:, i], rcond=None)[0]
                r2 = 1 - ((cs[:, i] - pred) ** 2).sum() / ((cs[:, i] - cs[:, i].mean()) ** 2).sum()
                vif[c] = round(1 / (1 - r2) if r2 < 0.999 else float("inf"), 2)
            else:
                vif[c] = 1.0
        print(f"\nCovariates {cov_cols}  VIF {vif}")
        _pre_cache = {}

        def pre_pca_primary(model, idx):
            if (model, tuple(idx)) not in _pre_cache:
                E_tr, ids_tr = load_emb(model, "primary", "train")
                have = np.array([p in cov.index for p in ids_tr])
                assert have.mean() > 0.9, f"{model}: only {have.mean():.0%} of train ids have covariates"
                c_tr = cov.loc[[p for p, h in zip(ids_tr, have) if h], [cov_cols[i] for i in idx]].to_numpy(float)
                c_tr = (c_tr - c_tr.mean(0)) / np.where(c_tr.std(0) > 0, c_tr.std(0), 1.0)
                E, ids = RAW[(model, "primary")]
                _pre_cache[(model, tuple(idx))] = al.resid_then_pca(
                    E_tr[have], al.design_matrix(c_tr, 1), al.strict_subset(E, ids, core_ids, name=model),
                    al.design_matrix(cs[:, idx], 1), PCA_DIM, name=model)
            return _pre_cache[(model, tuple(idx))]
        for mA, mB, key in CROSS_PAIRS:
            XA, XB = P[(mA, "primary")], P[(mB, "primary")]
            raw = al.cka_debiased(Kraw[(mA, "primary")], Kraw[(mB, "primary")])
            per = {}
            for i, c in enumerate(cov_cols):
                r = al.cka_debiased(al.gram(ols_resid_mat(XA, cs[:, i:i+1])), al.gram(ols_resid_mat(XB, cs[:, i:i+1])))
                per[c] = dict(cka_resid=round(r, 4), delta=round(raw - r, 4))
            rj = al.cka_debiased(al.gram(ols_resid_mat(XA, cs)), al.gram(ols_resid_mat(XB, cs)))
            per["_joint"] = dict(cka_resid=round(rj, 4), delta=round(raw - rj, 4))
            # sensitivity: residualize the RAW embeddings first, then PCA-64 fit on train residuals
            pre = {}
            for label, idx in [(c, [i]) for i, c in enumerate(cov_cols)] + [("_joint", list(range(len(cov_cols))))]:
                a, b = pre_pca_primary(mA, idx), pre_pca_primary(mB, idx)
                r = float(al.cka_feature(a, b)); pre[label] = dict(cka_resid=round(r, 4), delta=round(raw - r, 4))
            results["covariate_residualization"][key] = {"per_cov": per, "vif": vif, "pre_pca": pre,
                "pre_pca_note": "RAW embeddings OLS-residualized within each split, then PCA-64 fit on train residuals"}
            print(f"  {key}: raw={raw:.4f} joint resid(post-PCA)={rj:.4f}  joint resid(pre-PCA)={pre['_joint']['cka_resid']:.4f}")
else:
    results["covariate_residualization"]["status"] = "skipped: covariates.tsv missing"

# ── low-similarity subset n_AB (same calibration as run_leakage_r2.py) ───────
print("\n=== Low-similarity subset n_AB (ligand Tanimoto only; not a protein-leakage control) ===")
ids_path = ROOT / "data" / "splits" / "test_low_sim_ids.txt"
assert ids_path.exists(), "run run_write_lowsim_ids.py first"
lowsim = set(ids_path.read_text().split())
mask = np.array([p in lowsim for p in core_ids])
n_ls = int(mask.sum())
print(f"  low-sim in core: {n_ls}/{n}")
ls = {}
for mA, mB, key in CROSS_PAIRS:
    if n_ls < 50:
        ls[key] = {"status": "skipped: too few low-sim complexes", "n_lowsim": n_ls}
        continue
    fl = float(np.mean([al.cka_feature(P[(mA, t)][mask], P[(mB, t)][mask]) for t in UNTRAINED_TAGS]))
    cA = al.cka_feature(P[(mA, "primary")][mask], P[(mA, "anchor")][mask])
    cB = al.cka_feature(P[(mB, "primary")][mask], P[(mB, "anchor")][mask])
    ce = float(min(cA, cB))
    ob = float(al.cka_feature(P[(mA, "primary")][mask], P[(mB, "primary")][mask]))
    d = ce - fl
    ok = np.isfinite(d) and d >= MIN_DEN
    nab = round(float((ob - fl) / d), 6) if ok else None
    print(f"  {key}: floor={fl:.4f} ceil={ce:.4f} obs={ob:.4f} n_AB={nab}")
    ls[key] = dict(n_lowsim=n_ls, floor=round(fl, 4), ceiling=round(ce, 4), obs_cka=round(ob, 4),
                   denom=round(d, 4), n_ab=nab, n_floor_inits=len(UNTRAINED_TAGS),
                   status="ok" if ok else "undefined: denom < 0.05")
results["low_sim_nab"] = ls
results["_meta"] = dict(method="per_tag_train_pca64; strict alignment; zero/NaN reference rows dropped; "
                               f"reference kernels partialled out in feature space ({REF_VAR_FRAC:.0%} variance, <= {REF_MAX_COMP} comps)",
                        deviations=["CKA on per-tag train-fit PCA-64, retained for computational cost (GNINA is 27,648-d; 10k permutation/bootstrap draws); config says cka mean_center_only"])

out = LOG / "residualization_r2.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out}")
