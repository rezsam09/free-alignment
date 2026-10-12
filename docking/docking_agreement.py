#!/usr/bin/env python
"""docking_agreement.py -- agreement between each trained model's predicted affinity and the AutoDock Vina score.

For each model M (EGNN, Distance-GNN, voxel CNN of the GNINA type), on the complexes that received a Vina score:
    x = M's own predicted affinity (its readout head applied to the stored test embeddings, verified against the model's
        saved test predictions)
    y = negated Vina score of the top-ranked pose (higher = better docking)
    z = ligand heavy-atom count
Reports the partial Spearman correlation of x and y given z, with a 95% protein-cluster bootstrap interval (10,000
resamples). Also reports, on the same complexes, the Spearman correlation of the negated Vina score with measured
affinity (raw and given ligand heavy-atom count) and with ligand heavy-atom count, and the number of complexes whose top
pose lies within 2 A of the crystal pose.

Inputs:  logs/vina_results.json   (docking/make_vina_results.py, from data/docking_results.tsv)
         data/covariates.tsv      (gpu_runs/make_covariates.py)
         data/clusters.tsv        (gpu_runs/make_clusters.py)
         data/embeddings/, data/checkpoints/   (reference-model pipeline)
Output:  logs/docking_agreement.json
"""
import json, os, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
assert ROOT.exists(), f"PROJECT_ROOT not found: {ROOT}"
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
import pandas as pd
from src import prereg
from src import alignlib as al

prereg.verify_lock(ROOT)

cfg         = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
SPLIT_COL   = cfg["split"]["split_column"]
AFF_COL     = cfg["split"]["affinity_column"]
RMSD_THRESH = float(cfg["vina"]["pose_success_rmsd_A"])  # 2.0
N_BOOT_A    = 10_000
_seeds = cfg.get("seeds", {})
BOOT_SEED = _seeds.get("bootstrap", 20260920)
ALPHAS_WIDE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0]

EMB_DIR  = ROOT / "data" / "embeddings"
CKPT_DIR = ROOT / "data" / "checkpoints"

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    print("WARN: torch not found; readout heads will use the ridge fallback if ALLOW_RIDGE_FALLBACK=1")

# ── prerequisite checks ───────────────────────────────────────────────────────
vina_json = ROOT / "logs" / "vina_results.json"
assert vina_json.exists(), "logs/vina_results.json missing — run run_vina_docking.py first"
cov_path = ROOT / "data" / "covariates.tsv"
assert cov_path.exists(), "data/covariates.tsv missing — run make_covariates.py first"

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
split = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0)
split.index = split.index.astype(str).str.lower()
test_pids_set = set(pid for pid in split.index if split.loc[pid, SPLIT_COL] == "test")

vina = json.loads(vina_json.read_text())
print(f"  Vina results: {len(vina)} complexes")

cov = pd.read_csv(cov_path, sep="\t", index_col=0)
cov.index = cov.index.astype(str).str.lower()
print(f"  Covariates: {cov.shape}")

# ── load clusters (explicit column; required) ─────────────────────────────────
CLUSTER_COL = os.environ.get("CLUSTER_COL", "cluster_id")   # named, never guessed
clusters_path = ROOT / "data" / "clusters.tsv"
assert clusters_path.exists(), "data/clusters.tsv missing"
cl = pd.read_csv(clusters_path, sep="\t", index_col=0)
assert CLUSTER_COL in cl.columns, f"cluster column {CLUSTER_COL!r} not in clusters.tsv ({list(cl.columns)}); set CLUSTER_COL"
groups_map = {str(pid).lower(): int(v) for pid, v in cl[CLUSTER_COL].items()}
print(f"  Clusters ({CLUSTER_COL}): {len(groups_map)} entries")

def get_group(pid):
    return groups_map.get(pid, -1)

# ── embedding loaders ─────────────────────────────────────────────────────────
def load_emb(model, tag, sp):
    npy  = EMB_DIR / f"{model}_{tag}_{sp}.npy"
    idsf = EMB_DIR / f"{model}_{tag}_{sp}_ids.txt"
    if not npy.exists():
        return None, []
    E = np.load(npy).astype(np.float64)
    ids = idsf.read_text().splitlines() if idsf.exists() else []
    return E, ids

# ── checkpoint finder ──────────────────────────────────────────────────────────
def find_ckpt(prefix):
    for ext in (".pt", ".pth", "_best.pt", "_best.pth"):
        p = CKPT_DIR / f"{prefix}{ext}"
        if p.exists():
            return p
    hits = sorted(CKPT_DIR.glob(f"*{prefix}*"))
    return hits[0] if hits else None

# ── ridge fallback (only when readout head unavailable) ───────────────────────
def predict_ridge(model_name, tag="primary", pca_dim=None):
    """Ridge fallback: fit on train, predict test. Logs as deviation."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.decomposition import PCA

    E_tr, tr_ids = load_emb(model_name, tag, "train")
    E_te, te_ids = load_emb(model_name, tag, "test")
    if E_tr is None or E_te is None:
        return {}

    tr_pids = [p for p in tr_ids if p in split.index
               and split.loc[p, SPLIT_COL] == "train"
               and np.isfinite(split.loc[p, AFF_COL])]
    te_pids = [p for p in te_ids if p in split.index
               and split.loc[p, SPLIT_COL] == "test"
               and np.isfinite(split.loc[p, AFF_COL])]

    tr_rev = {p: i for i, p in enumerate(tr_ids)}
    te_rev = {p: i for i, p in enumerate(te_ids)}

    Xtr = E_tr[[tr_rev[p] for p in tr_pids]]
    Xte = E_te[[te_rev[p] for p in te_pids]]
    y_tr = np.array([split.loc[p, AFF_COL] for p in tr_pids])

    if pca_dim and Xtr.shape[1] > pca_dim:
        from sklearn.decomposition import PCA as _PCA
        mu = Xtr.mean(0)
        pca = _PCA(n_components=pca_dim, random_state=0).fit(Xtr - mu)
        Xtr = pca.transform(Xtr - mu)
        Xte = pca.transform(Xte - mu)

    grps = np.array([get_group(p) for p in tr_pids])
    n_splits = min(5, len(set(grps)))
    gkf = GroupKFold(n_splits=n_splits)
    ridge = RidgeCV(alphas=ALPHAS_WIDE, cv=gkf.split(Xtr, y_tr, grps), scoring="r2")
    ridge.fit(Xtr, y_tr)
    y_pred = ridge.predict(Xte)
    return {p: float(y_pred[i]) for i, p in enumerate(te_pids)}

# ── apply model's own readout head to precomputed test embeddings ──────────────
ALLOW_RIDGE_FALLBACK = os.environ.get("ALLOW_RIDGE_FALLBACK") == "1"   # documented deviation, off by default
HEAD_ACTIVATION = "silu"          # MUST match the model definition (verified by the checks below)
GNINA_HEAD_PREFIXES: list = [s.strip() for s in os.environ.get("GNINA_HEAD_PREFIXES", "").split(",") if s.strip()]   # e.g. GNINA_HEAD_PREFIXES=output_fc ; inspect checkpoint keys with check_gnina_ckpt.py
MIN_HEAD_PEARSON = 0.3            # reconstructed predictions must correlate with labels at least this much
HEAD_CHECKS: dict = {}
SAVED_PREDS: dict = {}            # model -> saved test predictions (row order of *_test_ids.txt)

def _act(x):
    if HEAD_ACTIVATION == "silu":
        return x * (1.0 / (1.0 + np.exp(-x)))
    raise ValueError(HEAD_ACTIVATION)

def _ridge_or_fail(model_name, tag, why):
    if not ALLOW_RIDGE_FALLBACK:
        raise RuntimeError(f"{model_name}/{tag}: {why}. the agreement analysis requires the model's OWN head (not a ridge refit). "
                           "Set ALLOW_RIDGE_FALLBACK=1 only to accept a documented deviation.")
    print(f"  DEVIATION: {model_name}/{tag}: {why} - using ridge fallback")
    return predict_ridge(model_name, tag, 64 if model_name == "gnina" else None), "ridge_fallback"

def _forward(x, layers):
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = _act(x)
    return x.squeeze(-1)

def apply_model_head(model_name, tag="primary"):
    """Returns ({pid: pred}, source) using the model's own readout weights (not ridge)."""
    emb_path = EMB_DIR / f"{model_name}_{tag}_test.npy"
    ids_path = EMB_DIR / f"{model_name}_{tag}_test_ids.txt"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"{emb_path} / ids not found")
    emb = np.load(emb_path).astype(np.float64)
    ids = ids_path.read_text().splitlines()
    assert len(ids) == emb.shape[0], f"{model_name}: ids/rows mismatch"

    ckpt = find_ckpt(f"{model_name}_{tag}")
    if ckpt is None or not _HAS_TORCH:
        return _ridge_or_fail(model_name, tag, "no checkpoint / torch")

    sd = torch.load(str(ckpt), map_location="cpu")
    for key in ("state_dict", "model", "net"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]

    def _collect(sd, patterns):
        res = {}
        for k, v in sd.items():
            if not hasattr(v, "float") or not k.endswith((".weight", ".bias")):
                continue
            if not any(p in k.lower() for p in patterns):
                continue
            res.setdefault(k.rsplit(".", 1)[0], {})[k.rsplit(".", 1)[1]] = v.float().numpy()
        # keep only Linear layers (2-D weight); drops norm layers whose names contain the pattern
        return {k: d for k, d in res.items() if "weight" in d and d["weight"].ndim == 2}

    if model_name == "gnina" and GNINA_HEAD_PREFIXES:
        patterns, head_source = tuple(p.lower() for p in GNINA_HEAD_PREFIXES), "gnina_explicit_prefixes"
    else:
        patterns, head_source = ("readout",), "readout_head"
    layer_dict = _collect(sd, patterns)
    if not layer_dict and model_name == "gnina":
        keys2d = [k for k, v in sd.items() if hasattr(v, "ndim") and v.ndim == 2]
        raise RuntimeError(f"GNINA: no head layers matched {patterns} in {ckpt.name}. 2-D weights present: "
                           f"{keys2d[:60]}. Set GNINA_HEAD_PREFIXES to the affinity-head layers explicitly.")
    if not layer_dict:
        layer_dict, head_source = _collect(sd, ("head", "output")), "head_output_fallback"
    if not layer_dict:
        return _ridge_or_fail(model_name, tag, "no readout/head/output Linear layers")

    sorted_keys = sorted(layer_dict, key=lambda t: [int(u) if u.isdigit() else u for u in re.split(r"(\d+)", t)])
    layers = [(layer_dict[k]["weight"].astype(np.float64),
               layer_dict[k].get("bias", np.zeros(layer_dict[k]["weight"].shape[0])).astype(np.float64))
              for k in sorted_keys]
    print(f"  [head] {model_name} layers: {sorted_keys}  shapes: {[W.shape for W, _ in layers]}  emb: {emb.shape}")
    if layers[0][0].shape[1] != emb.shape[1]:
        raise RuntimeError(f"Dim mismatch: emb has {emb.shape[1]} features, first head layer expects "
                           f"{layers[0][0].shape[1]}. Check which tensor was extracted.")
    ambiguous = layers[0][0].shape[0] == layers[0][0].shape[1] == emb.shape[1]
    if ambiguous:
        print("  WARNING: first head layer is square, so shapes cannot rule out that the embedding is already "
              "the OUTPUT of readout.0 (double application). Relying on the checks below.")

    preds = _forward(emb, layers)

    # CHECK 1 (REQUIRED): exact match with the model's own saved test predictions (written by the extractor from the SAME
    # forward pass as the embeddings, in the same row order as *_test_ids.txt). A label-correlation check is far too weak to
    # catch a double-applied or wrong head, so without the file the run aborts unless ALLOW_NO_SAVED_PRED=1 is set explicitly.
    saved = EMB_DIR / f"{model_name}_{tag}_test_pred.npy"
    check = {"ambiguous_square_first_layer": bool(ambiguous)}
    if not saved.exists() and os.environ.get("ALLOW_NO_SAVED_PRED") != "1":
        raise RuntimeError(f"{saved.name} not found. The extractor must write the model's own test predictions "
                           "(one float per row of *_test_ids.txt) so the reconstructed head can be verified exactly. "
                           "Set ALLOW_NO_SAVED_PRED=1 only to accept the weak label-correlation check (recorded in the output).")
    if saved.exists():
        sp = np.load(saved).astype(np.float64).reshape(-1)
        assert sp.shape[0] == preds.shape[0], "saved predictions length mismatch"
        assert np.isfinite(sp).all() and sp.std() > 0, f"{saved.name}: non-finite or constant predictions"
        md = float(np.max(np.abs(sp - preds)))
        assert md < 1e-3, f"{model_name}: reconstructed head predictions differ from saved model predictions (max abs diff {md:.4g})"
        check["saved_prediction_max_abs_diff"] = md
        check["verification"] = "exact_match_with_saved_predictions"
        SAVED_PREDS[model_name] = sp
    else:
        check["verification"] = "label_correlation_only_WEAK"
        # CHECK 2 (no saved predictions): reconstructed predictions must track the labels, and the alternative
        # interpretation (embedding = output of readout.0) must not fit clearly better.
        y = np.array([split.loc[p, AFF_COL] if p in split.index else np.nan for p in ids], dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(preds)
        r_main = float(np.corrcoef(preds[ok], y[ok])[0, 1])
        check["pearson_vs_labels"] = r_main
        if r_main < MIN_HEAD_PEARSON:
            raise RuntimeError(f"{model_name}: head reconstruction fails sanity check (Pearson {r_main:.3f} < {MIN_HEAD_PEARSON}). "
                               "Embedding/head mismatch (double application? wrong activation?). Write *_test_pred.npy in the extractor and re-run.")
        if len(layers) >= 2 and ambiguous:
            alt = _act(emb) @ layers[1][0].T + layers[1][1] if len(layers) == 2 else None
            if alt is None:
                x = _act(emb)
                for i, (W, b) in enumerate(layers[1:]):
                    x = x @ W.T + b
                    if i < len(layers) - 2:
                        x = _act(x)
                alt = x
            alt = np.asarray(alt).reshape(-1)
            r_alt = float(np.corrcoef(alt[ok], y[ok])[0, 1])
            check["pearson_alt_interpretation_embedding_is_readout0_output"] = r_alt
            if r_alt > r_main + 0.1:
                raise RuntimeError(f"{model_name}: alternative interpretation (embedding already passed readout.0) fits labels "
                                   f"much better (r={r_alt:.3f} vs {r_main:.3f}); readout.0 is probably applied twice.")
    check.update(activation=HEAD_ACTIVATION, layers=sorted_keys)
    HEAD_CHECKS[model_name] = check
    print(f"  [head] checks: {check}")
    return ({pid: float(p) for pid, p in zip(ids, preds) if pid in test_pids_set}, head_source)

# ── statistical helpers ────────────────────────────────────────────────────────
def partial_spearman_rho(x, y, z):
    """Partial Spearman rho(x, y | z) via rank residuals."""
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz = rankdata(z).astype(float)
    rz_c = rz - rz.mean()
    def resid(r):
        r_c = r - r.mean()
        beta = np.dot(rz_c, r_c) / (np.dot(rz_c, rz_c) + 1e-30)
        return r_c - beta * rz_c
    ex, ey = resid(rx), resid(ry)
    denom = np.sqrt(np.dot(ex, ex) * np.dot(ey, ey))
    return float(np.dot(ex, ey) / denom) if denom > 1e-15 else float("nan")


def bootstrap_rho_ci(x, y, z, groups=None, n_boot=N_BOOT_A, seed=BOOT_SEED):
    """Percentile CI of the partial Spearman rho; resamples protein clusters if groups is given."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if groups is not None:
        labs, inv = np.unique(groups, return_inverse=True)
        mem = [np.flatnonzero(inv == k) for k in range(len(labs))]
    boots = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([mem[k] for k in rng.integers(0, len(mem), len(mem))])
        rho = partial_spearman_rho(x[idx], y[idx], z[idx])
        if np.isfinite(rho):
            boots.append(rho)
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))



# ── load all model predictions ─────────────────────────────────────────────────
print("\nExtracting per-model predictions via model head (or ridge fallback) ...")
arch_preds: dict[str, dict] = {}
arch_head_source: dict[str, str] = {}
MODEL_CFG = [("egnn", "primary"), ("gnina", "primary"), ("distance_gnn", "primary")]
for model_name, tag in MODEL_CFG:
    short = "dnn" if model_name == "distance_gnn" else model_name
    print(f"  {short} ...")
    _res = apply_model_head(model_name, tag)
    if isinstance(_res, tuple):
        arch_preds[short], arch_head_source[short] = _res
    else:
        arch_preds[short] = _res
        arch_head_source[short] = "ridge_fallback"
    print(f"    → {len(arch_preds[short])} predictions  (source: {arch_head_source[short]})")

# ── cross-model sanity: three independently trained encoders must NOT give (near-)identical predictions ─────────
# (an earlier version produced agreement rho identical to 5 digits for two models: a symptom of one head/embedding used twice)
PRED_CROSS = {}
_shorts = list(arch_preds)
for _i in range(len(_shorts)):
    for _j in range(_i + 1, len(_shorts)):
        a, b = _shorts[_i], _shorts[_j]
        common_ids = sorted(set(arch_preds[a]) & set(arch_preds[b]))
        xa = np.array([arch_preds[a][p] for p in common_ids]); xb = np.array([arch_preds[b][p] for p in common_ids])
        r_p = float(np.corrcoef(xa, xb)[0, 1]); r_s = float(np.corrcoef(rankdata(xa), rankdata(xb))[0, 1])
        mad = float(np.max(np.abs(xa - xb)))
        PRED_CROSS[f"{a}_vs_{b}"] = dict(pearson=round(r_p, 6), spearman=round(r_s, 6), max_abs_diff=round(mad, 6), n=len(common_ids))
        print(f"  prediction agreement {a} vs {b}: pearson={r_p:.5f} spearman={r_s:.5f} max|diff|={mad:.5g}")
        if (r_p > 0.999 or mad < 1e-6) and os.environ.get("ACCEPT_IDENTICAL_PREDS") != "1":
            raise SystemExit(f"{a} and {b} predictions are suspiciously identical (pearson={r_p:.6f}, max|diff|={mad:.3g}): "
                             "check that the two models' embeddings/checkpoints/heads are really different. "
                             "ACCEPT_IDENTICAL_PREDS=1 overrides (the agreement statistics are recorded in the output).")
_ea = np.load(EMB_DIR / "egnn_primary_test.npy"); _eb = np.load(EMB_DIR / "distance_gnn_primary_test.npy")
if _ea.shape == _eb.shape and np.allclose(_ea, _eb) and os.environ.get("ACCEPT_IDENTICAL_PREDS") != "1":
    raise SystemExit("EGNN and Distance-GNN primary test embeddings are identical arrays: extraction bug.")

# ── build agreement set ──────────────────────────────────────────────────────────────
print("\nBuilding agreement set ...")
core_ids = set(arch_preds.get("egnn", {})) & \
           set(arch_preds.get("gnina", {})) & \
           set(arch_preds.get("dnn", {}))

# pose success must be defined on the SAME pose as vina_score (config: top1_only). A best-of-N RMSD would not be.
RMSD_KEY = "top1_rmsd" if any("top1_rmsd" in r for r in vina.values()) else "best_rmsd"
if RMSD_KEY == "best_rmsd" and os.environ.get("ACCEPT_BEST_RMSD") != "1":
    raise SystemExit("vina_results.json has no 'top1_rmsd'. 'best_rmsd' may be the best of 20 poses, while vina_score is the "
                     "top-1 score and config.vina.top1_only is true. Confirm in the docking script that best_rmsd is the "
                     "top-1 pose RMSD (then set ACCEPT_BEST_RMSD=1) or write top1_rmsd.")
print(f"  pose-success RMSD key: {RMSD_KEY}")
vina_ok_ids = {pid for pid, r in vina.items()
               if r.get("error") is None
               and r.get(RMSD_KEY) is not None
               and r.get("vina_score") is not None}

h3_ids = sorted(core_ids & vina_ok_ids & set(cov.index))
n_h3   = len(h3_ids)
print(f"  core={len(core_ids)}  vina_docked={len(vina_ok_ids)}  agreement set={n_h3}")
assert n_h3 > 100, f"agreement set too small ({n_h3}); check inputs"

# arrays aligned to h3_ids
y_true      = np.array([split.loc[pid, AFF_COL]
                         for pid in h3_ids], dtype=np.float64)

# R² sanity check: head predictions vs ground-truth affinity labels
print("\nR\u00b2 check: head predictions vs affinity labels (sanity)")
for _sh in ("egnn", "gnina", "dnn"):
    _pm = arch_preds.get(_sh, {})
    _src = arch_head_source.get(_sh, "?")
    _pa = np.array([_pm.get(pid, float("nan")) for pid in h3_ids])
    _vld = np.isfinite(_pa) & np.isfinite(y_true)
    if _vld.sum() > 10:
        _yh = _pa[_vld]; _yt = y_true[_vld]
        _ss_res = ((_yt - _yh)**2).sum(); _ss_tot = ((_yt - _yt.mean())**2).sum()
        _r2 = float(1 - _ss_res/_ss_tot) if _ss_tot > 0 else float("nan")
        print(f"  {_sh}: R\u00b2={_r2:.4f} (source={_src}, n={int(_vld.sum())})")

vina_score  = np.array([vina[pid]["vina_score"]  for pid in h3_ids], dtype=np.float64)
neg_vina    = -vina_score                         # negate: higher = better docking
vina_rmsd   = np.array([vina[pid][RMSD_KEY]   for pid in h3_ids], dtype=np.float64)
lig_heavy   = np.array([cov.loc[pid, "ligand_heavy_atom_count"]
                         for pid in h3_ids], dtype=np.float64)
vina_success = vina_rmsd <= RMSD_THRESH           # True = pose success
h3_groups = al.fill_singleton_groups(np.array([get_group(p) for p in h3_ids]))   # unclustered -> singleton

print(f"  neg_vina: mean={neg_vina.mean():.2f}  std={neg_vina.std():.2f}")
print(f"  Vina pose success: {vina_success.sum()}/{n_h3} ({100*vina_success.mean():.1f}%)")


# ── agreement with the Vina score ─────────────────────────────────────────────
print("\n=== partial Spearman: predicted affinity vs -Vina score, given ligand heavy-atom count ===")
MODELS_AGREE = [("egnn", "egnn"), ("gnina", "gnina"), ("distance_gnn", "dnn")]
results = {}
for model_name, short in MODELS_AGREE:
    preds_map = arch_preds[short]
    preds = np.array([preds_map.get(pid, float("nan")) for pid in h3_ids])
    valid = np.isfinite(preds) & np.isfinite(neg_vina) & np.isfinite(lig_heavy)
    x, y, z = preds[valid], neg_vina[valid], lig_heavy[valid]
    rho = partial_spearman_rho(x, y, z)
    ci_lo, ci_hi = bootstrap_rho_ci(x, y, z, groups=h3_groups[valid])
    print(f"  {short}: n={int(valid.sum())}  rho={rho:.4f}  95% CI ({ci_lo:.4f}, {ci_hi:.4f})")
    results[short] = dict(partial_rho=round(rho, 6), ci_lo=round(ci_lo, 6), ci_hi=round(ci_hi, 6), n=int(valid.sum()),
                          n_boot=N_BOOT_A, bootstrap="protein cluster", head_source=arch_head_source[short])

# ── the Vina score against measured affinity and size, on the same complexes ───
def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
vina_context = dict(
    n=int(n_h3),
    spearman_negvina_affinity=round(spearman(neg_vina, y_true), 4),
    partial_spearman_negvina_affinity_given_size=round(partial_spearman_rho(neg_vina, y_true, lig_heavy), 4),
    spearman_negvina_heavy_atoms=round(spearman(neg_vina, lig_heavy), 4),
    n_pose_within_threshold=int(vina_success.sum()), rmsd_threshold_A=RMSD_THRESH)
print("\nVina score context:", vina_context)

out = dict(agreement=results, vina_context=vina_context,
           meta=dict(x="model_predicted_affinity_own_head", y="negated_vina_score", covariate="ligand_heavy_atom_count",
                     head_checks=HEAD_CHECKS, prediction_cross_model_agreement=PRED_CROSS, pose_rmsd_key=RMSD_KEY))
out_path = ROOT / "logs" / "docking_agreement.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved {out_path}")
