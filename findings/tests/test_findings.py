"""Unit tests (run inside a fixture project: PROJECT_ROOT=<fixture> python findings/tests/test_findings.py ; tests/ is copied by run_e2e.sh)."""
import os, sys, io, tempfile, json, shutil
import numpy as np
ROOT = os.environ["PROJECT_ROOT"]; sys.path.insert(0, os.path.join(ROOT, "findings")); sys.path.insert(0, ROOT)
import fcommon as fc, randnets as rn, fprobe as fp, sweeplib as sl, build_graphs as bg, a4_ablation as a4, a3_floor_model as a3, c3_output_agreement as c3
al = fc.al
OK = []
def check(name, cond, info=""):
    assert cond, f"FAIL {name} {info}"; OK.append(name); print("ok  ", name)

rng = np.random.default_rng(0)
def cloud(n=70):
    return rng.normal(size=(n, 3)) * 4, rng.integers(0, 6, n), np.arange(n) < 9
items = [cloud(int(rng.integers(40, 90))) for _ in range(5)]
CFG = dict(width=16, depth=3, radius=5.0, gain=1.0, readout="mean", n_rbf=8, aggr="sum")
def emb(cfg, arch, it, seed=0, dt=np.float64): return rn.embed_all(it, cfg, arch, seed, 6, 1, dt)

# ---------- random networks
def rot():
    q, _ = np.linalg.qr(rng.normal(size=(3, 3))); return q * np.sign(np.linalg.det(q))
for arch in rn.ARCHS:
    for ro in ("mean", "sum", "max"):
        for ag in ("sum", "mean"):
            c = {**CFG, "readout": ro, "aggr": ag}; E0 = emb(c, arch, items)
            R, t = rot(), rng.normal(size=3) * 10
            E1 = emb(c, arch, [(x @ R.T + t, e, l) for x, e, l in items])
            check(f"rot/trans invariance {arch} {ro} {ag}", np.allclose(E0, E1, rtol=1e-8, atol=1e-8), np.abs(E0 - E1).max())
            perm = [rng.permutation(len(e)) for _, e, _ in items]
            E2 = emb(c, arch, [(x[p], e[p], l[p]) for (x, e, l), p in zip(items, perm)])
            check(f"atom-permutation invariance {arch} {ro} {ag}", np.allclose(E0, E2, rtol=1e-8, atol=1e-8))
    check(f"deterministic {arch}", np.array_equal(emb(CFG, arch, items), emb(CFG, arch, items)))
    check(f"seeds differ {arch}", not np.allclose(emb(CFG, arch, items, 0), emb(CFG, arch, items, 1)))
    one = np.vstack([emb(CFG, arch, [it]) for it in items])
    check(f"batch composition invariance {arch}", np.allclose(one, emb(CFG, arch, items), atol=1e-10))
    check(f"float32 close to float64 {arch}", np.allclose(emb(CFG, arch, items, dt=np.float32), emb(CFG, arch, items), rtol=1e-3, atol=1e-3))
check("egnn ignores n_rbf (shared cache key)", rn.cfg_id({**CFG, "n_rbf": 8}, "egnn") == rn.cfg_id({**CFG, "n_rbf": 32}, "egnn") and np.allclose(emb({**CFG, "n_rbf": 8}, "egnn", items), emb({**CFG, "n_rbf": 32}, "egnn", items)))
check("distance_gnn uses n_rbf", rn.cfg_id({**CFG, "n_rbf": 8}, "distance_gnn") != rn.cfg_id({**CFG, "n_rbf": 32}, "distance_gnn"))
check("geometry matters (coordinate noise changes egnn)", not np.allclose(emb(CFG, "egnn", items), emb(CFG, "egnn", [(x + rng.normal(size=x.shape) * 2, e, l) for x, e, l in items])))

def naive(cfg, arch, it, seed):
    """Per-node / per-edge python loops, mirrors the documented equations."""
    P = rn.make_params(cfg, arch, seed, 7, 1, np.float64); xyz, el, lig = it; n = len(el)
    F = np.zeros((n, 7)); F[np.arange(n), el] = 1; F[:, 6] = lig; h = F @ P["W_in"]; x = xyz.copy(); r = cfg["radius"]
    nb = [[j for j in range(n) if j != i and np.linalg.norm(xyz[i] - xyz[j]) <= r] for i in range(n)]
    cen = np.linspace(0, r, cfg["n_rbf"]); gam = 0.5 / (cen[1] - cen[0]) ** 2; d0 = lambda i, j: np.linalg.norm(xyz[i] - xyz[j])
    for L in P["layers"]:
        M = {}; hn = h.copy(); xn = x.copy()
        for i in range(n):
            for j in nb[i]:
                e = np.array([np.sum((x[i] - x[j]) ** 2) / r ** 2]) if arch == "egnn" else np.exp(-gam * (d0(i, j) - cen) ** 2)
                pre = h[i] @ L["Wi"] + h[j] @ L["Wj"] + e @ L["We"]
                M[(i, j)] = rn.silu(rn.silu(pre) @ L["W2"])
        for i in range(n):
            agg = sum((M[(i, j)] for j in nb[i]), np.zeros(cfg["width"]))
            if cfg["aggr"] == "mean": agg = agg / max(len(nb[i]), 1)
            hn[i] = h[i] + rn.silu(np.concatenate([h[i], agg]) @ L["Wh1"]) @ L["Wh2"]
            if arch == "egnn" and nb[i]:
                xn[i] = x[i] + sum((x[i] - x[j]) * np.tanh(M[(i, j)] @ L["wx"]) for j in nb[i]) / len(nb[i])
        h, x = hn, xn
    return {"mean": h.mean(0), "sum": h.sum(0), "max": h.max(0)}[cfg["readout"]]
for arch in rn.ARCHS:
    for ag, ro in (("sum", "mean"), ("mean", "max"), ("sum", "sum")):
        c = {**CFG, "depth": 2, "aggr": ag, "readout": ro, "width": 8}
        ref = naive(c, arch, items[0], 3); got = rn.embed_all([items[0]], c, arch, 3, 6, 1, np.float64)[0]
        check(f"vectorised == naive loops {arch} {ag} {ro}", np.allclose(ref, got, rtol=1e-9, atol=1e-9), np.abs(ref - got).max())
try:
    rn.embed_all(items, {**CFG, "gain": 1e6, "depth": 6}, "egnn", 0, 6, 1); bad = False
except fc.FindingsError: bad = True
check("non-finite embedding raises", bad)

# ---------- CKA / kNN
X = rng.normal(size=(60, 7)); Y = X @ rng.normal(size=(7, 5)) + .5 * rng.normal(size=(60, 5))
gs = fc.GramStats(X).cka(fc.GramStats(Y))
check("GramStats == alignlib.cka_feature", abs(gs - al.cka_feature(X, Y)) < 1e-10)
check("GramStats == Gram-based debiased CKA", abs(gs - al.cka_debiased(al.gram(X), al.gram(Y))) < 1e-10)
a_ = sl.knn_idx(X, 6); b_ = sl.knn_idx(Y, 6)
check("mutual kNN == alignlib.mutual_knn", abs(sl.mutual(a_, b_, 6) - al.mutual_knn(X, Y, 6, center=True)[1]) < 1e-12)
check("prep PCA reduces width", sl.prep(rng.normal(size=(50, 30)), 8).shape == (50, 8) and sl.prep(rng.normal(size=(50, 6)), 8).shape == (50, 6))

# ---------- probes / metrics
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import Ridge
y = rng.normal(size=80); p = y + rng.normal(size=80); yb = (rng.random(80) < .4).astype(float); sb = rng.normal(size=80) + yb; sb[:10] = 0.0
check("r2_w == sklearn", abs(fp.r2_w(y, p) - r2_score(y, p)) < 1e-12)
inv, nu = fp.auc_prep(sb); check("auc_w == sklearn (with ties)", abs(fp.auc_w(yb, inv, nu) - roc_auc_score(yb, sb)) < 1e-12)
w = rng.integers(0, 4, 80).astype(float); idx = np.repeat(np.arange(80), w.astype(int))
check("weighted R2 == duplicated rows", abs(fp.r2_w(y, p, w) - fp.r2_w(y[idx], p[idx])) < 1e-12)
check("weighted AUC == duplicated rows", abs(fp.auc_w(yb, inv, nu, w) - roc_auc_score(yb[idx], sb[idx])) < 1e-12)
Xr = rng.normal(size=(50, 6)); yr = Xr @ rng.normal(size=6) + rng.normal(size=50)
check("SVD ridge == sklearn Ridge", np.allclose(fp.ridge_solver(Xr, yr)(Xr[:9], 3.0), Ridge(alpha=3.0).fit(Xr, yr).predict(Xr[:9])))
n, ng = 400, 80; grp = np.repeat(np.arange(ng), n // ng)
Xc = np.eye(ng)[grp] @ rng.normal(size=(ng, 20)); yc = np.repeat(rng.normal(size=ng), n // ng)       # y is a function of the CLUSTER only
fl = fp.folds_for(grp, 5)
for tr, te in fl: assert not set(grp[tr]) & set(grp[te])
check("folds are grouped (no cluster in train and test)", True)
Pg = fp.oof_ridge(Xc, yc, grp, [.1, 1, 10], 5, 3, fl); Pu = fp.oof_ridge(Xc, yc, np.arange(n), [.1, 1, 10], 5, 3)
check("LEAKAGE: grouped CV cannot memorise cluster identity", fp.r2_w(yc, Pg) < 0.1, fp.r2_w(yc, Pg))
check("LEAKAGE control: ungrouped CV does (test is sensitive)", fp.r2_w(yc, Pu) > 0.3, fp.r2_w(yc, Pu))
Xs = rng.normal(size=(400, 10)); ys = Xs[:, 0] * 2 + rng.normal(size=400)
check("probe recovers planted linear signal", fp.r2_w(ys, fp.oof_ridge(Xs, ys, grp, [.1, 1, 10], 5, 3)) > 0.6)
yn = rng.normal(size=400); check("probe ~0 R2 on noise", fp.r2_w(yn, fp.oof_ridge(Xs, yn, grp, [.1, 1, 10], 5, 3)) < 0.05)
null = fp.shuffled_null(Xs, ys, grp, fp.Scorer(ys, "continuous"), [.1, 1, 10], 5, 3, fp.folds_for(grp, 5), 8, 0)
check("shuffled-target null centred near/below 0", null.mean() < 0.03 and null.max() < 0.15, null)
ybin = (Xs[:, 0] + .5 * rng.normal(size=400) > 0).astype(float); sc = fp.Scorer(ybin, "binary")
check("binary probe AUROC high on planted signal", sc(fp.oof_ridge(Xs, ybin, grp, [.1, 1, 10], 5, 3)) > 0.85)
ctl = fp.control_target(Xs, "continuous", 0.5, 20, 0)
check("control target is a function of input cells", len(np.unique(ctl)) <= 20 and ctl.shape == (400,))

# ---------- parsers
sdf = """lig
 x

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.5000    0.0000    0.0000 N   0  3  0  0  0  0  0  0  0  0  0  0
    0.0000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  1  5  1  0
M  END
$$$$
"""
d = tempfile.mkdtemp(); open(f"{d}/a.sdf", "w").write(sdf)
xyz, el, ch, heavy, bonds = bg.parse_sdf(f"{d}/a.sdf")
check("SDF: H removed, 4 heavy atoms, charge from atom block", len(el) == 4 and ch == 1 and el[3] == "N")
check("rotatable bonds: butane-like chain -> 1", bg.rotatable_bonds(heavy, bonds) == 1)
def rb(n, edges): return bg.rotatable_bonds(list(range(n)), [(a, b, 1) for a, b in edges])
check("rotatable: cyclohexane-like ring -> 0", rb(6, [(i, (i + 1) % 6) for i in range(6)]) == 0)
check("rotatable: ring-bond-ring (biphenyl-like) -> 1", rb(8, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (2, 6), (5, 7)]) == 1 + 0 + 0 + 0)
def _l(rec, i, nm, res, x, e): return "%-6s%5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (rec, i, nm, res, 1, x, 2.0, 3.0, e)
open(f"{d}/p.pdb", "w").write("\n".join([_l("ATOM", 1, "N", "ALA", 1.0, "N"), _l("ATOM", 2, "HB1", "ALA", 1.0, "H"), _l("HETATM", 3, "O", "HOH", 0.0, "O"), _l("ATOM", 4, "CA", "ALA", 2.0, "C")]) + "\n")
px, pe, miss, tot = bg.parse_pdb(f"{d}/p.pdb")
check("PDB: waters and H skipped, elements capitalised", pe == ["N", "C"] and px.shape == (2, 3) and tot == 3 and miss == 0, (pe, tot))

# ---------- ablations
plan = fc.load_plan(); ids5 = ["a", "b", "c", "d", "e"]
for kind in ("shuffle_types", "shuffle_types_within_role", "random_coords", "none"):
    A = a4.ablate(items, ids5, kind, 1); A2 = a4.ablate(items, ids5, kind, 1)
    check(f"ablation {kind}: deterministic", all(np.array_equal(x[0], y_[0]) and np.array_equal(x[1], y_[1]) for x, y_ in zip(A, A2)))
    for (x0, e0, l0), (x1, e1, l1) in zip(items, A):
        if kind in ("shuffle_types", "shuffle_types_within_role"):
            assert np.array_equal(x0, x1) and np.array_equal(l0, l1) and sorted(e0) == sorted(e1)
        if kind == "shuffle_types_within_role": assert sorted(e0[l0]) == sorted(e1[l1])
        if kind == "random_coords":
            assert np.array_equal(e0, e1) and abs(np.sqrt(((x0 - x0.mean(0)) ** 2).sum(1).mean()) - np.sqrt(((x1 - x1.mean(0)) ** 2).sum(1).mean())) / np.sqrt(((x0 - x0.mean(0)) ** 2).sum(1).mean()) < 0.25
    check(f"ablation {kind}: invariants kept", True)
check("shuffle_types actually changes types", any(not np.array_equal(a[1], b[1]) for a, b in zip(items, a4.ablate(items, ids5, "shuffle_types", 1))))

# ---------- sweep grid / floor model helpers
cf = sl.make_configs(plan); idsc = [rn.cfg_id(c) for c in cf]
check("grid: base first, ids unique", idsc[0] == rn.cfg_id(plan["sweep"]["base"]) and len(set(idsc)) == len(idsc), len(cf))
check("grid: deterministic", idsc == [rn.cfg_id(c) for c in sl.make_configs(plan)])
Xlin = rng.normal(size=(40, 4)); yl = Xlin[:, 0] + .1 * rng.normal(size=40)
Pcv = a3.cv_predict(Xlin, yl, 1.0, 5, 3, 0); check("floor-model CV: every config held out in every repeat", np.isfinite(Pcv).all() and a3.r2(yl, Pcv.mean(0)) > 0.8)
check("floor-model: R2 on noise <= ~0", a3.r2(yl, a3.cv_predict(rng.normal(size=(40, 4)), yl, 1.0, 5, 3, 0).mean(0)) < 0.1)

# ---------- agreement stats
a = rng.normal(size=100); b = a + rng.normal(size=100); w1 = np.ones(100)
check("wcorr == numpy corrcoef", abs(c3.wcorr(a, b, w1) - np.corrcoef(a, b)[0, 1]) < 1e-12)
zz = rng.normal(size=100); aa, bb = a + zz, b + zz
pc = (np.corrcoef(aa, bb)[0, 1] - np.corrcoef(aa, zz)[0, 1] * np.corrcoef(bb, zz)[0, 1]) / np.sqrt((1 - np.corrcoef(aa, zz)[0, 1] ** 2) * (1 - np.corrcoef(bb, zz)[0, 1] ** 2))
check("partial corr formula", abs(c3.partial(aa, bb, zz, w1) - pc) < 1e-12)
print(f"\n{len(OK)} unit checks passed")
