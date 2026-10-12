"""make_fixture_findings.py <dest> <base_dir>  -- SYNTHETIC project for the Findings package (meaningless results; plants known signals so the pipeline can be checked):
 builds on the base-project fixture (real prereg lock), writes synthetic PDBbind pocket/ligand files, graph-informed embeddings
   untrained = f(size-like geometry) + noise   (so a high floor is explained by kernels)
   trained   = f(size-like geometry) + latent affinity factor  (so 'affinity' has a real training gain, size targets do not)
 plus a SMALL plan (few configs/boots) so the e2e run takes minutes."""
import os, sys, shutil, subprocess, json
dest, base = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); here = os.path.dirname(os.path.abspath(__file__)); pkg = os.path.dirname(here)
subprocess.run([sys.executable, os.path.join(base, "tests", "make_fixture.py"), dest], check=True, cwd=base)
os.chdir(dest); sys.path.insert(0, dest)
import numpy as np, pandas as pd, yaml
from scipy.spatial import cKDTree
rng = np.random.default_rng(5)
sp = pd.read_csv("data/splits/split_v1.tsv", sep="\t", index_col=0)
ids = {s: sp.index[sp.new_split == s].tolist() for s in ("train", "test")}
EL = ["C", "N", "O", "S", "P", "F", "Cl"]; PR = [.6, .12, .15, .04, .02, .04, .03]
def fmt_pdb(i, x, e, k):
    return "ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" % (i, e.upper()[:4], "ALA", k, x[0], x[1], x[2], e.upper())
S = {}
os.makedirs("PDBbind_2020", exist_ok=True)
for pid in ids["train"] + ids["test"]:
    nl, npk = int(rng.integers(8, 26)), int(rng.integers(60, 130)); sc = rng.uniform(.8, 1.4)
    L = rng.normal(size=(nl, 3)) * 2.0; Pk = rng.normal(size=(npk, 3)) * 6.5 * sc
    el_l = rng.choice(EL, nl, p=PR); el_p = rng.choice(EL, npk, p=PR)
    d = f"PDBbind_2020/{pid}"; os.makedirs(d, exist_ok=True)
    lines = [fmt_pdb(i + 1, Pk[i], el_p[i], i // 8 + 1) for i in range(npk)]
    lines.insert(3, "HETATM%5d  O   HOH A 999    %8.3f%8.3f%8.3f  1.00  0.00           O" % (9999, 0, 0, 0))     # water, must be skipped
    lines.insert(5, fmt_pdb(8888, Pk[0], "H", 3))                                                                   # hydrogen, must be skipped
    open(f"{d}/{pid}_pocket.pdb", "w").write("\n".join(lines) + "\nEND\n")
    nH = 2; bonds = [(i, i + 1, 1) for i in range(nl - 1)] + ([(0, 4, 1)] if nl > 6 else [])
    chg = int(rng.integers(0, 3) == 0)
    sdf = ["lig", "  fixture", "", "%3d%3d  0  0  0  0  0  0  0  0999 V2000" % (nl + nH, len(bonds))]
    for i in range(nl): sdf.append("%10.4f%10.4f%10.4f %-3s%2d%3d  0  0  0  0  0  0  0  0  0  0" % (*L[i], el_l[i], 0, 3 if (chg and i == 1) else 0))
    for i in range(nH): sdf.append("%10.4f%10.4f%10.4f %-3s 0  0  0  0  0  0  0  0  0  0  0  0" % (*(L[i] + 1.0), "H"))
    for a, b, t in bonds: sdf.append("%3d%3d%3d  0" % (a + 1, b + 1, t))
    if chg: sdf.append("M  CHG  1%4d%4d" % (2, 1))
    sdf += ["M  END", "$$$$"]; open(f"{d}/{pid}_ligand.sdf", "w").write("\n".join(sdf) + "\n")
    cont = sum(len(x) for x in cKDTree(Pk).query_ball_point(L, 4.5))
    S[pid] = (nl, npk + nl, cont)
cov = pd.DataFrame({p: dict(ligand_heavy_atom_count=S[p][0], complex_atom_count=S[p][1]) for p in S}).T
trm = cov.loc[ids["train"]]; z = (cov - trm.mean()) / trm.std()
old = pd.read_csv("data/covariates.tsv", sep="\t", index_col=0); old.index.name = "pdb_id"
new = old.copy(); new.loc[z.index, ["ligand_heavy_atom_count", "complex_atom_count"]] = z[["ligand_heavy_atom_count", "complex_atom_count"]].values
new.to_csv("data/covariates.tsv", sep="\t")
geo = {p: np.array([S[p][0], S[p][1], S[p][2]], float) for p in S}
G = {s: np.array([geo[p] for p in ids[s]]) for s in ids}; mu, sd = G["train"].mean(0), G["train"].std(0)
G = {s: (G[s] - mu) / sd for s in G}
lat = {s: rng.normal(size=len(ids[s])) for s in ids}
for s in ids: sp.loc[ids[s], "value"] = 0.3 * G[s][:, 0] + lat[s] + 0.3 * rng.normal(size=len(ids[s]))
sp.to_csv("data/splits/split_v1.tsv", sep="\t")
dims = {"egnn": 64, "distance_gnn": 64, "gnina": 300}
Wg = {m: rng.normal(size=(3, d)) for m, d in dims.items()}; Wl = {m: rng.normal(size=(1, d)) for m, d in dims.items()}
E = "data/embeddings/"
def wr(m, tag, s, A): np.save(f"{E}{m}_{tag}_{s}.npy", A); open(f"{E}{m}_{tag}_{s}_ids.txt", "w").write("\n".join(ids[s]))
for m, d in dims.items():
    for s in ids:
        n = len(ids[s])
        for tag in ["primary", "anchor"]: wr(m, tag, s, G[s] @ Wg[m] + 1.5 * lat[s][:, None] @ Wl[m] + .5 * rng.normal(size=(n, d)))
        for tag in ["untrained"] + [f"untrained_init{i}" for i in range(2, 6)]: wr(m, tag, s, 0.4 * G[s] @ Wg[m] + 1.0 * rng.normal(size=(n, d)))

te = ids["test"]
# findings package + SMALL plan
shutil.copytree(pkg, os.path.join(dest, "findings"), ignore=shutil.ignore_patterns("__pycache__", "tests"))
plan = yaml.safe_load(open("findings/plan/findings_plan.yaml"))
plan["n_boot"] = 40
plan["sweep"].update(n_seeds=3, n_random=20, vary=dict(width=[32], depth=[2, 4], radius=[4.0, 6.0], gain=[0.5], readout=["sum", "max"], n_rbf=[8], aggr=["mean"]), n_jobs=2)
plan["ablation"]["n_boot"] = 20
plan["floor_model"].update(cv_repeats=4, n_boot=100, min_configs=20)
plan["probe"].update(n_perm=5, mlp=dict(hidden=8, max_iter=60, alpha=0.001, seed=0), dims=[16, 64, "full"], min_n=100)
yaml.safe_dump(plan, open("findings/plan/findings_plan.yaml", "w"), sort_keys=False)
# re-lock the base config with the new split file (fixture only)
from src import prereg
os.remove("config/config.lock.json"); prereg.lock({"split": "data/splits/split_v1.tsv"}, root=dest)
print("fixture ready:", dest)
