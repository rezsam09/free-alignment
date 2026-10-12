#!/usr/bin/env python
"""run_t3_anchors.py -- T3 + T4: finish EGNN and Distance-GNN anchor training, then gate,
invariance test, extraction, health check, and commit.

Usage:
    python run_t3_anchors.py [--root /path/to/pdbbind_data]

Safe to re-run: finished anchor checkpoints are detected (is_best=True) and skipped.
Primary and untrained checkpoints are NEVER retrained; they are only loaded.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--root", default=os.environ.get("PROJECT_ROOT",
    "/home/ec2-user/SageMaker/pdbbind_data"))
args = parser.parse_args()
ROOT = Path(args.root).resolve()
assert ROOT.exists(), f"Project root not found: {ROOT}"
os.chdir(ROOT)
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml as _yaml
from src import prereg, seeds as seedmod, embed_utils as eu

prereg.verify_lock()
cfg = _yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
GLOBAL_SEED = int(cfg["seeds"]["global"])
ANCHOR_SEED  = int(cfg["seeds"]["anchor"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE} | global seed: {GLOBAL_SEED} | anchor seed: {ANCHOR_SEED}")

# ---------------------------------------------------------------------------
# split + PDBbind index
# ---------------------------------------------------------------------------
split = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0)
split.index = split.index.astype(str).str.lower()
SPLIT_COL = "split"
AFF       = cfg["split"]["affinity_column"]
PDBBIND_DIR = Path(cfg["paths"]["pdbbind_dir"])

def build_index(base):
    idx = {}
    for b in [base] + [d for d in base.iterdir() if d.is_dir() and len(d.name) != 4]:
        for d in b.iterdir():
            if d.is_dir() and len(d.name) == 4:
                idx[d.name.lower()] = d
    return idx

IDX = build_index(PDBBIND_DIR)
print(f"split: {split[SPLIT_COL].value_counts().to_dict()} | PDBbind dirs: {len(IDX)}")

# ---------------------------------------------------------------------------
# graph construction (identical to notebook 2a)
# ---------------------------------------------------------------------------
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HALOGENS = {"F", "CL", "BR", "I"}
METALS   = {"ZN", "MG", "CA", "MN", "FE", "NA", "K", "CU", "NI", "CO", "CD", "HG", "MO"}
ELEMENT_BUCKETS = ["C", "N", "O", "S", "P", "HAL", "METAL", "OTHER"]
N_TYPES = len(ELEMENT_BUCKETS) * 2  # 16

def element_bucket(sym):
    s = sym.strip().upper()
    if s == "C":   return 0
    if s == "N":   return 1
    if s == "O":   return 2
    if s == "S":   return 3
    if s == "P":   return 4
    if s in HALOGENS: return 5
    if s in METALS:   return 6
    return 7

def node_type(sym, is_ligand):
    return element_bucket(sym) * 2 + (1 if is_ligand else 0)

def parse_pocket(pdb):
    out = []
    with open(pdb) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")): continue
            if line[17:20].strip() in ("HOH", "WAT", "H2O"): continue
            el = line[76:78].strip() if len(line) >= 78 else ""
            if not el:
                el = line[12:16].strip().lstrip("0123456789")[:1]
            if el.upper() in ("H", "D") or not el: continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError: continue
            out.append((el, x, y, z))
    return out

def parse_ligand(lig):
    path = str(lig)
    mol = (Chem.SDMolSupplier(path, removeHs=False, sanitize=False)[0]
           if path.endswith(".sdf")
           else Chem.MolFromMol2File(path, removeHs=False, sanitize=False))
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"unreadable ligand: {lig}")
    conf = mol.GetConformer()
    return [(a.GetSymbol(), *conf.GetAtomPosition(a.GetIdx()))
            for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

GRAPH_CACHE = ROOT / "data" / "graph_cache"
GRAPH_CACHE.mkdir(parents=True, exist_ok=True)

def get_graph(pid):
    cache = GRAPH_CACHE / f"{pid}.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        return d["z"], d["x"], d["y"]
    d = IDX.get(pid)
    if d is None: raise FileNotFoundError(f"no dir for {pid}")
    pocket = d / f"{pid}_pocket.pdb"
    lig = (d / f"{pid}_ligand.sdf" if (d / f"{pid}_ligand.sdf").exists()
           else d / f"{pid}_ligand.mol2")
    if not pocket.exists() or not lig.exists():
        raise FileNotFoundError(f"missing pocket/ligand for {pid}")
    prot = parse_pocket(pocket); lig_atoms = parse_ligand(lig)
    zs, xs = [], []
    for el, x, y, z in prot:   zs.append(node_type(el, False)); xs.append((x, y, z))
    for el, x, y, z in lig_atoms: zs.append(node_type(el, True));  xs.append((x, y, z))
    Z = torch.tensor(zs, dtype=torch.long)
    X = torch.tensor(xs, dtype=torch.float32)
    Y = float(split.loc[pid, AFF])
    torch.save({"z": Z, "x": X, "y": Y}, cache)
    return Z, X, Y

# precompute (idempotent: cached files are skipped)
print("Precomputing graph cache ...")
CACHED_IDS = {}
for sp in ("train", "val", "test"):
    ids = split.index[split[SPLIT_COL] == sp].tolist()
    ok, bad = [], []
    for pid in ids:
        try:
            get_graph(pid); ok.append(pid)
        except Exception as e:
            bad.append((pid, str(e)[:120]))
    if bad:
        prereg.log_exclusions(bad, stage="graph_build", root=ROOT)
    CACHED_IDS[sp] = ok
    print(f"  {sp}: {len(ok)} ok, {len(bad)} failed")

# ---------------------------------------------------------------------------
# dataset / model config
# ---------------------------------------------------------------------------
EGNN_HP = cfg["training"]["egnn"]
DGNN_HP = cfg["training"]["distance_gnn"]
CUTOFF_A = float(DGNN_HP.get("cutoff_A", 5.0))
N_RBF    = int(DGNN_HP.get("n_rbf", 16))
GATE_MARGIN = 0.10

MODEL_SPECS = {
    "egnn":         dict(cls=eu.EGNNAffinity, hp=EGNN_HP,
                         kwargs=dict(n_types=N_TYPES, hidden=EGNN_HP["hidden"],
                                     n_layers=EGNN_HP["n_layers"])),
    "distance_gnn": dict(cls=eu.DistanceGNN,  hp=DGNN_HP,
                         kwargs=dict(n_types=N_TYPES, hidden=DGNN_HP["hidden"],
                                     n_layers=DGNN_HP["n_layers"],
                                     n_rbf=N_RBF, cutoff=CUTOFF_A)),
}

class ComplexDataset(Dataset):
    def __init__(self, ids): self.ids = list(ids)
    def __len__(self):  return len(self.ids)
    def __getitem__(self, i):
        pid = self.ids[i]; z, x, y = get_graph(pid); return pid, z, x, y

def collate(batch):
    pids, zs, xs, ys, bs = [], [], [], [], []
    for gi, (pid, z, x, y) in enumerate(batch):
        pids.append(pid); zs.append(z); xs.append(x); ys.append(y)
        bs.append(torch.full((z.shape[0],), gi, dtype=torch.long))
    z = torch.cat(zs); x = torch.cat(xs); b = torch.cat(bs)
    y = torch.tensor(ys, dtype=torch.float32)
    ei = eu.radius_edges(x, b, cutoff=CUTOFF_A)
    return pids, z, x, ei, b, len(batch), y

# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues, total, n = [], [], 0.0, 0
    for _, z, x, ei, b, ng, y in loader:
        z, x, ei, b, y = z.to(DEVICE), x.to(DEVICE), ei.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
        out = model(z, x, ei, b, ng)
        total += nn.functional.mse_loss(out, y, reduction="sum").item(); n += ng
        preds.append(out.cpu().numpy()); trues.append(y.cpu().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    r = float(pearsonr(preds, trues)[0]) if len(preds) > 1 else float("nan")
    return dict(val_loss=total / max(n, 1), pearson_r=r,
                rmse=float(np.sqrt(np.mean((preds - trues) ** 2))), n=n)

# ---------------------------------------------------------------------------
# train_one (identical to notebook 2a; adds is_best on every improvement)
# ---------------------------------------------------------------------------
CKPT_DIR = ROOT / "data" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def train_one(name, seed, untrained=False, tag=""):
    spec = MODEL_SPECS[name]; hp = spec["hp"]
    ckpt = CKPT_DIR / f"{name}_{tag or seed}.pt"
    seedmod.seed_everything(seed)
    model = spec["cls"](**spec["kwargs"]).to(DEVICE)

    if untrained:
        torch.save(dict(epoch=0, model=model.state_dict(), optimizer=None,
                        best_val_loss=None, patience_ctr=0, seed=seed, untrained=True), ckpt)
        print(f"[{name}/{tag or seed}] untrained anchor saved -> {ckpt.name}")
        return model, {}

    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    start, best_loss, patience_ctr, history = 0, float("inf"), 0, []
    best_ckpt = CKPT_DIR / f"{name}_{tag or seed}_best.pt"

    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE)
        # If already finished with best weights, skip re-training (Issue 9)
        if state.get("is_best") and state.get("patience_ctr", 0) >= hp["early_stopping_patience"]:
            model.load_state_dict(state["model"])
            print(f"[{name}/{tag or seed}] already finished (is_best=True, "
                  f"epoch {state['epoch']}, best_val_loss={state['best_val_loss']:.4f}) -- skipping")
            return model, {"history": state.get("history", []),
                           "best_val_loss": state["best_val_loss"]}
        model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            opt.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        best_loss = state.get("best_val_loss", float("inf"))
        patience_ctr = state.get("patience_ctr", 0)
        history = state.get("history", [])
        print(f"[{name}/{tag or seed}] resuming from epoch {start} (best val_loss={best_loss:.4f})")

    # Restore best_state from disk if a prior run was interrupted after an improvement
    best_state = None
    if best_ckpt.exists():
        bs = torch.load(best_ckpt, map_location=DEVICE)
        best_state = bs["model"]
        best_loss = min(best_loss, bs.get("best_val_loss", best_loss))
        print(f"[{name}/{tag or seed}] restored best_state from {best_ckpt.name} "
              f"(best_val_loss={bs.get('best_val_loss', '?')})")

    train_loader = DataLoader(ComplexDataset(CACHED_IDS["train"]), batch_size=hp["batch_size"],
                              shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(ComplexDataset(CACHED_IDS["val"]),   batch_size=hp["batch_size"],
                              shuffle=False, collate_fn=collate, num_workers=0)

    for epoch in range(start, hp["max_epochs"]):
        model.train()
        t0 = time.time()
        for _, z, x, ei, b, ng, y in train_loader:
            z, x, ei, b, y = z.to(DEVICE), x.to(DEVICE), ei.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            nn.functional.mse_loss(model(z, x, ei, b, ng), y).backward()
            opt.step()
        vm = evaluate(model, val_loader)
        history.append(dict(epoch=epoch, **vm, sec=round(time.time() - t0, 1)))
        improved = vm["val_loss"] < best_loss - 1e-6
        if improved:
            best_loss = vm["val_loss"]; patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # Write to disk immediately so crashes don't lose the best weights
            torch.save(dict(epoch=epoch, model=best_state,
                            best_val_loss=best_loss, seed=seed), best_ckpt)
        else:
            patience_ctr += 1
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                        best_val_loss=best_loss, patience_ctr=patience_ctr,
                        history=history, seed=seed, untrained=False), ckpt)
        if epoch % 5 == 0 or improved:
            print(f"[{name}/{tag or seed}] epoch {epoch}: val_loss={vm['val_loss']:.4f} "
                  f"r={vm['pearson_r']:.3f} (patience {patience_ctr}/{hp['early_stopping_patience']})")
        if patience_ctr >= hp["early_stopping_patience"]:
            print(f"[{name}/{tag or seed}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                        best_val_loss=best_loss, patience_ctr=patience_ctr,
                        history=history, seed=seed, untrained=False, is_best=True), ckpt)
        best_ckpt.unlink(missing_ok=True)
        print(f"[{name}/{tag or seed}] wrote is_best=True checkpoint")
    return model, {"history": history, "best_val_loss": best_loss}

# ---------------------------------------------------------------------------
# T3: train anchors (skip if already finished)
# ---------------------------------------------------------------------------
print("\n=== T3: anchor training ===")
MODELS = {}
for name in ("egnn", "distance_gnn"):
    print(f"\n--- {name}, anchor seed {ANCHOR_SEED} ---")
    m, h = train_one(name, ANCHOR_SEED, untrained=False, tag="anchor")
    MODELS[(name, "anchor")] = m

# train primary + untrained from scratch if not transferred from Drive
for name in ("egnn", "distance_gnn"):
    for tag, is_ut in (("primary", False), ("untrained", True)):
        ckpt = CKPT_DIR / f"{name}_{tag}.pt"
        if not ckpt.exists():
            print(f"\n--- {name}, {tag} (training from scratch, not in Drive) ---")
            m, h = train_one(name, GLOBAL_SEED, untrained=is_ut, tag=tag)
            MODELS[(name, tag)] = m

# load primaries and untrained without re-training (Issue 9)
for name in ("egnn", "distance_gnn"):
    for tag in ("primary", "untrained"):
        ckpt = CKPT_DIR / f"{name}_{tag}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing {ckpt.name} -- transfer from Drive first.")
        state = torch.load(ckpt, map_location=DEVICE)
        spec = MODEL_SPECS[name]
        seedmod.seed_everything(state.get("seed", GLOBAL_SEED))
        m = spec["cls"](**spec["kwargs"]).to(DEVICE)
        m.load_state_dict(state["model"])
        MODELS[(name, tag)] = m
        print(f"[{name}/{tag}] loaded from {ckpt.name} "
              f"(epoch {state.get('epoch')}, is_best={state.get('is_best', 'MISSING')})")

# ---------------------------------------------------------------------------
# T4a: Table 1 gate
# ---------------------------------------------------------------------------
print("\n=== T4a: Table 1 gate ===")
TABLE1 = ROOT / "logs" / "table1_performance.tsv"
val_loader = DataLoader(ComplexDataset(CACHED_IDS["val"]), batch_size=32,
                        shuffle=False, collate_fn=collate, num_workers=0)
rows = []
for name in ("egnn", "distance_gnn"):
    metrics = {}
    for tag in ("primary", "untrained", "anchor"):
        if (name, tag) not in MODELS: continue
        m = evaluate(MODELS[(name, tag)], val_loader)
        metrics[tag] = m
        rows.append(dict(model=name, seed_tag=tag, split="val", **m))
        print(f"{name:14s} {tag:10s} val_loss={m['val_loss']:.4f} "
              f"r={m['pearson_r']:.3f} rmse={m['rmse']:.3f}")
    if "primary" in metrics:
        r_tr = metrics["primary"]["pearson_r"]
        r_un = metrics.get("untrained", {}).get("pearson_r", float("nan"))
        gate = (r_tr >= 0.4) and (r_tr - r_un >= GATE_MARGIN)
        tag_label = "PASS" if gate else "FAIL"
        print(f"  GATE ({name}, primary): r={r_tr:.3f} vs untrained={r_un:.3f} -> {tag_label}")
        if not gate:
            raise RuntimeError(f"Gate FAILED for {name} primary -- "
                               "check checkpoint and re-run.")
    if "anchor" in metrics:
        r_an = metrics["anchor"]["pearson_r"]
        r_un = metrics.get("untrained", {}).get("pearson_r", float("nan"))
        gate_a = (r_an >= 0.4) and (r_an - r_un >= GATE_MARGIN)
        print(f"  GATE ({name}, anchor):  r={r_an:.3f} -> {'PASS' if gate_a else 'FAIL'}")

table1 = pd.DataFrame(rows)
header = not TABLE1.exists()
table1.to_csv(TABLE1, sep="\t", mode="a", header=header, index=False)
print(f"appended to {TABLE1.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# T4b: invariance test (float64)
# ---------------------------------------------------------------------------
print("\n=== T4b: invariance test ===")
for name in ("egnn", "distance_gnn"):
    model = MODELS[(name, "primary")]
    spec = MODEL_SPECS[name]
    model64 = spec["cls"](**spec["kwargs"]).double()
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    model64.eval()
    g = torch.Generator().manual_seed(42)
    sizes = (30, 45, 62, 51)
    zs, xs, bs = [], [], []
    for i, n in enumerate(sizes):
        zs.append(torch.randint(0, N_TYPES, (n,), generator=g))
        xs.append(torch.randn(n, 3, generator=g, dtype=torch.float64) * 4.0)
        bs.append(torch.full((n,), i, dtype=torch.long))
    z, x, b = torch.cat(zs), torch.cat(xs), torch.cat(bs)
    ei = eu.radius_edges(x, b, cutoff=CUTOFF_A)
    with eu.PreHeadCapture(model64, "readout.0") as cap:
        model64(z, x, ei, b, len(sizes)); e1 = cap.value.clone()
    R = eu.random_rotation(seed=99, dtype=torch.float64)
    t = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    x2 = x @ R.T + t
    ei2 = eu.radius_edges(x2, b, cutoff=CUTOFF_A)
    with eu.PreHeadCapture(model64, "readout.0") as cap:
        model64(z, x2, ei2, b, len(sizes)); e2 = cap.value.clone()
    diff = (e1 - e2).abs().max().item()
    status = "PASS" if diff < 1e-10 else "FAIL"
    print(f"[{name}] rotation+translation invariance float64: max|diff|={diff:.2e} -> {status}")
    if status == "FAIL":
        raise RuntimeError(f"Invariance check FAILED for {name}")

# ---------------------------------------------------------------------------
# T4c: extraction
# ---------------------------------------------------------------------------
print("\n=== T4c: extraction ===")
EMB_DIR = ROOT / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)
manifest_path = ROOT / "logs" / "embeddings_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

@torch.no_grad()
def extract(name, model, split_name):
    spec = MODEL_SPECS[name]
    model64 = spec["cls"](**spec["kwargs"]).double()
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    model64.eval()
    ids = CACHED_IDS[split_name]
    loader = DataLoader(ComplexDataset(ids), batch_size=1, shuffle=False,
                        collate_fn=collate, num_workers=0)
    rows, ordered = [], []
    for pids, z, x, ei, b, ng, _ in loader:
        x = x.double()
        with eu.PreHeadCapture(model64, "readout.0") as cap:
            model64(z, x, ei, b, ng)
            rows.append(cap.value.cpu().numpy()[0])
        ordered.append(pids[0])
    E = np.stack(rows).astype(np.float64)
    assert E.shape[0] == len(ordered) == len(CACHED_IDS[split_name])
    return E, ordered

for (name, tag), model in MODELS.items():
    for sp in ("train", "val", "test"):
        E, ids_order = extract(name, model, sp)
        npy = EMB_DIR / f"{name}_{tag}_{sp}.npy"
        ids_f = EMB_DIR / f"{name}_{tag}_{sp}_ids.txt"
        np.save(npy, E)
        ids_f.write_text("\n".join(ids_order))
        sha = hashlib.sha256(E.tobytes()).hexdigest()
        key = f"{name}/{tag}/{sp}"
        manifest[key] = dict(shape=list(E.shape), sha256=sha,
                             npy=str(npy.relative_to(ROOT)),
                             ids=str(ids_f.relative_to(ROOT)))
        print(f"{key:30s} shape={E.shape} sha256={sha[:12]}")

manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"wrote {manifest_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# T4d: health check
# ---------------------------------------------------------------------------
print("\n=== T4d: health check ===")
cov_cols = ["complex_atom_count", "ligand_heavy_atom_count"]
health = {}
for (name, tag) in MODELS:
    E = np.load(EMB_DIR / f"{name}_{tag}_test.npy")
    ids = (EMB_DIR / f"{name}_{tag}_test_ids.txt").read_text().splitlines()
    Z = split.loc[ids, cov_cols].to_numpy()
    h = eu.embedding_sanity(E, Z)
    health[f"{name}/{tag}/test"] = h
    print(f"{name}/{tag}: PR={h['participation_ratio']:.2f} "
          f"dead={h['dead_dim_frac']:.1%} cov_R2={h.get('frac_var_explained_by_covariates', float('nan')):.3f}")

(ROOT / "logs" / "embedding_health_2a.json").write_text(json.dumps(health, indent=2))

# ---------------------------------------------------------------------------
# T4e: commit
# ---------------------------------------------------------------------------
print("\n=== T4e: commit ===")
def sh(*a):
    r = subprocess.run(a, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()

sh("git", "config", "user.email", "t3runner@local")
sh("git", "config", "user.name",  "t3runner")
sh("git", "add",    "logs", "data/splits")
rc, msg = sh("git", "commit", "-m",
             "T3/T4: EGNN + Distance-GNN anchors trained, gated, extracted")
print(msg[:400] or "(nothing to commit)")

print("\n=== T3/T4 complete ===")
print("Next: run run_t5_untrained_inits.py, then switch instance to ml.m5.4xlarge.")
