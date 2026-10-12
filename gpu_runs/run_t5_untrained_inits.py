#!/usr/bin/env python
"""run_t5_untrained_inits.py -- T5: build untrained inits 2-5 for EGNN, Distance-GNN, and GNINA.

Untrained init 1 was written by the primary training loop (egnn_untrained.pt /
distance_gnn_untrained.pt / gnina_untrained.pt) and is NEVER re-generated here.

Seeds:
  init 2 = 20260924, init 3 = 20260925, init 4 = 20260926, init 5 = 20260927

Each init: instantiate the model with the seed, DO NOT train, extract embeddings
for train/val/test splits, save .npy + ids file.

GNINA requires molgrid and gnina_torch; those are already installed on the GNINA
SageMaker instance.  numpy<2 must be preserved.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import torch
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
# These seeds are not in config.yaml; they are defined in the pre-registration doc.
UNTRAINED_INIT_SEEDS = [20260924, 20260925, 20260926, 20260927]
print(f"untrained init seeds: {UNTRAINED_INIT_SEEDS}")

# ---------------------------------------------------------------------------
# model config (same as T3 script)
# ---------------------------------------------------------------------------
EGNN_HP = cfg["training"]["egnn"]
DGNN_HP = cfg["training"]["distance_gnn"]
CUTOFF_A = float(DGNN_HP.get("cutoff_A", 5.0))
N_RBF    = int(DGNN_HP.get("n_rbf", 16))
HALOGENS = {"F", "CL", "BR", "I"}
METALS   = {"ZN", "MG", "CA", "MN", "FE", "NA", "K", "CU", "NI", "CO", "CD", "HG", "MO"}
N_TYPES  = 16  # 8 element buckets × {protein, ligand}

MODEL_SPECS = {
    "egnn":         dict(cls=eu.EGNNAffinity,
                         kwargs=dict(n_types=N_TYPES, hidden=EGNN_HP["hidden"],
                                     n_layers=EGNN_HP["n_layers"])),
    "distance_gnn": dict(cls=eu.DistanceGNN,
                         kwargs=dict(n_types=N_TYPES, hidden=DGNN_HP["hidden"],
                                     n_layers=DGNN_HP["n_layers"],
                                     n_rbf=N_RBF, cutoff=CUTOFF_A)),
}

EMB_DIR  = ROOT / "data" / "embeddings"
CKPT_DIR = ROOT / "data" / "checkpoints"
EMB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# IDs from split (must be prebuilt by run_t3_anchors.py)
# ---------------------------------------------------------------------------
import pandas as pd
split = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0)
split.index = split.index.astype(str).str.lower()
SPLIT_COL = "split"
AFF       = cfg["split"]["affinity_column"]
GRAPH_CACHE = ROOT / "data" / "graph_cache"

def get_graph(pid):
    cache = GRAPH_CACHE / f"{pid}.pt"
    if not cache.exists():
        raise FileNotFoundError(f"graph cache missing for {pid} -- run T3 first")
    d = torch.load(cache, map_location="cpu")
    return d["z"], d["x"], d["y"]

CACHED_IDS = {}
for sp in ("train", "val", "test"):
    ids_f = EMB_DIR / f"egnn_primary_{sp}_ids.txt"
    if ids_f.exists():
        CACHED_IDS[sp] = ids_f.read_text().splitlines()
    else:
        ids = split.index[split[SPLIT_COL] == sp].tolist()
        ok = [pid for pid in ids if (GRAPH_CACHE / f"{pid}.pt").exists()]
        CACHED_IDS[sp] = ok
    print(f"  {sp}: {len(CACHED_IDS[sp])} complexes")

# ---------------------------------------------------------------------------
# dataset / collate
# ---------------------------------------------------------------------------
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
# extraction helper for EGNN / DNN
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_gnn(name, model, split_name):
    spec = MODEL_SPECS[name]
    model64 = spec["cls"](**spec["kwargs"]).double()
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    model64.eval()
    loader = DataLoader(ComplexDataset(CACHED_IDS[split_name]), batch_size=1,
                        shuffle=False, collate_fn=collate, num_workers=0)
    rows, ids_out = [], []
    for pids, z, x, ei, b, ng, _ in loader:
        x = x.double()
        with eu.PreHeadCapture(model64, "readout.0") as cap:
            model64(z, x, ei, b, ng)
            rows.append(cap.value.cpu().numpy()[0])
        ids_out.append(pids[0])
    E = np.stack(rows).astype(np.float64)
    return E, ids_out

# ---------------------------------------------------------------------------
# T5: EGNN and Distance-GNN untrained inits 2-5 (CPU, no molgrid)
# ---------------------------------------------------------------------------
print("\n=== T5: EGNN + Distance-GNN untrained inits 2-5 ===")
manifest_path = ROOT / "logs" / "embeddings_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

for i, seed in enumerate(UNTRAINED_INIT_SEEDS, start=2):
    for name in ("egnn", "distance_gnn"):
        tag = f"untrained_init{i}"
        ckpt_path = CKPT_DIR / f"{name}_{tag}.pt"

        # check if already done
        already = all(
            (EMB_DIR / f"{name}_{tag}_{sp}.npy").exists()
            for sp in ("train", "val", "test")
        )
        if already:
            print(f"[{name}/{tag}] embeddings already present -- skipping")
            continue

        seedmod.seed_everything(seed)
        spec = MODEL_SPECS[name]
        model = spec["cls"](**spec["kwargs"])  # CPU is fine for extraction
        model.eval()

        # save checkpoint so it's reproducible
        torch.save(dict(epoch=0, model=model.state_dict(), seed=seed, untrained=True,
                        init_index=i), ckpt_path)
        print(f"[{name}/{tag}] model instantiated, checkpoint -> {ckpt_path.name}")

        for sp in ("train", "val", "test"):
            E, ids_out = extract_gnn(name, model, sp)
            npy = EMB_DIR / f"{name}_{tag}_{sp}.npy"
            ids_f = EMB_DIR / f"{name}_{tag}_{sp}_ids.txt"
            np.save(npy, E)
            ids_f.write_text("\n".join(ids_out))
            sha = hashlib.sha256(E.tobytes()).hexdigest()
            key = f"{name}/{tag}/{sp}"
            manifest[key] = dict(shape=list(E.shape), sha256=sha,
                                 npy=str(npy.relative_to(ROOT)),
                                 ids=str(ids_f.relative_to(ROOT)))
            print(f"  {key:35s} shape={E.shape} sha256={sha[:12]}")

manifest_path.write_text(json.dumps(manifest, indent=2))
print("updated embeddings_manifest.json (EGNN/DNN inits 2-5)")

# ---------------------------------------------------------------------------
# T5: GNINA untrained inits 2-5
# ---------------------------------------------------------------------------
print("\n=== T5: GNINA untrained inits 2-5 ===")

GNINA_TYPES_FILE = ROOT / "data" / "gnina_types" / "test.types"
if not GNINA_TYPES_FILE.parent.exists():
    print("[WARN] data/gnina_types/ not found -- skipping GNINA inits (copy from Drive first)")
    gnina_available = False
else:
    gnina_available = True

try:
    try:
        import gnina_torch          # editable install name used on SageMaker
    except ImportError:
        import gninatorch as gnina_torch  # fallback package name
    import molgrid
    gnina_available = gnina_available and True
    print(f"gnina_torch: {getattr(gnina_torch, '__version__', 'unknown')}  "
          f"molgrid: {molgrid.__version__}")
except ImportError as e:
    print(f"[WARN] gnina_torch/molgrid not importable: {e}")
    gnina_available = False

if gnina_available:
    # load existing untrained_init1 to get architecture reference
    GNINA_CKPT = CKPT_DIR / "gnina_untrained.pt"
    if not GNINA_CKPT.exists():
        raise FileNotFoundError(f"Missing {GNINA_CKPT.name} -- transfer from Drive first")
    ref_state = torch.load(GNINA_CKPT, map_location="cpu")

    # build type-file lists for each split
    GNINA_TYPES = {
        sp: ROOT / "data" / "gnina_types" / f"{sp}.types"
        for sp in ("train", "val", "test")
    }
    for sp, tp in GNINA_TYPES.items():
        if not tp.exists():
            raise FileNotFoundError(f"Missing {tp} -- transfer from Drive first")

    _gnina_feat_layer = [None]  # cached across splits (same arch for all inits)

    def extract_gnina_untrained(model, split_name):
        """Voxelize with molgrid, run through GNINA, capture shared feature vector.

        IDs: pre-read from types file (receptor path → parent dir name).
        Feature layer: auto-discovered via find_shared_input_heads on first call.
        """
        types_file_path = GNINA_TYPES[split_name]

        # Pre-read IDs from types file (format: label rmsd receptor ligand …)
        ids_ordered = []
        with open(types_file_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#"):
                    continue
                _parts = _line.split()
                if len(_parts) >= 3:
                    ids_ordered.append(Path(_parts[2]).parent.name.lower())

        ep = molgrid.ExampleProvider(data_root=str(PDBBIND_DIR),
                                     shuffle=False, default_batch_size=1,
                                     iteration_scheme=molgrid.IterationScheme.LargeEpoch)
        ep.populate(str(types_file_path))
        gmaker = molgrid.GridMaker()
        dims = gmaker.grid_dimensions(ep.num_types())
        total = ep.size()

        # Auto-discover the shared feature layer (input to parallel heads) once
        if _gnina_feat_layer[0] is None:
            dummy = torch.zeros(1, *dims, dtype=torch.float32)
            groups = eu.find_shared_input_heads(model, dummy)
            if groups:
                best = max(groups, key=lambda g: g[0][-1])
                _gnina_feat_layer[0] = best[1][0]
                print(f"  GNINA feature layer (shared input): '{_gnina_feat_layer[0]}' "
                      f"feat_dim={best[0][-1]}")
            else:
                # Fallback: widest Linear layer by input dimension
                order, _ = eu.trace_linear_layers(model, dummy)
                if not order:
                    raise RuntimeError("No Linear layers found in GNINA model; "
                                       "inspect architecture manually.")
                name, in_f, _out_f, _ = max(order, key=lambda t: t[1])
                _gnina_feat_layer[0] = name
                print(f"  GNINA feature layer (widest Linear): '{name}' in_dim={in_f}")

        feat_layer = _gnina_feat_layer[0]
        model.eval()
        rows, ids_out = [], []
        with torch.no_grad():
            for idx in range(total):
                batch = ep.next_batch(1)
                input_tensor = torch.zeros(1, *dims, dtype=torch.float32)
                gmaker.forward(batch, input_tensor, 0, random_rotation=False)
                with eu.PreHeadCapture(model, feat_layer) as cap:
                    model(input_tensor)
                rows.append(cap.value.cpu().numpy().reshape(-1))
                ids_out.append(ids_ordered[idx] if idx < len(ids_ordered)
                                else f"unk_{idx}")
        return np.stack(rows).astype(np.float64), ids_out

    PDBBIND_DIR = Path(cfg["paths"]["pdbbind_dir"])

    # Discover GNINA model class (try multiple attribute paths).
    # Primary target: Default2018Affinity (crossdock_default2018 member).
    gnina_model_cls = None
    for _attr_path in [
        ["models", "Default2018Affinity"],
        ["models", "Default2017"],
        ["gnina", "Default2018Affinity"],
        ["gnina", "Default2017"],
        ["models", "DefaultGnina"],
        ["gnina", "DefaultGnina"],
    ]:
        try:
            obj = gnina_torch
            for _attr in _attr_path:
                obj = getattr(obj, _attr)
            if callable(obj):
                gnina_model_cls = obj
                print(f"  GNINA class: gninatorch.{'.'.join(_attr_path)}")
                break
        except AttributeError:
            continue
    if gnina_model_cls is None:
        raise ImportError(
            "Cannot locate GNINA model class in gninatorch. "
            "Check: python -c \"import gninatorch; print(dir(gninatorch.models))\""
        )

    for i, seed in enumerate(UNTRAINED_INIT_SEEDS, start=2):
        tag = f"untrained_init{i}"
        already = all(
            (EMB_DIR / f"gnina_{tag}_{sp}.npy").exists()
            for sp in ("train", "val", "test")
        )
        if already:
            print(f"[gnina/{tag}] embeddings already present -- skipping")
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = gnina_model_cls()

        gnina_ckpt = CKPT_DIR / f"gnina_{tag}.pt"
        torch.save(dict(epoch=0, model=model.state_dict(), seed=seed,
                        untrained=True, init_index=i), gnina_ckpt)
        print(f"[gnina/{tag}] model instantiated -> {gnina_ckpt.name}")

        for sp in ("train", "val", "test"):
            E, ids_out = extract_gnina_untrained(model, sp)
            npy = EMB_DIR / f"gnina_{tag}_{sp}.npy"
            ids_f = EMB_DIR / f"gnina_{tag}_{sp}_ids.txt"
            np.save(npy, E)
            ids_f.write_text("\n".join(ids_out))
            sha = hashlib.sha256(E.tobytes()).hexdigest()
            key = f"gnina/{tag}/{sp}"
            manifest[key] = dict(shape=list(E.shape), sha256=sha,
                                 npy=str(npy.relative_to(ROOT)),
                                 ids=str(ids_f.relative_to(ROOT)))
            print(f"  {key:35s} shape={E.shape} sha256={sha[:12]}")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("updated embeddings_manifest.json (GNINA inits 2-5)")

else:
    print("[SKIP] GNINA untrained inits 2-5: run after transferring gnina_types/ from Drive")
    print("       Then re-run:  python run_t5_untrained_inits.py")

# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------
import subprocess
def sh(*a):
    r = subprocess.run(a, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()

sh("git", "config", "user.email", "t5runner@local")
sh("git", "config", "user.name",  "t5runner")
sh("git", "add", "logs", "data/checkpoints")
rc, msg = sh("git", "commit", "-m", "T5: EGNN + DNN + GNINA untrained inits 2-5 extracted")
print(msg[:400] or "(nothing to commit)")

print("\n=== T5 complete ===")
print("All untrained baseline embeddings ready.")
print("Next steps after both runs complete:")
print("  1. Switch instance to ml.m5.4xlarge (compute-only, no GPU needed)")
print("  3. Proceed to T8 (CKA analysis)")
