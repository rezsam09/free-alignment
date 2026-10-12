import sys, subprocess

def pip(*pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    print(pkgs, "ok" if r.returncode == 0 else "FAILED: " + r.stderr.strip().splitlines()[-1][:300])

pip("numpy<2")
pip("pyyaml", "pandas", "scipy", "tqdm", "scikit-learn")

import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import molgrid
from gninatorch.gnina import setup_gnina_model
print("molgrid and gninatorch importable.")


from pathlib import Path
import os

PROJECT_DIR = "/home/ec2-user/SageMaker/pdbbind_data"   # <-- same path you used in notebook 2b
ROOT = Path(PROJECT_DIR).resolve()
assert ROOT.exists(), f"{ROOT} not found."
os.chdir(ROOT)
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import prereg
from src import embed_utils as eu

cfg = prereg.load_config(ROOT)
prereg.verify_lock(ROOT)
print("root:", ROOT)

# v2: extraction device. Default CPU (as in v1). EXTRACT_ON_GPU=1 runs the float64 forward pass on the GPU.
# Use ONE device for all checkpoints of the run; do not mix.
EXTRACT_DEVICE = DEVICE if (os.environ.get("EXTRACT_ON_GPU", "0") == "1" and torch.cuda.is_available()) else torch.device("cpu")
print("extraction device:", EXTRACT_DEVICE)


import pandas as pd

split = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0)
split.index = split.index.astype(str).str.lower()

TYPES_DIR = ROOT / "data" / "gnina_types"
CKPT_DIR = ROOT / "data" / "checkpoints"
assert TYPES_DIR.exists(), "data/gnina_types missing -- run (or sync the output of) notebook 2b's types-file cell first."

TYPES, IDS_BY_SPLIT = {}, {}
for s in ("train", "val", "test"):
    TYPES[s] = TYPES_DIR / f"{s}.types"
    IDS_BY_SPLIT[s] = (TYPES_DIR / f"{s}_ids.txt").read_text().splitlines()
    assert TYPES[s].exists(), f"{TYPES[s]} missing."
    print(s, "n =", len(IDS_BY_SPLIT[s]))

MC = cfg["embeddings"]["models"]["gnina"]
RES, DIM_A, EXPECTED_DIM = MC["grid_resolution_A"], MC["grid_dimension_A"], MC["dim"]
FROZEN_LAYER = MC["layer"]
CNN = MC["cnn"]
N_ROTATIONS = MC.get("n_rotations", 1)
assert MC.get("orientation") == "fixed_canonical", (
    "config.embeddings.models.gnina.orientation is not fixed_canonical -- this notebook only "
    "implements the canonical (no-augmentation) extraction path. If you locked mean_over_rotations, "
    "extend the extraction loop to average over rotations before saving."
)

receptor_typer, ligand_typer = molgrid.defaultGninaReceptorTyper, molgrid.defaultGninaLigandTyper
gmaker = molgrid.GridMaker(resolution=RES, dimension=DIM_A)

CKPTS = {}
for tag in ("primary", "untrained", "anchor"):
    p = CKPT_DIR / f"gnina_{tag}.pt"
    if p.exists():
        CKPTS[tag] = p
print("checkpoints found:", list(CKPTS))
assert "primary" in CKPTS, "gnina_primary.pt is missing -- notebook 2b must produce this before extraction."


def load_gnina(tag):
    ckpt = torch.load(CKPTS[tag], map_location="cpu")
    assert ckpt.get("untrained") or ckpt.get("is_best"), (
        f"{CKPTS[tag].name} is an in-progress checkpoint (not the best-validation weights). Finish 2b first.")
    model = setup_gnina_model(CNN)
    if isinstance(model, tuple):
        model, is_ensemble = model
        if is_ensemble:
            model = model[0] if isinstance(model, (list, tuple)) else model
    model.load_state_dict(ckpt["model"])
    return model.double().eval().to(EXTRACT_DEVICE)   # double precision for extraction, matching config.embeddings.dtype

MODELS = {tag: load_gnina(tag) for tag in CKPTS}
print("loaded:", list(MODELS))


def make_provider(types_path, batch_size, shuffle):
    provider = molgrid.ExampleProvider(receptor_typer, ligand_typer, default_batch_size=batch_size,
                                       shuffle=shuffle, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
    provider.populate(str(types_path))
    return provider

def next_grid_batch_double(provider, gmaker, batch_size):
    """Canonical orientation only: no random rotation, no translation."""
    batch = provider.next_batch(batch_size)
    dims = gmaker.grid_dimensions(provider.num_types())
    grid = torch.zeros((batch_size,) + dims, dtype=torch.float32)
    gmaker.forward(batch, grid, random_translation=0.0, random_rotation=False)
    return grid.double().to(EXTRACT_DEVICE)

check_provider = make_provider(TYPES["test"], batch_size=4, shuffle=False)
check_grid = next_grid_batch_double(check_provider, gmaker, 4)
with eu.PreHeadCapture(MODELS["primary"], FROZEN_LAYER) as cap, torch.no_grad():
    MODELS["primary"](check_grid)
    dim_seen = cap.value.shape[-1]
print("hook width on loaded checkpoint:", dim_seen, "| config expects:", EXPECTED_DIM)
assert dim_seen == EXPECTED_DIM, "Dimension mismatch on the loaded checkpoint -- stop and investigate."
del check_provider, check_grid
print("OK -- proceeding to full extraction.")


import numpy as np, hashlib, json, time
from tqdm.auto import tqdm

EMB_DIR = ROOT / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)
manifest_path = ROOT / "logs" / "embeddings_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

@torch.no_grad()
def extract_gnina_split(tag, split_name):
    model = MODELS[tag]
    ids = IDS_BY_SPLIT[split_name]
    n = len(ids)
    out_npy = EMB_DIR / f"gnina_{tag}_{split_name}.npy"
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float64, shape=(n, EXPECTED_DIM))
    provider = make_provider(TYPES[split_name], batch_size=1, shuffle=False)
    t0 = time.time()
    for i in tqdm(range(n), desc=f"{tag}/{split_name}", total=n):
        grid = next_grid_batch_double(provider, gmaker, 1)
        with eu.PreHeadCapture(model, FROZEN_LAYER) as cap:
            model(grid)
            arr[i] = cap.value.cpu().numpy()[0]
        if i == 19:
            _r = (time.time() - t0) / 20
            print(f"  first 20 complexes: {_r:.2f} s/complex on {EXTRACT_DEVICE} -> about {_r*n/60:.0f} min for this split")
        if i % 500 == 0:
            arr.flush()
    arr.flush()
    del arr
    print(f"{tag}/{split_name}: wrote {out_npy.name} in {(time.time()-t0)/60:.1f} min")
    (EMB_DIR / f"gnina_{tag}_{split_name}_ids.txt").write_text("\n".join(ids))
    h = sha256_file(out_npy)
    key = f"gnina/{tag}/{split_name}"
    manifest[key] = dict(shape=[n, EXPECTED_DIM], sha256=h,
                         npy=str(out_npy.relative_to(ROOT)),
                         ids=str((EMB_DIR / f"gnina_{tag}_{split_name}_ids.txt").relative_to(ROOT)))
    print(f"  sha256={h[:12]}")
    return out_npy

for tag in MODELS:
    for split_name in ("train", "val", "test"):
        extract_gnina_split(tag, split_name)

manifest_path.write_text(json.dumps(manifest, indent=2))
print("wrote", manifest_path.relative_to(ROOT))


_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
HEALTH_ON_TRAIN = {"1": True, "0": False}.get(os.environ.get("HEALTH_ON_TRAIN"), _ram_gb >= 32)   # v2: auto by RAM, env override
print(f"RAM {_ram_gb:.0f} GB -> HEALTH_ON_TRAIN = {HEALTH_ON_TRAIN}")

cov_cols = ["complex_atom_count", "ligand_heavy_atom_count"]
health = {}
for tag in MODELS:
    for split_name in ("val", "test") + (("train",) if HEALTH_ON_TRAIN else ()):
        key = f"gnina/{tag}/{split_name}"
        E = np.load(EMB_DIR / f"gnina_{tag}_{split_name}.npy", mmap_mode="r")
        ids_order = (EMB_DIR / f"gnina_{tag}_{split_name}_ids.txt").read_text().splitlines()
        Z = split.loc[ids_order, cov_cols].to_numpy()
        # embedding_sanity does a full SVD; for train-sized arrays this is the memory-heavy step --
        # that's what HEALTH_ON_TRAIN gates.
        h = eu.embedding_sanity(np.asarray(E), Z)
        health[key] = h
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k != "pc_R2_on_covariates"})
        del E

(ROOT / "logs" / "embedding_health_2c.json").write_text(json.dumps(health, indent=2))
if not HEALTH_ON_TRAIN:
    print("\nTrain-split health skipped (HEALTH_ON_TRAIN=False). Re-run this cell with it set to True "
          "on a higher-memory instance whenever convenient -- it only needs the .npy files, not the checkpoints.")


for tag in MODELS:
    for split_name in ("train", "val", "test"):
        ids_order = (EMB_DIR / f"gnina_{tag}_{split_name}_ids.txt").read_text().splitlines()
        expected = IDS_BY_SPLIT[split_name]
        assert ids_order == expected, f"row-order mismatch for gnina/{tag}/{split_name}"
print("Row alignment OK for every (tag, split) pair.")

# Optional: sync back to S3. Uncomment and edit.
# subprocess.run(["aws", "s3", "sync", str(EMB_DIR), "s3://YOUR-BUCKET/pdbbind_cka/data/embeddings"], check=True)
# subprocess.run(["aws", "s3", "sync", str(ROOT / "logs"), "s3://YOUR-BUCKET/pdbbind_cka/logs"], check=True)
print("\nRemember to sync data/embeddings/ and logs/ back to your canonical project copy and commit there.")

