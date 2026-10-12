import sys, subprocess

def pip(*pkgs, extra_index=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(pkgs, "ok" if r.returncode == 0 else "FAILED: " + r.stderr.strip().splitlines()[-1][:300])

# numpy MUST be pinned below 2 before molgrid is imported anywhere in this process --
# molgrid warns/misbehaves on numpy>=2.1 (same note as Step 0).
pip("numpy<2")
pip("pyyaml", "pandas", "scipy", "tqdm")

import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no CUDA device visible. Check the instance type and drivers before training.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import molgrid
    print("molgrid:", molgrid.__version__ if hasattr(molgrid, "__version__") else "(version unknown)")
except ImportError:
    print("molgrid not importable. On most AMIs: `pip install molgrid` (a compiled wheel matching your "
          "CUDA + Python version), or build from source per github.com/gnina/libmolgrid if no wheel matches.")
    raise


import subprocess
r = subprocess.run(["pip", "show", "gninatorch"], capture_output=True, text=True)
if r.returncode != 0:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/RMeli/gnina-torch", "/tmp/gnina-torch"], check=True)
    pip("-e", "/tmp/gnina-torch")
import gninatorch
from gninatorch.gnina import setup_gnina_model
print("gninatorch:", gninatorch.__file__)


from pathlib import Path
import os

PROJECT_DIR = "/home/ec2-user/SageMaker/pdbbind_data"   # <-- edit to wherever you put the synced project on this instance

# Optional: pull the project down from S3 first. Uncomment and set your bucket/prefix.
# subprocess.run(["aws", "s3", "sync", "s3://YOUR-BUCKET/pdbbind_cka", PROJECT_DIR], check=True)

ROOT = Path(PROJECT_DIR).resolve()
assert ROOT.exists(), f"{ROOT} not found. Sync/copy the project here first (see markdown above)."
os.chdir(ROOT)
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
print("root:", ROOT)


try:
    from src import prereg
    from src import seeds as seedmod
    from src import embed_utils as eu
except ImportError as e:
    raise ImportError("src/prereg.py, src/seeds.py or src/embed_utils.py missing under ROOT/src. "
                      "Sync the whole project folder, not just the data.") from e

cfg = prereg.load_config(ROOT)
rec = prereg.verify_lock(ROOT)   # raises if config.yaml has drifted from config.lock.json
GLOBAL_SEED = int(cfg["seeds"]["global"])

# The anchor arm runs by default in this notebook -- set RUN_ANCHOR = False below to skip it.
# ANCHOR_SEED falls back to 20260923 (the value already used elsewhere in this project's deviation
# log draft) if the frozen config doesn't have seeds.anchor set yet.
ANCHOR_SEED = int(cfg["seeds"].get("anchor", 20260923))
RUN_ANCHOR = os.environ.get("RUN_ANCHOR", "1") == "1"   # v2: env var RUN_ANCHOR=0 skips the anchor arm (staged run)

cfg_says_anchor_on = bool(cfg["controls"].get("same_arch_seed_anchor"))
print("primary seed:", GLOBAL_SEED, "| anchor seed:", ANCHOR_SEED, "| running anchor arm:", RUN_ANCHOR)
if RUN_ANCHOR and not cfg_says_anchor_on:
    print("NOTE: controls.same_arch_seed_anchor is still false/unset in the frozen config -- this "
          "notebook runs the anchor arm anyway (RUN_ANCHOR env var is 1). Log the "
          "same_arch_seed_anchor deviation (and seeds.anchor if it's not in the config yet) so the "
          "record matches what actually ran.")


DEV_LOG = ROOT / "logs" / "deviation_log.md"
assert DEV_LOG.exists(), "logs/deviation_log.md not found -- sync it from the project root."
dev_txt = DEV_LOG.read_text().lower()

required_markers = ["training.gnina", "random_default2018affinity"]
missing = [m for m in required_markers if m not in dev_txt]
assert not missing, (
    "The deviation log does not (yet) record training GNINA from random init. "
    f"Missing marker(s): {missing}. Log it (prereg.apply_deviation or log_deviation_note) before "
    "spending GPU hours here -- this mirrors the discipline Step 1b enforced for the prep ladder."
)
print("GNINA-from-scratch deviation: logged, proceeding.")

if RUN_ANCHOR:
    anchor_markers = ["same_arch_seed_anchor"]
    anchor_missing = [m for m in anchor_markers if m not in dev_txt]
    if anchor_missing:
        print(f"WARNING: RUN_ANCHOR is true from the config, but the deviation log doesn't mention "
              f"{anchor_missing}. The anchor-seed section below will still run since the config says to -- "
              f"but log the deviation for the record.")


import pandas as pd

split = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0)
split.index = split.index.astype(str).str.lower()
AFF = cfg["split"]["affinity_column"]
PDBBIND_DIR = Path(cfg["paths"]["pdbbind_dir"])

def build_index(base: Path):
    bases = [base] + [d for d in base.iterdir() if d.is_dir() and len(d.name) != 4]
    idx = {}
    for b in bases:
        for d in b.iterdir():
            if d.is_dir() and len(d.name) == 4:
                idx[d.name.lower()] = d
    return idx

IDX = build_index(PDBBIND_DIR)
print("complex dirs found:", len(IDX))

TYPES_DIR = ROOT / "data" / "gnina_types"
TYPES_DIR.mkdir(parents=True, exist_ok=True)

def complex_files(pid):
    d = IDX.get(pid)
    if d is None:
        return None
    pocket = d / f"{pid}_pocket.pdb"
    lig = d / f"{pid}_ligand.sdf"
    if not lig.exists():
        lig = d / f"{pid}_ligand.mol2"
    if not pocket.exists() or not lig.exists():
        return None
    return pocket, lig

def write_types_file(split_name):
    out_path = TYPES_DIR / f"{split_name}.types"
    ids = sorted(split.index[split.split == split_name].tolist())
    lines, kept_ids, bad = [], [], []
    for pid in ids:
        f = complex_files(pid)
        if f is None:
            bad.append((pid, "gnina_types: missing pocket/ligand file")); continue
        pocket, lig = f
        lines.append(f"{split.loc[pid, AFF]} {pocket} {lig}")
        kept_ids.append(pid)
    out_path.write_text("\n".join(lines) + "\n")
    if bad:
        prereg.log_exclusions(bad, stage="gnina_types", root=ROOT)
    (TYPES_DIR / f"{split_name}_ids.txt").write_text("\n".join(kept_ids))
    print(f"{split_name}: wrote {len(kept_ids)} lines to {out_path.relative_to(ROOT)} ({len(bad)} excluded)")
    return out_path, kept_ids

TYPES = {}
IDS_BY_SPLIT = {}
for s in ("train", "val", "test"):
    TYPES[s], IDS_BY_SPLIT[s] = write_types_file(s)


CNN = cfg["embeddings"]["models"]["gnina"]["cnn"]

def get_fresh_gnina_arch():
    """Always call setup_gnina_model fresh (new pretrained-then-to-be-wiped weights) and return a
    single nn.Module, member 0 if it's an ensemble. Centralized here so every call site (smoke test,
    train_gnina) unpacks an ensemble the same way -- two different unpacking patterns is exactly how
    this kind of bug (indexing the wrong thing) sneaks in."""
    result = setup_gnina_model(CNN)
    model, is_ens = result if isinstance(result, tuple) else (result, False)
    if is_ens and isinstance(model, (list, tuple)):
        model = model[0]
    return model, is_ens

gmodel, is_ensemble = get_fresh_gnina_arch()
if is_ensemble:
    print("setup_gnina_model returned an ensemble -- using member 0 only, per the methods "
          "('choose one member network, not the ensemble'). Verify this indexing matches your "
          "installed gnina-torch version.")
print(gmodel.__class__.__name__)
del gmodel  # only needed to print the architecture name / confirm ensemble-ness; not reused

def reinit_weights(model, seed):
    """Random-initialize every parametric submodule in place. Keeps the exact pretrained
    architecture but wipes pretrained weights, per the from-scratch deviation entry."""
    torch.manual_seed(seed)
    n = 0
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
            n += 1
    print(f"reinitialized {n} submodules with reset_parameters() under seed {seed}")
    return model


MC = cfg["embeddings"]["models"]["gnina"]
RES, DIM_A = MC["grid_resolution_A"], MC["grid_dimension_A"]
EXPECTED_DIM = MC["dim"]
FROZEN_LAYER = MC["layer"]

receptor_typer, ligand_typer = molgrid.defaultGninaReceptorTyper, molgrid.defaultGninaLigandTyper
gmaker = molgrid.GridMaker(resolution=RES, dimension=DIM_A)

def make_provider(types_path, batch_size, shuffle):
    provider = molgrid.ExampleProvider(receptor_typer, ligand_typer, default_batch_size=batch_size,
                                       shuffle=shuffle, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
    provider.populate(str(types_path))
    return provider

def next_grid_batch(provider, gmaker, batch_size, random_rotation, random_translation):
    """Returns (grid: (B,C,D,D,D) float32 tensor on DEVICE, labels: (B,) float32 tensor on DEVICE)."""
    batch = provider.next_batch(batch_size)
    dims = gmaker.grid_dimensions(provider.num_types())
    grid = torch.zeros((batch_size,) + dims, dtype=torch.float32, device=DEVICE)
    gmaker.forward(batch, grid, random_translation=random_translation, random_rotation=random_rotation)
    labels = torch.zeros(batch_size, dtype=torch.float32, device=DEVICE)
    batch.extract_label(0, labels)
    return grid, labels


# --- mandatory smoke test: 30-50 train complexes, dimension + shared-head check ---
SMOKE_N = 32
smoke_provider = make_provider(TYPES["train"], batch_size=SMOKE_N, shuffle=False)
grid, labels = next_grid_batch(smoke_provider, gmaker, SMOKE_N, random_rotation=False, random_translation=0.0)
print("voxel grid shape:", tuple(grid.shape))

test_arch, _ = get_fresh_gnina_arch()
test_model = reinit_weights(test_arch, seed=GLOBAL_SEED).to(DEVICE).eval()

groups = eu.find_shared_input_heads(test_model, grid)
print("parallel heads sharing one input tensor:", [(shape, names) for shape, names in groups])
assert groups, "No shared-input head group found -- pose and affinity heads may not share a pooled " \
    "vector in this gnina-torch version. Inspect eu.trace_linear_layers(test_model, grid) and adjust FROZEN_LAYER."
shared_shape = groups[0][0]
gnina_dim = shared_shape[-1]
print(f"shared input width: {gnina_dim}  (config expects {EXPECTED_DIM})")
assert gnina_dim == EXPECTED_DIM, (
    f"Dimension mismatch: hook sees {gnina_dim}, config.embeddings.models.gnina.dim = {EXPECTED_DIM}. "
    "Do not proceed -- this must be resolved (fix the model/config, or run apply_deviation with a reason) "
    "before any training."
)
assert FROZEN_LAYER in groups[0][1], (
    f"config layer '{FROZEN_LAYER}' is not among the shared-input heads found ({groups[0][1]}). "
    "The frozen layer name may not match this gnina-torch version's module names."
)

with torch.no_grad():
    out = test_model(grid)
print("model output type:", type(out), "-- shape(s):", [tuple(o.shape) for o in out] if isinstance(out, (tuple, list)) else tuple(out.shape))
print("\nSMOKE TEST PASSED: dimension stable at", gnina_dim, "| pose+affinity heads share one input tensor.")


def pick_affinity(output):
    if torch.is_tensor(output):
        return output.squeeze(-1) if output.dim() > 1 else output
    if isinstance(output, (tuple, list)):
        candidates = [o for o in output if o.dim() >= 1 and (o.dim() == 1 or o.shape[-1] == 1)]
        if len(candidates) == 1:
            return candidates[0].squeeze(-1) if candidates[0].dim() > 1 else candidates[0]
        raise RuntimeError(
            f"Could not uniquely identify the affinity output among {[tuple(o.shape) for o in output]}. "
            "Inspect the model's forward() source and hard-code the index here."
        )
    raise RuntimeError(f"Unexpected model output type: {type(output)}")

with torch.no_grad():
    aff_check = pick_affinity(test_model(grid))
print("picked affinity output shape:", tuple(aff_check.shape), "-- should be (batch,)")
assert aff_check.shape == (SMOKE_N,), "pick_affinity did not return one scalar per complex -- fix before training."
del test_model, grid, labels, smoke_provider
if torch.cuda.is_available():
    torch.cuda.empty_cache()


from scipy.stats import pearsonr
import numpy as np, time

CKPT_DIR = ROOT / "data" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GATE_MARGIN = 0.10  # same documented assumption as notebook 2a; no exact margin is specified in the methods

HP = cfg["training"]["gnina"]
AUG_TRANSLATE_A = 2.0  # per the logged deviation: "random rotation and +-2 A translation augmentation"

def n_batches_per_epoch(n_examples, batch_size):
    return max(1, n_examples // batch_size)

@torch.no_grad()
def evaluate_gnina(model, types_path, n_examples, batch_size=16):
    model.eval()
    provider = make_provider(types_path, batch_size=batch_size, shuffle=False)
    preds, trues, seen = [], [], 0
    nb = -(-n_examples // batch_size)  # ceil, one deterministic pass since shuffle=False
    for _ in range(nb):
        this_bs = min(batch_size, n_examples - seen) if seen + batch_size > n_examples else batch_size
        grid, labels = next_grid_batch(provider, gmaker, batch_size, random_rotation=False, random_translation=0.0)
        out = pick_affinity(model(grid))
        preds.append(out[:this_bs].cpu().numpy()); trues.append(labels[:this_bs].cpu().numpy())
        seen += this_bs
        if seen >= n_examples:
            break
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    r = float(pearsonr(preds, trues)[0]) if len(preds) > 1 else float("nan")
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    loss = float(np.mean((preds - trues) ** 2))
    return dict(val_loss=loss, pearson_r=r, rmse=rmse, n=len(preds))

def train_gnina(seed, tag, untrained=False):
    ckpt_path = CKPT_DIR / f"gnina_{tag}.pt"
    arch, _ = get_fresh_gnina_arch()
    model = arch.to(DEVICE)
    reinit_weights(model, seed)

    if untrained:
        torch.save(dict(epoch=0, model=model.state_dict(), optimizer=None, seed=seed, untrained=True), ckpt_path)
        print(f"[gnina/{tag}] untrained anchor saved -> {ckpt_path.name}")
        return model

    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    start_epoch, best_val_loss, patience_ctr, history = 0, float("inf"), 0, []
    best_state = None
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model"])
        if state.get("finished"):   # v2: a completed run is never resumed/extended
            print(f"[gnina/{tag}] already finished (is_best={state.get('is_best')}); loaded, not retraining")
            return model
        if state.get("optimizer") is not None:
            opt.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state.get("best_val_loss", float("inf"))
        patience_ctr = state.get("patience_ctr", 0)
        history = state.get("history", [])
        best_state = state.get("best_state")   # v2: best-val weights survive a resume
        print(f"[gnina/{tag}] resuming from epoch {start_epoch} (best val_loss={best_val_loss:.4f})")

    n_train = len(IDS_BY_SPLIT["train"])
    n_val = len(IDS_BY_SPLIT["val"])
    nb_train = n_batches_per_epoch(n_train, HP["batch_size"])
    epoch = start_epoch - 1

    for epoch in range(start_epoch, HP["max_epochs"]):
        model.train()
        molgrid.set_random_seed(seed + epoch)   # v2: per-epoch seeding => batch order/augmentation reproducible, also across resumes
        torch.manual_seed(seed + epoch)
        train_provider = make_provider(TYPES["train"], batch_size=HP["batch_size"], shuffle=True)
        t0 = time.time()
        for _ in range(nb_train):
            grid, labels = next_grid_batch(train_provider, gmaker, HP["batch_size"],
                                           random_rotation=True, random_translation=AUG_TRANSLATE_A)
            opt.zero_grad()
            out = pick_affinity(model(grid))
            loss = torch.nn.functional.mse_loss(out, labels)
            loss.backward()
            opt.step()
        val_metrics = evaluate_gnina(model, TYPES["val"], n_val, batch_size=HP["batch_size"])
        history.append(dict(epoch=epoch, **val_metrics, sec=round(time.time() - t0, 1)))
        improved = val_metrics["val_loss"] < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_metrics["val_loss"]; patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                        best_val_loss=best_val_loss, patience_ctr=patience_ctr, history=history,
                        seed=seed, untrained=False, best_state=best_state, finished=False), ckpt_path)
        if epoch % 2 == 0 or improved:
            print(f"[gnina/{tag}] epoch {epoch}: val_loss={val_metrics['val_loss']:.4f} "
                  f"r={val_metrics['pearson_r']:.3f} rmse={val_metrics['rmse']:.3f} "
                  f"({time.time()-t0:.0f}s/epoch, patience {patience_ctr}/{HP['early_stopping_patience']})")
        if patience_ctr >= HP["early_stopping_patience"]:
            print(f"[gnina/{tag}] early stopping at epoch {epoch}")
            break

    assert best_state is not None, "no best-validation weights recorded"
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                        best_val_loss=best_val_loss, patience_ctr=patience_ctr, history=history,
                        seed=seed, untrained=False, is_best=True, finished=True), ckpt_path)
    return model


print("=== training GNINA, primary seed", GLOBAL_SEED, "===")
gnina_primary = train_gnina(GLOBAL_SEED, tag="primary", untrained=False)


print("=== untrained anchor for GNINA, seed", GLOBAL_SEED, "===")
gnina_untrained = train_gnina(GLOBAL_SEED, tag="untrained", untrained=True)


if RUN_ANCHOR:
    print("=== training GNINA, anchor seed", ANCHOR_SEED, "===")
    gnina_anchor = train_gnina(ANCHOR_SEED, tag="anchor", untrained=False)
else:
    print("Skipping the GNINA anchor-seed arm (RUN_ANCHOR = False in Section 1). This is explicitly "
          "allowed to be skipped -- the methods note that if the GNINA anchor is unaffordable, GNINA "
          "pairs simply report obs/floor/identity-null and n_AB is left undefined for them. You can "
          "flip RUN_ANCHOR back on and re-run this cell later; it won't touch the primary/untrained "
          "checkpoints already on disk.")


TABLE1_PATH = ROOT / "logs" / "table1_performance.tsv"
rows = []
n_val = len(IDS_BY_SPLIT["val"])
arms = {"primary": gnina_primary, "untrained": gnina_untrained}
if RUN_ANCHOR:
    arms["anchor"] = gnina_anchor

for tag, model in arms.items():
    m = evaluate_gnina(model, TYPES["val"], n_val, batch_size=HP["batch_size"])
    rows.append(dict(model="gnina", seed_tag=tag, split="val", **m))
    print(f"gnina {tag:10s} val_loss={m['val_loss']:.4f} r={m['pearson_r']:.3f} rmse={m['rmse']:.3f} n={m['n']}")

r_trained = next(r["pearson_r"] for r in rows if r["seed_tag"] == "primary")
r_untrained = next(r["pearson_r"] for r in rows if r["seed_tag"] == "untrained")
gate = (r_trained >= 0.4) and (r_trained - r_untrained >= GATE_MARGIN)
print(f"\nGATE (gnina, primary): r={r_trained:.3f} >= 0.4 and beats untrained ({r_untrained:.3f}) "
      f"by >= {GATE_MARGIN}: {'PASS' if gate else 'FAIL'}")
rows.append(dict(model="gnina", seed_tag="gate_primary", split="val", val_loss=None,
                 pearson_r=r_trained, rmse=None, n=None, gate_pass=gate))

table1 = pd.DataFrame(rows)
write_header = not TABLE1_PATH.exists()
table1.to_csv(TABLE1_PATH, sep="\t", mode="a", header=write_header, index=False)
print("appended to", TABLE1_PATH.relative_to(ROOT))
if not gate:
    print("\nNOTE: from-scratch models on this split may fail the gate -- the methods say this "
          "explicitly for the test-R2 gate. A GNINA gate failure here is reported and flagged, "
          "not silently re-tuned.")


n_test = len(IDS_BY_SPLIT["test"])
test_rows = []
for tag, model in arms.items():
    m = evaluate_gnina(model, TYPES["test"], n_test, batch_size=HP["batch_size"])
    test_rows.append(dict(model="gnina", seed_tag=tag, split="test", **m))
    print(f"gnina {tag:10s} TEST r={m['pearson_r']:.3f} rmse={m['rmse']:.3f} n={m['n']}")
pd.DataFrame(test_rows).reindex(columns=table1.columns).to_csv(TABLE1_PATH, sep="\t", mode="a", header=False, index=False)
print("appended test rows to", TABLE1_PATH.relative_to(ROOT))


# Optional: push results back to S3. Uncomment and edit.
# subprocess.run(["aws", "s3", "sync", str(ROOT / "data" / "checkpoints"), "s3://YOUR-BUCKET/pdbbind_cka/data/checkpoints"], check=True)
# subprocess.run(["aws", "s3", "sync", str(ROOT / "data" / "gnina_types"), "s3://YOUR-BUCKET/pdbbind_cka/data/gnina_types"], check=True)
# subprocess.run(["aws", "s3", "sync", str(ROOT / "logs"), "s3://YOUR-BUCKET/pdbbind_cka/logs"], check=True)
print("Checkpoints in:", CKPT_DIR)
print("Remember to sync data/checkpoints/, data/gnina_types/ and logs/ back to your canonical project copy.")

