# Free Alignment

Code, analysis plan and result tables for **Free Alignment: What Determines Similarity Between Untrained Geometric
Networks of Protein-Ligand Complexes**. The study measures the representational similarity of untrained geometric
networks of protein-ligand complexes (random-network sweep, input-feature kernels, input ablations, probe panel). A
version of this work is under review at the UniReps workshop at NeurIPS.

## Layout

| Path | Content |
|---|---|
| `findings/` | analysis scripts (`a1` sweep, `a3` regression on setting-level scores, `a4` ablations, `a5_a7` real untrained networks, `b0`/`b_probe_panel` probes, `c3` output agreement) and shared helpers |
| `findings/plan/` | analysis plan (`findings_plan.yaml`) |
| `findings/tests/` | tests and an end-to-end run on synthetic data |
| `findings_out/` | result tables (`res_*.tsv`), `targets_meta.json` |
| `results/` | `residualization_r2.json`, the output of the subspace projection |
| `logs/` | stdout of the run stages |
| `paper_analysis/` | `analysis.py` and `analysis2.py` recompute every number, table and figure of the paper from `findings_out/` |
| `src/`, `config/`, `gpu_runs/`, `notebooks/`, `docking/`, `data/` | reference-model pipeline, docking and supporting inputs (below) |

## Running

`findings/lock_freeze.py` records checksums of the analysis plan and of the analysis code, and the result-producing
scripts check them before they run. Run order:

```
python3 findings/lock_freeze.py lock
python3 findings/build_graphs.py && python3 findings/input_kernels.py && python3 findings/b0_make_targets.py
python3 findings/lock_freeze.py freeze
for s in a1_floor_sweep a4_ablation a3_floor_model a5_a7_real_untrained b_probe_panel c3_output_agreement; do
  python3 findings/$s.py; done
```

All scripts read `PROJECT_ROOT` (default: current directory).

## Inputs that are not in this repository

* PDBbind v2020 structures (`PDBbind_2020/`) and the LP-PDBBind split, obtained from their original sources
  under their own terms of use. No structure files are redistributed (`targets.tsv` and `kernels.npz` are regenerated
  by `b0_make_targets.py` and `input_kernels.py`).
* Embeddings and checkpoints of the trained and untrained reference models (`data/embeddings/`,
  `data/checkpoints/`) and the split, cluster and covariate tables (`data/`). They are produced by the pipeline below
  and are large, so they are not included. `data/docking_results.tsv` holds the raw per-complex docking output
  (identifiers and numeric results only), and its `pid` column is the list of docked test complexes.

## Reference-model pipeline

* `src/` holds the similarity library (`alignlib.py`), the configuration checksum helper (`prereg.py`), seeds and embedding helpers.
* `config/` holds the reference-model configuration and its checksum file. Paths in `config.yaml` follow a Colab and
  Google Drive layout and should be set to your project root.
* `gpu_runs/` builds clusters, covariates and input kernels and runs the second-seed and untrained-initialisation extractions.
* `notebooks/` trains and extracts the reference models: `notebook_2a_egnn_distancegnn.ipynb` (EGNN and Distance-GNN,
  model classes in `src/embed_utils.py`) and the voxel-CNN notebooks 2b and 2c (`.py` and `.ipynb`).
* `docking/` holds the docking preparation (`dockprep.py`, `prepladder.py`) and runner (`dockrun.py`).

## Docking

`docking/dockrun.py` is the docking runner (fixed preparation ladder, AutoDock Vina 1.2.7, crystal-referenced RMSD,
one result file per complex). It imports `dockprep.py`, `prepladder.py` and `src/prereg.py`, and reads a queue table
listing the complexes to dock.

## Docking agreement and subspace projection

* `docking/make_vina_results.py` turns `data/docking_results.tsv` into `logs/vina_results.json`, and
  `docking/docking_agreement.py` computes the partial Spearman correlation of each model's predicted affinity with the
  negated Vina score given ligand heavy-atom count (protein-cluster bootstrap), together with the correlation of the
  Vina score with measured affinity and size, into `logs/docking_agreement.json`.
* `gpu_runs/run_residualization_r2.py` projects the ligand-fingerprint and pocket amino-acid-composition reference
  directions out of each embedding and records the CKA change and the variance share. Its output is
  `results/residualization_r2.json`, which holds the numbers reported in the paper.

## Recomputing the paper's numbers

```
python3 findings/b0_make_targets.py                      # needs PDBbind, writes findings_out/targets.tsv
REFERENCE_LOGS=<dir with h1_results.json, performance_table.json, embedding_health_2a.json> \
  python3 -I paper_analysis/analysis.py
python3 -I paper_analysis/analysis2.py
```

Outputs go to `paper_analysis/generated/` (override with `PAPER_OUT`). With the result tables in `findings_out/`
these scripts reproduce the paper's `numbers.json`, `numbers2.json` and tables.

## Environment

See `requirements.txt`. The sweep forward passes are NumPy float32 re-implementations, so the exact random
weights can depend on the NumPy version.

## Data terms

PDBbind and LP-PDBBind are used under their own terms of use. Only per-complex identifiers and numeric results are
included in this repository.
