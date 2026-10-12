"""b0_make_targets.py (B2) -- probe targets for the evaluation split -> findings_out/targets.tsv + targets_meta.json.
graph-derived (no RDKit): ligand_heavy_atoms, complex_atoms, contacts (ligand-pocket atom pairs < graphs.contact_cutoff A), pocket_volume_proxy (convex-hull volume of pocket atoms),
buried_fraction (fraction of ligand atoms with >= graphs.buried_min_neighbors pocket atoms within the cutoff), pocket_polar_frac (N/O share of pocket atoms),
rotatable_bonds (acyclic single bonds between heavy atoms of heavy degree > 1; a proxy for RDKit's strict definition), ligand_charge (sum of formal charges).
split: affinity.
Binary targets: column <= threshold_le -> 1.  Missing values stay NaN (the probe drops them per target; nothing is imputed)."""
import json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree, ConvexHull
import fcommon as fc
from build_graphs import Graphs

def graph_targets(g: Graphs, plan, pid):
    xyz, el, lig = g.get(pid); gp = plan["graphs"]; cut = gp["contact_cutoff"]
    P, L = xyz[~lig], xyz[lig]
    tp = cKDTree(P); nb = np.array([len(x) for x in tp.query_ball_point(L, cut)])
    try: vol = float(ConvexHull(P).volume)
    except Exception: vol = float("nan")
    names = gp["elements"]; pol = [names.index(e) for e in ("N", "O") if e in names]
    i = g.index[pid]
    return dict(ligand_heavy_atoms=float(lig.sum()), complex_atoms=float(len(el)), contacts=float(nb.sum()), pocket_volume_proxy=vol,
                buried_fraction=float((nb >= gp["buried_min_neighbors"]).mean()), pocket_polar_frac=float(np.isin(el[~lig], pol).mean()),
                rotatable_bonds=float(g.rot[i]), ligand_charge=float(g.lig_charge[i]))

def main():
    plan = fc.load_plan(); fc.require_lock()
    sp = fc.split_df(plan); g = Graphs()
    ids = [p for p in g.ids if sp.loc[p, plan["split_column"]] == plan["eval_split"]]
    T = pd.DataFrame(index=ids); meta = {}; specs = plan["targets"]["specs"]
    gt = pd.DataFrame([graph_targets(g, plan, p) for p in ids], index=ids)
    for name, s in specs.items():
        src = s["source"]; col = None
        if src == "split": col = sp.loc[ids, plan["affinity_column"]].astype(float)
        elif src == "graph": col = gt[name]
        else: raise fc.FindingsError(f"target {name}: unknown source {src!r}")
        if s["type"] == "binary":
            col = pd.Series(np.where(col.isna(), np.nan, (col <= s["threshold_le"]).astype(float)), index=ids)
        T[name] = col.astype(float).values; meta[name] = dict(type=s["type"], n_valid=int(T[name].notna().sum()), n_total=len(ids))
        if meta[name]["n_valid"] < plan["probe"]["min_n"]:
            fc.ledger("b0_make_targets", "warn", f"target {name} has only {meta[name]['n_valid']} valid values (< probe.min_n): will be skipped by the probe panel", "planned")
    T.index.name = "id"; T.to_csv(fc.OUT / "targets.tsv", sep="\t"); (fc.OUT / "targets_meta.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(meta).T.to_string())

def load_targets():
    T = pd.read_csv(fc.OUT / "targets.tsv", sep="\t", index_col=0); T.index = T.index.astype(str)
    return T, json.loads((fc.OUT / "targets_meta.json").read_text())

if __name__ == "__main__":
    main()
