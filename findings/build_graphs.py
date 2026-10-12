"""build_graphs.py -- parse PDBbind pocket + ligand files into one atom-graph archive (no RDKit needed).
Output: findings_out/graphs.npz  (ids, split, offsets, xyz, elem, is_lig, lig_charge, rot_bonds) + graphs_failed.txt
Atom rules mirror make_covariates.py: pocket = ATOM/HETATM minus waters (HOH/WAT/TIP/DOD), non-H (element cols 77-78, atom-name fallback,
asserted < 0.1 % of records); ligand = first molecule of the V2000 SDF, non-H. Failed complexes are listed, never imputed."""
import sys
import numpy as np
import fcommon as fc

WATER = {"HOH", "WAT", "TIP", "DOD"}

def parse_pdb(path):
    xyz, el, miss, tot = [], [], 0, 0
    for line in open(path):
        if not line.startswith(("ATOM  ", "HETATM")): continue
        if line[17:20].strip() in WATER: continue
        e = line[76:78].strip()
        tot += 1
        if not e:
            miss += 1; e = line[12:14].strip().lstrip("0123456789")
        e = e.capitalize()
        if e in ("H", "D"): continue
        xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54]))); el.append(e)
    return np.array(xyz, float).reshape(-1, 3), el, miss, tot

CHG = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}

def parse_sdf(path):
    L = open(path).read().split("\n")
    if len(L) < 5 or "V3000" in L[3]: raise ValueError("only V2000 SDF supported")
    na, nb = int(L[3][0:3]), int(L[3][3:6])
    xyz, el, chg = [], [], []
    for i in range(na):
        s = L[4 + i]
        xyz.append((float(s[0:10]), float(s[10:20]), float(s[20:30]))); el.append(s[31:34].strip().capitalize()); chg.append(CHG.get(int(s[36:39] or 0), 0))
    bonds = []
    for i in range(nb):
        s = L[4 + na + i]; bonds.append((int(s[0:3]) - 1, int(s[3:6]) - 1, int(s[6:9])))
    ov = {}
    for s in L[4 + na + nb:]:
        if s.startswith("M  CHG"):
            n = int(s[6:9])
            for j in range(n): ov[int(s[10 + 8 * j:14 + 8 * j]) - 1] = int(s[14 + 8 * j:18 + 8 * j])
        if s.startswith("$$$$") or s.startswith("M  END"): break
    for k, v in ov.items(): chg[k] = v
    heavy = [i for i, e in enumerate(el) if e not in ("H", "D")]
    return np.array(xyz, float)[heavy], [el[i] for i in heavy], int(sum(chg[i] for i in range(na))), heavy, bonds

def rotatable_bonds(n_heavy_idx, bonds):
    """Acyclic (bridge) single bonds between two heavy atoms that both have heavy degree > 1."""
    idx = {a: i for i, a in enumerate(n_heavy_idx)}
    edges = [(idx[a], idx[b], t) for a, b, t in bonds if a in idx and b in idx]
    n = len(idx); adj = [[] for _ in range(n)]
    for k, (a, b, _) in enumerate(edges): adj[a].append((b, k)); adj[b].append((a, k))
    disc, low, bridge, t = [-1] * n, [0] * n, set(), [0]
    for r in range(n):
        if disc[r] != -1: continue
        stack = [(r, -1, iter(adj[r]))]; disc[r] = low[r] = t[0]; t[0] += 1
        while stack:
            v, pe, it = stack[-1]
            for w, k in it:
                if k == pe: continue
                if disc[w] == -1:
                    disc[w] = low[w] = t[0]; t[0] += 1; stack.append((w, k, iter(adj[w]))); break
                low[v] = min(low[v], disc[w])
            else:
                stack.pop()
                if stack:
                    u = stack[-1][0]; low[u] = min(low[u], low[v])
                    if low[v] > disc[u]: bridge.add(pe)
    return sum(1 for k, (a, b, ty) in enumerate(edges) if ty == 1 and k in bridge and len(adj[a]) > 1 and len(adj[b]) > 1)

def main():
    plan = fc.load_plan(); fc.require_lock()
    elems = plan["graphs"]["elements"]; emap = {e: i for i, e in enumerate(elems)}; other = len(elems)
    sp = fc.split_df(plan)
    use = sp[sp[plan["split_column"]].isin(plan["graphs"]["splits"])]
    root = fc.ROOT / plan["pdbbind_dir"]
    ids, split, off, XYZ, EL, LG, CH, RB, failed = [], [], [0], [], [], [], [], [], []
    miss = tot = 0
    for pid, row in use.iterrows():
        d = root / pid
        try:
            px, pe, m, t = parse_pdb(d / plan["graphs"]["pocket_file"].format(pid=pid))
            lx, le, ch, heavy, bonds = parse_sdf(d / plan["graphs"]["ligand_file"].format(pid=pid))
            assert len(px) > 0 and len(lx) > 0
        except Exception as ex:
            failed.append(f"{pid}\t{type(ex).__name__}: {ex}"); continue
        miss += m; tot += t
        ids.append(pid); split.append(row[plan["split_column"]]); off.append(off[-1] + len(px) + len(lx))
        XYZ.append(np.vstack([px, lx])); EL += [emap.get(e, other) for e in pe + le]
        LG += [0] * len(px) + [1] * len(lx); CH.append(ch); RB.append(rotatable_bonds(heavy, bonds))
    fc.OUT.mkdir(exist_ok=True)
    (fc.OUT / "graphs_failed.txt").write_text("\n".join(failed))
    if tot and miss / tot >= 0.001: raise fc.FindingsError(f"{miss / tot:.2%} of pocket records lack an element column")
    if len(failed) > 0.005 * len(use): raise fc.FindingsError(f"{len(failed)} / {len(use)} complexes failed to parse (see graphs_failed.txt)")
    np.savez_compressed(fc.OUT / "graphs.npz", ids=np.array(ids), split=np.array(split), offsets=np.array(off, np.int64),
                        xyz=np.vstack(XYZ).astype(np.float32), elem=np.array(EL, np.int8), is_lig=np.array(LG, bool),
                        lig_charge=np.array(CH, np.int16), rot_bonds=np.array(RB, np.int16), n_elem=len(elems) + 1)
    fc.ledger("build_graphs", "graphs", f"{len(ids)} complexes, {len(failed)} failed", "planned")
    print(f"graphs: {len(ids)} complexes, {len(failed)} failed")

class Graphs:
    """Loaded archive with per-complex slicing."""
    def __init__(self, path=None):
        z = np.load(path or fc.OUT / "graphs.npz", allow_pickle=False)
        self.ids = [s for s in z["ids"].tolist()]; self.split = z["split"]; self.off = z["offsets"]; self.xyz = z["xyz"].astype(np.float64)
        self.elem = z["elem"].astype(np.int64); self.is_lig = z["is_lig"]; self.lig_charge = z["lig_charge"]; self.rot = z["rot_bonds"]
        self.n_elem = int(z["n_elem"]); self.index = {p: i for i, p in enumerate(self.ids)}
    def get(self, pid):
        i = self.index[pid]; s = slice(self.off[i], self.off[i + 1])
        return self.xyz[s], self.elem[s], self.is_lig[s]
    def rows(self, pids):
        miss = [p for p in pids if p not in self.index]
        if miss: raise fc.FindingsError(f"{len(miss)} ids have no parsed graph (e.g. {miss[:3]})")
        return [self.index[p] for p in pids]

if __name__ == "__main__":
    main()
