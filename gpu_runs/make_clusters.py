#!/usr/bin/env python
"""make_clusters.py -- protein clusters for the cluster bootstrap: MMseqs2 at config analysis_rules.cluster_mmseqs
(min_seq_id 0.3, coverage 0.8, bidirectional coverage).  Output: data/clusters.tsv (index pdb_id, column cluster_id) + logs/clusters_meta.json.

Every protein chain of every complex ({pid}_protein.pdb, CA atoms -> sequence) is clustered; a complex belongs to ONE cluster: complexes that
share any chain cluster are merged (union-find), i.e. the coarsest/most conservative grouping, so homologous complexes are never split across
bootstrap units. Needs the `mmseqs` binary on PATH. Uses ROOT/PDBbind_2020/<pid>/<pid>_protein.pdb for all ids in the split file.
NOTE: if your prereg lock pins data/clusters.tsv, record the change with prereg.apply_deviation / re-lock.
"""
import json, os, subprocess, sys, shutil, tempfile
from pathlib import Path
import pandas as pd, yaml

ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/ec2-user/SageMaker/pdbbind_data")).resolve()
for p in (str(ROOT), str(ROOT / "src")):
    if p in sys.path: sys.path.remove(p)
    sys.path.insert(0, p)
from src import prereg
prereg.verify_lock(ROOT)
cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
MM = cfg["analysis_rules"]["cluster_mmseqs"]; SEQ_ID, COV = MM["min_seq_id"], MM["coverage"]
PDB_DIR = Path(os.environ.get("PDBBIND_DIR", ROOT / "PDBbind_2020"))
assert shutil.which("mmseqs"), "mmseqs not on PATH"
AA = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M",
      "PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","MSE":"M","HID":"H","HIE":"H","HIP":"H","CYX":"C"}

def chains(pdb):
    seq = {}; seen = set()
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ch, key = line[21], (line[21], line[22:27])
            if key in seen: continue
            seen.add(key); seq.setdefault(ch, []).append(AA.get(line[17:20].strip(), "X"))
    return {c: "".join(s) for c, s in seq.items() if len(s) >= 20}

split = pd.read_csv(ROOT / cfg["split"]["file"], sep="\t", index_col=0); pids = [str(p).lower() for p in split.index]
tmp = Path(tempfile.mkdtemp()); fa = tmp / "chains.fasta"; n_chain = 0; no_seq = []
with open(fa, "w") as f:
    for pid in pids:
        pdb = PDB_DIR / pid / f"{pid}_protein.pdb"
        ch = chains(pdb) if pdb.exists() else {}
        if not ch: no_seq.append(pid); continue
        for c, s in ch.items(): f.write(f">{pid}|{c}\n{s}\n"); n_chain += 1
print(f"{len(pids)-len(no_seq)} complexes, {n_chain} chains; {len(no_seq)} without a protein sequence (left unclustered = singleton later)")
subprocess.run(["mmseqs", "easy-cluster", str(fa), str(tmp / "out"), str(tmp / "work"), "--min-seq-id", str(SEQ_ID), "-c", str(COV),
                "--cov-mode", "0"], check=True)
pairs = [l.rstrip("\n").split("\t")[:2] for l in open(tmp / "out_cluster.tsv")]   # representative \t member
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
for rep, mem in pairs:
    find(rep); find(mem)
    # every chain joins its representative's group; complex nodes join through their chains
    parent[find(mem)] = find(rep)
    parent[find("cx:" + mem.split("|")[0])] = find(rep)
    parent[find("cx:" + rep.split("|")[0])] = find(rep)
roots = {}; rows = []
for pid in pids:
    if "cx:" + pid not in parent: continue
    rows.append((pid, roots.setdefault(find("cx:" + pid), len(roots))))
out = pd.DataFrame(rows, columns=["pdb_id", "cluster_id"]).set_index("pdb_id")
out.to_csv(ROOT / "data" / "clusters.tsv", sep="\t")
meta = dict(tool="mmseqs easy-cluster", min_seq_id=SEQ_ID, coverage=COV, cov_mode=0, n_complexes=len(out), n_clusters=int(out.cluster_id.nunique()),
            n_without_sequence=len(no_seq), complex_rule="complexes sharing any chain cluster are merged (union-find)",
            mmseqs_version=subprocess.run(["mmseqs", "version"], capture_output=True, text=True).stdout.strip())
(ROOT / "logs" / "clusters_meta.json").write_text(json.dumps(meta, indent=2)); print(meta)
