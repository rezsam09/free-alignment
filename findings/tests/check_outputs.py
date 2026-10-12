"""Checks of the planted truths in the synthetic fixture (affinity has a real training gain; size targets do not)."""
import os, sys
import numpy as np, pandas as pd
O = os.path.join(os.environ["PROJECT_ROOT"], "findings_out")
rd = lambda n: pd.read_csv(os.path.join(O, n), sep="\t")
def check(name, cond, info=""):
    assert cond, f"FAIL {name} {info}"; print("ok  ", name)
P = rd("res_sweep_pairs.tsv"); check("sweep: all CKA finite and <= 1", np.isfinite(P.cka).all() and (P.cka <= 1.0 + 1e-9).all())
check("sweep: 25 cross pairs per config (3 seeds -> 9)", (P[P.kind == "cross"].groupby("cfg_id").size() == 9).all())
check("sweep: matched flag marks the diagonal only", (P[(P.kind == "cross")].eval("matched == (i == j)")).all())
V = rd("res_floor_model_verdict.tsv"); check("floor model: 3 responses", len(V) == 3 and V.n_configs.min() >= 20)
KK = rd("res_kernel_kernel.tsv"); check("kernel-kernel CKA table complete (6 pairs)", len(KK) == 6 and np.isfinite(KK.cka).all())
F = rd("res_real_floor_pairing.tsv"); check("real floor table: 3 model pairs, matched ~ mismatched", len(F) == 3 and (F.matched_minus_mismatched.abs() < 0.15).all())
AB = rd("res_ablation_boot.tsv"); check("ablation: 4 conditions with CI", len(AB) == 4 and AB.ci_lo.notna().all())
M = rd("res_probe_metrics.tsv"); G = rd("res_probe_gain.tsv")
m = lambda inp, tg: M[(M.input == inp) & (M.target == tg)].iloc[0]
check("size kernel decodes ligand_heavy_atoms (R2>0.95)", m("kernel/size", "ligand_heavy_atoms").metric > 0.95)
check("kernel decodes contacts partly", m("kernel/all_kernels", "contacts").metric > 0.2)
g = lambda mod, tg, rep="primary": G[(G.model == mod) & (G.target == tg) & (G.replicate == rep)].iloc[0]
for mod in ("egnn", "distance_gnn", "gnina"):
    a = g(mod, "affinity"); check(f"planted: {mod} affinity gain present", a.gain > 0.15 and a.verdict == "gain", a.to_dict())
    h = g(mod, "ligand_heavy_atoms"); check(f"planted: {mod} NO gain on ligand_heavy_atoms (kernel covers it)", h.verdict != "gain" and h.gain < 0.05, h.to_dict())
t = m("egnn/primary", "affinity"); check("probe above shuffled-target null", t.above_null and t.metric > t.null_p95)
Mc, Mb = M[M.type == "continuous"], M[M.type == "binary"]
check("shuffled null: R2 near/below 0, AUROC near 0.5", Mc.null_mean.abs().max() < 0.3 and (Mb.null_mean - 0.5).abs().max() < 0.15)
check("probe tables complete (dims, mlp)", len(rd("res_probe_dims.tsv")) > 50 and len(rd("res_probe_mlp.tsv")) > 50)
check("dims table has the 'full' setting for graph models", (rd("res_probe_dims.tsv").dim_setting == "full").any())
A = rd("res_output_agreement.tsv"); check("agreement: finite values, 3 pairs x 5 stats", len(A) == 15 and A.value.notna().all())
L = pd.read_csv(os.path.join(O, "ledger.tsv"), sep="\t"); check("ledger written, all rows 'planned'", len(L) > 10 and (L.status == "planned").all())
print("\nALL OUTPUT CHECKS PASSED")
