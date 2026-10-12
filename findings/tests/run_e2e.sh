#!/bin/sh
# End-to-end test on SYNTHETIC data (planted signals; results are meaningless except for the checks in check_outputs.py).
# Usage: sh findings/tests/run_e2e.sh <base_project_dir> [workdir]     (base_project_dir = the reference-model project that contains tests/make_fixture.py)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); BASE=$(cd "$1" && pwd); W=${2:-/tmp/findings_e2e}
python3 "$HERE/make_fixture_findings.py" "$W" "$BASE"
mkdir -p "$W/findings/tests"; cp "$HERE"/test_findings.py "$HERE"/check_outputs.py "$W/findings/tests/"
export PROJECT_ROOT="$W"; cd "$W"
python3 findings/tests/test_findings.py
# order discipline: nothing result-producing may run before lock / freeze
if python3 findings/a1_floor_sweep.py >/dev/null 2>&1; then echo "FAIL: a1 ran without lock"; exit 1; fi
python3 findings/lock_freeze.py lock
if python3 findings/a1_floor_sweep.py >/dev/null 2>&1; then echo "FAIL: a1 ran without freeze"; exit 1; fi
python3 findings/build_graphs.py; python3 findings/input_kernels.py; python3 findings/b0_make_targets.py >/dev/null
python3 findings/a1_floor_sweep.py --estimate
python3 findings/lock_freeze.py freeze
for s in a1_floor_sweep a4_ablation a3_floor_model a5_a7_real_untrained b_probe_panel c3_output_agreement; do
  echo "=== $s"; python3 findings/$s.py > logs/_f_$s.out 2>&1 || { tail -30 logs/_f_$s.out; echo "FAILED: $s"; exit 1; }
done
# frozen-code / plan tamper checks
echo "# tamper" >> findings/c3_output_agreement.py
if python3 findings/c3_output_agreement.py >/dev/null 2>&1; then echo "FAIL: ran after code changed post-freeze"; exit 1; fi
sed -i '$ d' findings/c3_output_agreement.py
python3 findings/c3_output_agreement.py >/dev/null
sed -i 's/n_boot: 40/n_boot: 41/' findings/plan/findings_plan.yaml
if python3 findings/c3_output_agreement.py >/dev/null 2>&1; then echo "FAIL: ran after plan changed post-lock"; exit 1; fi
sed -i 's/n_boot: 41/n_boot: 40/' findings/plan/findings_plan.yaml
python3 findings/tests/check_outputs.py
echo "FINDINGS E2E OK"
