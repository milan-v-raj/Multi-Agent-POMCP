#!/usr/bin/env bash
# Run the route and network experiments with the configured output directory.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export DOS_RL_OUTPUT_BASE="${DOS_RL_OUTPUT_BASE:-$PWD/runs/full_suite}"
case "$(realpath -m "$DOS_RL_OUTPUT_BASE")" in
  "$(realpath "$PWD")") echo "Refusing to overwrite historical results" >&2; exit 1 ;;
esac
mkdir -p "$DOS_RL_OUTPUT_BASE/results" "$DOS_RL_OUTPUT_BASE/figures" "$DOS_RL_OUTPUT_BASE/stages"
if [[ "${1:-}" != '--resume' ]] && find "$DOS_RL_OUTPUT_BASE/results" -maxdepth 1 -name '*.csv' -print -quit | grep -q .; then
  echo "Output directory already has CSVs; use --resume or a new DOS_RL_OUTPUT_BASE" >&2
  exit 1
fi
printf 'running\n' > "$DOS_RL_OUTPUT_BASE/status.txt"
trap 'printf "failed (exit %s)\n" "$?" > "$DOS_RL_OUTPUT_BASE/status.txt"' ERR
run_stage() {
  local name=$1; shift
  if [[ -f "$DOS_RL_OUTPUT_BASE/stages/$name.done" ]]; then
    echo "=== $name: already complete ==="
  else
    echo "=== $name ==="
    "$@"
    printf 'complete\n' > "$DOS_RL_OUTPUT_BASE/stages/$name.done"
  fi
}
run_stage full_state "$PY" -m full_state_experiment.run --seeds 8 --episodes 3000 --test-episodes 30 --output "$DOS_RL_OUTPUT_BASE/results/full_state_comparison.csv"
run_stage e0 "$PY" -m experiments.e0_learning_curves --episodes 3000 --seeds 8 --test-episodes 30
# E0b defaults to 30 test episodes; its CLI does not expose a test-episodes flag.
run_stage e0b "$PY" -m experiments.e0b_coldstart --episodes 1500 --seeds 10
run_stage e2_e3 "$PY" -m experiments.e2_e3_network_benchmark --seeds 10 --episodes 2000
run_stage e4 "$PY" -m experiments.e4_parameter_sweep --seeds 6 --episodes 2500
run_stage e5 "$PY" -m experiments.e5_environment_change --seeds 8 --episodes 2000
run_stage e6 "$PY" -m experiments.e6_ablation_and_stats --seeds 10 --episodes 2000 --test-seeds 30
run_stage e7 "$PY" -m experiments.e7_generalisation --train-episodes 2500 --test-seeds 20
run_stage figures "$PY" -m experiments.make_figures
run_stage report "$PY" -m experiments.corrected_report
printf 'complete\n' > "$DOS_RL_OUTPUT_BASE/status.txt"
echo 'CORRECTED NETWORK SUITE COMPLETE'
