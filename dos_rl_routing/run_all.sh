#!/usr/bin/env bash
# Full experiment suite. Every script is seeded and writes CSV + PNG artefacts.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
# Keep archived results/ and figures/ intact. This optional all-experiments
# runner also includes gridworld and telemetry; for the core-only corrected
# network suite use run_corrected_network.sh instead.
export DOS_RL_OUTPUT_BASE="${DOS_RL_OUTPUT_BASE:-$PWD/full_suite_output}"
if [[ "$(realpath -m "$DOS_RL_OUTPUT_BASE")" == "$(realpath "$PWD")" ]]; then
  echo "Refusing to overwrite historical outputs" >&2; exit 1
fi
if [[ -d "$DOS_RL_OUTPUT_BASE/results" ]] && find "$DOS_RL_OUTPUT_BASE/results" -maxdepth 1 -name '*.csv' -print -quit | grep -q .; then
  echo "Use an empty DOS_RL_OUTPUT_BASE to avoid overwriting another run" >&2; exit 1
fi

echo "=== E1 gridworld replication ==="
$PY -m experiments.e1_gridworld_replication --episodes 2000 --seeds 10

echo "=== E0 learning curves ==="
$PY -m experiments.e0_learning_curves --episodes 3000 --seeds 8

echo "=== E0b cold-start learning ==="
$PY -m experiments.e0b_coldstart --episodes 1500 --seeds 10

echo "=== E2/E3 network benchmark ==="
$PY -m experiments.e2_e3_network_benchmark --seeds 10 --episodes 2000

echo "=== E4 hyperparameter sweep ==="
$PY -m experiments.e4_parameter_sweep --seeds 6 --episodes 2500

echo "=== E5 environmental change ==="
$PY -m experiments.e5_environment_change --seeds 8 --episodes 2000

echo "=== E6 ablation + statistics ==="
$PY -m experiments.e6_ablation_and_stats --seeds 10 --episodes 2000 --test-seeds 30

echo "=== E7 generalisation ==="
$PY -m experiments.e7_generalisation --train-episodes 2500 --test-seeds 20

echo "=== E8 partial observability ==="
$PY -m experiments.e8_partial_observability --seeds 8 --episodes 2000

echo "=== figures for report ==="
$PY -m experiments.make_figures

echo "=== aggregate every quoted number ==="
$PY -m experiments.aggregate --csv

echo "ALL DONE"
