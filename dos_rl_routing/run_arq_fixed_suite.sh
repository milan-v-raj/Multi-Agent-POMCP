#!/usr/bin/env bash
# Full network suite with fixed expected ARQ attempts, in a fresh output folder.
# Existing results/ and corrected_suite/ are not overwritten.
set -euo pipefail
cd "$(dirname "$0")"
export DOS_RL_OUTPUT_BASE="${DOS_RL_OUTPUT_BASE:-$PWD/arq_fixed_suite}"
output=$(realpath -m "$DOS_RL_OUTPUT_BASE")
for protected in "$PWD" "$PWD/results" "$PWD/figures" "$PWD/corrected_suite"; do
  protected_path=$(realpath -m "$protected")
  if [[ "$output" == "$protected_path" ]] || { [[ "$protected_path" != "$PWD" ]] && [[ "$output" == "$protected_path/"* ]]; }; then
    echo "Refusing to use protected historical output: $output" >&2
    exit 1
  fi
done
mkdir -p "$output"
# Keep one writer per output directory, including during --resume.
lock_key=$(printf '%s' "$output" | sha256sum | cut -d' ' -f1)
exec 9>"${TMPDIR:-/tmp}/arq-suite-$lock_key.lock"
flock -n 9 || { echo 'A suite is already running for this output directory' >&2; exit 1; }
model=truncated-geometric-v1
fingerprint=$(sha256sum dos_rl/network_sim.py | cut -d' ' -f1)
source_fingerprint=$((find dos_rl experiments full_state_experiment -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum; sha256sum run_corrected_network.sh) | sha256sum | cut -d' ' -f1)
versions=$(${PY:-python3} -c 'import sys, numpy, scipy, matplotlib; print(f"Python {sys.version}\nNumPy {numpy.__version__}\nSciPy {scipy.__version__}\nMatplotlib {matplotlib.__version__}")')
if [[ -f "$output/arq_model.txt" ]]; then
  [[ "$(cat "$output/arq_model.txt")" == "$model" ]] || { echo 'ARQ model marker mismatch' >&2; exit 1; }
  [[ -f "$output/network_sim.sha256" && "$(cat "$output/network_sim.sha256")" == "$fingerprint" ]] || {
    echo 'Simulator code changed since this run started; choose a fresh output directory' >&2; exit 1;
  }
  [[ -f "$output/suite_source.sha256" && "$(cat "$output/suite_source.sha256")" == "$source_fingerprint" ]] || {
    echo 'Suite source changed since this run started; choose a fresh output directory' >&2; exit 1;
  }
  [[ -f "$output/software_versions.txt" && "$(cat "$output/software_versions.txt")" == "$versions" ]] || {
    echo 'Software versions changed since this run started; choose a fresh output directory' >&2; exit 1;
  }
else
  if find "$output" -type f -print -quit | grep -q .; then
    echo 'Refusing to mark existing files as ARQ-fixed; choose an empty directory' >&2
    exit 1
  fi
  printf '%s\n' "$model" > "$output/arq_model.txt"
  printf '%s\n' "$fingerprint" > "$output/network_sim.sha256"
  printf '%s\n' "$source_fingerprint" > "$output/suite_source.sha256"
  printf '%s\n' "$versions" > "$output/software_versions.txt"
fi
bash run_corrected_network.sh "$@"
