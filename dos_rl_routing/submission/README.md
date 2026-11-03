# Reinforcement learning for sensor-network routing

This folder contains the routing simulator, the learning algorithms, experiment scripts, tests and saved ARQ-corrected results. The simulator is an analytical flow model written in Python; it is not a physical or packet-level network simulation.

## Folder guide

| Path | Contents |
| --- | --- |
| `dos_rl/mdp.py` | Route episodes, next-hop actions, rewards and transitions |
| `dos_rl/agents/tabular.py` | Q-learning, SARSA and Monte Carlo control |
| `dos_rl/agents/linear_fa.py` | Linear Q-function approximation |
| `dos_rl/network_sim.py` | Network flow, queue, retransmission and energy model |
| `dos_rl/policy.py` | Learned and non-RL routing policies |
| `experiments/` | Learning curves, sweeps, failures, ablations and transfer experiments |
| `full_state_experiment/` | Separate route-snapshot state comparison |
| `tests/` | Tests for the formulation, agents and simulator |
| `results/` | Saved CSVs from the completed corrected suite |
| `figures/` | Selected learning and network figures |
| `RESULTS.md` | Guide to the saved results and their interpretation |

## Setup

Use Python 3.11 or newer. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The Python experiment and test commands below also work on Windows. The full-suite shell runner requires Bash and GNU/Linux utilities (`flock`, `realpath`, `sha256sum`, `find` and `xargs`).

## Run the tests

```bash
python -m unittest tests.test_formulation tests.test_implementation_audit tests.test_paper_fidelity full_state_experiment.test_full_state -v
python -m tests.test_network_sim
```

The second command runs the standalone network-simulator checks. Test success provides regression coverage, not proof of routing performance.

## Run an experiment

A short learning-curve run compares DOS Q-learning, weighted-sum Q-learning, SARSA, Monte Carlo control and a linear approximator:

```bash
python -m experiments.e0_learning_curves --quick --test-episodes 30
```

New results and plots go to `runs/experiments/` by default, not to the saved `results/` and `figures/` folders. A custom output folder can be selected before running an experiment:

```bash
DOS_RL_OUTPUT_BASE="$PWD/runs/another_run" python -m experiments.e0b_coldstart --quick
```

The saved CSV `e0_curves_raw.csv` is named after the original experiment output; its values are 20-episode moving averages, not individual unsmoothed episode returns.

To run the complete corrected network suite on GNU/Linux:

```bash
bash run_arq_fixed_suite.sh
```

It writes to `runs/full_suite/`, runs the separate full-state experiment, all main network experiments and figure/report stages, and can take substantial time. Use `bash run_arq_fixed_suite.sh --resume` after an interruption. Resume is allowed only when source and dependency fingerprints match. To use a different empty output directory, set `DOS_RL_OUTPUT_BASE` before running the script.

## Rebuild the two comparison plots

```bash
python plot_comparisons.py
```

The script reads the saved CSVs in `results/` and creates the offered-load and link-outage plots in `runs/comparison_plots/`. It does not overwrite the saved figures.

## Reading the results

- The main tabular key contains only the current relay ID; it omits changing network information and is not proven to be a complete Markov state. The full-snapshot experiment is separate and models a route episode, not the full deployed network.
- Route tests use new random episodes on the same topology used to train the agent. In the deployment battery, each RL agent is retrained on the deployment being evaluated. The generalisation experiment separately tests transfer without retraining.
- Routes can be replanned during network evaluation, but the reported runs keep the learned Q-values fixed. PDR and delay are analytical estimates. The reported energy-efficiency metric is the PDR fraction divided by average network power.
- The saved outputs use the corrected expected number of ARQ attempts. Previous outputs produced with the earlier attempt-count formula are not included. `reference_run_versions.txt` records the original run's software versions; different versions can produce slightly different results or plots.
- To make the code easier to read, module descriptions and comments were shortened in this copy. The core simulator and agent logic was kept unchanged; experiment output defaults were redirected to `runs/` so saved results are not overwritten.

The routing formulation is motivated by Godfrey et al., *Sensors* 23, 8435 (2023), https://doi.org/10.3390/s23208435.
