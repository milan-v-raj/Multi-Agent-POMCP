# CMOMDP-based routing for wireless sensor networks

> **ARQ correction:** The full network suite was rerun with the corrected
> expected attempt count. Earlier `corrected_suite/` network CSVs are preserved
> as historical; see the [measured impact](ARQ_IMPACT.md) and
> [fidelity audit](PAPER_FIDELITY.md).


A course project based on Godfrey et al., *An Energy-Efficient Routing Protocol
with Reinforcement Learning in Software-Defined Wireless Sensor Networks*,
[doi:10.3390/s23208435](https://doi.org/10.3390/s23208435). A central controller
chooses relay next hops using energy, load and link-quality objectives.

## Start here

- **Final submission report:** [PDF](FINAL_REPORT.pdf) · [editable Markdown](FINAL_REPORT.md).
- [Concise report](REPORT.md) — formulation, corrected experiments and conclusions.
- [Rubric map](RUBRIC.md) — where each requirement is answered.
- [ARQ-fixed full result tables](arq_fixed_suite/RESULTS.md) and [figures](arq_fixed_suite/figures/).
- [Measured ARQ impact](ARQ_IMPACT.md) — earlier versus fixed runs.
- [Earlier CMOMDP work](CORRECTED_CMOMDP.md) — prior plan and historical results.
- [Formal CMOMDP](MDP.md), [paper/code comparison](CMOMDP_COMPARISON.md), and
  [full paper-fidelity audit](PAPER_FIDELITY.md).
- [Viva notes](VIVA.md).

The optional [partial-observability extension](POMDP_EXTENSION.md) and
[expanded-state experiment](full_state_experiment/) are separate from the core
paper-focused model.

## Core model

| Component | Implementation |
|---|---|
| State | Current relay, as in the paper; this compact key is not proven Markov. |
| Action | An eligible next-hop relay or the directly reachable gateway. |
| Rewards | Separate energy, load and PRR objectives, with potential shaping in the corrected updates. |
| DOS rule | The default selects the objective with the highest estimated Q-value, not highest certainty. |
| Algorithms | DOS Q-learning, weighted-sum Q-learning, SARSA and Monte Carlo; a linear approximation is also evaluated. |

## Selected corrected results

| Comparison | Result | Interpretation |
|---|---|---|
| Cold-start route evaluation | DOS 82.0% success; weighted-sum Q 79.7% | Descriptive means over 10 seeds, with 30 test episodes per seed; not proof of a significant difference. |
| Shaping ablation | DOS PDR 84.14% with shaping; 84.46% without | Small mean difference over 10 deployments; no significance established. |
| 40% transient link outage | DOS post-onset PDR 74.12% vs shortest path 69.25%; p = 0.078 | The corrected comparison is not significant at the 0.05 level. |
| Full route-snapshot keys | 0 / 1,095 test decisions reused a training key | Exact tabular snapshots do not generalise in this experiment. |

The simulator is a custom flow model, not ns-3. The paper's SDN-Q baseline is
not implemented, and PDR per watt differs from the paper's energy metric.

## Run and test

```bash
cd dos_rl_routing
python3 -m pip install numpy scipy matplotlib
python3 -m unittest tests.test_implementation_audit tests.test_formulation tests.test_paper_fidelity full_state_experiment.test_full_state -v
python3 -m tests.test_network_sim

# Run into a new directory to preserve the saved corrected suite:
DOS_RL_OUTPUT_BASE="$PWD/my_new_run" bash run_arq_fixed_suite.sh
# Resume the same directory after an interruption:
DOS_RL_OUTPUT_BASE="$PWD/my_new_run" bash run_arq_fixed_suite.sh --resume
```

The earlier suite is preserved in `corrected_suite/`; the new ARQ-fixed full
suite writes to `arq_fixed_suite/`. Individual
experiment modules default to `results/` and `figures/`, so set
`DOS_RL_OUTPUT_BASE` to a separate directory when rerunning them. `run_all.sh`
also runs the optional gridworld and telemetry studies and writes separately
by default.

## Historical results

`results/`, `figures/`, [QUANTITATIVE_RESULTS.md](QUANTITATIVE_RESULTS.md) and
[REPORT_full.md](REPORT_full.md) describe older runs. Those runs used a network
baseline that learned only the first objective despite its weighted-sum label;
network Q-learning and SARSA also omitted shaping, and the linear agent had
update defects. The corrected code and suite must not be mixed with those
historical numbers.
