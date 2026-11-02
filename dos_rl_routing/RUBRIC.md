# Rubric map: corrected CMOMDP project

> **ARQ correction:** The full network suite was rerun with the corrected
> expected attempt count. Earlier `corrected_suite/` network CSVs are preserved
> as historical; see the [measured impact](ARQ_IMPACT.md) and
> [fidelity audit](PAPER_FIDELITY.md).


| Requirement | Evidence |
|---|---|
| Problem understanding and literature | [Report §1](REPORT.md): SDWSN routing and the framework paper. |
| CMOMDP formulation and rewards | [Report §2](REPORT.md), [MDP.md](MDP.md), and [paper/code comparison](CMOMDP_COMPARISON.md). |
| Working implementation | [Report §3](REPORT.md), `dos_rl/`, and the regression tests. |
| Reward versus training episode and independent tests | [Report §4](REPORT.md), corrected E0/E0b figures and per-seed results; each evaluation uses at least 30 route episodes per seed. |
| Interpretation and a parameter or environmental change | [Report §§4–5](REPORT.md): shaping ablation, learning-rate sweep and transient link outage. |

## Corrected results and tests

The completed full network suite and the separate full-snapshot comparison are
in [`arq_fixed_suite/`](arq_fixed_suite/). [RESULTS.md](arq_fixed_suite/RESULTS.md)
contains the saved quantitative tables. E0 compares Q-learning, weighted-sum
Q-learning, SARSA, Monte Carlo and linear approximation; E6 compares shaping
and other design choices. E6 retrains on each evaluated deployment, whereas
E7 separately evaluates transfer to new deployments.

Run the regression checks from `dos_rl_routing/`:

```bash
python3 -m unittest tests.test_implementation_audit tests.test_formulation tests.test_paper_fidelity full_state_experiment.test_full_state -v
python3 -m tests.test_network_sim
```

The suites pass **39 checks**, including three ARQ regression tests. The corrected network suite can be reproduced with
`bash run_arq_fixed_suite.sh` into an empty output directory; use `--resume`
for completed stages in the same directory. Historical `results/` and
`figures/` are not replaced.

## Scope and limitations

- The paper's relay-only state is an approximation in this simulator; the full
  route-snapshot experiment is separate and does not model the full physical network.
- Value-based DOS is the chosen default, not a certainty-based selector.
- The simulator does not reproduce ns-3 or the paper's SDN-Q baseline; the
  existing PDR-per-watt energy metric differs from the paper's metric.
- The optional [POMDP extension](POMDP_EXTENSION.md) is separate, not a full
  POMDP solver. Its historical E8 results were not rerun in the core suite.
- Earlier network results used an incorrectly scalarised baseline and omitted
  shaping from Q updates; do not use them as corrected comparisons. Historical
  material is retained in [REPORT_full.md](REPORT_full.md).
