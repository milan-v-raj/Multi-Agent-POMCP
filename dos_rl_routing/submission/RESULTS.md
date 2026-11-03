# Results guide

The CSVs in `results/` are from the completed ARQ-corrected network suite. The expected number of transmission attempts is `sum((1-p)**k for k in range(R))`, where `p` is per-attempt success probability and `R` is the maximum number of attempts. The simulator estimates flows, queueing, energy and delay; it does not simulate physical packets.

## Route-task comparison

Source: [`e0_summary.csv`](results/e0_summary.csv). Each learner was trained for 3,000 episodes and tested on 30 independently sampled episodes per seed, across eight seeds. Test episodes share their agent's training topology.

| Learner | Mean test success | Mean test return |
| --- | ---: | ---: |
| DOS Q-learning | 93.33% | 10.48 |
| Weighted-sum Q-learning | 94.17% | 10.67 |
| SARSA | 95.00% | 10.84 |
| Monte Carlo | 95.00% | 10.85 |
| Linear Q approximation | 92.92% | 10.46 |

These are descriptive means, not evidence of a statistically significant ranking. `e0_curves_raw.csv` contains 20-episode moving averages.

## Matched deployment battery

Sources: [`e6_test_battery.csv`](results/e6_test_battery.csv) and [`e6_statistics.csv`](results/e6_statistics.csv). The battery contains 30 deployments; each RL agent was trained on the deployment being evaluated.

| Policy | Mean PDR | Mean delay | Mean PDR/W | Mean energy |
| --- | ---: | ---: | ---: | ---: |
| DOS Q-learning | 87.15% | 62.46 ms | 22.02 | 8.65 J |
| Weighted-sum Q-learning | 87.07% | 61.90 ms | 21.76 | 8.71 J |
| Linear approximation | 86.76% | 61.75 ms | 21.41 | 8.81 J |
| Re-planned shortest path | 86.68% | 61.47 ms | 21.60 | 8.76 J |
| Static OSPF-like | 86.68% | 61.47 ms | 21.60 | 8.76 J |
| Energy-aware greedy | 74.80% | 51.67 ms | 21.37 | 7.50 J |

PDR/W is the mean of per-run PDR fractions divided by average network power, not lifetime. DOS minus shortest path has a mean paired PDR difference of +0.46 percentage points; the Wilcoxon p-value is 0.325, so an advantage is not established. Delay is an analytical estimate, and low energy consumption is not useful on its own if delivery is poor.

## Temporary link outages

Sources: [`e5b_link_failure.csv`](results/e5b_link_failure.csv) and [`e5_paired_tests.csv`](results/e5_paired_tests.csv). Each condition contains eight paired deployments. Post-onset PDR covers the outage and the subsequent recovery period.

| Temporarily unavailable links | DOS PDR | Shortest-path PDR | Paired difference | Wilcoxon p-value |
| --- | ---: | ---: | ---: | ---: |
| 20% | 75.06% | 72.19% | +2.87 pp | 0.109 |
| 40% | 74.12% | 69.25% | +4.87 pp | 0.078 |

DOS has higher mean PDR in these samples, but neither result meets the conventional 0.05 significance threshold. Report this as a possible scenario-specific benefit, not a proven universal advantage.

## Other files

- `e0b_coldstart.csv`: harder route task and learning progress.
- `e2_e3_raw.csv` and `e2_e3_paired_tests.csv`: load, source count and relay-density sweeps.
- `e4_parameter_sweep.csv`: learning rate, discount and exploration sweeps.
- `e5*.csv`: arrival-rate and failure scenarios.
- `e6_ablation.csv`: shaping, objective selection and action-mask variants.
- `e7_generalisation.csv`: zero-shot transfer to new deployments.
- `full_state_comparison.csv`: separate relay-only versus expanded route-state experiment.

The model uses an approximate relay-only tabular state, a restricted action set and analytical network dynamics. The expanded-state comparison is limited to the route-episode model. See `README.md` for how to run the code and tests.
