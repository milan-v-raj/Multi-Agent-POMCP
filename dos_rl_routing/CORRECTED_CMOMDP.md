# Corrected CMOMDP work

> **Historical network results:** Values in this document predate the ARQ
> correction. The completed [ARQ-fixed results](arq_fixed_suite/RESULTS.md) and
> [impact comparison](ARQ_IMPACT.md) are separate; see also the
> [fidelity audit](PAPER_FIDELITY.md).


## Selected plan

- **Main model:** relay-only key from the paper; its compact representation is
  not proven Markov. Compare separately with the exact route-episode snapshot.
- **DOS:** retain the value-based selector as the default. It is not a
  certainty-based rule. E6 also reports alternative selection modes.
- **Shaping:** use it in Q-learning and SARSA updates and compare with the
  no-shaping setting. Terminal potential is set to zero in corrected episodes.
- **Experiments:** rerun the full applicable network suite in a separate folder;
  E1 gridworld and E8 telemetry remain separate, historical studies.

## Code corrections

The network single-objective Q learner now applies normalized configured
weights to all three rewards. Network Q-learning, SARSA and the linear learner
now include shaping in their updates. The linear update bootstraps per objective
and uses special gateway features instead of indexing the last relay. The E6
reward ablation now compares the default with `corrected_reward` rather than
repeating the default. Regression tests cover these changes; no old result was
relabelled as a corrected experiment.

## Results and status

**The full corrected network suite completed successfully.** The corrected suite writes to [`corrected_suite/`](corrected_suite/), not to
`results/`. Its [status file](corrected_suite/status.txt) and completed-stage
markers show whether the full run finished. Generate or refresh the
[quantitative tables](corrected_suite/RESULTS.md) with:

```bash
DOS_RL_OUTPUT_BASE="$PWD/corrected_suite" python3 -m experiments.corrected_report
```

The report shows only saved stages, with missing stages explicitly marked.
Start the full suite with `bash run_corrected_network.sh`; after an interrupted
stage, resume with `bash run_corrected_network.sh --resume`. Do not run it into
an existing unrelated output folder. The first attempted run stopped at E0b
because an unsupported CLI flag was supplied; completed full-state and E0
stages were preserved, the runner was fixed, and the suite was resumed.

### Completed: relay-only vs full route snapshot

Eight matched topology seeds, 3,000 training episodes and 30 held-out route
episodes per seed. Mean test metrics:

| Agent state | Test success | Test return | Evaluation decisions using a previously seen state |
|---|---:|---:|---:|
| Relay only | 91.7% | 10.52 | 1,102 / 1,102 |
| Full route snapshot | 90.4% | 10.24 | 0 / 1,095 |

The full key did not recur in these test episodes, so its policy mostly falls
back to untrained values. This comparison does **not** establish superiority of
either state representation or a deployable full-state network policy. The
snapshot covers the fixed-topology route-episode model, not the complete
physical network. Raw data: [`full_state_comparison.csv`](corrected_suite/results/full_state_comparison.csv).

### Completed: cold-start routing (E0b)

Ten topology seeds, 1,500 training episodes and 30 held-out route episodes per
seed. Training returns are descriptive averages, not a monotonicity claim.

| Learner | Training return, first 50 → last 100 | Held-out success | Held-out return |
|---|---:|---:|---:|
| Value-based DOS Q-learning | −3.30 → 1.38 | 82.0% | 8.88 |
| Weighted-sum Q-learning | −2.96 → 0.82 | 79.7% | 8.02 |

These are small-sample descriptive results, not a proven performance advantage.
Both policies use shaping. See the [figure](corrected_suite/figures/e0b_coldstart.png)
and [per-seed CSV](corrected_suite/results/e0b_coldstart.csv). E0 also contains
Q-learning, SARSA, Monte Carlo and linear-agent comparisons.

### Completed: algorithm comparison (E0)

Eight topology seeds; each learner was evaluated on 30 independently sampled
route episodes per seed, on the same topology used for its training.

| Learner | Held-out success | Mean held-out return |
|---|---:|---:|
| DOS Q-learning | 93.3% | 10.48 |
| Weighted-sum Q-learning | 94.2% | 10.67 |
| SARSA | 95.0% | 10.84 |
| Monte Carlo | 95.0% | 10.85 |
| Linear approximation | 92.9% | 10.46 |

The easy default task has high success for all policies; these are descriptive
means, not evidence of a statistically significant ranking. See the
[learning curve](corrected_suite/figures/e0_learning_curves.png) and the
[per-seed results](corrected_suite/results/e0_summary.csv).

### Completed: shaping and design ablation (E6)

Ten matched deployments at 60 packets/s per source. PDR is a percentage;
energy efficiency here is PDR fraction per watt.

| Variant | PDR | Energy efficiency |
|---|---:|---:|
| Value-based DOS, shaping | 77.19% | 12.17 |
| Value-based DOS, no shaping | 77.49% | 12.17 |
| Weighted-sum Q-learning, shaping | 76.97% | 12.16 |
| `argmin` confidence rule, shaping | 76.91% | 12.11 |
| No loop-free mask, shaping | 62.00% | 9.39 |

The shaping/no-shaping difference is small in these means; no significance or
causality is established from this table alone. The mask variants alter action
availability, not just the learning algorithm. See the [ablation CSV](corrected_suite/results/e6_ablation.csv).

### Completed: an environmental change (E5)

At a 40% transient link outage, across eight paired deployments, the DOS
post-onset PDR mean was 74.12% versus 69.25% for re-planned shortest path. The
paired difference was +4.87 percentage points, with Wilcoxon p = 0.078; this
sample does **not** establish a difference at the 0.05 level. `pdr_post`
includes the outage and recovery interval. The older report's significant
outage result belongs to the historical implementation and is not the
corrected result.

| Setting | DOS post-onset PDR | Shortest-path post-onset PDR | Paired difference | p-value | Deployments |
|---|---:|---:|---:|---:|---:|
| 40% transient link outage | 74.12% | 69.25% | +4.87 pp | 0.078 | 8 |

All saved quantitative tables, including the other environment and parameter
settings and the candidate paper-style energy metric, are in
[`corrected_suite/RESULTS.md`](corrected_suite/RESULTS.md).

## Interpretation boundaries

- The custom flow simulator is not ns-3; the paper's SDN-Q baseline is absent.
- PDR fraction/W is not the paper's PDR divided by average relay energy.
- The E6 30-deployment battery retrains on each evaluated topology; it is not
  a zero-shot held-out topology evaluation. E7 is a separate transfer test.
- Shaped and unshaped returns use different reward definitions. Compare success
  and network metrics as well; do not treat return differences alone as proof.
- E8 remains a limited telemetry experiment using privileged action masking;
  it is not part of the core CMOMDP suite.
