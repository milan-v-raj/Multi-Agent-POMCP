# Measured effect of the ARQ correction

The earlier `corrected_suite/` used an incorrect attempt-count formula. The new [ARQ-fixed full results](arq_fixed_suite/RESULTS.md) use the truncated-geometric expectation. Historical CSVs are preserved. These are descriptive before/after means for matching deployment seeds, not confidence intervals or proof of policy superiority. Battery depletion and routes can change, so the effect need not be uniform. The suite source checksum and software versions are in `arq_fixed_suite/`.

Each cell is **old → fixed (fixed minus old)**. PDR differences are in percentage points; delay is milliseconds; energy is total joules; attempts are the simulator-reported mean per link.

## E2 offered load: selected operating points

Five sources, 10 deployments per operating point. DOS and scalar are Q-learning policies. The complete sweep (including source and density changes) is in the full results.

| Packets/s/source | Policy | Deployments | PDR % | Delay ms | Energy J | Attempts |
|---|---|---:|---:|---:|---:|---:|
| 10 | dos_rl | 10 | 91.58 → 91.58 (+0.00) | 48.01 → 36.92 (-11.08) | 2.89 → 1.64 (-1.25) | 2.32 → 1.32 (-1.01) |
| 10 | scalar | 10 | 91.45 → 91.45 (+0.00) | 47.89 → 37.04 (-10.85) | 2.87 → 1.64 (-1.23) | 2.32 → 1.32 (-0.99) |
| 10 | shortest_path | 10 | 91.63 → 91.63 (+0.00) | 47.94 → 37.21 (-10.73) | 2.89 → 1.65 (-1.24) | 2.30 → 1.33 (-0.98) |
| 10 | ospf | 10 | 91.63 → 91.63 (+0.00) | 47.94 → 37.21 (-10.73) | 2.89 → 1.65 (-1.24) | 2.30 → 1.33 (-0.98) |
| 60 | dos_rl | 10 | 77.19 → 84.14 (+6.95) | 92.80 → 81.90 (-10.90) | 14.64 → 9.57 (-5.06) | 2.32 → 1.32 (-1.00) |
| 60 | scalar | 10 | 76.97 → 83.98 (+7.01) | 92.48 → 81.61 (-10.87) | 14.56 → 9.59 (-4.97) | 2.32 → 1.32 (-0.99) |
| 60 | shortest_path | 10 | 76.13 → 83.81 (+7.67) | 95.72 → 84.99 (-10.73) | 14.43 → 9.61 (-4.82) | 2.30 → 1.33 (-0.98) |
| 60 | ospf | 10 | 75.03 → 83.81 (+8.77) | 95.72 → 84.99 (-10.73) | 14.33 → 9.61 (-4.72) | 2.30 → 1.33 (-0.98) |
| 80 | dos_rl | 10 | 70.01 → 82.60 (+12.59) | 93.82 → 82.71 (-11.11) | 16.94 → 12.56 (-4.38) | 2.32 → 1.32 (-1.00) |
| 80 | scalar | 10 | 69.12 → 82.51 (+13.40) | 93.58 → 82.06 (-11.53) | 16.73 → 12.57 (-4.16) | 2.32 → 1.32 (-1.00) |
| 80 | shortest_path | 10 | 68.14 → 82.12 (+13.98) | 95.99 → 85.26 (-10.73) | 16.42 → 12.59 (-3.83) | 2.30 → 1.33 (-0.98) |
| 80 | ospf | 10 | 65.03 → 82.12 (+17.08) | 95.99 → 85.26 (-10.73) | 16.06 → 12.59 (-3.47) | 2.30 → 1.33 (-0.98) |

## E6 retrained deployment battery

The RL agents are retrained on each deployment, not tested zero-shot. `full` is DOS Q-learning and `no_dos` is weighted-sum Q-learning.

| Scenario | Policy | Deployments | PDR % | Delay ms | Energy J | Attempts |
|---|---|---:|---:|---:|---:|---:|
| battery | full | 30 | 82.49 → 87.15 (+4.66) | 71.86 → 62.46 (-9.40) | 14.18 → 8.65 (-5.52) | 2.30 → 1.33 (-0.97) |
| battery | no_dos | 30 | 82.72 → 87.07 (+4.36) | 71.22 → 61.90 (-9.32) | 14.27 → 8.71 (-5.56) | 2.30 → 1.33 (-0.97) |
| battery | linear_fa | 30 | 82.00 → 86.76 (+4.76) | 70.83 → 61.75 (-9.08) | 14.16 → 8.81 (-5.35) | 2.28 → 1.34 (-0.94) |
| battery | shortest_path | 30 | 79.91 → 86.68 (+6.77) | 70.51 → 61.47 (-9.04) | 13.66 → 8.76 (-4.90) | 2.27 → 1.35 (-0.92) |
| battery | ospf | 30 | 78.98 → 86.68 (+7.70) | 70.46 → 61.47 (-8.99) | 13.47 → 8.76 (-4.71) | 2.27 → 1.35 (-0.92) |
| battery | energy_aware_greedy | 30 | 69.46 → 74.80 (+5.34) | 61.08 → 51.67 (-9.41) | 12.40 → 7.50 (-4.90) | 2.39 → 1.60 (-0.78) |

## Remaining limitations

The corrected attempt count does not make this an ns-3 reproduction. Retry-delay handling and source/delivery-weighted aggregation remain approximations; paper energy efficiency and the project’s PDR/W are different metrics. The default value-based DOS is not the paper’s confidence rule. There is no SDN-Q baseline; deployed Q values are frozen while routes can be replanned. See [the paper-fidelity audit](PAPER_FIDELITY.md). E0/E0b and the separate full-state route experiment do not directly use the flow-level ARQ formula.
