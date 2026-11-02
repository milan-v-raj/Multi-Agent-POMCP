# Quantitative results and paper metric audit

> **Fidelity audit:** These historical network runs used an incorrect expected ARQ attempt count. The code has been fixed; do not mix their energy, delay or potentially affected PDR with the separate ARQ-fixed rerun. See `PAPER_FIDELITY.md`.

All numbers below come from the saved CSVs, not from new experiments. The study uses a custom flow-level simulator, not ns-3; it does not implement the paper’s SDN-Q baseline. Results are not a direct reproduction of the paper’s numbers. The E6 battery uses 30 new deployments relative to the ablation experiment, but RL agents are trained separately on each deployment before it is evaluated; it is not a zero-shot held-out topology test.

> **Audit:** In the saved network experiments, labels `scalar`, `weighted_sum` and `no_dos` refer to a one-objective learner that updates the first (energy) reward component, not a weighted sum. Network Q-learning also omits shaping from its update, and the linear agent has TD-update defects. See `CMOMDP_COMPARISON.md`; these results need a corrected rerun.

## Paper metric coverage

| Paper metric | Coverage and limitation |
| --- | --- |
| Network PDR | Calculated as delivered/offered packets in the flow model. |
| Network end-to-end delay | Calculated using analytical queueing and service/retry time; it is not the paper’s ns-3 packet-level measurement and does not explicitly include propagation delay. |
| Network energy efficiency | Existing `energy_efficiency` is PDR as a fraction divided by total network average power (W), not the paper’s PDR divided by average relay energy. See the derived comparison below. |
| Gridworld reward and episode length | Collected during training and held-out testing; training plots are in `figures/e1_gridworld_replication.png`. |
| Gridworld energy and obstacle hits | Collected and plotted during training, and test values exist in memory but were not saved to the E1 CSV. `final_energy_used` is 5 minus final energy, i.e. net depletion, not cumulative energy consumed when recharging is possible. |

### Paper-style energy-efficiency calculation (E6 only)

Derived per deployment as `PDR_percent / (total_energy_j / 30)` and then averaged across the 30 deployments. The paper describes PDR divided by average relay energy; because its precise averaging and units are not fully specified, treat this as a candidate paper-style calculation rather than an exact reproduction. Existing EE is `(PDR_percent / 100) / (total_energy_j / 200 s)`, averaged per run.

| policy | deployments | PDR % | delay ms | total J | mean J/relay | PDR fraction/W | paper-style %/J | Jain fairness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ospf | 30 | 78.982 | 70.464 | 13.469 | 0.44898 | 12.727 | 190.9 | 0.1949 |
| shortest_path | 30 | 79.912 | 70.513 | 13.659 | 0.45531 | 12.72 | 190.79 | 0.19703 |
| energy_aware_greedy | 30 | 69.463 | 61.084 | 12.396 | 0.41321 | 11.96 | 179.41 | 0.16417 |
| full | 30 | 82.408 | 70.639 | 14.165 | 0.47217 | 12.594 | 188.92 | 0.20109 |
| no_dos | 30 | 81.811 | 70.911 | 14.071 | 0.46902 | 12.577 | 188.66 | 0.19816 |
| linear_fa | 30 | 81.944 | 75.435 | 14.345 | 0.47815 | 12.253 | 183.8 | 0.19582 |

## Complete saved-result summaries

Each row below is the arithmetic mean of finite values over saved seed rows, except E4 and paired-statistics tables, which are already aggregated. `n` counts CSV rows, not necessarily independent episodes. Missing data are shown as —. Do not interpret averages of individual within-run standard deviations or confidence intervals as confidence intervals for the pooled mean. Exact per-seed values are in the linked CSV files.

Network fields: `pdr_pct` is percent; `e2e_delay_ms` is ms; `energy_efficiency` is PDR fraction/W; `avg_power_w` is watts; `total_energy_j` is joules; `fairness` is Jain index of energy drain; `load_imbalance` is utilisation standard deviation divided by mean; `mean_hops`, `mean_attempts`, and `throughput_pps` are simulator summaries. E5 `pdr_post` averages from failure onset to run end, including the outage and recovery; `pdr_min` is the minimum in the outage interval. `recovery_s` follows the experiment’s criterion.

E0’s per-seed, 20-episode moving-average training curves are in [`e0_curves_raw.csv`](results/e0_curves_raw.csv). E0b and E1 full per-episode data and E4 curves are not persisted; their summary results and available figures are shown here. E5 original time traces are not saved.

### E0 training and held-out tests

Source: [`e0_summary.csv`](results/e0_summary.csv).

| agent | n | train_seconds | q_entries | final_train_return | final_train_success_pct | test_success_rate | test_return_mean | test_return_std | test_return_ci95 | test_length_mean | test_n_episodes | test_term_goal | test_term_hop_limit | test_term_loop | test_term_dead_end | test_term_energy_exhausted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dos_rl | 8 | 1.3013 | 59.75 | 10.708 | 90.5 | 93.75 | 11.426 | 4.8788 | 1.7459 | 4.0708 | 30 | 28.125 | 0 | 0 | 1.875 | 0 |
| weighted_sum | 8 | 1.0775 | 59.75 | 10.864 | 91.375 | 95 | 11.675 | 4.2201 | 1.5101 | 4.0875 | 30 | 28.5 | 0 | 0 | 1.5 | 0 |
| sarsa | 8 | 1.3088 | 59.75 | 11.118 | 92.5 | 95 | 11.693 | 4.1872 | 1.4984 | 4.1083 | 30 | 28.5 | 0 | 0 | 1.5 | 0 |
| mc_control | 8 | 1.2537 | 59.75 | 11.139 | 92.625 | 95 | 11.676 | 4.473 | 1.6006 | 4.0917 | 30 | 28.5 | 0 | 0 | 1.5 | 0 |
| linear_fa | 8 | 1.8288 | — | 11.167 | 93 | 94.583 | 11.531 | 4.5623 | 1.6326 | 4.0917 | 30 | 28.375 | 0 | 0 | 1.625 | 0 |

### E0b harder routing task

Source: [`e0b_coldstart.csv`](results/e0b_coldstart.csv).

| policy | n | early_return | late_return | early_success | late_success | test_success_pct | test_return_mean | test_return_ci95 | test_length_mean | test_n_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dos_rl | 10 | -2.8912 | 0.54059 | 24.8 | 40.1 | 63.667 | 5.6268 | 2.6615 | 4.9633 | 30 |
| scalar | 10 | -2.9389 | 1.7083 | 24.8 | 46.4 | 81.667 | 9.3522 | 2.784 | 5.0533 | 30 |

### E1 gridworld

Source: [`e1_gridworld.csv`](results/e1_gridworld.csv).

| variant | n | test_return | test_success_pct | test_length | test_n_episodes | final_return | final_success_pct | final_episode_length | final_obstacle_hits | final_energy_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DOS-RL (shaped) | 10 | -9.7091 | 0 | 23.1 | 30 | 2.7253 | 14 | 36.698 | 0.098 | 1.6403 |
| DOS-RL (unshaped) | 10 | -10.023 | 0 | 23.1 | 30 | 2.3705 | 14 | 36.698 | 0.098 | 1.6403 |
| Weighted-sum RL | 10 | -9.6749 | 10 | 23.5 | 30 | -8.3732 | 16.4 | 21.54 | 0.594 | 1.2273 |
| DOS[value] | 10 | -9.7091 | 0 | — | 30 | 2.7253 | 14 | — | — | — |
| DOS[argmax] | 10 | -14.977 | 0 | — | 30 | 3.6409 | 10.1 | — | — | — |
| DOS[argmin] | 10 | -0.54833 | 0 | — | 30 | 2.6101 | 0.6 | — | — | — |

### E2/E3 network benchmark

Source: [`e2_e3_raw.csv`](results/e2_e3_raw.csv).

| sweep | value | policy | n | lambda_pps | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| load | 10 | ospf | 10 | 10 | 91.625 | 47.944 | 76.397 | 0.014462 | 2.8925 | 0.19282 | 2.0788 | 4.62 | 2.3043 | 45.813 |
| load | 10 | shortest_path | 10 | 10 | 91.625 | 47.944 | 76.397 | 0.014462 | 2.8925 | 0.19282 | 2.0788 | 4.62 | 2.3043 | 45.813 |
| load | 10 | dos_rl | 10 | 10 | 91.731 | 47.953 | 76.464 | 0.0145 | 2.9 | 0.19218 | 1.9784 | 4.62 | 2.3266 | 45.865 |
| load | 10 | scalar | 10 | 10 | 91.65 | 47.956 | 75.972 | 0.014563 | 2.9125 | 0.18971 | 1.9962 | 4.62 | 2.3222 | 45.825 |
| load | 20 | ospf | 10 | 20 | 91.026 | 56.252 | 38.065 | 0.028917 | 5.7833 | 0.19284 | 2.0688 | 4.62 | 2.3043 | 91.026 |
| load | 20 | shortest_path | 10 | 20 | 91.026 | 56.252 | 38.065 | 0.028917 | 5.7833 | 0.19284 | 2.0688 | 4.62 | 2.3043 | 91.026 |
| load | 20 | dos_rl | 10 | 20 | 91.14 | 55.456 | 38.1 | 0.028992 | 5.7984 | 0.1922 | 1.9712 | 4.62 | 2.3266 | 91.14 |
| load | 20 | scalar | 10 | 20 | 91.059 | 55.489 | 37.854 | 0.029118 | 5.8235 | 0.18972 | 1.989 | 4.62 | 2.3222 | 91.059 |
| load | 30 | ospf | 10 | 30 | 89.044 | 67.162 | 24.902 | 0.043311 | 8.6621 | 0.19301 | 2.0074 | 4.62 | 2.3043 | 133.57 |
| load | 30 | shortest_path | 10 | 30 | 89.044 | 67.162 | 24.902 | 0.043311 | 8.6621 | 0.19301 | 2.0074 | 4.62 | 2.3043 | 133.57 |
| load | 30 | dos_rl | 10 | 30 | 90.014 | 63.399 | 25.214 | 0.043446 | 8.6892 | 0.1923 | 1.9218 | 4.62 | 2.3266 | 135.02 |
| load | 30 | scalar | 10 | 30 | 89.943 | 63.49 | 25.052 | 0.043634 | 8.7269 | 0.18984 | 1.939 | 4.62 | 2.3222 | 134.91 |
| load | 45 | ospf | 10 | 45 | 85.322 | 77.946 | 16.494 | 0.061577 | 12.315 | 0.1935 | 1.8779 | 4.4194 | 2.3043 | 191.98 |
| load | 45 | shortest_path | 10 | 45 | 85.322 | 77.946 | 16.494 | 0.061577 | 12.315 | 0.1935 | 1.8779 | 4.4194 | 2.3043 | 191.98 |
| load | 45 | dos_rl | 10 | 45 | 86.144 | 78.191 | 16.605 | 0.062372 | 12.474 | 0.19473 | 1.8406 | 4.4612 | 2.3256 | 193.82 |
| load | 45 | scalar | 10 | 45 | 86.027 | 78.631 | 16.494 | 0.062627 | 12.525 | 0.19275 | 1.8535 | 4.4612 | 2.321 | 193.56 |
| load | 60 | ospf | 10 | 60 | 75.033 | 95.722 | 12.091 | 0.071654 | 14.331 | 0.19585 | 1.7846 | 3.9657 | 2.3043 | 225.1 |
| load | 60 | shortest_path | 10 | 60 | 76.134 | 95.725 | 12.089 | 0.072135 | 14.427 | 0.19781 | 1.7633 | 3.9967 | 2.3044 | 228.4 |
| load | 60 | dos_rl | 10 | 60 | 77.488 | 93.478 | 12.172 | 0.073447 | 14.689 | 0.19999 | 1.7372 | 4.0469 | 2.3237 | 232.46 |
| load | 60 | scalar | 10 | 60 | 77.342 | 93.201 | 12.097 | 0.073637 | 14.727 | 0.19823 | 1.7445 | 4.0391 | 2.3192 | 232.03 |
| load | 80 | ospf | 10 | 80 | 65.034 | 95.986 | 8.9976 | 0.080285 | 16.057 | 0.19736 | 1.6632 | 3.3987 | 2.3046 | 260.14 |
| load | 80 | shortest_path | 10 | 80 | 68.137 | 95.99 | 8.9925 | 0.082118 | 16.424 | 0.20302 | 1.6276 | 3.4839 | 2.3044 | 272.55 |
| load | 80 | dos_rl | 10 | 80 | 70.334 | 95.967 | 9.0478 | 0.085265 | 17.053 | 0.20464 | 1.6132 | 3.5994 | 2.3234 | 281.34 |
| load | 80 | scalar | 10 | 80 | 69.453 | 95.377 | 8.9617 | 0.084721 | 16.944 | 0.20498 | 1.6116 | 3.5475 | 2.321 | 277.81 |
| sources | 2 | ospf | 10 | 10 | 91.821 | 43.831 | 203.79 | 0.0054369 | 1.0874 | 0.16104 | 2.3559 | 4.4 | 2.2706 | 18.364 |
| sources | 2 | shortest_path | 10 | 10 | 91.821 | 43.831 | 203.79 | 0.0054369 | 1.0874 | 0.16104 | 2.3559 | 4.4 | 2.2706 | 18.364 |
| sources | 2 | dos_rl | 10 | 10 | 91.069 | 43.822 | 196.35 | 0.0054882 | 1.0976 | 0.16243 | 2.2105 | 4.4 | 2.2867 | 18.214 |
| sources | 2 | scalar | 10 | 10 | 91.374 | 43.825 | 196.99 | 0.0054996 | 1.0999 | 0.16028 | 2.2689 | 4.4 | 2.2839 | 18.275 |
| sources | 4 | ospf | 10 | 10 | 91.107 | 49.279 | 90.891 | 0.012148 | 2.4296 | 0.1868 | 2.1923 | 4.825 | 2.2899 | 36.443 |
| sources | 4 | shortest_path | 10 | 10 | 91.107 | 49.279 | 90.891 | 0.012148 | 2.4296 | 0.1868 | 2.1923 | 4.825 | 2.2899 | 36.443 |
| sources | 4 | dos_rl | 10 | 10 | 91.293 | 49.506 | 90.982 | 0.01219 | 2.438 | 0.17587 | 2.1716 | 4.825 | 2.3235 | 36.517 |
| sources | 4 | scalar | 10 | 10 | 91.353 | 49.44 | 90.688 | 0.01221 | 2.4419 | 0.18385 | 2.0817 | 4.825 | 2.3198 | 36.541 |
| sources | 6 | ospf | 10 | 10 | 91.474 | 49.207 | 65.154 | 0.017523 | 3.5046 | 0.1993 | 2.0279 | 4.6667 | 2.299 | 54.885 |
| sources | 6 | shortest_path | 10 | 10 | 91.474 | 49.207 | 65.154 | 0.017523 | 3.5046 | 0.1993 | 2.0279 | 4.6667 | 2.299 | 54.885 |
| sources | 6 | dos_rl | 10 | 10 | 91.398 | 49.116 | 65.23 | 0.01763 | 3.5261 | 0.19776 | 1.9515 | 4.6667 | 2.3169 | 54.839 |
| sources | 6 | scalar | 10 | 10 | 91.511 | 49.181 | 64.71 | 0.017745 | 3.549 | 0.19903 | 1.9354 | 4.6667 | 2.3229 | 54.907 |
| sources | 8 | ospf | 10 | 10 | 91.733 | 51.521 | 49.763 | 0.022948 | 4.5896 | 0.20871 | 2.0429 | 4.6 | 2.3086 | 73.386 |
| sources | 8 | shortest_path | 10 | 10 | 91.733 | 51.521 | 49.763 | 0.022948 | 4.5896 | 0.20871 | 2.0429 | 4.6 | 2.3086 | 73.386 |
| sources | 8 | dos_rl | 10 | 10 | 91.798 | 51.255 | 48.795 | 0.023153 | 4.6307 | 0.20092 | 1.9132 | 4.6 | 2.3397 | 73.439 |
| sources | 8 | scalar | 10 | 10 | 91.823 | 51.3 | 48.475 | 0.023277 | 4.6554 | 0.20194 | 1.9344 | 4.6 | 2.3501 | 73.458 |
| sources | 10 | ospf | 10 | 10 | 91.227 | 54.35 | 37.171 | 0.028558 | 5.7116 | 0.22017 | 1.9495 | 4.57 | 2.2935 | 91.227 |
| sources | 10 | shortest_path | 10 | 10 | 91.227 | 54.35 | 37.171 | 0.028558 | 5.7116 | 0.22017 | 1.9495 | 4.57 | 2.2935 | 91.227 |
| sources | 10 | dos_rl | 10 | 10 | 91.172 | 53.896 | 36.498 | 0.028865 | 5.7731 | 0.21541 | 1.8517 | 4.57 | 2.326 | 91.172 |
| sources | 10 | scalar | 10 | 10 | 91.208 | 53.821 | 36.48 | 0.028876 | 5.7752 | 0.21691 | 1.8407 | 4.57 | 2.3225 | 91.208 |
| nodes | 30 | ospf | 10 | 10 | 91.625 | 47.944 | 76.397 | 0.014462 | 2.8925 | 0.19282 | 2.0788 | 4.62 | 2.3043 | 45.813 |
| nodes | 30 | shortest_path | 10 | 10 | 91.625 | 47.944 | 76.397 | 0.014462 | 2.8925 | 0.19282 | 2.0788 | 4.62 | 2.3043 | 45.813 |
| nodes | 30 | dos_rl | 10 | 10 | 91.731 | 47.953 | 76.464 | 0.0145 | 2.9 | 0.19218 | 1.9784 | 4.62 | 2.3266 | 45.865 |
| nodes | 30 | scalar | 10 | 10 | 91.65 | 47.956 | 75.972 | 0.014563 | 2.9125 | 0.18971 | 1.9962 | 4.62 | 2.3222 | 45.825 |
| nodes | 45 | ospf | 10 | 10 | 91.645 | 43.093 | 71.046 | 0.013059 | 2.6118 | 0.16104 | 2.4079 | 4.26 | 2.2856 | 45.823 |
| nodes | 45 | shortest_path | 10 | 10 | 91.645 | 43.093 | 71.046 | 0.013059 | 2.6118 | 0.16104 | 2.4079 | 4.26 | 2.2856 | 45.823 |
| nodes | 45 | dos_rl | 10 | 10 | 90.866 | 42.8 | 70.983 | 0.012916 | 2.5832 | 0.17601 | 2.1625 | 4.26 | 2.2874 | 45.433 |
| nodes | 45 | scalar | 10 | 10 | 90.914 | 42.838 | 71.9 | 0.012792 | 2.5584 | 0.16764 | 2.2905 | 4.26 | 2.2835 | 45.457 |
| nodes | 60 | ospf | 10 | 10 | 91.472 | 37.339 | 86.611 | 0.010878 | 2.1756 | 0.12863 | 2.8594 | 3.78 | 2.2138 | 45.736 |
| nodes | 60 | shortest_path | 10 | 10 | 91.472 | 37.339 | 86.611 | 0.010878 | 2.1756 | 0.12863 | 2.8594 | 3.78 | 2.2138 | 45.736 |
| nodes | 60 | dos_rl | 10 | 10 | 90.972 | 37.539 | 84.116 | 0.011059 | 2.2119 | 0.13198 | 2.6175 | 3.78 | 2.2724 | 45.486 |
| nodes | 60 | scalar | 10 | 10 | 90.931 | 37.489 | 83.242 | 0.011167 | 2.2334 | 0.12391 | 2.5688 | 3.78 | 2.2908 | 45.465 |

### E2 paired comparisons

Source: [`e2_e3_paired_tests.csv`](results/e2_e3_paired_tests.csv).

| sweep | value | policy | n | mean_delta_pp | ci95_pp | p_value | effect_size | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| load | 30 | dos_rl | 1 | 0.96962 | 1.2595 | 0.16016 | 0.52727 | 10 |
| load | 30 | scalar | 1 | 0.89865 | 1.1576 | 0.19336 | 0.49091 | 10 |
| load | 45 | dos_rl | 1 | 0.82144 | 1.5996 | 0.55664 | 0.23636 | 10 |
| load | 45 | scalar | 1 | 0.70427 | 1.5 | 0.69531 | 0.16364 | 10 |
| load | 60 | dos_rl | 1 | 1.3541 | 1.653 | 0.32227 | 0.38182 | 10 |
| load | 60 | scalar | 1 | 1.2087 | 1.642 | 0.19336 | 0.49091 | 10 |
| load | 80 | dos_rl | 1 | 2.1977 | 3.1859 | 0.10547 | 0.6 | 10 |
| load | 80 | scalar | 1 | 1.3164 | 1.8089 | 0.13086 | 0.56364 | 10 |

### E4 parameter sweep

Source: [`e4_parameter_sweep.csv`](results/e4_parameter_sweep.csv).

| sweep | value | n | final_return | final_return_std | final_success_pct | learning_auc | convergence_episode | pdr_delta_vs_baseline_pp | pdr_delta_ci95 | q_table_entries | n_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alpha | 0.02 | 1 | 11.147 | 0.60998 | 91.583 | 11.16 | 411.6 | -0.091889 | 1.6914 | 394.67 | 6 |
| alpha | 0.05 | 1 | 11.337 | 0.79407 | 92.417 | 11.008 | 722.2 | 0.62641 | 1.7301 | 393.33 | 6 |
| alpha | 0.1 | 1 | 11.055 | 0.57861 | 91.083 | 11.105 | 908 | -0.19797 | 1.6354 | 393.33 | 6 |
| alpha | 0.3 | 1 | 11.145 | 0.83576 | 91.583 | 11.058 | 827.25 | 0.71166 | 2.0281 | 396.5 | 6 |
| alpha | 0.6 | 1 | 11.111 | 0.3757 | 91.417 | 11.208 | 517.33 | -0.038697 | 1.4268 | 395.33 | 6 |
| alpha | 0.9 | 1 | 11.001 | 0.47838 | 90.917 | 11.224 | 831 | 0.66071 | 2.3922 | 398 | 6 |
| gamma | 0.5 | 1 | 9.8851 | 0.4267 | 90.25 | 10.106 | 939 | 0.17466 | 2.0285 | 394.5 | 6 |
| gamma | 0.7 | 1 | 10.884 | 0.34386 | 92.417 | 10.632 | 350 | 0.5033 | 2.0129 | 391.33 | 6 |
| gamma | 0.85 | 1 | 11.145 | 0.83576 | 91.583 | 11.058 | 827.25 | 0.71166 | 2.0281 | 396.5 | 6 |
| gamma | 0.95 | 1 | 11.332 | 0.72996 | 91 | 11.385 | 249.25 | 0.11487 | 2.4946 | 393.67 | 6 |
| gamma | 0.99 | 1 | 11.149 | 0.77646 | 89.75 | 11.475 | 598.67 | 0.24241 | 1.674 | 396.17 | 6 |
| epsilon | 0.02 | 1 | 11.322 | 0.48812 | 92.333 | 11.243 | 357.33 | 0.40808 | 1.6258 | 386 | 6 |
| epsilon | 0.05 | 1 | 11.167 | 0.42825 | 91.667 | 11.036 | 536.2 | 0.6848 | 1.9156 | 387.17 | 6 |
| epsilon | 0.1 | 1 | 11.51 | 0.41202 | 93.25 | 11.302 | 946.6 | 0.0023007 | 1.0936 | 396.67 | 6 |
| epsilon | 0.3 | 1 | 11.145 | 0.83576 | 91.583 | 11.058 | 827.25 | 0.71166 | 2.0281 | 396.5 | 6 |
| epsilon | 0.5 | 1 | 10.923 | 0.8989 | 90.583 | 10.931 | 946 | 0.58949 | 2.5168 | 399 | 6 |
| epsilon | 0.8 | 1 | 11.171 | 0.89018 | 91.667 | 11.162 | 822.2 | 0.51237 | 2.3553 | 402.67 | 6 |
| epsilon_decay | 0.9 | 1 | 11.392 | 0.66113 | 92.667 | 11.075 | 564.67 | -0.033128 | 2.6663 | 383.5 | 6 |
| epsilon_decay | 0.995 | 1 | 11.357 | 0.61683 | 92.583 | 11.312 | 245 | 0.39459 | 2.0362 | 390 | 6 |
| epsilon_decay | 1 | 1 | 11.145 | 0.83576 | 91.583 | 11.058 | 827.25 | 0.71166 | 2.0281 | 396.5 | 6 |

### E5 arrival-rate sweep

Source: [`e5a_arrival_rate.csv`](results/e5a_arrival_rate.csv).

| sweep | value | policy | n | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arrival | 5 | ospf | 8 | 91.472 | 48.476 | 143.88 | 0.0076658 | 1.5332 | 0.19632 | 2.078 | 4.825 | 2.3188 | 22.868 |
| arrival | 5 | shortest_path | 8 | 91.472 | 48.476 | 143.88 | 0.0076658 | 1.5332 | 0.19632 | 2.078 | 4.825 | 2.3188 | 22.868 |
| arrival | 5 | dos_rl | 8 | 91.754 | 48.608 | 145.65 | 0.0076149 | 1.523 | 0.20356 | 1.9363 | 4.825 | 2.3344 | 22.938 |
| arrival | 5 | scalar | 8 | 91.649 | 48.617 | 144.44 | 0.0076526 | 1.5305 | 0.19987 | 1.9602 | 4.825 | 2.3302 | 22.912 |
| arrival | 10 | ospf | 8 | 91.472 | 50.282 | 71.939 | 0.015332 | 3.0663 | 0.19632 | 2.078 | 4.825 | 2.3188 | 45.736 |
| arrival | 10 | shortest_path | 8 | 91.472 | 50.282 | 71.939 | 0.015332 | 3.0663 | 0.19632 | 2.078 | 4.825 | 2.3188 | 45.736 |
| arrival | 10 | dos_rl | 8 | 91.754 | 50.265 | 72.827 | 0.01523 | 3.046 | 0.20356 | 1.9363 | 4.825 | 2.3344 | 45.877 |
| arrival | 10 | scalar | 8 | 91.649 | 50.285 | 72.222 | 0.015305 | 3.061 | 0.19987 | 1.9602 | 4.825 | 2.3302 | 45.825 |
| arrival | 20 | ospf | 8 | 90.728 | 59.557 | 35.805 | 0.030653 | 6.1307 | 0.19634 | 2.0655 | 4.825 | 2.3188 | 90.728 |
| arrival | 20 | shortest_path | 8 | 90.728 | 59.557 | 35.805 | 0.030653 | 6.1307 | 0.19634 | 2.0655 | 4.825 | 2.3188 | 90.728 |
| arrival | 20 | dos_rl | 8 | 91.015 | 58.968 | 36.248 | 0.03045 | 6.09 | 0.20358 | 1.9274 | 4.825 | 2.3344 | 91.015 |
| arrival | 20 | scalar | 8 | 90.911 | 59.022 | 35.946 | 0.030601 | 6.1202 | 0.19989 | 1.9512 | 4.825 | 2.3302 | 90.911 |
| arrival | 40 | ospf | 8 | 87.458 | 76.7 | 17.494 | 0.060179 | 12.036 | 0.19672 | 1.8983 | 4.7616 | 2.3188 | 174.92 |
| arrival | 40 | shortest_path | 8 | 87.458 | 76.7 | 17.494 | 0.060179 | 12.036 | 0.19672 | 1.8983 | 4.7616 | 2.3188 | 174.92 |
| arrival | 40 | dos_rl | 8 | 88.681 | 74.28 | 17.9 | 0.060204 | 12.041 | 0.20387 | 1.8075 | 4.7869 | 2.3344 | 177.36 |
| arrival | 40 | scalar | 8 | 88.577 | 74.599 | 17.75 | 0.0605 | 12.1 | 0.20021 | 1.8288 | 4.7869 | 2.3302 | 177.15 |
| arrival | 60 | ospf | 8 | 72.808 | 102.72 | 11.301 | 0.074261 | 14.852 | 0.19973 | 1.7636 | 4.0608 | 2.3188 | 218.42 |
| arrival | 60 | shortest_path | 8 | 74.184 | 102.72 | 11.298 | 0.074862 | 14.972 | 0.20219 | 1.7369 | 4.0996 | 2.3189 | 222.55 |
| arrival | 60 | dos_rl | 8 | 75.395 | 98.946 | 11.553 | 0.075059 | 15.012 | 0.20651 | 1.7043 | 4.1086 | 2.3344 | 226.19 |
| arrival | 60 | scalar | 8 | 75.211 | 98.722 | 11.46 | 0.075284 | 15.057 | 0.20301 | 1.7174 | 4.0989 | 2.3302 | 225.63 |
| arrival | 90 | ospf | 8 | 57.764 | 100.81 | 7.4214 | 0.086135 | 17.227 | 0.20277 | 1.5946 | 3.284 | 2.3229 | 259.94 |
| arrival | 90 | shortest_path | 8 | 63.742 | 100.5 | 7.4225 | 0.090521 | 18.104 | 0.21211 | 1.546 | 3.3959 | 2.3168 | 286.84 |
| arrival | 90 | dos_rl | 8 | 64.53 | 102.75 | 7.557 | 0.091119 | 18.224 | 0.21739 | 1.5364 | 3.4336 | 2.3332 | 290.39 |
| arrival | 90 | scalar | 8 | 64.174 | 103.73 | 7.47 | 0.091587 | 18.317 | 0.21333 | 1.5495 | 3.4198 | 2.332 | 288.78 |
| arrival | 120 | ospf | 8 | 45.988 | 95.167 | 5.4959 | 0.091259 | 18.252 | 0.20576 | 1.4963 | 2.819 | 2.3376 | 275.93 |
| arrival | 120 | shortest_path | 8 | 55.139 | 94.452 | 5.5135 | 0.10083 | 20.165 | 0.22317 | 1.4356 | 2.8642 | 2.316 | 330.83 |
| arrival | 120 | dos_rl | 8 | 56.519 | 96.954 | 5.606 | 0.10254 | 20.508 | 0.23209 | 1.4271 | 2.9362 | 2.3315 | 339.11 |
| arrival | 120 | scalar | 8 | 54.817 | 99.176 | 5.5667 | 0.10071 | 20.143 | 0.22392 | 1.4336 | 2.8765 | 2.3289 | 328.9 |

### E5 link outages

Source: [`e5b_link_failure.csv`](results/e5b_link_failure.csv).

| sweep | value | policy | n | pdr_pre | pdr_post | pdr_min | dip_area | recovery_s | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| link_failure | 0 | ospf | 8 | 91.472 | 90.484 | 89.072 | 0.0010032 | 7.0896 | 90.97 | 50.282 | 71.645 | 0.015332 | 3.0663 | 0.19632 | 2.078 | 4.825 | 2.3188 | 45.485 |
| link_failure | 0 | shortest_path | 8 | 91.472 | 90.484 | 89.072 | 0.0010032 | 7.0896 | 90.97 | 50.324 | 71.862 | 0.015262 | 3.0523 | 0.19658 | 2.0768 | 4.8328 | 2.3209 | 45.485 |
| link_failure | 0 | dos_rl | 8 | 91.754 | 91.774 | 91.753 | 2.6145e-08 | 1.4925 | 91.764 | 50.334 | 72.74 | 0.015259 | 3.0518 | 0.20414 | 1.9346 | 4.8302 | 2.3352 | 45.882 |
| link_failure | 0 | scalar | 8 | 91.649 | 91.649 | 91.649 | 2.6145e-08 | 1.4925 | 91.649 | 50.349 | 72.134 | 0.015333 | 3.0667 | 0.2003 | 1.9584 | 4.8302 | 2.3308 | 45.825 |
| link_failure | 0.2 | ospf | 8 | 91.472 | 65.571 | 28.57 | 0.026287 | 46.269 | 78.328 | 50.282 | 62.779 | 0.015332 | 3.0663 | 0.19632 | 2.078 | 4.825 | 2.3188 | 39.164 |
| link_failure | 0.2 | shortest_path | 8 | 91.472 | 72.192 | 44.644 | 0.01957 | 35.075 | 81.688 | 49.981 | 72.357 | 0.013153 | 2.6307 | 0.20124 | 2.0549 | 4.2795 | 2.331 | 40.844 |
| link_failure | 0.2 | dos_rl | 8 | 91.754 | 75.259 | 51.55 | 0.016801 | 29.478 | 83.383 | 50.149 | 72.718 | 0.013354 | 2.6707 | 0.20977 | 1.9237 | 4.3026 | 2.3459 | 41.692 |
| link_failure | 0.2 | scalar | 8 | 91.649 | 75.08 | 51.41 | 0.016816 | 29.478 | 83.241 | 50.242 | 72.057 | 0.013451 | 2.6902 | 0.20375 | 1.9429 | 4.3131 | 2.3394 | 41.621 |
| link_failure | 0.4 | ospf | 8 | 91.472 | 60.652 | 16.624 | 0.03128 | 46.269 | 75.832 | 50.282 | 60.481 | 0.015332 | 3.0663 | 0.19632 | 2.078 | 4.825 | 2.3188 | 37.916 |
| link_failure | 0.4 | shortest_path | 8 | 91.472 | 69.245 | 37.471 | 0.022568 | 40.672 | 80.193 | 49.803 | 72.673 | 0.013038 | 2.6075 | 0.197 | 2.0645 | 4.4114 | 2.3419 | 40.096 |
| link_failure | 0.4 | dos_rl | 8 | 91.754 | 75.245 | 51.662 | 0.016755 | 35.075 | 83.376 | 50.181 | 72.235 | 0.013488 | 2.6975 | 0.21439 | 1.8931 | 4.468 | 2.3568 | 41.688 |
| link_failure | 0.4 | scalar | 8 | 91.649 | 75.164 | 51.613 | 0.016732 | 35.075 | 83.284 | 50.209 | 71.843 | 0.013554 | 2.7109 | 0.21689 | 1.8774 | 4.4759 | 2.3526 | 41.642 |

### E5 node outages

Source: [`e5c_node_failure.csv`](results/e5c_node_failure.csv).

| sweep | value | policy | n | pdr_pre | pdr_post | pdr_min | dip_area | recovery_s | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| node_failure | 0 | ospf | 8 | 91.472 | 88.104 | 83.293 | 0.0034181 | 7.0896 | 89.763 | 50.045 | 72.073 | 0.014787 | 2.9574 | 0.19587 | 2.0808 | 4.7414 | 2.323 | 44.882 |
| node_failure | 0 | shortest_path | 8 | 91.472 | 88.104 | 83.293 | 0.0034181 | 7.0896 | 89.763 | 50.045 | 72.073 | 0.014787 | 2.9574 | 0.19587 | 2.0808 | 4.7414 | 2.323 | 44.882 |
| node_failure | 0 | dos_rl | 8 | 91.754 | 88.258 | 83.263 | 0.0035481 | 7.0896 | 89.979 | 50.027 | 72.978 | 0.014688 | 2.9375 | 0.20294 | 1.9388 | 4.7414 | 2.3378 | 44.99 |
| node_failure | 0 | scalar | 8 | 91.649 | 88.194 | 83.257 | 0.0035072 | 7.0896 | 89.896 | 50.05 | 72.364 | 0.014758 | 2.9516 | 0.19932 | 1.9628 | 4.7414 | 2.3338 | 44.948 |
| node_failure | 0.1 | ospf | 8 | 91.472 | 74.039 | 49.134 | 0.017693 | 40.672 | 82.625 | 49.969 | 72.624 | 0.013547 | 2.7093 | 0.19453 | 2.0922 | 4.4484 | 2.336 | 41.313 |
| node_failure | 0.1 | shortest_path | 8 | 91.472 | 79.653 | 62.763 | 0.011998 | 23.881 | 85.474 | 50.233 | 71.731 | 0.013981 | 2.7961 | 0.20297 | 2.0604 | 4.4981 | 2.3273 | 42.737 |
| node_failure | 0.1 | dos_rl | 8 | 91.754 | 79.766 | 62.623 | 0.012174 | 23.881 | 85.671 | 50.229 | 72.615 | 0.013891 | 2.7782 | 0.20746 | 1.9102 | 4.4981 | 2.3419 | 42.835 |
| node_failure | 0.1 | scalar | 8 | 91.649 | 79.726 | 62.675 | 0.012109 | 23.881 | 85.599 | 50.25 | 71.958 | 0.013971 | 2.7941 | 0.20334 | 1.9387 | 4.4981 | 2.3381 | 42.799 |
| node_failure | 0.2 | ospf | 8 | 91.472 | 72.131 | 44.501 | 0.01963 | 40.672 | 81.657 | 49.706 | 72.991 | 0.013211 | 2.6421 | 0.19218 | 2.0969 | 4.3683 | 2.3381 | 40.829 |
| node_failure | 0.2 | shortest_path | 8 | 91.472 | 73.088 | 46.826 | 0.018658 | 40.672 | 82.143 | 49.727 | 72.84 | 0.013251 | 2.6503 | 0.19244 | 2.0952 | 4.3705 | 2.3356 | 41.071 |
| node_failure | 0.2 | dos_rl | 8 | 91.754 | 73.162 | 46.604 | 0.018869 | 40.672 | 82.319 | 49.718 | 73.687 | 0.013172 | 2.6344 | 0.20115 | 1.9552 | 4.3705 | 2.3509 | 41.16 |
| node_failure | 0.2 | scalar | 8 | 91.649 | 73.141 | 46.702 | 0.018784 | 40.672 | 82.257 | 49.746 | 73.05 | 0.013241 | 2.6482 | 0.19666 | 1.9805 | 4.3705 | 2.3472 | 41.129 |
| node_failure | 0.3 | ospf | 8 | 91.472 | 64.224 | 25.298 | 0.027655 | 46.269 | 77.645 | 49.814 | 72.378 | 0.01287 | 2.5739 | 0.19345 | 2.0846 | 4.3609 | 2.3441 | 38.822 |
| node_failure | 0.3 | shortest_path | 8 | 91.472 | 73.826 | 48.617 | 0.01791 | 46.269 | 82.517 | 49.937 | 71.724 | 0.013424 | 2.6847 | 0.19997 | 2.0131 | 4.4393 | 2.3413 | 41.259 |
| node_failure | 0.3 | dos_rl | 8 | 91.754 | 73.92 | 48.443 | 0.0181 | 46.269 | 82.703 | 49.956 | 72.585 | 0.013353 | 2.6706 | 0.20726 | 1.9105 | 4.4393 | 2.3562 | 41.352 |
| node_failure | 0.3 | scalar | 8 | 91.649 | 73.852 | 48.428 | 0.018063 | 46.269 | 82.618 | 49.958 | 72.18 | 0.013393 | 2.6786 | 0.206 | 1.9379 | 4.4393 | 2.3518 | 41.309 |

### E5 repeated outages

Source: [`e5d_repeated_outages.csv`](results/e5d_repeated_outages.csv).

| sweep | value | policy | n | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeated | 0 | ospf | 8 | 91.472 | 50.282 | 71.939 | 0.015332 | 3.0663 | 0.19632 | 2.078 | 4.825 | 2.3188 | 45.736 |
| repeated | 0 | shortest_path | 8 | 91.472 | 50.282 | 71.939 | 0.015332 | 3.0663 | 0.19632 | 2.078 | 4.825 | 2.3188 | 45.736 |
| repeated | 0 | dos_rl | 8 | 91.754 | 50.265 | 72.827 | 0.01523 | 3.046 | 0.20356 | 1.9363 | 4.825 | 2.3344 | 45.877 |
| repeated | 0 | scalar | 8 | 91.649 | 50.285 | 72.222 | 0.015305 | 3.061 | 0.19987 | 1.9602 | 4.825 | 2.3302 | 45.825 |
| repeated | 2 | ospf | 8 | 83.923 | 50.136 | 69.594 | 0.014574 | 2.9149 | 0.1955 | 2.0804 | 4.6738 | 2.3252 | 41.961 |
| repeated | 2 | shortest_path | 8 | 86.221 | 50.041 | 72.182 | 0.014174 | 2.8347 | 0.19787 | 2.071 | 4.6352 | 2.3265 | 43.11 |
| repeated | 2 | dos_rl | 8 | 86.861 | 50.07 | 73.075 | 0.014223 | 2.8445 | 0.20684 | 1.9262 | 4.6471 | 2.3433 | 43.431 |
| repeated | 2 | scalar | 8 | 86.918 | 50.176 | 72.352 | 0.014321 | 2.8641 | 0.20357 | 1.9434 | 4.657 | 2.3386 | 43.459 |
| repeated | 4 | ospf | 8 | 74.976 | 50.007 | 65.809 | 0.013802 | 2.7604 | 0.19486 | 2.0846 | 4.5126 | 2.3312 | 37.488 |
| repeated | 4 | shortest_path | 8 | 79.744 | 49.947 | 72.331 | 0.012891 | 2.5782 | 0.2018 | 2.0416 | 4.3571 | 2.338 | 39.872 |
| repeated | 4 | dos_rl | 8 | 80.76 | 50.037 | 73.054 | 0.01301 | 2.602 | 0.2133 | 1.9015 | 4.3718 | 2.3527 | 40.38 |
| repeated | 4 | scalar | 8 | 80.824 | 50.155 | 72.335 | 0.013111 | 2.6222 | 0.20803 | 1.9223 | 4.3831 | 2.3479 | 40.412 |
| repeated | 6 | ospf | 8 | 66.005 | 49.932 | 61.92 | 0.012906 | 2.5812 | 0.19428 | 2.088 | 4.2432 | 2.3346 | 33.003 |
| repeated | 6 | shortest_path | 8 | 73.53 | 49.741 | 72.551 | 0.011621 | 2.3242 | 0.20548 | 2.018 | 4.0514 | 2.349 | 36.765 |
| repeated | 6 | dos_rl | 8 | 75.071 | 49.92 | 72.811 | 0.011838 | 2.3676 | 0.221 | 1.8722 | 4.0705 | 2.3629 | 37.535 |
| repeated | 6 | scalar | 8 | 75.008 | 50.114 | 72.041 | 0.011915 | 2.383 | 0.21305 | 1.9028 | 4.0789 | 2.3571 | 37.504 |
| repeated | 8 | ospf | 8 | 57.493 | 49.792 | 57.834 | 0.012066 | 2.4133 | 0.19401 | 2.0896 | 4.0422 | 2.3428 | 28.747 |
| repeated | 8 | shortest_path | 8 | 66.954 | 49.509 | 72.88 | 0.010249 | 2.0499 | 0.20942 | 1.9947 | 3.7019 | 2.3611 | 33.477 |
| repeated | 8 | dos_rl | 8 | 68.434 | 49.739 | 73.091 | 0.010507 | 2.1014 | 0.22676 | 1.854 | 3.7167 | 2.3732 | 34.217 |
| repeated | 8 | scalar | 8 | 68.787 | 50.144 | 72.039 | 0.010688 | 2.1375 | 0.21921 | 1.8817 | 3.748 | 2.3667 | 34.394 |

### E5 paired comparisons

Source: [`e5_paired_tests.csv`](results/e5_paired_tests.csv).

| sweep | value | comparison | n | mean_delta_pp | p_value | effect_size | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| link_failure | 0.2 | dos_rl - shortest_path | 1 | 3.0665 | 0.078125 | 0.72222 | 8 |
| link_failure | 0.4 | dos_rl - shortest_path | 1 | 6.0002 | 0.0078125 | 1 | 8 |
| node_failure | 0.1 | dos_rl - shortest_path | 1 | 0.11337 | 0.74219 | 0.16667 | 8 |
| node_failure | 0.2 | dos_rl - shortest_path | 1 | 0.074145 | 0.74219 | 0.16667 | 8 |
| node_failure | 0.3 | dos_rl - shortest_path | 1 | 0.093741 | 0.64062 | 0.22222 | 8 |
| repeated_outages | 0 | dos_rl - shortest_path | 1 | 0.28144 | 0.54688 | 0.27778 | 8 |
| repeated_outages | 2 | dos_rl - shortest_path | 1 | 0.64044 | 0.46094 | 0.33333 | 8 |
| repeated_outages | 4 | dos_rl - shortest_path | 1 | 1.016 | 0.078125 | 0.72222 | 8 |
| repeated_outages | 6 | dos_rl - shortest_path | 1 | 1.5406 | 0.015625 | 0.94444 | 8 |
| repeated_outages | 8 | dos_rl - shortest_path | 1 | 1.48 | 0.015625 | 0.94444 | 8 |

### E6 ablations

Source: [`e6_ablation.csv`](results/e6_ablation.csv).

| ablation | n | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 10 | 77.488 | 93.478 | 12.172 | 0.073447 | 14.689 | 0.19999 | 1.7372 | 4.0469 | 2.3237 | 232.46 |
| no_shaping | 10 | 77.488 | 93.478 | 12.172 | 0.073447 | 14.689 | 0.19999 | 1.7372 | 4.0469 | 2.3237 | 232.46 |
| no_dos | 10 | 77.342 | 93.201 | 12.097 | 0.073637 | 14.727 | 0.19823 | 1.7445 | 4.0391 | 2.3192 | 232.03 |
| argmax_confidence | 10 | 77.488 | 93.478 | 12.172 | 0.073447 | 14.689 | 0.19999 | 1.7372 | 4.0469 | 2.3237 | 232.46 |
| argmin_confidence | 10 | 76.806 | 93.663 | 12.095 | 0.073327 | 14.665 | 0.20127 | 1.722 | 4.0391 | 2.3108 | 230.42 |
| no_loop_free | 10 | 69.999 | 109.84 | 8.1311 | 0.098824 | 19.765 | 0.26415 | 1.5521 | 5.6505 | 2.4625 | 210 |
| visited_loop_free | 10 | 55.654 | 96.767 | 6.1152 | 0.077207 | 15.441 | 0.18448 | 1.2393 | 3.7364 | 1.9519 | 166.96 |
| paper_reward | 10 | 77.488 | 93.478 | 12.172 | 0.073447 | 14.689 | 0.19999 | 1.7372 | 4.0469 | 2.3237 | 232.46 |

### E6 deployment battery

Source: [`e6_test_battery.csv`](results/e6_test_battery.csv).

| policy | n | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ospf | 30 | 78.982 | 70.464 | 12.727 | 0.067346 | 13.469 | 0.1949 | 1.7565 | 3.7915 | 2.2682 | 236.95 |
| shortest_path | 30 | 79.912 | 70.513 | 12.72 | 0.068296 | 13.659 | 0.19703 | 1.7472 | 3.839 | 2.2685 | 239.74 |
| energy_aware_greedy | 30 | 69.463 | 61.084 | 11.96 | 0.061981 | 12.396 | 0.16417 | 1.952 | 3.6039 | 2.3855 | 208.39 |
| full | 30 | 82.408 | 70.639 | 12.594 | 0.070826 | 14.165 | 0.20109 | 1.7099 | 3.9307 | 2.304 | 247.22 |
| no_dos | 30 | 81.811 | 70.911 | 12.577 | 0.070353 | 14.071 | 0.19816 | 1.7015 | 3.9128 | 2.2975 | 245.43 |
| linear_fa | 30 | 81.944 | 75.435 | 12.253 | 0.071723 | 14.345 | 0.19582 | 1.7138 | 3.9741 | 2.3113 | 245.83 |

### E6 paired statistics

Source: [`e6_statistics.csv`](results/e6_statistics.csv).

| policy_a | policy_b | metric | n | mean_a | mean_b | delta | delta_ci95 | p_value | rank_biserial | cliffs_delta | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| energy_aware_greedy | full | pdr_pct | 1 | 69.463 | 82.408 | -12.945 | 6.4151 | 6.9104e-07 | -0.91398 | -0.39111 | 30 |
| energy_aware_greedy | full | energy_efficiency | 1 | 11.96 | 12.594 | -0.63394 | 1.7702 | 0.1706 | -0.29032 | -0.066667 | 30 |
| energy_aware_greedy | full | e2e_delay_ms | 1 | 61.084 | 70.639 | -9.5558 | 7.8064 | 0.022099 | -0.47527 | -0.25333 | 30 |
| energy_aware_greedy | full | fairness | 1 | 0.16417 | 0.20109 | -0.036921 | 0.021296 | 0.0023675 | -0.6172 | -0.40444 | 30 |
| energy_aware_greedy | linear_fa | pdr_pct | 1 | 69.463 | 81.944 | -12.482 | 6.4998 | 0.00066655 | -0.68172 | -0.3 | 30 |
| energy_aware_greedy | linear_fa | energy_efficiency | 1 | 11.96 | 12.253 | -0.29265 | 1.6239 | 0.28009 | -0.23011 | -0.024444 | 30 |
| energy_aware_greedy | linear_fa | e2e_delay_ms | 1 | 61.084 | 75.435 | -14.351 | 8.4284 | 4.9684e-05 | -0.78925 | -0.38222 | 30 |
| energy_aware_greedy | linear_fa | fairness | 1 | 0.16417 | 0.19581 | -0.031647 | 0.022029 | 0.01546 | -0.50108 | -0.33778 | 30 |
| energy_aware_greedy | no_dos | pdr_pct | 1 | 69.463 | 81.811 | -12.348 | 6.2291 | 5.7183e-07 | -0.91828 | -0.37556 | 30 |
| energy_aware_greedy | no_dos | energy_efficiency | 1 | 11.96 | 12.577 | -0.6167 | 1.7968 | 0.17719 | -0.28602 | -0.062222 | 30 |
| energy_aware_greedy | no_dos | e2e_delay_ms | 1 | 61.084 | 70.911 | -9.8274 | 8.1487 | 0.012834 | -0.51398 | -0.25333 | 30 |
| energy_aware_greedy | no_dos | fairness | 1 | 0.16417 | 0.19815 | -0.033986 | 0.017721 | 0.00060841 | -0.68602 | -0.38 | 30 |
| energy_aware_greedy | ospf | pdr_pct | 1 | 69.463 | 78.982 | -9.5198 | 6.1694 | 0.0003179 | -0.75862 | -0.28111 | 30 |
| energy_aware_greedy | ospf | energy_efficiency | 1 | 11.96 | 12.727 | -0.7663 | 1.9116 | 0.17655 | -0.29741 | -0.032222 | 30 |
| energy_aware_greedy | ospf | e2e_delay_ms | 1 | 61.084 | 70.464 | -9.3806 | 7.6657 | 0.052956 | -0.40517 | -0.27 | 30 |
| energy_aware_greedy | ospf | fairness | 1 | 0.16417 | 0.1949 | -0.030734 | 0.019841 | 0.01897 | -0.48276 | -0.29889 | 30 |
| energy_aware_greedy | shortest_path | pdr_pct | 1 | 69.463 | 79.912 | -10.45 | 6.3827 | 1.9496e-05 | -0.90086 | -0.31444 | 30 |
| energy_aware_greedy | shortest_path | energy_efficiency | 1 | 11.96 | 12.72 | -0.75915 | 1.9145 | 0.18357 | -0.2931 | -0.032222 | 30 |
| energy_aware_greedy | shortest_path | e2e_delay_ms | 1 | 61.084 | 70.513 | -9.4292 | 7.6887 | 0.050359 | -0.41379 | -0.27222 | 30 |
| energy_aware_greedy | shortest_path | fairness | 1 | 0.16417 | 0.19703 | -0.032863 | 0.019903 | 0.013292 | -0.50862 | -0.32333 | 30 |
| full | linear_fa | pdr_pct | 1 | 82.408 | 81.944 | 0.46329 | 1.0704 | 0.087939 | 0.35914 | 0.13333 | 30 |
| full | linear_fa | energy_efficiency | 1 | 12.594 | 12.253 | 0.34129 | 0.29624 | 0.016431 | 0.49677 | 0.053333 | 30 |
| full | linear_fa | e2e_delay_ms | 1 | 70.639 | 75.435 | -4.7956 | 2.8327 | 0.0001699 | -0.74194 | -0.15111 | 30 |
| full | linear_fa | fairness | 1 | 0.20109 | 0.19581 | 0.0052749 | 0.013936 | 0.9193 | 0.023656 | 0.026667 | 30 |
| full | no_dos | pdr_pct | 1 | 82.408 | 81.811 | 0.59676 | 1.2012 | 0.12287 | 0.35135 | 0.035556 | 30 |
| full | no_dos | energy_efficiency | 1 | 12.594 | 12.577 | 0.017238 | 0.12623 | 0.73171 | 0.10811 | 0.022222 | 30 |
| full | no_dos | e2e_delay_ms | 1 | 70.639 | 70.911 | -0.27161 | 2.4937 | 0.79707 | 0.067568 | 0.026667 | 30 |
| full | no_dos | fairness | 1 | 0.20109 | 0.19815 | 0.0029352 | 0.012312 | 0.42371 | 0.15315 | 0.033333 | 30 |
| full | ospf | pdr_pct | 1 | 82.408 | 78.982 | 3.4253 | 1.8939 | 0.00023164 | 0.72903 | 0.17333 | 30 |
| full | ospf | energy_efficiency | 1 | 12.594 | 12.727 | -0.13236 | 0.22772 | 0.96767 | 0.010753 | 0.0088889 | 30 |
| full | ospf | e2e_delay_ms | 1 | 70.639 | 70.464 | 0.17519 | 2.3056 | 0.52911 | 0.13548 | 0.011111 | 30 |
| full | ospf | fairness | 1 | 0.20109 | 0.1949 | 0.0061878 | 0.017448 | 0.12935 | 0.32043 | 0.10222 | 30 |
| full | shortest_path | pdr_pct | 1 | 82.408 | 79.912 | 2.4952 | 1.6683 | 0.002061 | 0.65517 | 0.13 | 30 |
| full | shortest_path | energy_efficiency | 1 | 12.594 | 12.72 | -0.12521 | 0.22845 | 0.77035 | 0.073276 | 0.012222 | 30 |
| full | shortest_path | e2e_delay_ms | 1 | 70.639 | 70.513 | 0.12651 | 2.3039 | 0.58137 | 0.12069 | 0.01 | 30 |
| full | shortest_path | fairness | 1 | 0.20109 | 0.19703 | 0.0040579 | 0.017171 | 0.19824 | 0.27586 | 0.083333 | 30 |
| linear_fa | no_dos | pdr_pct | 1 | 81.944 | 81.811 | 0.13347 | 1.5374 | 0.27101 | -0.23441 | -0.091111 | 30 |
| linear_fa | no_dos | energy_efficiency | 1 | 12.253 | 12.577 | -0.32405 | 0.31549 | 0.02341 | -0.47097 | -0.031111 | 30 |
| linear_fa | no_dos | e2e_delay_ms | 1 | 75.435 | 70.911 | 4.524 | 1.861 | 3.7905e-06 | 0.87097 | 0.15333 | 30 |
| linear_fa | no_dos | fairness | 1 | 0.19581 | 0.19815 | -0.0023397 | 0.012726 | 0.73034 | 0.075269 | -0.013333 | 30 |
| linear_fa | ospf | pdr_pct | 1 | 81.944 | 78.982 | 2.962 | 2.2602 | 0.063557 | 0.38925 | 0.06 | 30 |
| linear_fa | ospf | energy_efficiency | 1 | 12.253 | 12.727 | -0.47365 | 0.47847 | 0.26212 | -0.23871 | -0.033333 | 30 |
| linear_fa | ospf | e2e_delay_ms | 1 | 75.435 | 70.464 | 4.9708 | 3.5898 | 0.0020202 | 0.62581 | 0.14222 | 30 |
| linear_fa | ospf | fairness | 1 | 0.19581 | 0.1949 | 0.00091289 | 0.016449 | 0.62635 | 0.10538 | 0.042222 | 30 |
| linear_fa | shortest_path | pdr_pct | 1 | 81.944 | 79.912 | 2.0319 | 2.1891 | 0.46883 | 0.15086 | -0.0055556 | 30 |
| linear_fa | shortest_path | energy_efficiency | 1 | 12.253 | 12.72 | -0.4665 | 0.4791 | 0.30437 | -0.21121 | -0.032222 | 30 |
| linear_fa | shortest_path | e2e_delay_ms | 1 | 75.435 | 70.513 | 4.9221 | 3.5942 | 0.0041691 | 0.59914 | 0.13889 | 30 |
| linear_fa | shortest_path | fairness | 1 | 0.19581 | 0.19703 | -0.001217 | 0.016436 | 0.80362 | 0.060345 | 0.012222 | 30 |
| no_dos | ospf | pdr_pct | 1 | 81.811 | 78.982 | 2.8285 | 1.5152 | 0.0001886 | 0.73763 | 0.14222 | 30 |
| no_dos | ospf | energy_efficiency | 1 | 12.577 | 12.727 | -0.1496 | 0.20033 | 0.61201 | -0.10968 | 0.0066667 | 30 |
| no_dos | ospf | e2e_delay_ms | 1 | 70.911 | 70.464 | 0.4468 | 2.8991 | 0.73034 | 0.075269 | -0.013333 | 30 |
| no_dos | ospf | fairness | 1 | 0.19815 | 0.1949 | 0.0032526 | 0.011522 | 0.096102 | 0.35054 | 0.08 | 30 |
| no_dos | shortest_path | pdr_pct | 1 | 81.811 | 79.912 | 1.8984 | 1.2746 | 0.0034752 | 0.5957 | 0.10444 | 30 |
| no_dos | shortest_path | energy_efficiency | 1 | 12.577 | 12.72 | -0.14245 | 0.20049 | 0.87121 | -0.036559 | 0.011111 | 30 |
| no_dos | shortest_path | e2e_delay_ms | 1 | 70.911 | 70.513 | 0.39812 | 2.899 | 0.82358 | 0.049462 | -0.015556 | 30 |
| no_dos | shortest_path | fairness | 1 | 0.19815 | 0.19703 | 0.0011226 | 0.011274 | 0.1706 | 0.29032 | 0.06 | 30 |
| ospf | shortest_path | pdr_pct | 1 | 78.982 | 79.912 | -0.93007 | 1.0075 | 0.043114 | -1 | -0.038889 | 30 |
| ospf | shortest_path | energy_efficiency | 1 | 12.727 | 12.72 | 0.0071509 | 0.0085526 | 0.043114 | 1 | 0.0077778 | 30 |
| ospf | shortest_path | e2e_delay_ms | 1 | 70.464 | 70.513 | -0.048683 | 0.0588 | 0.043114 | -1 | -0.0055556 | 30 |
| ospf | shortest_path | fairness | 1 | 0.1949 | 0.19703 | -0.0021299 | 0.002493 | 0.043114 | -1 | -0.025556 | 30 |

### E7 deployment generalisation

Source: [`e7_generalisation.csv`](results/e7_generalisation.csv).

| family | policy | n | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_train_30 | ospf | 20 | 79.13 | 69.013 | 12.998 | 0.067521 | 13.504 | 0.20249 | 1.7314 | 3.7737 | 2.272 | 237.39 |
| A_train_30 | shortest_path | 20 | 79.953 | 69.022 | 12.997 | 0.068185 | 13.637 | 0.2044 | 1.7208 | 3.8112 | 2.2721 | 239.86 |
| A_train_30 | tabular | 20 | 82.522 | 69.599 | 12.812 | 0.070479 | 14.096 | 0.19973 | 1.7275 | 3.9443 | 2.2711 | 247.57 |
| A_train_30 | linear_fa | 20 | 81.7 | 77.306 | 12.345 | 0.071621 | 14.324 | 0.18285 | 1.7853 | 3.971 | 2.3207 | 245.1 |
| B_unseen_45 | ospf | 20 | 80.379 | 62.23 | 13.904 | 0.060283 | 12.057 | 0.12614 | 2.3126 | 3.4643 | 2.2577 | 241.14 |
| B_unseen_45 | shortest_path | 20 | 82.977 | 62.739 | 13.824 | 0.06318 | 12.636 | 0.13145 | 2.2697 | 3.6237 | 2.2587 | 248.93 |
| B_unseen_45 | tabular | 20 | 84.875 | 61.135 | 13.778 | 0.064429 | 12.886 | 0.13033 | 2.2296 | 3.6861 | 2.2831 | 254.63 |
| B_unseen_45 | linear_fa | 20 | 85.492 | 66.738 | 13.397 | 0.066531 | 13.306 | 0.12134 | 2.2345 | 3.7246 | 2.3462 | 256.48 |
| C_unseen_60 | ospf | 20 | 82.561 | 56.966 | 13.112 | 0.064616 | 12.923 | 0.11849 | 2.4295 | 3.7015 | 2.241 | 247.68 |
| C_unseen_60 | shortest_path | 20 | 83.54 | 57.107 | 13.09 | 0.065663 | 13.133 | 0.11947 | 2.4268 | 3.7185 | 2.239 | 250.62 |
| C_unseen_60 | tabular | 20 | 86.144 | 58.29 | 13.065 | 0.067958 | 13.592 | 0.10488 | 2.5407 | 3.8552 | 2.2606 | 258.43 |
| C_unseen_60 | linear_fa | 20 | 86.295 | 63.213 | 12.882 | 0.069173 | 13.835 | 0.10782 | 2.5185 | 3.8979 | 2.2903 | 258.88 |
| D_unseen_clutter | ospf | 20 | 82.422 | 88.9 | 11.177 | 0.078625 | 15.725 | 0.20166 | 1.6852 | 4.2333 | 2.5175 | 247.27 |
| D_unseen_clutter | shortest_path | 20 | 85.822 | 89.727 | 11.092 | 0.082969 | 16.594 | 0.21237 | 1.634 | 4.4415 | 2.5165 | 257.47 |
| D_unseen_clutter | tabular | 20 | 86.983 | 91.787 | 11.104 | 0.084453 | 16.891 | 0.20797 | 1.6693 | 4.5376 | 2.524 | 260.95 |
| D_unseen_clutter | linear_fa | 20 | 85.674 | 94.669 | 11.007 | 0.084307 | 16.861 | 0.20513 | 1.6816 | 4.5136 | 2.5375 | 257.02 |

### E8 observation profiles

Source: [`e8_partial_observability.csv`](results/e8_partial_observability.csv).

| profile | policy | n | pdr_pct | e2e_delay_ms | energy_efficiency | avg_power_w | total_energy_j | fairness | load_imbalance | mean_hops | mean_attempts | throughput_pps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | dos_rl | 8 | 75.395 | 98.946 | 11.553 | 0.075059 | 15.012 | 0.20651 | 1.7043 | 4.1086 | 2.3344 | 226.19 |
| full | linear_fa | 8 | 74.396 | 102.12 | 11.179 | 0.075351 | 15.07 | 0.19373 | 1.7715 | 4.1213 | 2.3241 | 223.19 |
| full | shortest_path | 8 | 74.184 | 102.72 | 11.298 | 0.074862 | 14.972 | 0.20219 | 1.7369 | 4.0996 | 2.3189 | 222.55 |
| delay | dos_rl | 8 | 75.395 | 98.946 | 11.553 | 0.075059 | 15.012 | 0.20651 | 1.7043 | 4.1086 | 2.3344 | 226.19 |
| delay | linear_fa | 8 | 74.079 | 101.88 | 11.163 | 0.075232 | 15.046 | 0.19351 | 1.7603 | 4.1116 | 2.3257 | 222.24 |
| delay | shortest_path | 8 | 74.184 | 102.72 | 11.298 | 0.074862 | 14.972 | 0.20219 | 1.7369 | 4.0996 | 2.3189 | 222.55 |
| quantised | dos_rl | 8 | 75.395 | 98.946 | 11.553 | 0.075059 | 15.012 | 0.20651 | 1.7043 | 4.1086 | 2.3344 | 226.19 |
| quantised | linear_fa | 8 | 74.306 | 102.2 | 11.167 | 0.075322 | 15.064 | 0.19508 | 1.7653 | 4.122 | 2.3212 | 222.92 |
| quantised | shortest_path | 8 | 74.184 | 102.72 | 11.298 | 0.074862 | 14.972 | 0.20219 | 1.7369 | 4.0996 | 2.3189 | 222.55 |
| noisy | dos_rl | 8 | 75.395 | 98.946 | 11.553 | 0.075059 | 15.012 | 0.20651 | 1.7043 | 4.1086 | 2.3344 | 226.19 |
| noisy | linear_fa | 8 | 74.384 | 102.52 | 11.362 | 0.074869 | 14.974 | 0.19487 | 1.7907 | 4.1213 | 2.3124 | 223.15 |
| noisy | shortest_path | 8 | 74.184 | 102.72 | 11.298 | 0.074862 | 14.972 | 0.20219 | 1.7369 | 4.0996 | 2.3189 | 222.55 |
| severe | dos_rl | 8 | 75.395 | 98.946 | 11.553 | 0.075059 | 15.012 | 0.20651 | 1.7043 | 4.1086 | 2.3344 | 226.19 |
| severe | linear_fa | 8 | 73.243 | 104.66 | 11.264 | 0.074654 | 14.931 | 0.18837 | 1.804 | 4.1121 | 2.309 | 219.73 |
| severe | shortest_path | 8 | 74.184 | 102.72 | 11.298 | 0.074862 | 14.972 | 0.20219 | 1.7369 | 4.0996 | 2.3189 | 222.55 |
