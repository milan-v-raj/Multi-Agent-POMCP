# Separate route-snapshot state experiment

This experiment compares the relay-only Q-table key with a key containing the full synthetic route-episode snapshot. It does not replace the main agent's state or represent the complete deployed physical network.

The snapshot includes the current relay, visited relays, hop count, energies, queue loads, link PRRs, eligibility, hop map and link status. Topology and configuration are fixed for a run. Exact snapshot keys can be very large and rarely repeat, limiting tabular generalisation.

Run a small test from the submission folder:

```bash
python -m unittest full_state_experiment.test_full_state -v
python -m full_state_experiment.run --seeds 3 --episodes 100 --test-episodes 30 --output runs/full_state/results.csv
```

The saved full-suite comparison is `results/full_state_comparison.csv`. Test episodes use a separate random stream on the same topology used in training.
