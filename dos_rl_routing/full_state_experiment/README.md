# Separate experiment: a full routing-episode state

This is an opt-in experiment. It does **not** change `dos_rl/mdp.py`, the
original experiment scripts, their results, or their report claims.

## What is the state?

For **one routing episode on a fixed topology and fixed configuration**, the
experimental `FullStateRoutingMDP` uses:

```
s = (current relay, successful-hop count, visited relays,
     energy of all relays, queues of all relays, relay loads,
     link PRR matrix, eligible-relay mask, hop-distance map,
     up/down status of all links)
```

These are exact values from the episode snapshot, not discretised buckets. The
key uses immutable byte strings for the arrays, so later mutations do not change
previous Q-table keys. Each terminal outcome has a separate absorbing key.

### Why this is Markov for this episode model

With a fixed topology, radio and configuration, inspection of the existing
`RoutingMDP.actions`, `rewards` and `step` shows that all their mutable inputs
are included in the state above. Given this state and an action, the
probability of the next state, terminal outcome and reward is determined by the
existing transition rules and a fresh random draw. A random transition can still
be Markov. The implementation tests equal states with equal random draws and
checks that the listed inputs affect the key. Tests are **not a mathematical
proof** and must be updated if the transition rules gain new dependencies.

The guarantee has a narrow scope:

- This is the **synthetic route-computation episode**, not the complete
  closed-loop simulator. It does not model future arrivals, the simulator's
  failure-event schedule, controller-observer history or route updates.
- The state exposes the episode's entire snapshot, including fields normally
  hidden from an agent. The episode calls these values `belief`; they may be
  synthetic estimates and are not necessarily true physical network conditions.
- Do not reuse one Q-table with different topologies or configurations. These
  are fixed parameters of each MDP, not included in the key.
- This does not make realistic delayed or noisy telemetry fully observable, and
  it does not establish a full Markov model for the physical world.

The actions, objective rewards and dynamics are inherited from the existing
model. This also inherits existing trainer limitations, including that the
network Q-learning/SARSA updates do not include the logged shaping term.

## Run the tests

From `dos_rl_routing/`:

```bash
python -m unittest full_state_experiment.test_full_state -v
```

The checks cover state fields, snapshot immutability, action/reward/transition
consistency under matched random draws, terminal keys and integration with the
existing Q-learning trainer.

## Small isolated comparison

```bash
python -m full_state_experiment.run --seeds 3 --episodes 100 --test-episodes 30
```

The script compares the existing node-only key with the full key using the same
configurations and seeds. It writes **only** `full_state_experiment/results.csv`
(and will overwrite that separate file). Each evaluation uses 30 independently
sampled route episodes on the *same topology as training*; these are not unseen
network deployments. It also records how often a test decision's state key was
present in the training Q-table. A present key need not have been updated.

The checked-in run is a **small smoke test**, not a controlled, adequately
powered comparison of policy quality. Across its three seeds, the full-snapshot
agent reused a training-state key in **0 of 421** test decisions; node-only
reused one in **415 of 415** decisions. Exact continuous snapshots almost never
repeat, so tabular Q-learning cannot generalise to unseen snapshots even though
the state is more complete. The test returns in the CSV should not be treated as
evidence that either representation is better.

A practical next step is to use an approximator over the full information, or
to design a smaller approximate state deliberately; the latter may again lose
the formal Markov guarantee. A proper study would also use more seeds, test
episodes and independent deployments before replacing the original results.
