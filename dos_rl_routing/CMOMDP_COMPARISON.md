# CMOMDP: proposed description vs paper and implementation

> **Historical network results:** Values in this document predate the ARQ
> correction. The completed [ARQ-fixed results](arq_fixed_suite/RESULTS.md) and
> [impact comparison](ARQ_IMPACT.md) are separate; see also the
> [fidelity audit](PAPER_FIDELITY.md).


This audit distinguishes **the proposed description**, **the framework paper**
(Godfrey et al., Sensors 2023, 23, 8435; §3.2, pp. 7–13) and **the saved
implementation**. The separate observation study is in
[`POMDP_EXTENSION.md`](POMDP_EXTENSION.md). No existing experiment or numerical
result in the historical `results/` folder has been replaced. Since that audit,
the reward/update defects described below have been corrected in code. A new
network suite writes separately to `corrected_suite/`; see `CORRECTED_CMOMDP.md`.

| Item | Followed? | What the paper and code actually do |
|---|---|---|
| Central SDN controller | Broadly | A central controller computes forwarding tables; the simulator is a simplified implementation, not the paper's ns-3 system. `dos_rl/policy.py` |
| State includes energy, queues and link quality | **Not as the main agent's state** | Paper §3.2.3 says each state corresponds to a relay node. The default Q-table key is `(node,)`; optional `(node, energy_bin, queue_bin)` omits link and neighbours. The rewards/action mask read additional information, so neither compact key is guaranteed Markov. `dos_rl/mdp.py:encode` |
| Complete network state | Separate experiment only | `full_state_experiment/` includes network-wide episode snapshot values. Its Markov scope is one routing episode on a fixed topology/configuration; it is not used for the main reported results. |
| Next-hop relay action | Yes | `RoutingMDP.actions` lists eligible neighbours; if the gateway is directly reachable, the only action is the gateway. |
| Geographically closer on every hop | No | Default `dag` normally requires fewer BFS hops to the gateway, with a geographic fallback. `stp` permits hop slack; neither is a literal implementation of the paper's STP pre-pass. Strict geographic progress is not the general rule. |
| `R_energy = E_j/E_max` | No, neither exact formula | The code defaults to `0.5*(1 + e_j - e_i)`, where `e_i` and `e_j` are normalized energies. The paper's Eq. 4 prints `E_j - E_i` (its surrounding prose reverses node labels). The proposal depends only on `e_j`. The optional `corrected` mode uses hop cost and `e_j`. |
| `R_load = 1 - Q_j/Q_max` | Partly | The code uses available buffer but also subtracts current relay load: `0.5*(1 + (1 - q_j) - load_i)`. The proposal omits `load_i` and the rescaling; the paper's Eq. 5 includes both quantities. |
| Link quality by delivery ratio / ETX | Partly | The code uses packet reception ratio (PRR), with smoothed estimates in controller telemetry. It does not implement an ETX- or RSSI-based reward. The paper's Eq. 6 uses EWMA PRR. |
| Vector of three objective rewards | Yes for multi-objective Q-learning | It stores three Q values per `(state, action)` and updates per-objective rewards. Scalarised baselines are also legitimate multi-objective methods; fixed weights do not inherently fail. |
| Parallel Q values and TD-error statistics | Yes | `Q` and `VAR` store vectors per state/action; `VAR` tracks an EWMA of squared TD errors. It is not guaranteed to be calibrated statistical uncertainty. `dos_rl/agents/tabular.py` |
| Choose highest-certainty objective | **Not by default** | The default tabular setting `dos_select="value"` picks the objective whose best available Q-value is largest, without using variance. Other settings compute alternative variance/confidence rules. Paper prose says highest certainty, while Algorithm 1 says `argmin` of its Eq. 3; the paper's formula and prose are not automatically equivalent. The linear agent now also uses this value rule by default. |
| DOS-RL expansion | No | The paper calls it **Dynamic Objective Selection** RL, not Degree of Certainty RL. Certainty is its stated selection motivation. |
| Potential-based shaping in Q-learning | **Corrected** | The current trainer passes the shaping term into network Q-learning, SARSA, and linear updates; terminal potential is zero. Historical saved network Q updates omitted it, so the old E6 `no_shaping` comparison is invalid. The corrected suite tests it anew. |

## Important correction to saved network results

The historical network rows and figures labeled `scalar`, `weighted_sum` or
`no_dos` used `QLearningAgent(n_objectives=1)`. At the time, its vector update
wrote only index 0. Thus those learners trained on **the energy component**,
not an equal-weight reward. The corrected code now applies the configured
normalized weighted sum. The
logged episode return is separately scalarised, so its label does not prove how
the policy was trained. For example, with rewards `[0.1, 0.9, 0.9]`, zero
initial Q, and learning rate 0.3, the saved network update is **0.03**, not the
**0.19** produced by equal-weight scalarisation.

Consequently, statements that the network results establish DOS vs a
weighted-sum learner, or demonstrate the effect of shaping, are **withdrawn**.
The numerical results remain available as historical results for the code that
actually ran. The corrected implementation has regression tests and has been rerun separately
without substituting historical results. The gridworld has its own scalarisation path; this specific
network finding should not be automatically applied to it. However, on gridworld
terminal steps the ordinary vector update and `_weighted_update` both run, and
the scalar update omits the terminal bonus; this is not a clean standard
weighted-sum Q-learning baseline either.

The historical linear network learner had additional defects: it did not
bootstrap under the trainer's call, used the first objective's current value for
all objectives, and indexed the last relay for gateway features. These defects
have been corrected and regression-tested; **old** linear results must not be
interpreted as evidence for the corrected implementation.

## A possible simplified design

Your three proposed rewards form a reasonable **alternative design**, provided
link quality is consistently defined (e.g. larger PRR is better, whereas lower
ETX is better and must be transformed). They are not the paper's exact Eqs. 4–6
and are not the formulas currently used in the reported network experiments.
Likewise, adding their input features to an agent representation does not by
itself guarantee the Markov property: other network and episode variables can
influence transitions and available actions.
