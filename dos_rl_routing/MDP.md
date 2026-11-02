# CMOMDP: formal routing formulation

**Project:** RL-based energy-efficient routing for SDWSN-IoT
**Framework paper:** Godfrey et al., *Sensors* **2023**, 23, 8435 (DOS-RL)
**Implementation:** `dos_rl/mdp.py` — every symbol below maps to a function there.

This document supplies what the framework paper omits: a complete, checkable
decision-process specification with states, actions, rewards, transitions and
terminal conditions. The optional telemetry extension is in
[`POMDP_EXTENSION.md`](POMDP_EXTENSION.md). See
[`CMOMDP_COMPARISON.md`](CMOMDP_COMPARISON.md) for a paper-to-code audit.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `V = {1..N}` | set of relay nodes (RNs) |
| `G` | gateway / base station (mains-powered) |
| `N(i)` | static radio neighbours of relay `i` (within `R_c = 450 m`) |
| `d_ij`, `d_iG` | Euclidean distances (m) |
| `E_i(t)` | true residual energy of relay `i` (J), `E_init = 5 J` |
| `Ẽ_i = E_i / E_init` | normalised residual energy |
| `θ = 0.10` | eligibility threshold: relays below `θ·E_init` are not routed through |
| `p_ij` | per-frame packet reception ratio on link `(i,j)` |
| `K = 3` | MAC-layer ARQ attempt limit |
| `λ_i` | aggregated offered load at relay `i` (pkt/s) |
| `μ = 1/(T_frame + T_turn)` | service rate of each relay queue |
| `B = 20` | relay buffer size (packets, M/M/1/K) |
| `γ` | discount factor (0.85 network task) |

---

## 1. The correlated multi-objective MDP (CMOMDP)

A finite-horizon, stochastic, **multi-objective** MDP

```
M = ( S, A, P, R, γ, S_T )
```

### 1.1 State space `S`

For a full simulator state, a suitable conceptual choice is
`X_t = (i_t, E_t, q_t, load_t, PRR_t, link_up_t, eligible_t, visited_t, h_t)`:
the packet location, network-wide energy, queues and loads, link and eligibility
status, visited nodes and hop count. Fixed topology and configuration are part
of the environment. A Markov model of deployed dynamics must also account for
relevant simulator and observation-channel history. This is a formulation, **not**
the representation stored by the tabular agents.

The implemented tabular observation (Q-table key) is either `z_t = (i_t,)`
(default), or the compact representation:

```
z_t = ( i_t , e_t , q_t )
```

| Component | Meaning | Cardinality |
|---|---|---|
| `i_t ∈ V` | relay currently holding the packet | `N` |
| `e_t ∈ {0..2}` | coarse residual-energy bucket of `i_t` (empty of the *network*, not of the packet) | 3 |
| `q_t ∈ {0..2}` | coarse congestion bucket of `i_t` (from the belief) | 3 |

The framework paper defines the state as **the relay node alone** (Sec. 3.2.3,
item 2: "Each state in the state space corresponds to an RN"). That is the
degenerate case `e_t = q_t = 0`, implemented as `RoutingMDP.state_mode="node"`
and used as the paper-faithful default. This is an observation/key, not a
complete description of the physical state.

Both representations alias network situations. In particular, even for the same
`(i,j)` and the same local bins, the next relay's battery and queue and link PRR
may differ, changing reward and transition probabilities. Action `j` is already
part of the Q-table key; merely appending `j` to the state would not fix this.
The bins add information but **do not establish the Markov property**. The 9×
figure is a nominal upper bound on local bin combinations per node, not a
measured table-size increase. This approximation and the effect of alternative
representations require a controlled, held-out experiment.

> **Implementation:** `RoutingMDP.encode(node, mode)`

### 1.2 Action space `A`

```
A(s) = { j ∈ N(i) : E_j > θE_init , alive(j) , link(i,j) up , loop-free(j) }
```

* Training eligibility is derived from sampled energy against the configured
  threshold (default 10%). In deployment `from_metrics` uses reported `alive`
  and smoothed energy above 2%; the simulator also passes its true eligibility
  mask. These are not identical tests.
* **Loop restriction:** in default `dag` mode, candidates normally have lower
  BFS hop distance to `G`; a geographic fallback is used if none remains.
  This is an approximation of the paper's STP pre-pass, not an actual STP
  implementation or an unconditional guarantee of strict BFS descent.
* If `d_iG ≤ R_c` the only action is the terminal action `G` (action id `-1`).

The actual candidate set is `A(X)`, not generally a function of the compact
`z` alone. Direct gateway reachability is a special case. In `dag` mode, the
fallback can use a geographically closer eligible neighbour if no lower-hop
neighbour remains; hence strict hop descent is not guaranteed in that case.
The `stp` option also permits hop slack and is not necessarily acyclic.

> **Implementation:** `RoutingMDP.actions(node)`

### 1.3 Transition rule `P(s_{t+1} | s_t, a_t)`

Given `s_t = (i, ·, ·)` and action `a_t = j`:

```
1. energy feasibility
   if  E_i − c(i,j) ≤ θE_init          → terminal EXHAUSTED
      where c(i,j) = P_tx(d_ij)·T_frame + P_rx·T_frame

2. link outcome (ARQ, K attempts)
   success  w.p.  1 − (1 − p_ij)^K     → packet moves to j
   failure  w.p.  (1 − p_ij)^K         → terminal DEAD_END ("link_failure")

3. on success
   s_{t+1} = ( j , bin(Ẽ_j) , bin(q_j) )
   E_i ← E_i − c(i,j)                (sender pays)
   load_i ← clip(load_i + 0.05, 0, 1)
   q_j ← clip(q_j + 0.05, 0, 1)     (or q_i for gateway action)

4. terminal checks (see §1.5)
```

`p_ij` is derived from the link budget, not fitted:

```
PL(d)   = PL_1m + 10·n·log10(d)                     (n = 2.7, PL_1m = 40 dB)
PRR(d)  = 1 / (1 + exp(−(P_max − PL(d) − P_sens)/w))    (w = 6 dB)
```

**Modelling assumption (stated explicitly):** the transition is a *routing*
transition. Queue occupancy and energy are aggregated analytically rather than
simulated packet by packet; the aggregate is the M/M/1/K closed form

```
ρ_i = λ_i / μ
P_block,i = [ (1−ρ_i) / (1−ρ_i^{B+1}) ] · ρ_i^B          (ρ_i ≠ 1)
```

which is exact for Poisson arrivals and exponential service — the standard
assumption for CBR sensor traffic. `tests/test_network_sim.py` checks the
analytic model against a packet-level reference simulation.

> **Implementation:** `RoutingMDP.step()`, `NetworkSimulator.evaluate()`

### 1.4 Reward function `R` — three correlated objectives

Re-using the paper's objective triple, normalised to `[0,1]` so that the
objectives have comparable ranges. This does not make the confidence comparison
scale-invariant: Q-values and the exponential variance term remain scale-sensitive:

| Objective | Equation | Paper Eq. | Implementation |
|---|---|---|---|
| `o1` energy | `R_E = ½[1 + (Ẽ_j − Ẽ_i)]` `(paper)`<br>`R_E = clip(1 − c̃(i,j) − (1−Ẽ_j), 0, 1)` `(corrected)` | Eq. 4 | `reward_mode` |
| `o2` load | `R_Q = ½[1 + (buf_j/B − load_i/L_max)]` | Eq. 5 | default |
| `o3` link | `R_LQ = α_w·P̂RR_ij(k−1) + (1−α_w)·PRR_ij(k)` | Eq. 6 | training copies model PRR; deployment smooths telemetry |

**Potential-based shaping** (Eq. 8–9):

```
F_t = γ·φ(s_{t+1}) − φ(s_t),     φ(s) = 1 − d(s,G)/d_max
```

The corrected network code uses zero potential for every terminal outcome, so
`F_t = -φ(s_t)` at a terminal transition. The usual policy-invariance theorem
has further assumptions and is not established for the aliased relay-only key.
In the historical E6 results Q-learning did not receive `F_t`; those results
are not a valid shaping ablation. New runs use the corrected updates.

**Terminal rewards:** `+10` for reaching `G`, and generally `−10` for failure
transitions. If there is no action at the initial relay, the trainer exits
without adding a failure update.

**Logged scalar episode reward:**

```
r_t = wᵀR_t / ‖w‖₁ + F_t + r_terminal
```

Network Q-learning and SARSA now update each objective with `R_o + F_t` and the
terminal reward. The single-objective Q learner uses the normalized weighted
sum of the objective vector, with the same shaping and terminal terms. Monte
Carlo and the linear agent also receive shaping. The old `results/` and linked
report tables remain historical; corrected outputs go to `corrected_suite/`.

> **Implementation:** `RoutingMDP.rewards()`, `RoutingMDP.shaping()`, `trainer.run_episode`

### 1.5 Terminal set `S_T`

| Code | Condition | Reward | Interpretation |
|---|---|---|---|
| `GOAL` | packet reaches `G` | `+10` | successful delivery |
| `EXHAUSTED` | forwarding would drop `i` below `θE_init` | `−10` | energy-exhaustion |
| `DEAD_END` | no eligible neighbour, or all `K` ARQ attempts fail | `−10` | route/​link failure |
| `LOOP` | revisit of an already-visited relay | `−10` | routing cycle |
| `HOP_LIMIT` | `hops ≥ H_max` (default 20) or nonpositive downstream hop map | `−10` | runaway or unreachable path |

`S_T = {GOAL} ∪ {EXHAUSTED, DEAD_END, LOOP, HOP_LIMIT}`. Episodes are also
truncated at `H_max` decisions. **Only `GOAL` is a success.**

> **Implementation:** terminal constants and `RoutingMDP.step()`

### 1.6 Horizon and objective

One episode = **one route computation** for one sampled source. The agent chooses
relays hop by hop; the learning objective uses the discounted sum of per-hop rewards:

```
G_t = Σ_{k≥0} γ^k r_{t+k}
```

with `γ = 0.85` (paper Table 2). Plotted episode returns are undiscounted sums.
The corrected **network** Q-learning update
(non-terminal) includes `F_t` as in paper Eq. 8:

```
Q_o(s_t,a_t) ← Q_o(s_t,a_t) + α[ r_{o,t} + F_t + γ max_a Q_o(s_{t+1},a) − Q_o(s_t,a_t) ]
```

---

## 2. Dynamic Objective Selection (DOS)

Per objective `o`, a variance is tracked (paper Eq. 2):

```
VAR_o(s,a) ← (1−ψ)VAR_o(s,a) + ψ·δ_o²            δ_o = TD error
```

and a confidence is formed (paper Eq. 3):

```
C_k = K(μ_k + e^{σ²_k}) / Σ_j (μ_j + e^{σ²_j})
```

with `μ_k = max_a Q_k(s,a)` and `σ²_k = mean_a VAR_k(s,a)`. The action is then
chosen greedily on the single objective `o_best`.

**Four readings are implemented and compared** (`RLConfig.dos_select`), because
the paper's prose, its Eq. 3 and its Algorithm 1 do not agree:

| Mode | Rule | What actually gets selected |
|---|---|---|
| `argmax` | `o* = argmax_k C_k` (Eq. 3 literal) | can favour high variance when `e^{σ²}` dominates; value terms also matter |
| `argmin` | `o* = argmin_k C_k` (Algorithm 1 literal) | again variance-driven, by the same exponential term |
| `var` | `o* = argmin_k σ²_k` | the most stable objective |
| **`value`** (default) | `o* = argmax_k μ_k` | the objective with the best estimated value |

E6 includes the `argmax` and `argmin` alternatives alongside the selected default;
the exponential term can dominate, but high variance need not win for all Q-values.

---

## 3. Optional partial-observability extension

The paper's core model is a CMOMDP. The separate observation/telemetry study is
in [`POMDP_EXTENSION.md`](POMDP_EXTENSION.md); it is not a complete POMDP
solver and is not required to define the core routing task.

---

## 4. Closed-loop deployment

The controller is evaluated as deployed, not as a one-shot shortest-path
computation:

```
for each routing epoch (3 s, 200 s total):
    collect  → controller telemetry (full profile by default)
    update   → smoothed network estimates
    plan     → choose a next hop per relay
    repair   → loop-check the table (NetworkSimulator.resolve)
    evaluate → M/M/1/K blocking, ARQ, energy drain, E2E delay
    advance  → energy, queues, failures
```

**Observation leakage:** deployment passes the true eligibility mask and checks
`sim.link_up` when building actions; some baselines also use true energy. Thus
The optional E8 extension therefore does not establish robustness to hidden
failures; see `POMDP_EXTENSION.md`. The node-only Q-table key does not directly
encode the smoothed estimates.

Policies that cannot adapt (`OSPFStaticPolicy`) install their table once and
lose routes permanently when a relay dies — the paper's stated OSPF failure mode.
Policies that re-plan (`GreedyHopPolicy`, all RL agents) react every epoch.

---

## 5. Mapping to the assignment rubric

| Rubric item | Where |
|---|---|
| 1. Problem understanding, literature | `REPORT.md` §1–2 |
| 2. CMOMDP formulation + reward design | **this document**, `REPORT.md` §2 |
| 3. Working implementation | `dos_rl/`, `experiments/`, `tests/` |
| 4. Interpretation of results | `REPORT.md` §5–6 |
| 5. One parameter / environmental change | `REPORT.md` §5.4–5.5 |
