# Project Plan — RL-Based Energy-Efficient Routing for SDWSN-IoT

> Historical design plan. The selected current CMOMDP plan and corrected-run
> status are in [`CORRECTED_CMOMDP.md`](CORRECTED_CMOMDP.md); previous
> implementation and result claims here are not a current audit.


**Course project:** RL-based performance optimisation
**Framework paper:** Godfrey, Suh, Lim, Lee & Kim, *"An Energy-Efficient Routing Protocol with
Reinforcement Learning in Software-Defined Wireless Sensor Networks"*, **Sensors 2023, 23, 8435**
(DOI: 10.3390/s23208435) — the **DOS-RL** protocol.
**Status:** plan for approval · **Owner:** _(student)_ · **Supervisor:** _(course faculty)_

---

## 0. Where this lives

This repository currently holds an unrelated capstone (**Deep-POMCP**, multi-agent pursuit–evasion).
To avoid touching that work, everything for this course project goes into a **new, fully
self-contained top-level package** `dos_rl_routing/`. Zero files outside it are modified.

---

## 1. What the paper actually does (and what it leaves to us)

### 1.1 Paper in one paragraph
DOS-RL is a **centralised, software-defined routing protocol** for IoT wireless sensor networks.
An SDN controller maintains a global view of the relay-node (RN) graph and runs **Q-learning**
where *states = RNs*, *actions = candidate next hops*, and the reward is a **vector of three
correlated objectives**: (o1) residual energy, (o2) load balance via available buffer, (o3) link
quality via EWMA packet-reception-ratio. Instead of scalarising, it tracks **per-objective
Q-tables plus a variance estimate**, and before each decision picks the single objective it is
*most confident* about (**Dynamic Objective Selection**, DOS, adapted from Brys et al. 2014).
A **potential-based shaping reward** `F = γφ(s') − φ(s)` with `φ = 1 − d(s,G)/d_max` speeds up
learning while (by Ng et al. 1999) leaving the optimal policy unchanged. Evaluated on a 5×5
dynamic gridworld and on ns-3 network simulations (30/45/60 RNs, 2–10 traffic sources, 200 s,
10 pkt/s, 512-bit packets, 5 J initial energy) against **OSPF** and **SDN-Q**, reporting PDR,
end-to-end delay and energy efficiency.

### 1.2 Report/defence gaps this project closes

| # | Gap in the paper | What we will do |
|---|---|---|
| G1 | No explicit, gradeable **MDP/PO-MDP specification** (only a prose 4-tuple; no terminal conditions, no transition kernel, no observability model) | Write a formal MDP **and** PO-MDP spec (states, actions, rewards, transition rule, terminal set, observation model, belief) — deliverable §3 below |
| G2 | The ns-3 setup is **not reproducible** for us (no ns-3 / ns3-ai here) | Build a **transparent, dependency-free packet-level simulator** (numpy only) with a first-order radio energy model, log-distance/shadowing PRR, M/M/1/K queues, ARQ — fully seeded and auditable |
| G3 | Network-level results are reported only as **steady-state bar charts**; the graded "reward vs. training episode" curve exists only for the toy gridworld | Produce learning curves for **both** the gridworld replication **and** the real network task |
| G4 | **No hyperparameter sensitivity study** (paper's own future-work item) | Full α / γ / ε / ε-decay sweep with learning-curve AUC and convergence-episode metrics |
| G5 | Environmental dynamics are "varied sources / varied RNs" only; **no failure events**, no adaptation-latency measurement | Arrival-rate sweep **+ mid-run link/node failure injection** with measured recovery time |
| G6 | Reward equations have **internal inconsistencies** (see §9) | Implement the paper's literal equations *and* corrected variants; ablate the difference — this is the qualitative core of the report |
| G7 | Only tabular Q-learning; no function approximation, so nothing generalises to unseen topologies | Add a **linear function approximator** over belief features (rubric explicitly allows this) and compare tabular vs. LFA generalisation |

---

## 2. Real-life problem statement

> A solar/battery-powered **landslide & rainfall early-warning sensor network** deployed over a
> 1.5 km × 1.5 km hillslope (Western Ghats context). ~30–60 battery relay nodes forward soil-moisture
> and rainfall samples multi-hop to one gateway. Batteries are non-replaceable, radio traffic is
> bursty during monsoon, and links fade with foliage/rain. **Objective: maximise the fraction of
> warnings that actually reach the gateway, for as many months as possible, without draining any
> single relay.** Routing that ignores residual energy kills nodes; routing that ignores queueing
> drops packets; routing that ignores PRR retransmits into dead links. All three must be traded
> off online, under delayed and noisy controller telemetry — a partially observable decision problem.

This maps 1:1 onto the paper's SDWSN-IoT framing, so the paper's framework is used as intended,
not as a toy.

---

## 3. Formal MDP / PO-MDP (deliverable #2 of the rubric)

### 3.1 MDP — `M = (S, A, P, R, γ, S_T)`

| Element | Definition |
|---|---|
| **State** `s_t` | Current relay node `i ∈ V` where the packet sits, augmented with a *state summary* `x_t = (hop count, remaining energy bucket of i, queue occupancy bucket of i, belief over neighbours)`. The paper's pure-node state is the default implementation. Local augmentation is approximate and does not guarantee a Markov representation; see MDP.md for the current audit. |
| **Action** `a_t` | Choose next hop `j ∈ N_elig(i)`, where `N_elig(i)` = neighbours with reported residual energy ≥ `E_thresh` and a live radio link. Discrete, `|A| = deg(i)`. |
| **Reward** `R_t` | Vector `r_t = (R_E, R_Q, R_LQ)` + shaping `F_t` (see §3.3). Each component normalised to `[0,1]` so the DOS confidence comparison is scale-invariant. |
| **Transition** `P(s_{t+1} \| s_t, a_t)` | Topological but **stochastic**: `s_{t+1} = j` if the transmission succeeds; if the link drops the packet (prob `1 − PRR_ij`) the packet is retried up to `K_arq` times and then discarded; node `j` may become ineligible mid-episode if its energy falls below threshold or it dies; queue occupancy and residual energy evolve as a Markov chain induced by the Poisson traffic. |
| **Discount** `γ` | `0.85` (paper, network task) / `0.99` (paper, gridworld). Swept in E4. |
| **Terminal set `S_T`** | `(T1)` reach the base station → `+R_goal`; `(T2)` hop count > `H_max = 2·max_dist(G)` → `−R_dead`; `(T3)` revisit of a node (loop) → `−R_dead`; `(T4)` no eligible neighbour (dead end) → `−R_dead`; `(T5)` forwarded into a node whose true residual energy < `E_thresh` (energy-exhaustion) → `−R_exhaust`. Episodes are also time-limited to `H_max` decisions. |
| **Horizon** | One *routing epoch* = one complete path computation from a sampled source to the gateway. |

### 3.2 PO-MDP — `M = (S, A, P, R, Ω, O, γ)`

The controller **cannot see the true state**. It receives telemetry that is *delayed, noisy,
coarsely quantised, and sometimes lost*:

| Element | Definition |
|---|---|
| **Hidden state** `s_t` | True residual energies, true queue occupancies, true link PRRs, node alive/dead flags. |
| **Observation** `o_t` | `Ω` emits, per node/link: residual energy quantised into `5 %` buckets with additive noise; queue length with noise; PRR measured from the last `W` transmissions; each report delayed by `D ∈ {0,1,2}` epochs and dropped with probability `p_loss`. |
| **Belief** `b_t` | Vector of per-node/per-link **EWMA estimators with tracked variance**: `b_t = {Ê_i, σ²_E,i, Q̂_i, σ²_Q,i, P̂RR_ij, σ²_PRR,ij}` (paper Eq. 2 style variance tracking, but honest running variance). |
| **Policy** | `π(b_t) → a_t`. Two realisations: a **tabulated belief-state** agent (bucketed belief) and a **linear function approximator** `Q(b,a) = θᵀφ(b,a)` over belief features — this is both the POMDP solution (Q-MDP) *and* the rubric's "linear function approximator" option. |
| **Why it matters** | E8 compares full observability vs. the degraded channel, and shows the belief-tracking agent degrading gracefully instead of oscillating — the qualitative PO-MDP result. |

### 3.3 Reward functions (paper Eq. 4–9, with our corrections)

Let `Ẽ_i = E_i / E_init ∈ [0,1]`, `Buf_avail(j)`, `Load(i)`, `P̂RR_ij` = belief quantities.

```
o1  Energy    R_E(i,j)  = Ẽ_j − Ẽ_i                       (paper Eq. 4, literal, normalised to %)
                          + variant: Ẽ_j − c_hop(i,j)      (ours: physically meaningful per-hop cost)
o2  Load      R_Q(i,j)  = Buf_avail(j)/Buf_max − Load(i)/Max_Load          (paper Eq. 5, verbatim)
o3  Link      R_LQ(i,j) = α_EWMA·P̂RR_ij(k−1) + (1−α_EWMA)·PRR_sample(k)   (paper Eq. 6, verbatim)
Shaping       F_t       = γ·φ(s_{t+1}) − φ(s_t),  φ(s) = 1 − d(s,G)/d_max  (paper Eq. 9)
Total         r_t       = R^{o_best}_t + F_t                               (paper Eq. 8)
```

Terminal rewards: `R_goal = +10`, `R_dead = R_exhaust = −10` (settled by an ablation showing the
goal/dead ratio controls hop-count vs. reachability trade-off).

### 3.4 Algorithms (rubric #3)

| Agent | Type | Purpose |
|---|---|---|
| **Q-learning (tabular)** | off-policy TD(0), paper Eq. 7/8 | faithful DOS-RL reproduction |
| **SARSA (tabular)** | on-policy TD(0) | *safer* policies near depleted nodes — qualitative contrast |
| **First-visit MC control** | Monte Carlo | credit-assignment comparison (no bootstrapping) |
| **Linear FA (semi-gradient Q)** | `Q(b,a)=θᵀφ(b,a)` | scalability + unseen-topology generalisation |
| **DOS wrapper** | Alg. 1 over any above | dynamic objective selection with confidence metric `C_k` |
| Baselines | Dijkstra/OSPF, SDN-Q (single-objective Q-routing), weighted-sum RL | paper's comparison set |
| Oracle | LP/ILP min-energy max-PDR route | optimality gap (quantitative anchor) |

---

## 4. Repository layout

```
dos_rl_routing/
├── PLAN.md                     # this file
├── README.md                   # how to reproduce every figure
├── REPORT.md                   # main deliverable (formatted, figure-embedded)
├── MDP.md                      # formal spec, cross-referenced to code line numbers
├── dos_rl/
│   ├── topology.py             # deployment, log-distance+shadowing PRR, connectivity
│   ├── energy.py               # first-order radio model (Heinzelman)
│   ├── network_sim.py          # packet-level event simulator: M/M/1/K queues, ARQ, drops
│   ├── observer.py             # POMDP channel: delay, noise, quantisation, report loss
│   ├── belief.py               # EWMA + running-variance belief tracker
│   ├── mdp.py                  # RoutingMDP — single source of truth for S/A/R/terminals
│   ├── gridworld.py            # 5×5 dynamic gridworld (paper §4.2 replication)
│   ├── agents/                 # q_learning, sarsa, mc_control, linear_fa, dos
│   └── baselines/              # ospf, sdn_q, weighted_sum, oracle
├── experiments/                # e0…e8, one script per figure, all seeded + resumable
├── results/                    # raw per-episode CSV + aggregated CSV (committed)
└── figures/                    # 300 dpi PNGs (committed)
```

---

## 5. Experiment matrix

`N_test = 30` independent test episodes per configuration (rubric asks ≥ 20);
`n_seeds = 10` independent training seeds for every learning curve.

| ID | Experiment | Produces | Tier |
|---|---|---|---|
| **E0** | Learning curves, 6 agents × 10 seeds | **episode reward vs. training episode** (mean ± 95 % CI), energy/episode, episode length, dead-end/hop-limit frequency — mirrors paper Fig. 4 but on the network task | core |
| **E1** | 5×5 dynamic gridworld replication (obstacles appear every 100 eps, recharge points, 5-unit battery) | direct replication of paper Fig. 4 → validates our reading of the paper | core |
| **E2** | PDR / E2E / EE vs. number of traffic sources (2,4,6,8,10) | paper Figs. 5–7, done as 30-episode error-bar plots | core |
| **E3** | Same vs. number of RNs (30, 45, 60) | scalability | strong |
| **E4** | **Parameter study:** α ∈ {0.05,0.1,0.3,0.5,0.9}, γ ∈ {0.5,0.85,0.95,0.99}, ε ∈ {0.05,0.1,0.3,0.5}, plus ε-decay on/off | convergence episode, AUC, final return, seed variance + heatmaps | core (**rubric #5**) |
| **E5** | **Environmental change:** (a) arrival rate λ swept 2→20 pkt/s; (b) link availability: 20 %/40 % of links killed at t = 100 s, plus a node-failure storm | PDR-vs-λ curves; **recovery time** and pre/post-failure PDR dip area | core (**rubric #5**) |
| **E6** | Final test battery, 30 independent episodes × all agents | mean ± 95 % CI table + Wilcoxon signed-rank tests + effect sizes + optimality gap vs. oracle | core (**rubric #4**) |
| **E7** | Generalisation: train on topology family A, test on unseen B/C | tabular vs. LFA success + PDR drop | strong |
| **E8** | Partial observability ladder: full obs → delay only → noise only → all + loss | shows the belief layer earning its keep | strong |
| **E9** | Dashboard: energy-vs-PDR Pareto front, lifetime curve (nodes alive vs. time), Jain fairness index | qualitative story of *why* RL helps | strong |

**Reported metrics:** PDR (%), end-to-end delay (ms), energy efficiency (PDR per Joule, J/packet),
network lifetime (round of 1st / 50 % node death), Jain fairness index, load imbalance, hop count,
packet retransmissions, episode return, convergence episode, learning AUC, adaptation/recovery time.

---

## 6. Qualitative analysis we commit to reporting (rubric #1, #4)

1. **Why does DOS beat weighted-sum?** Show, per-state, which objective is selected and demonstrate on
   the trace that the confidence metric re-weights toward link quality in fading and toward buffer in
   congestion — i.e. the selection is *state-dependent*, which a fixed scalarisation cannot express.
2. **Does RL actually help?** Decompose the gain against OSPF/SDN-Q into (a) shorter paths, (b) better
   load spreading, (c) avoidance of low-energy nodes — quantify each with ablations that disable one
   objective at a time.
3. **Cost of learning.** Report offline training cost (episodes, wall-clock) vs. online benefit, and the
   price paid during the exploration phase — the honest trade-off graders ask about.
4. **Critical reading of the paper** (see §9) — where the formulation is inconsistent and what we changed.
5. **Failure modes** — where RL loses (e.g. large topologies for tabular, sparse reward, distribution
   shift when λ changes) and why.

---

## 7. Deliverables

1. `dos_rl_routing/REPORT.md` — full report: literature review → problem → formal MDP/PO-MDP →
   algorithms → experimental setup → results (all figures) → discussion → limitations → references.
   ~12–18 pages, figure-heavy.
2. `MDP.md` — the formal specification as a standalone, gradeable artefact.
3. `figures/` — 12–16 publication-quality 300 dpi plots.
4. `results/*.csv` — every number in the report is generated by a script and committed.
5. `README.md` — one command per figure (`python experiments/e2_traffic_sources.py`).
6. A 1-page **viva cheat-sheet** (formulation + headline numbers).
7. Optional on request: LaTeX source matching the department's existing report template.

---

## 8. Execution phases

| Phase | Work | Est. time |
|---|---|---|
| **P1** | Topology + energy + queue simulator, observation model, belief tracker, formal MDP doc, unit tests & sanity checks (energy conservation, PDR→1 in clean conditions, Dijkstra correctness) | ~1 build cycle |
| **P2** | All agents (Q / SARSA / MC / LFA / DOS wrapper), gridworld replication **E1** + convergence **E0** | ~1 cycle |
| **P3** | Baselines + oracle, network benchmark **E2**, **E3**, **E6** (30-episode battery) | ~1 cycle |
| **P4** | **E4** sweeps, **E5** failure/arrival studies, **E7**, **E8**, **E9** — run as background jobs | ~1–2 cycles, mostly compute |
| **P5** | Figures, tables, statistics, `REPORT.md`, viva sheet | ~1 cycle |

Everything is seeded and resumable, so any phase can be re-run reduced (`--quick`) for a live demo.

---

## 9. Critical notes on the paper (drafted; to be verified in code)

These are pre-identified and will be settled empirically, not just asserted:

1. **ε-greedy is inverted.** Paper §3.2.3 and Eq. 10 give *exploit* with probability `ε` and *explore*
   with `1 − ε`, i.e. a 30 %-exploitation agent. We implement the literal rule **and** the standard
   convention as a flag, and show the effect — expected to be large at `ε = 0.3`.
2. **Confidence metric sign.** Eq. 3 defines `C_k = K(µ_k + e^{σ_k²})/Σ`, where larger is *more*
   confident, but Algorithm 1 selects `o_best = argmin_k C_k`. Prose says "highest level of certainty".
   We implement **argmax** (prose-consistent) and report the argmin variant as an ablation.
3. **Eq. 2 is not a variance.** `VAR ← (1-ψ)VAR + ψδ²` tracks *squared TD-error magnitude*, which grows
   with both bad policy **and** high reward variance; it conflates instability with scale. We implement
   a true running variance as the alternative and compare selection behaviour.
4. **Eq. 4 is mis-specified.** `R_E = RemainingEnergy(s_j) − RemainingEnergy(s_i)` compares two
   *different nodes*, not the energy consumed by an action; it rewards picking a full neighbour and is
   invariant to link length, so it ignores the dominant `d²` term of radio cost. Corrected variant
   (`Ẽ_j − c_hop`) ablated against the literal one.
5. **Conflicting reward scales.** Three objectives on raw (%, packets, ratio) scales are compared through
   a "scale-invariant" confidence metric — but the *Q-values* are not scale-invariant. We normalise all
   three to `[0,1]` and report the effect.
6. **No terminal conditions or observability model** are stated anywhere (G1), so as published the
   formulation is not a complete MDP.
7. **Reproducibility.** Public availability = "available on request"; the ns-3/ns3-ai harness is not
   released. Our simulator makes the whole chain auditable.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Simulator runtime explodes (200 s × 10 sources × 30 episodes × many configs) | Learning happens over link-state snapshots (microseconds/episode); only the **evaluation** pass is packet-level. Tiered matrix + `--quick` mode + background execution. |
| "This is a toy network" criticism | Physically grounded radio + queue model, oracle optimality gap, OSPF/SDN-Q baselines, 30-episode CIs, paired significance tests. |
| Tabular Q cannot scale to 60 nodes | Exactly why the LFA variant exists (E7) — the limitation becomes a reported finding, not a hidden flaw. |
| Results don't beat baselines | Then we report it honestly and explain the mechanism — the rubric rewards interpretation, and a negative result with an oracle comparison is still a rigorous one. |

---

## 11. Decisions I need from you

1. **Scope** — faithful reproduction only, or faithful + the rubric-driven extensions (POMDP belief
   layer, SARSA/MC, linear FA, oracle)? I recommend the full version: it directly answers rubric
   items 1–4 rather than just re-running the paper.
2. **Report format** — Markdown (renders in the viewer, easiest to iterate) vs. LaTeX (matches the
   department template used by the capstone report) vs. both.
3. **Headline environmental change for rubric #5** — arrival rate, link availability/failures, or all
   three.
