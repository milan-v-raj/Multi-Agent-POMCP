# Paper fidelity audit: CMOMDP routing

**Verdict: paper-inspired implementation, not an exact reproduction.** This
review compares the current code and saved `corrected_suite/` runs with
Godfrey et al., *Sensors* 2023, 23, 8435,
[doi:10.3390/s23208435](https://doi.org/10.3390/s23208435). Page numbers below
refer to the 22-page paper supplied for the project. This audit originally
identified an ARQ issue; the code has since been fixed and the full network
suite was rerun separately in `arq_fixed_suite/`; its measured impact is in
[ARQ_IMPACT.md](ARQ_IMPACT.md).

## Resolved in code: retransmission-count error in earlier runs

The earlier `NetworkSimulator.evaluate()` computed expected attempts as
`(1 - p**R)/(1 - p)` for per-attempt success probability `p` and attempt limit
`R`. For independent attempts that stop after the first success, the expected
count is `1 + (1-p) + ... + (1-p)**(R-1)`. The difference is material:

| Per-attempt PRR | Attempt limit | Earlier simulator count | Fixed expected count |
|---:|---:|---:|---:|
| 0.1 | 3 | 1.11 | 2.71 |
| 0.5 | 3 | 1.75 | 1.75 |
| 0.9 | 3 | 2.71 | 1.11 |

At `p = 0.9`, the earlier implementation returned **2.71 attempts**, versus
**1.11** under the independent-attempt model. The current code now sums
`(1-p)**k` for `k = 0..R-1`; its relay and gateway calculations are covered by
passing low/high-PRR and multiple-attempt-limit tests in
`tests/test_paper_fidelity.py`. The separate hop-success formula was already
consistent with that model.

The count feeds transmission/reception energy, attempt statistics and
non-gateway retry delay. It can affect **PDR as well** through simulated battery
depletion and route eligibility. Therefore the `corrected_suite/` closed-loop
network energy, delay, PDR, outage and policy-comparison results remain **outputs
of the earlier flawed simulator**, not results of the fixed model. The
ARQ-fixed suite is complete and stored separately. E0/E0b and
the full-state comparison are route-episode experiments, not the flow-level
network evaluation that uses this formula; the ARQ-count error does not directly
change those saved route-episode results. Their route model has its own
approximations, including charging a single hop cost rather than sampling
per-attempt energy.

## Core framework and algorithm

| Paper reference | Paper | Current implementation | Assessment |
|---|---|---|---|
| §3.2.2, pp. 9–10 | Multiple distributed network controllers share a centralized network view; topology discovery and flow installation are described. | One simulated controller and routing table; no distributed control plane, OpenFlow messaging, or controller overhead simulation. | Partial architectural approximation. |
| §3.2.3, p. 12 | Each state corresponds to a relay node in the topology. | Default Q-table key `(node,)`; rewards and eligible actions depend on additional network variables and history. | Matches the stated key, but is not proven Markov in this simulator. The expanded snapshot remains separate. |
| §3.2.3–3.2.4, pp. 12–13 | Eligible next-hop actions and STP-derived loop-free routes. | Neighbours filtered by link, energy and BFS hop count; geographic fallback; direct-gateway shortcut. | Similar intent, not STP; loop-freedom is not guaranteed for every fallback/mode. |
| Eq. 4, p. 11 | `E_j − E_i` as printed; the explanatory node labels are reversed in the paper. It describes energy after transmission. | Default `0.5 * (1 + e_j − e_i)`, using normalized snapshot energies **before** the hop. | Rescaled and timed differently; not verbatim. The optional hop-cost mode is an alternative, not Eq. 4. |
| Eq. 5, p. 11 | Available buffer at `j` divided by maximum minus sender `i` load fraction. | `0.5 * (1 + available_buffer_fraction_j - load_i)`. | Same structure, affine rescaling; load and queue are simulator estimates. |
| Eq. 6, p. 11 | EWMA of sampled PRR. | Route-episode training uses copied distance-model PRR; deployment telemetry smooths PRR via the observer/belief tracker. Gateway PRR is directly modelled. | PRR objective followed in spirit, not packet-sampled EWMA throughout. |
| Eq. 2–3 and Algorithm 1, pp. 8–9 | EWMA of squared TD error and a confidence formula; prose says most confidence, pseudocode selects `argmin`. | Tracks squared TD error; default `dos_select="value"` picks the objective with the largest available Q-value. Optional `argmin` implements one interpretation using code-defined Q/variance summaries. | Default is **not** the paper's confidence rule; paper does not completely resolve the selection semantics. |
| Eqs. 7–9, p. 12 | Per-objective Q-learning with potential-based shaping. | Corrected network tabular updates receive vector rewards and shaping; terminal potential is set to zero. | Broadly followed; reward scales, episode dynamics, terminal handling and default selection differ. |
| Eq. 10, pp. 12–13 | As printed, exploit with probability ε and explore otherwise. | `NET_RL`: standard convention, explore with probability ε; `epsilon_convention="paper"` is available but not the default run. | Different default convention. The paper's network table does not specify ε; 0.3 is listed for its gridworld. |
| Algorithm 2, pp. 13–14 | Q-routing over source–destination pairs, then route installation and updates. | Trains on sampled synthetic route-episode snapshots for a single gateway; deployment recomputes choices but defaults to `online=False`, so Q-values are frozen during the reported network evaluation. | Partial. Dynamic action masks/replanning do not imply live RL learning. |

## Settings and coverage

| Item | Paper | Current project |
|---|---|---|
| Field, relay counts, packet size, nominal duration | 1,500 × 1,500 m, 30/45/60 relays, 512 bits, 200 s (§4.3, Table 2). | Same nominal values in defaults and/or E2/E3 sweeps. |
| Initial energy | Same 5 J for all relays (§4.3, Table 2). | Default deployment samples 95–100% of 5 J; route training also samples synthetic energy profiles. Nominal initial energy is 5 J, but nodes do not all start at 5 J. |
| Source count and rate | 2/4/6/8/10 sources; 10 packets/s; density figures use 5 sources (§4.3–4.4). | E2b sweeps those counts at 10 packets/s; E3 uses 5 sources. E2a is an extra offered-load sweep using 5 sources. E0 uses 10 sources. |
| Figure-label correction | E2a and E3 were previously captioned as 10 sources. | They used 5: captions were reconstructed from saved CSVs with correct labels, no experiment rerun. CSV numerical values were not modified. |
| Route lifetime | The paper illustrates a 3 s route entry lifetime (p. 10); it is an example, not clearly a mandatory experimental setting. | The simulator defaults to 3 s, but `NetworkSimulator.run()` uses its own argument default; changing `SimConfig.route_lifetime_s` alone does not change the actual run. |
| Energy threshold, radio, queues, traffic | Paper gives a threshold concept and some network settings but does not specify every value in Table 2. | Project selects a 10% threshold, 450 m range, custom log-distance radio/PRR, analytical M/M/1/K queues, and a near-sink traffic multiplier. These are modelling assumptions, not all paper parameters. Connectivity is repaired after random deployment. |
| Baselines | OSPF and SDN-Q for network experiments (§4.2, p. 15); weighted-sum RL also used in gridworld (§4.2, p. 16). | Static OSPF-like routing, adaptive BFS and a corrected weighted-sum Q control; **no SDN-Q**. These are not identical to the paper's baseline set. |
| Experimental settings | Paper reports 10 runs per network scenario (§4.3, p. 17). | E2/E3 use 10 seeds; E4/E5/E6 and route-episode tests have different sample counts as recorded in the tables. Extra load, failure, state and shaping studies are project extensions. |

## Paper metric coverage

| Paper metric | Project coverage | Important difference |
|---|---|---|
| Network PDR (§4.4, p. 18) | Flow model computes estimated delivered/offered traffic, saved as `pdr_pct`. | Analytic estimate, not ns-3 packet counts; the earlier ARQ-count error could indirectly alter it via battery depletion. |
| End-to-end delay (§4.4, p. 18) | Flow-model estimate `e2e_delay_ms`. | It uses analytic queue/service/retry delays, not packet-level measurements or explicit propagation delay. It averages source path delays rather than calculating exact delivery-weighted packet delay; earlier saved runs had incorrect retry counts. |
| Energy efficiency (§4.4, pp. 18–19) | `energy_efficiency` is PDR fraction divided by total average power (W). A per-run `PDR_percent / (total_energy_j / 30)` candidate is reported for E6. | PDR/W is not the paper's PDR divided by average relay energy. Candidate denominator/averaging is an approximation to the paper's description and inherits the retransmission-count error in the earlier saved runs. |
| Gridworld reward, energy, length, obstacle hits (§4.2, p. 16) | Historical E1 plots/tables exist; the core network rerun did not rerun E1. | Historical gridworld shaping and scalar baseline have known update issues; energy is net depletion, not necessarily cumulative consumption. These figures cannot validate paper fidelity. |

Additional fields such as fairness, throughput and outage recovery are project
extensions, not replacements for the paper's metrics. The old `results/` and
`corrected_suite/` remain unchanged and still contain the ARQ-count issue; the
fixed suite is separate. Even after that fix, do not claim exact paper reproduction
or paper-equivalent energy efficiency. Retry-delay handling and unequal-rate/source
weighting remain separate limitations.
