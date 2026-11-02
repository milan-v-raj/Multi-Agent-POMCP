# CMOMDP viva notes

> **ARQ correction:** The full network suite was rerun with the corrected
> expected attempt count. Earlier `corrected_suite/` network CSVs are preserved
> as historical; see the [measured impact](ARQ_IMPACT.md) and
> [fidelity audit](PAPER_FIDELITY.md).


**What is the problem?** The paper describes multiple SDN controllers sharing a network view; our
simulator uses one controller to route data through battery-powered relay nodes
while considering energy, congestion and reliability.

**What is the state?** The paper and main implementation use the current relay
as the Q-table key. This is a compact approximation: it omits other changing
network variables, so we cannot claim it is a complete Markov state. The full
route-snapshot experiment is separate.

**What are the action and rewards?** Choose an eligible next hop. The three
reward components represent energy difference, available buffer relative to
sender load, and packet reception ratio. The implemented formulas are rescaled
versions of the paper's expressions, not exactly the simpler alternatives
proposed in discussion. Potential shaping is included in corrected updates.

**What does DOS mean?** Dynamic Objective Selection. Our chosen default picks
the objective with the highest estimated Q-value; it is not a certainty-based
rule. The paper's confidence equation and selection pseudocode are ambiguous.

**Is this a POMDP?** The main model is a CMOMDP. The telemetry study is a
separate partial-observability extension, not a full POMDP solver.

**What did the corrected experiments show?**

| Experiment | Result | Interpretation |
|---|---|---|
| Cold-start learning | DOS training return −3.30 → 1.38; held-out success 82.0% | Learning evidence on a harder task; not proof DOS is superior. |
| Weighted-sum control | Held-out success 79.7% in the same cold-start experiment | Descriptive comparison, not a significant ranking. |
| Shaping ablation | PDR 84.14% with shaping vs 84.46% without | Small mean difference; no significance established here. |
| 40% transient link outage | DOS vs shortest-path post-onset PDR 74.12% vs 69.25%; p = 0.078 | No significant difference at the 0.05 level. |
| Full-state key | 0 / 1,095 evaluation decisions had a previously seen key | Exact snapshots do not generalise well with a tabular lookup. |

**Is DOS more energy-efficient?** In the ARQ-fixed deployment battery DOS has
higher mean PDR per watt than static OSPF on these deployments. This descriptive
comparison does not establish a general advantage, and PDR/W is not exactly
the paper's energy metric.

**What changed from the old implementation?** The old network baseline updated
only the energy component despite being called weighted-sum. Q-learning and
SARSA omitted shaping, and the linear update had defects. These are corrected;
the old results remain separately labelled. The corrected suite was rerun.

Full results and limits: [REPORT.md](REPORT.md),
[CORRECTED_CMOMDP.md](CORRECTED_CMOMDP.md), and
[arq_fixed_suite/RESULTS.md](arq_fixed_suite/RESULTS.md).
