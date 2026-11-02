"""Generate a comprehensive table of saved quantitative experiment results.

Uses the Python standard library only. Run:
    python -m experiments.quantitative_report
No experiments are rerun and source CSVs are never modified.
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUTPUT = ROOT / 'QUANTITATIVE_RESULTS.md'

# (label, CSV, grouping keys). Each result row is a group mean across saved
# seeds; pre-aggregated E4 and statistical tables remain as recorded.
TABLES = [
    ('E0 training and held-out tests', 'e0_summary.csv', ['agent']),
    ('E0b harder routing task', 'e0b_coldstart.csv', ['policy']),
    ('E1 gridworld', 'e1_gridworld.csv', ['variant']),
    ('E2/E3 network benchmark', 'e2_e3_raw.csv', ['sweep', 'value', 'policy']),
    ('E2 paired comparisons', 'e2_e3_paired_tests.csv', ['sweep', 'value', 'policy']),
    ('E4 parameter sweep', 'e4_parameter_sweep.csv', ['sweep', 'value']),
    ('E5 arrival-rate sweep', 'e5a_arrival_rate.csv', ['sweep', 'value', 'policy']),
    ('E5 link outages', 'e5b_link_failure.csv', ['sweep', 'value', 'policy']),
    ('E5 node outages', 'e5c_node_failure.csv', ['sweep', 'value', 'policy']),
    ('E5 repeated outages', 'e5d_repeated_outages.csv', ['sweep', 'value', 'policy']),
    ('E5 paired comparisons', 'e5_paired_tests.csv', ['sweep', 'value', 'comparison']),
    ('E6 ablations', 'e6_ablation.csv', ['ablation']),
    ('E6 deployment battery', 'e6_test_battery.csv', ['policy']),
    ('E6 paired statistics', 'e6_statistics.csv', ['policy_a', 'policy_b', 'metric']),
    ('E7 deployment generalisation', 'e7_generalisation.csv', ['family', 'policy']),
    ('E8 observation profiles', 'e8_partial_observability.csv', ['profile', 'policy']),
]


def numeric(value):
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) else None


def fmt(x):
    return '—' if x is None else f'{x:.5g}'


def table(headers, rows):
    def clean(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    out = ['| ' + ' | '.join(map(clean, headers)) + ' |',
           '| ' + ' | '.join('---' for _ in headers) + ' |']
    out.extend('| ' + ' | '.join(map(clean, row)) + ' |' for row in rows)
    return '\n'.join(out)


def main():
    lines = [
        '# Quantitative results and paper metric audit',
        '',
        '> **Fidelity audit:** The flow simulator has an incorrect expected ARQ '
        'attempt count; saved network energy and delay, and potentially PDR '
        'through battery depletion, require a fix and fresh reruns. See '
        '`PAPER_FIDELITY.md`.',
        '',
        'All numbers below come from the saved CSVs, not from new experiments. '
        'The study uses a custom flow-level simulator, not ns-3; it does not '
        'implement the paper’s SDN-Q baseline. Results are not a direct '
        'reproduction of the paper’s numbers. The E6 battery uses 30 new deployments '
        'relative to the ablation experiment, but RL agents are trained separately '
        'on each deployment before it is evaluated; it is not a zero-shot held-out '
        'topology test.',
        '',
        '> **Audit:** In the saved network experiments, labels `scalar`, '
        '`weighted_sum` and `no_dos` refer to a one-objective learner that '
        'updates the first (energy) reward component, not a weighted sum. '
        'Network Q-learning also omits shaping from its update, and the linear '
        'agent has TD-update defects. See '
        '`CMOMDP_COMPARISON.md`; these results need a corrected rerun.',
        '',
        '## Paper metric coverage',
        '',
        '| Paper metric | Coverage and limitation |',
        '| --- | --- |',
        '| Network PDR | Calculated as delivered/offered packets in the flow model. |',
        '| Network end-to-end delay | Calculated using analytical queueing and service/retry time; it is not the paper’s ns-3 packet-level measurement and does not explicitly include propagation delay. |',
        '| Network energy efficiency | Existing `energy_efficiency` is PDR as a fraction divided by total network average power (W), not the paper’s PDR divided by average relay energy. See the derived comparison below. |',
        '| Gridworld reward and episode length | Collected during training and held-out testing; training plots are in `figures/e1_gridworld_replication.png`. |',
        '| Gridworld energy and obstacle hits | Collected and plotted during training, and test values exist in memory but were not saved to the E1 CSV. `final_energy_used` is 5 minus final energy, i.e. net depletion, not cumulative energy consumed when recharging is possible. |',
        '',
        '### Paper-style energy-efficiency calculation (E6 only)',
        '',
        'Derived per deployment as `PDR_percent / (total_energy_j / 30)` and '
        'then averaged across the 30 deployments. The paper describes PDR '
        'divided by average relay energy; because its precise averaging and units '
        'are not fully specified, treat this as a candidate paper-style '
        'calculation rather than an exact reproduction. Existing EE is '
        '`(PDR_percent / 100) / (total_energy_j / 200 s)`, averaged per run.',
        '',
    ]
    with (RESULTS / 'e6_test_battery.csv').open(newline='') as f:
        battery = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in battery:
        grouped[row['policy']].append(row)
    out = []
    for name, rows in grouped.items():
        mean = lambda col: statistics.mean(float(r[col]) for r in rows)
        paper = statistics.mean(float(r['pdr_pct']) / (float(r['total_energy_j']) / 30) for r in rows)
        out.append([name, len(rows), fmt(mean('pdr_pct')), fmt(mean('e2e_delay_ms')),
                    fmt(mean('total_energy_j')), fmt(mean('total_energy_j') / 30),
                    fmt(mean('energy_efficiency')), fmt(paper), fmt(mean('fairness'))])
    lines += [table(['policy', 'deployments', 'PDR %', 'delay ms', 'total J', 'mean J/relay',
                     'PDR fraction/W', 'paper-style %/J', 'Jain fairness'], out), '',
              '## Complete saved-result summaries', '',
              'Each row below is the arithmetic mean of finite values over saved '
              'seed rows, except E4 and paired-statistics tables, which are already '
              'aggregated. `n` counts CSV rows, not necessarily independent episodes. '
              'Missing data are shown as —. Do not interpret averages of individual '
              'within-run standard deviations or confidence intervals as confidence '
              'intervals for the pooled mean. Exact per-seed values are in the linked '
              'CSV files.', '',
              'Network fields: `pdr_pct` is percent; `e2e_delay_ms` is ms; '
              '`energy_efficiency` is PDR fraction/W; `avg_power_w` is watts; '
              '`total_energy_j` is joules; `fairness` is Jain index of energy drain; '
              '`load_imbalance` is utilisation standard deviation divided by mean; '
              '`mean_hops`, `mean_attempts`, and `throughput_pps` are simulator '
              'summaries. E5 `pdr_post` averages from failure onset to run end, '
              'including the outage and recovery; `pdr_min` is the minimum in '
              'the outage interval. `recovery_s` follows the experiment’s criterion.', '',
              'E0’s per-seed, 20-episode moving-average training curves are in '
              '[`e0_curves_raw.csv`](results/e0_curves_raw.csv). E0b and E1 '
              'full per-episode data and E4 curves are not persisted; their '
              'summary results and available figures are shown here. E5 original '
              'time traces are not saved.', '']
    for title, filename, keys in TABLES:
        path = RESULTS / filename
        if not path.exists():
            lines += [f'### {title}', '', f'Missing source file: `{filename}`.', '']
            continue
        with path.open(newline='') as f:
            data = list(csv.DictReader(f))
        columns = [k for k in data[0] if k not in keys and k != 'seed' and
                   any(numeric(r.get(k)) is not None for r in data)] if data else []
        buckets = defaultdict(list)
        for row in data:
            buckets[tuple(row[k] for k in keys)].append(row)
        output = []
        for key, rows in buckets.items():
            values = []
            for col in columns:
                numbers = [v for row in rows if (v := numeric(row.get(col))) is not None]
                values.append(fmt(statistics.mean(numbers)) if numbers else '—')
            output.append([*key, len(rows), *values])
        lines += [f'### {title}', '', f'Source: [`{filename}`](results/{filename}).', '',
                  table([*keys, 'n', *columns], output), '']
    OUTPUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {OUTPUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
