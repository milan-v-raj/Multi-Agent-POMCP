"""Summarise whatever corrected network results have been saved so far.

Run with DOS_RL_OUTPUT_BASE=<suite directory> python -m experiments.corrected_report.
Never reads or overwrites the historical results/. Missing stages are explicit.
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.common import OUTPUT_BASE, RESULTS

TABLES = [
    ('Expanded route-state comparison', 'full_state_comparison.csv', ['state_mode']),
    ('E0 learning and held-out tests', 'e0_summary.csv', ['agent']),
    ('E0b cold-start task', 'e0b_coldstart.csv', ['policy']),
    ('E2/E3 network benchmark', 'e2_e3_raw.csv', ['sweep', 'value', 'policy']),
    ('E2 paired comparisons', 'e2_e3_paired_tests.csv', ['sweep', 'value', 'policy']),
    ('E4 parameter sweep', 'e4_parameter_sweep.csv', ['sweep', 'value']),
    ('E5 arrival rate', 'e5a_arrival_rate.csv', ['sweep', 'value', 'policy']),
    ('E5 link outages', 'e5b_link_failure.csv', ['sweep', 'value', 'policy']),
    ('E5 relay outages', 'e5c_node_failure.csv', ['sweep', 'value', 'policy']),
    ('E5 repeated outages', 'e5d_repeated_outages.csv', ['sweep', 'value', 'policy']),
    ('E5 paired comparisons', 'e5_paired_tests.csv', ['sweep', 'value', 'comparison']),
    ('E6 ablations', 'e6_ablation.csv', ['ablation']),
    ('E6 deployment battery', 'e6_test_battery.csv', ['policy']),
    ('E6 paired statistics', 'e6_statistics.csv', ['policy_a', 'policy_b', 'metric']),
    ('E7 deployment generalisation', 'e7_generalisation.csv', ['family', 'policy']),
]


def number(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def table(headers, rows):
    def clean(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(clean, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(map(clean, row)) + ' |' for row in rows])


def main():
    root, results = Path(OUTPUT_BASE), Path(RESULTS)
    # Refuse to overwrite the historical report by accident.
    if root == Path(__file__).resolve().parents[1]:
        raise SystemExit('Set DOS_RL_OUTPUT_BASE to the separate corrected suite directory')
    marker = root / 'arq_model.txt'
    arq_fixed = marker.exists() and marker.read_text().strip() == 'truncated-geometric-v1'
    title = '# ARQ-fixed CMOMDP network results' if arq_fixed else '# Earlier CMOMDP network results'
    audit = (
        '> **ARQ model:** This run uses the corrected truncated-geometric expected '
        'attempt count. Other simulator and paper-fidelity limitations remain; '
        'see `../PAPER_FIDELITY.md`.' if arq_fixed else
        '> **Fidelity audit:** These older saved network runs have an '
        'expected-ARQ-attempt formula error. Energy, delay and potentially PDR '
        'and policy comparisons require corrected reruns. See `../PAPER_FIDELITY.md`.'
    )
    lines = [
        title, '', audit, '',
        'These tables use only CSVs in this output directory. Historical results '
        'from other directories must not be mixed with them. '
        'A missing section means that stage was not complete when this report '
        'was generated. Check `status.txt` and regenerate after the suite ends.', '',
        'The main agent uses a relay-only key, value-based DOS selection, and '
        'rescaled paper rewards. The separate full-snapshot experiment is a '
        'route-episode comparison, not a closed-loop policy. Compact states are '
        'not proven Markov. The simulator is not ns-3 and does not implement SDN-Q.', '',
        'E0 and E0b use 30 held-out route episodes per seed; E6 uses 30 '
        'deployment runs with a separate agent trained on each deployment, not '
        '30 unseen-to-the-agent topologies. E7 is a separate zero-shot comparison. '
        'Counts `n` below are CSV rows, generally seeds or deployments, not episodes. '
        'Statistics tables and E4 are already aggregated. Confidence interval '
        'columns averaged across seeds are not pooled confidence intervals.', '',
        '`energy_efficiency` is PDR fraction per watt, not the paper’s '
        'PDR divided by average relay energy. Delay is a flow-model estimate. '
        'Comparisons of shaped and unshaped episode returns have different reward '
        'definitions; use success and network metrics as well.', '',
        'Training figures, when generated: [E0](figures/e0_learning_curves.png), '
        '[E0b](figures/e0b_coldstart.png); E0 CSV values are 20-episode moving averages.', '',
    ]
    battery = results / 'e6_test_battery.csv'
    if battery.exists():
        with battery.open(newline='') as f:
            data = list(csv.DictReader(f))
        groups = defaultdict(list)
        for row in data:
            groups[row['policy']].append(row)
        paper_style = []
        for policy, group in groups.items():
            # E6 uses 30 relays. Compute the ratio per deployment before averaging.
            ratios = [float(row['pdr_pct']) / (float(row['total_energy_j']) / 30.0)
                      for row in group if float(row['total_energy_j']) > 0]
            paper_style.append([policy, len(group),
                                f"{statistics.mean(float(row['pdr_pct']) for row in group):.5g}",
                                f"{statistics.mean(float(row['total_energy_j']) for row in group):.5g}",
                                f"{statistics.mean(ratios):.5g}" if ratios else '—'])
        lines.extend(['## Candidate paper-style energy metric (E6)', '',
                      'Per deployment: `PDR_percent / (total_energy_j / 30 relays)`, '
                      'then averaged. This is a candidate because the paper’s '
                      'precise averaging and units are not fully specified; it is '
                      'not a direct ns-3 replication.', '',
                      table(['policy', 'deployments', 'PDR %', 'total energy J',
                             'candidate PDR % / mean relay J'], paper_style), ''])
    for title, filename, keys in TABLES:
        path = results / filename
        lines.extend([f'## {title}', ''])
        if not path.exists():
            lines.extend(['Not completed or saved yet.', ''])
            continue
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            lines.extend(['No rows.', ''])
            continue
        cols = [k for k in rows[0] if k not in keys and k != 'seed'
                and any(number(row.get(k)) is not None for row in rows)]
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in keys)].append(row)
        out = []
        for key, group in groups.items():
            values = []
            for col in cols:
                nums = [v for row in group if (v := number(row.get(col))) is not None]
                values.append(f'{statistics.mean(nums):.5g}' if nums else '—')
            out.append([*key, len(group), *values])
        lines.extend([f'Source: [{filename}](results/{filename}).', '',
                      table([*keys, 'n', *cols], out), ''])
    path = root / 'RESULTS.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
