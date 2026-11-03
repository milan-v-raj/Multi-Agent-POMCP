"""Summarise saved experiment CSV files into a Markdown results report."""
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
    title = '# Network experiment results'
    audit = ('Expected ARQ attempts use the truncated-geometric formula.' if arq_fixed
             else 'ARQ model marker missing: verify the run configuration before interpreting results.')
    lines = [
        title, '', audit, '',
        'These tables use CSVs in this output directory. A missing section means '
        'that its experiment has not completed.', '',
        'The relay-only state is not proven Markov. Route tests use a separate '
        'random stream on the same training topology. E6 retrains agents on '
        'each evaluated deployment; E7 tests transfer without retraining.', '',
        'PDR and delay are analytical estimates. Energy efficiency is the PDR '
        'fraction divided by average network power. Shaped and unshaped episode '
        'returns have different definitions.', '',
    ]
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
