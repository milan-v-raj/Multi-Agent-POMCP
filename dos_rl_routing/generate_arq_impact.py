#!/usr/bin/env python3
"""Compare completed historical and ARQ-fixed CSVs without modifying either run."""
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'corrected_suite' / 'results'
NEW = ROOT / 'arq_fixed_suite' / 'results'
if (ROOT / 'arq_fixed_suite' / 'status.txt').read_text().strip() != 'complete':
    raise SystemExit('ARQ-fixed suite is not complete; refusing a final comparison')

def read(directory, name):
    with (directory / name).open(newline='') as f:
        return list(csv.DictReader(f))

def avg(rows, column):
    return mean(float(r[column]) for r in rows)

def table(lines, filename, filters, label, policies):
    old, new = read(OLD, filename), read(NEW, filename)
    for name, criteria in filters:
        for policy in policies:
            key = dict(criteria, **{label: policy})
            a = [r for r in old if all(r[k] == v for k, v in key.items())]
            b = [r for r in new if all(r[k] == v for k, v in key.items())]
            a_seeds = {r['seed'] for r in a}
            b_seeds = {r['seed'] for r in b}
            if not a or a_seeds != b_seeds or len(a) != len(a_seeds) or len(b) != len(b_seeds):
                raise ValueError(f'Mismatched run keys for {filename}: {key}')
            def numbers(column):
                x, y = avg(a, column), avg(b, column)
                return f'{x:.2f} → {y:.2f} ({y-x:+.2f})'
            lines.append(f'| {name} | {policy} | {len(a)} | {numbers("pdr_pct")} | '
                         f'{numbers("e2e_delay_ms")} | {numbers("total_energy_j")} | '
                         f'{numbers("mean_attempts")} |')

lines = [
    '# Measured effect of the ARQ correction', '',
    'The earlier `corrected_suite/` used an incorrect attempt-count formula. '
    'The new [ARQ-fixed full results](arq_fixed_suite/RESULTS.md) use the '
    'truncated-geometric expectation. Historical CSVs are preserved. These '
    'are descriptive before/after means for matching deployment seeds, not '
    'confidence intervals or proof of policy superiority. Battery depletion '
    'and routes can change, so the effect need not be uniform. The suite '
    'source checksum and software versions are in `arq_fixed_suite/`.', '',
    'Each cell is **old → fixed (fixed minus old)**. PDR differences are in '
    'percentage points; delay is milliseconds; energy is total joules; '
    'attempts are the simulator-reported mean per link.', '',
    '## E2 offered load: selected operating points', '',
    'Five sources, 10 deployments per operating point. DOS and scalar are '
    'Q-learning policies. The complete sweep (including source and density '
    'changes) is in the full results.', '',
    '| Packets/s/source | Policy | Deployments | PDR % | Delay ms | Energy J | Attempts |',
    '|---|---|---:|---:|---:|---:|---:|'
]
table(lines, 'e2_e3_raw.csv', [(v, {'sweep':'load', 'value':v}) for v in ('10','60','80')], 'policy', ('dos_rl','scalar','shortest_path','ospf'))
lines += ['', '## E6 retrained deployment battery', '',
    'The RL agents are retrained on each deployment, not tested zero-shot. '
    '`full` is DOS Q-learning and `no_dos` is weighted-sum Q-learning.', '',
    '| Scenario | Policy | Deployments | PDR % | Delay ms | Energy J | Attempts |',
    '|---|---|---:|---:|---:|---:|---:|']
table(lines, 'e6_test_battery.csv', [('battery', {})], 'policy', ('full','no_dos','linear_fa','shortest_path','ospf','energy_aware_greedy'))
lines += ['', '## Remaining limitations', '',
    'The corrected attempt count does not make this an ns-3 reproduction. '
    'Retry-delay handling and source/delivery-weighted aggregation remain '
    'approximations; paper energy efficiency and the project’s PDR/W are '
    'different metrics. The default value-based DOS is not the paper’s '
    'confidence rule. There is no SDN-Q baseline; deployed Q values are '
    'frozen while routes can be replanned. See [the paper-fidelity audit]'
    '(PAPER_FIDELITY.md). E0/E0b and the separate full-state route experiment '
    'do not directly use the flow-level ARQ formula.', '']
(ROOT / 'ARQ_IMPACT.md').write_text('\n'.join(lines))
print('Wrote ARQ_IMPACT.md')
