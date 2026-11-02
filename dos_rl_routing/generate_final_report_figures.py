#!/usr/bin/env python3
"""Make a legible report figure directly from the completed E2 raw CSV."""
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'arq_fixed_suite/results/e2_e3_raw.csv').open(newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['sweep'] == 'load']
loads = sorted({int(r['value']) for r in rows})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
for policy, label, color in [('dos_rl', 'DOS Q-learning', '#187fa3'),
                              ('shortest_path', 'Re-planned shortest path', '#bc6228')]:
    for ax, metric in zip(axes, ['pdr_pct', 'e2e_delay_ms']):
        values = []
        for load in loads:
            group = [r for r in rows if r['policy'] == policy and int(r['value']) == load]
            if len(group) != 10:
                raise ValueError(f'Unexpected E2 sample count: {load} {policy}: {len(group)}')
            values.append(mean(float(r[metric]) for r in group))
        ax.plot(loads, values, marker='o', label=label, color=color, linewidth=2, markersize=4)
for ax, ylab in zip(axes, ['Packet delivery ratio (%)', 'Estimated end-to-end delay (ms)']):
    ax.set_xlabel('Packets/s per source')
    ax.set_ylabel(ylab)
    ax.grid(alpha=.24)
    ax.set_xticks(loads)
axes[0].legend(fontsize=8, loc='best')
fig.suptitle('Offered-load experiment: five sources, 10 deployments per setting', fontsize=11)
out = root / 'report_assets/offered_load.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Wrote {out}')
