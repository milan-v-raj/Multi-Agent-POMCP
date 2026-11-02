"""Recreate summary-only plots from saved CSVs without rerunning experiments.

The E4 raw learning curves and E5 time traces were not saved, so these cannot
be reconstructed from their aggregate CSV files. The E4 plot here explicitly
shows only the persisted summary. Run ``python -m experiments.plot_saved_results``.
"""
import csv
import numpy as np
from experiments.common import RESULTS, fig_path, setup_mpl
from pathlib import Path


def read(name):
    with (Path(RESULTS) / name).open(newline='') as f:
        return list(csv.DictReader(f))


def summary_plot(filename, group, series, metrics, output, title):
    rows = read(filename)
    groups = list(dict.fromkeys(r[group] for r in rows))
    labels = ['all'] if group == series else list(dict.fromkeys(r[series] for r in rows))
    plt = setup_mpl()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        width = 0.8 / len(labels)
        for k, label in enumerate(labels):
            mean, err = [], []
            for g in groups:
                a = np.array([float(r[metric]) for r in rows if r[group] == g and (group == series or r[series] == label)])
                a = a[np.isfinite(a)]
                mean.append(a.mean() if len(a) else np.nan)
                err.append(1.96*a.std(ddof=1)/np.sqrt(len(a)) if len(a) > 1 else 0.0)
            ax.bar(np.arange(len(groups)) + (k-(len(labels)-1)/2)*width, mean, width,
                   yerr=err, label=label, error_kw={'lw': .7})
        ax.set_xticks(range(len(groups)), groups, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(metric)
    if group != series:
        axes[0, 0].legend(fontsize=6)
    fig.suptitle(title + ' — reconstructed from saved CSV')
    fig.savefig(fig_path(output), bbox_inches='tight')
    plt.close(fig)


def main():
    summary_plot('e6_ablation.csv', 'ablation', 'ablation', ['pdr_pct', 'energy_efficiency', 'fairness'], 'e6_ablation.png', 'E6 ablation')
    summary_plot('e6_test_battery.csv', 'policy', 'policy', ['pdr_pct', 'energy_efficiency'], 'e6_test_battery.png', 'E6 test battery')
    summary_plot('e7_generalisation.csv', 'family', 'policy', ['pdr_pct', 'energy_efficiency', 'mean_hops'], 'e7_generalisation.png', 'E7 transfer')
    summary_plot('e8_partial_observability.csv', 'profile', 'policy', ['pdr_pct', 'energy_efficiency', 'load_imbalance'], 'e8_partial_observability.png', 'E8 telemetry')
    rows = read('e4_parameter_sweep.csv')
    plt = setup_mpl()
    sweeps = list(dict.fromkeys(r['sweep'] for r in rows))
    fig, axes = plt.subplots(1, len(sweeps), figsize=(4*len(sweeps), 3.5), squeeze=False)
    for ax, sweep in zip(axes[0], sweeps):
        r = [x for x in rows if x['sweep'] == sweep]
        x = np.array([float(a['value']) for a in r])
        y = np.array([float(a['pdr_delta_vs_baseline_pp']) for a in r])
        err = np.array([float(a['pdr_delta_ci95']) for a in r])
        ax.errorbar(x, y, yerr=err, marker='o', capsize=2)
        ax.axhline(0, color='black', ls='--', lw=.7)
        ax.set_xlabel(sweep)
        ax.set_ylabel('PDR gain vs baseline (pp)')
    fig.suptitle('E4: saved summary only (learning curves not saved)')
    fig.savefig(fig_path('e4_parameter_sweep.png'), bbox_inches='tight')
    plt.close(fig)
    print('Recreated E4 and E6–E8 summary figures from saved CSVs.')


if __name__ == '__main__':
    main()
