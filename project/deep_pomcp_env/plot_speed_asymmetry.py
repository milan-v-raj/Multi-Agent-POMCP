"""
Publication Plotting Script for Speed Asymmetry Benchmark (1.0x to 2.0x).
Generates speed_asymmetry_curves.png.
"""

import os
import sys
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

def plot_speed_asymmetry(csv_path: str, output_img: str):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    name_map = {'Deep-POMCP': 'Deep-POMCP (Ours)'}
    df['policy'] = df['policy'].replace(name_map)

    policies = ['Reactive A*', 'Vanilla POMCP', 'Heuristic POMCP', 'Deep-POMCP (Ours)']
    colors = {
        'Reactive A*': '#4A90E2',
        'Vanilla POMCP': '#E67E22',
        'Heuristic POMCP': '#F39C12',
        'Deep-POMCP (Ours)': '#2ECC71'
    }
    markers = {
        'Reactive A*': 's',
        'Vanilla POMCP': '^',
        'Heuristic POMCP': 'd',
        'Deep-POMCP (Ours)': 'o'
    }

    speeds = sorted(df['speed_mult'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Speed Asymmetry Benchmark: Resilience Against Faster Evaders ($v_{evader} = 1.0\\times$ to $2.0\\times$)', fontweight='bold', y=1.02)

    # 1. Win Rate vs Speed
    ax = axes[0]
    for p in policies:
        pdf = df[df['policy'] == p]
        if len(pdf) == 0: continue
        win_rates = []
        for s in speeds:
            sub = pdf[pdf['speed_mult'] == s]
            wr = sub['captured'].mean() * 100.0 if len(sub) > 0 else 0.0
            win_rates.append(wr)
        ax.plot(speeds, win_rates, marker=markers[p], color=colors[p], linewidth=2.5, markersize=8, label=p)

    ax.set_xlabel('Evader Speed Multiplier ($v_{evader} / v_{hunter}$)', fontweight='bold')
    ax.set_ylabel('Capture Success Rate (%)', fontweight='bold')
    ax.set_title('Capture Rate vs. Evader Speed', fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', framealpha=0.9)

    # 2. Time-to-Capture vs Speed
    ax = axes[1]
    for p in policies:
        pdf = df[df['policy'] == p]
        if len(pdf) == 0: continue
        ttc_means = []
        for s in speeds:
            sub = pdf[(pdf['speed_mult'] == s) & (pdf['captured'] == True)]
            mean_t = sub['steps'].mean() if len(sub) > 0 else 1500.0
            ttc_means.append(mean_t)
        ax.plot(speeds, ttc_means, marker=markers[p], color=colors[p], linewidth=2.5, markersize=8, label=p)

    ax.set_xlabel('Evader Speed Multiplier ($v_{evader} / v_{hunter}$)', fontweight='bold')
    ax.set_ylabel('Mean Time-to-Capture (Steps)', fontweight='bold')
    ax.set_title('Mean Time-to-Capture (Lower is Better)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 3. Wall Collisions vs Speed
    ax = axes[2]
    for p in policies:
        pdf = df[df['policy'] == p]
        if len(pdf) == 0: continue
        wall_means = []
        for s in speeds:
            sub = pdf[pdf['speed_mult'] == s]
            mean_w = sub['wall_hits'].mean() if len(sub) > 0 else 0.0
            wall_means.append(mean_w)
        ax.plot(speeds, wall_means, marker=markers[p], color=colors[p], linewidth=2.5, markersize=8, label=p)

    ax.set_xlabel('Evader Speed Multiplier ($v_{evader} / v_{hunter}$)', fontweight='bold')
    ax.set_ylabel('Wall Collisions / Episode', fontweight='bold')
    ax.set_title('Collision Frequency vs. Evader Speed', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Speed asymmetry curve saved to: {output_img}")


if __name__ == "__main__":
    csv_file = os.path.join(parent_dir, "speed_asymmetry_results.csv")
    out_img = os.path.join(parent_dir, "speed_asymmetry_curves.png")
    plot_speed_asymmetry(csv_file, out_img)
