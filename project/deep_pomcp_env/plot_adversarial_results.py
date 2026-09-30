"""
Plotting script for Adversarial Evader Benchmark.
Generates adversarial_evader_benchmark.png.
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

def plot_adversarial_benchmark(csv_path: str, output_img: str):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    name_map = {'Deep-POMCP': 'Deep-POMCP (Ours)'}
    df['policy'] = df['policy'].replace(name_map)

    policies = ['Reactive A*', 'Vanilla POMCP', 'Heuristic POMCP', 'Deep-POMCP (Ours)']
    evader_types = ['Standard Raycast', 'Strategic Adversarial']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adversarial Benchmark: Performance Against Strategic Occlusion-Seeking Evader', fontweight='bold', y=0.98)

    x = np.arange(len(policies))
    width = 0.35

    # 1. Win Rate Comparison
    ax = axes[0, 0]
    std_wr = [df[(df['policy'] == p) & (df['evader_type'] == 'Standard Raycast')]['captured'].mean() * 100.0 for p in policies]
    adv_wr = [df[(df['policy'] == p) & (df['evader_type'] == 'Strategic Adversarial')]['captured'].mean() * 100.0 for p in policies]

    bars1 = ax.bar(x - width/2, std_wr, width, label='Standard Raycast', color='#4A90E2', edgecolor='black', linewidth=1.1)
    bars2 = ax.bar(x + width/2, adv_wr, width, label='Strategic Adversarial', color='#E74C3C', edgecolor='black', linewidth=1.1)

    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Capture Success Rate (Higher is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)

    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., h + 2, f"{h:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., h + 2, f"{h:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#922B21')

    # 2. Mean Time-to-Capture
    ax = axes[0, 1]
    std_ttc = [df[(df['policy'] == p) & (df['evader_type'] == 'Standard Raycast') & (df['captured'] == True)]['steps'].mean() for p in policies]
    adv_ttc = [df[(df['policy'] == p) & (df['evader_type'] == 'Strategic Adversarial') & (df['captured'] == True)]['steps'].mean() for p in policies]

    bars1 = ax.bar(x - width/2, std_ttc, width, label='Standard Raycast', color='#4A90E2', edgecolor='black', linewidth=1.1)
    bars2 = ax.bar(x + width/2, adv_ttc, width, label='Strategic Adversarial', color='#E74C3C', edgecolor='black', linewidth=1.1)

    ax.set_ylabel('Steps (Frames)', fontweight='bold')
    ax.set_title('Mean Time-to-Capture (Lower is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)

    # 3. Line-of-Sight Occlusion Breakdown (%)
    ax = axes[1, 0]
    std_los = [df[(df['policy'] == p) & (df['evader_type'] == 'Standard Raycast')]['los_break_pct'].mean() for p in policies]
    adv_los = [df[(df['policy'] == p) & (df['evader_type'] == 'Strategic Adversarial')]['los_break_pct'].mean() for p in policies]

    bars1 = ax.bar(x - width/2, std_los, width, label='Standard Raycast', color='#4A90E2', edgecolor='black', linewidth=1.1)
    bars2 = ax.bar(x + width/2, adv_los, width, label='Strategic Adversarial', color='#E74C3C', edgecolor='black', linewidth=1.1)

    ax.set_ylabel('Time Unobserved / Blind (%)', fontweight='bold')
    ax.set_title('Occlusion Evasion Rate (% Time Out of LOS)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)

    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. Wall Collisions / Episode
    ax = axes[1, 1]
    std_walls = [df[(df['policy'] == p) & (df['evader_type'] == 'Standard Raycast')]['wall_hits'].mean() for p in policies]
    adv_walls = [df[(df['policy'] == p) & (df['evader_type'] == 'Strategic Adversarial')]['wall_hits'].mean() for p in policies]

    bars1 = ax.bar(x - width/2, std_walls, width, label='Standard Raycast', color='#4A90E2', edgecolor='black', linewidth=1.1)
    bars2 = ax.bar(x + width/2, adv_walls, width, label='Strategic Adversarial', color='#E74C3C', edgecolor='black', linewidth=1.1)

    ax.set_ylabel('Wall Collisions / Episode', fontweight='bold')
    ax.set_title('Pursuer Safety & Wall Collisions', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Adversarial benchmark plot saved to: {output_img}")


if __name__ == "__main__":
    csv_file = os.path.join(parent_dir, "adversarial_benchmark_results.csv")
    out_img = os.path.join(parent_dir, "adversarial_evader_benchmark.png")
    plot_adversarial_benchmark(csv_file, out_img)
