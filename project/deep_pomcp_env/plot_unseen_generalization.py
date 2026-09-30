"""
Plotting script for Experiment A: Unseen Map Generalization Results.
Generates publication-quality figures:
1. unseen_generalization_metrics.png (4-panel bar chart)
2. unseen_map_montage.png (Visualization of procedural map layouts)
"""

import os
import sys
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from deep_pomcp_env.scenarios import ScenarioConfig, ScenarioGenerator

# Matplotlib styling for scientific papers
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

def plot_metrics(csv_path: str, output_img: str):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    policies = ['Reactive A*', 'Vanilla POMCP', 'Heuristic POMCP', 'Deep-POMCP (Ours)']
    # Map possible naming differences
    name_map = {'Deep-POMCP': 'Deep-POMCP (Ours)'}
    df['policy'] = df['policy'].replace(name_map)

    colors = ['#4A90E2', '#E67E22', '#F39C12', '#2ECC71']  # Blue, Orange, Amber, Emerald

    summary = []
    for p in policies:
        pdf = df[df['policy'] == p]
        if len(pdf) == 0:
            continue
        win_rate = pdf['captured'].mean() * 100.0
        wins_df = pdf[pdf['captured'] == True]
        mean_ttc = wins_df['steps'].mean() if len(wins_df) > 0 else 1500.0
        std_ttc = wins_df['steps'].std() if len(wins_df) > 0 else 0.0
        mean_walls = pdf['wall_hits'].mean()
        mean_lat = pdf['mean_latency_ms'].mean()
        summary.append({
            'policy': p,
            'win_rate': win_rate,
            'ttc_mean': mean_ttc,
            'ttc_std': std_ttc,
            'walls': mean_walls,
            'latency': mean_lat
        })

    sum_df = pd.DataFrame(summary)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Experiment A: Generalization across 20 Unseen Procedural Maps', fontweight='bold', y=0.98)

    # 1. Win Rate
    ax = axes[0, 0]
    bars = ax.bar(sum_df['policy'], sum_df['win_rate'], color=colors[:len(sum_df)], width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Capture Success Rate (Higher is Better)', fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')

    # 2. Time-to-Capture
    ax = axes[0, 1]
    bars = ax.bar(sum_df['policy'], sum_df['ttc_mean'], yerr=sum_df['ttc_std'], capsize=5,
                  color=colors[:len(sum_df)], width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Steps (Frames)', fontweight='bold')
    ax.set_title('Mean Time-to-Capture (Lower is Better)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 25, f"{h:.0f}", ha='center', va='bottom', fontweight='bold')

    # 3. Wall Collisions
    ax = axes[1, 0]
    bars = ax.bar(sum_df['policy'], sum_df['walls'], color=colors[:len(sum_df)], width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Wall Hits / Episode', fontweight='bold')
    ax.set_title('Collision Frequency (Lower is Better)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.3, f"{h:.1f}", ha='center', va='bottom', fontweight='bold')

    # 4. Latency
    ax = axes[1, 1]
    bars = ax.bar(sum_df['policy'], sum_df['latency'], color=colors[:len(sum_df)], width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Latency per Decision (ms)', fontweight='bold')
    ax.set_title('Real-Time Decision Latency (Lower is Better)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.1, f"{h:.2f}ms", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Metrics plot saved to: {output_img}")


def plot_map_montage(seeds: List[int], output_img: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Sample Unseen Procedural Obstacle Layouts (Zero-Shot Benchmark)', fontweight='bold', y=0.98)

    for idx, (ax, seed) in enumerate(zip(axes.flatten(), seeds)):
        preset_name = f"procedural_{seed}"
        config = ScenarioConfig(width=800, height=600, density_preset=preset_name)
        obstacles, h_spawns, e_spawn = ScenarioGenerator.generate(config, np.random.default_rng(seed))

        ax.set_xlim(0, 800)
        ax.set_ylim(0, 600)
        ax.set_aspect('equal')
        ax.set_title(f"Unseen Map #{idx + 1} (Seed {seed})", fontweight='bold')
        ax.set_facecolor('#1E1E24')

        # Draw Obstacles
        for obs in obstacles:
            rect = patches.Rectangle((obs.x, obs.y), obs.width, obs.height, linewidth=1, edgecolor='#A0A0A0', facecolor='#4A4E69')
            ax.add_patch(rect)

        # Draw Hunters Spawns
        for i, h_pos in enumerate(h_spawns):
            c = '#00D2FF' if i == 0 else '#3A7BD5'
            ax.scatter(h_pos[0], h_pos[1], color=c, s=120, edgecolors='white', linewidth=1.5, zorder=5, label=f"Hunter {i + 1}" if idx == 0 else "")

        # Draw Evader Spawn
        ax.scatter(e_spawn[0], e_spawn[1], color='#FF416C', s=140, edgecolors='white', linewidth=1.5, marker='*', zorder=5, label="Evader Target" if idx == 0 else "")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

    axes[0, 0].legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Map montage saved to: {output_img}")


if __name__ == "__main__":
    csv_file = os.path.join(parent_dir, "unseen_generalization_results.csv")
    metrics_img = os.path.join(parent_dir, "unseen_generalization_metrics.png")
    montage_img = os.path.join(parent_dir, "unseen_map_montage.png")

    plot_metrics(csv_file, metrics_img)
    plot_map_montage([2001, 2005, 2011, 2017], montage_img)
