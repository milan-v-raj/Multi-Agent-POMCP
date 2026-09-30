"""
Plotting script for Formal Bimodal KL Divergence & Action Selection Error.
Generates bimodal_kl_analysis.png.
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

def plot_bimodal_analysis(csv_path: str, output_img: str):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    name_map = {'Deep-POMCP': 'Deep-POMCP (Ours)'}
    df['policy'] = df['policy'].replace(name_map)

    policies = ['Reactive A*', 'Vanilla POMCP', 'Heuristic POMCP', 'Deep-POMCP (Ours)']
    colors = ['#4A90E2', '#E67E22', '#F39C12', '#2ECC71']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Formal Analysis of Multimodal Belief Representation & Action Selection Error', fontweight='bold', y=1.02)

    # 1. KL Divergence Comparison (Gaussian vs GMM-2)
    ax = axes[0]
    kl_g = df['kl_gaussian'].dropna()
    kl_gmm = df['kl_gmm2'].dropna()
    
    bplot = ax.boxplot([kl_g, kl_gmm], tick_labels=['Gaussian $\\mathcal{N}(\\mu, \\Sigma)$', 'Multi-Modal (GMM-2)'],
                       patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    bplot['boxes'][0].set_facecolor('#E74C3C') # Red for high loss
    bplot['boxes'][1].set_facecolor('#2ECC71') # Green for low loss

    mean_g = kl_g.mean()
    mean_gmm = kl_gmm.mean()
    ax.text(1, mean_g + 0.05, f"$\\mu = {mean_g:.2f}$ nats", ha='center', fontweight='bold', color='#922B21')
    ax.text(2, mean_gmm + 0.05, f"$\\mu = {mean_gmm:.2f}$ nats", ha='center', fontweight='bold', color='#196F3D')

    ax.set_ylabel('KL Divergence $D_{KL}(P_{true} \\parallel Q)$ (nats)', fontweight='bold')
    ax.set_title('Information Loss at Bifurcation (Lower is Better)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Action Selection Error Rate (% Steering into Wall)
    ax = axes[1]
    err_rates = []
    for p in policies:
        pdf = df[df['policy'] == p]
        tot = len(pdf)
        err = pdf['action_steers_into_wall'].sum() if tot > 0 else 0
        rate = (err / tot * 100.0) if tot > 0 else 0.0
        err_rates.append(rate)

    bars = ax.bar(policies, err_rates, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
    ax.set_ylabel('Action Error Rate (%)', fontweight='bold')
    ax.set_title('Steering Error into Solid Barrier (%)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 2, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')

    # 3. Spatial Bifurcation Schematic
    ax = axes[2]
    ax.set_xlim(150, 650)
    ax.set_ylim(100, 500)
    ax.set_aspect('equal')
    ax.set_title('Spatial Belief Bifurcation Anatomy', fontweight='bold')
    ax.set_facecolor('#1E1E24')

    # Central Barrier Wall
    wall_rect = patches.Rectangle((240, 260), 320, 80, linewidth=1.5, edgecolor='#E74C3C', facecolor='#78281F')
    ax.add_patch(wall_rect)
    ax.text(400, 300, 'SOLID WALL (Mean Collapse Trap)', color='white', ha='center', va='center', fontsize=9, fontweight='bold')

    # Particle Mode 1 (Top Corridor)
    top_x = np.random.normal(380, 40, 50)
    top_y = np.random.normal(190, 25, 50)
    ax.scatter(top_x, top_y, color='#00FF88', s=15, alpha=0.7, label='Mode 1 (Top Passage)')

    # Particle Mode 2 (Bottom Corridor)
    bot_x = np.random.normal(380, 40, 50)
    bot_y = np.random.normal(410, 25, 50)
    ax.scatter(bot_x, bot_y, color='#00FF88', s=15, alpha=0.7, label='Mode 2 (Bottom Passage)')

    # Gaussian Mean Point (Inside Wall)
    ax.scatter(380, 300, color='#FFCC00', s=120, edgecolors='black', linewidth=1.5, marker='X', zorder=10, label='Gaussian Mean $\\mu$ (Inside Wall!)')

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Bimodal analysis plot saved to: {output_img}")


if __name__ == "__main__":
    csv_file = os.path.join(parent_dir, "bimodal_kl_results.csv")
    out_img = os.path.join(parent_dir, "bimodal_kl_analysis.png")
    plot_bimodal_analysis(csv_file, out_img)
