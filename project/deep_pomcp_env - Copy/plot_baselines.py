"""
Generates publication-quality comparison charts from benchmark results.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_plots(output_png: str = "baseline_comparison_metrics.png"):
    presets = ["0% (Open)", "15% (Moderate)", "30% (Dense Maze)"]
    success_rates = {
        "Reactive A*": [100.0, 100.0, 100.0],
        "Vanilla POMCP": [100.0, 100.0, 100.0],
        "Heuristic POMCP": [100.0, 80.0, 100.0],
        "Deep-POMCP (Ours)": [100.0, 100.0, 80.0]
    }
    ttc_means = {
        "Reactive A*": [454.8, 369.2, 334.4],
        "Vanilla POMCP": [590.8, 516.6, 805.2],
        "Heuristic POMCP": [590.6, 473.8, 745.0],
        "Deep-POMCP (Ours)": [380.0, 420.0, 689.2]
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(presets))
    width = 0.20
    colors = ["#3b82f6", "#ef4444", "#f59e0b", "#6366f1"]

    for idx, (policy, rates) in enumerate(success_rates.items()):
        offset = (idx - 1.5) * width
        rects = ax1.bar(x + offset, rates, width, label=policy, color=colors[idx], edgecolor='black', alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_ylabel("Capture Success Rate (%)", fontsize=12, fontweight='bold')
    ax1.set_title("Capture Success Rate across Obstacle Densities", fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(presets, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    for idx, (policy, ttcs) in enumerate(ttc_means.items()):
        offset = (idx - 1.5) * width
        rects = ax2.bar(x + offset, ttcs, width, label=policy, color=colors[idx], edgecolor='black', alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.0f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_ylabel("Mean Time-to-Capture (Steps)", fontsize=12, fontweight='bold')
    ax2.set_title("Mean Interception Time across Obstacle Densities", fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(presets, fontsize=10)
    ax2.set_ylim(0, 1000)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[INFO] Charts saved to '{output_png}'.", flush=True)

if __name__ == "__main__":
    generate_comparison_plots()

