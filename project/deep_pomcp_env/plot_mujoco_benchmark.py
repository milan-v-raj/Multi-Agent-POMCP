"""
Publication-Grade Visualizer for MuJoCo Continuous Multi-Body Physics Benchmark.

Generates `project/mujoco_continuous_physics_benchmark.png` featuring:
  Panel A: 3D Photorealistic MuJoCo Simulation Environment (Perspective Raycasted View)
  Panel B: Capture Win Rate (%) in Continuous Physics Dynamics
  Panel C: Mean Time-to-Capture (TTC) & Wall Collision Impedance
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def generate_plots():
    csv_path = os.path.join(parent_dir, "mujoco_benchmark_results.csv")
    render_path = os.path.join(parent_dir, "mujoco_render_u_trap_perspective.png")
    if not os.path.exists(render_path):
        render_path = os.path.join(parent_dir, "mujoco_render_dense_maze_perspective.png")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    # Parse CSV
    policy_stats = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pol = row["policy"]
            if pol not in policy_stats:
                policy_stats[pol] = {"wins": [], "ttc": [], "walls": [], "latencies": []}
            policy_stats[pol]["wins"].append(int(row["captured"]))
            policy_stats[pol]["ttc"].append(float(row["ttc"]))
            policy_stats[pol]["walls"].append(float(row["wall_contacts"]))
            policy_stats[pol]["latencies"].append(float(row["mean_latency_ms"]))

    # Policies in order
    pol_names = [
        "Reactive A*",
        "Tree Deep-POMCP (N=20)",
        "ET-DAG-Deep-POMCP (N=20)",
        "CD-ET-DAG-Deep-POMCP (N=20)"
    ]
    short_labels = [
        "Reactive A*",
        "Tree POMCP\n(N=20)",
        "ET-DAG\n(Unimodal)",
        "CD-ET-DAG\n(Counter-Deceptive)"
    ]

    win_rates = [np.mean(policy_stats[p]["wins"]) * 100.0 for p in pol_names]
    mean_ttcs = [np.mean(policy_stats[p]["ttc"]) for p in pol_names]
    mean_walls = [np.mean(policy_stats[p]["walls"]) for p in pol_names]
    mean_lats = [np.mean(policy_stats[p]["latencies"]) for p in pol_names]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "figure.dpi": 300
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2))

    # -------------------------------------------------------------------------
    # Panel A: MuJoCo 3D Physics Engine Simulation Rendering
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    if os.path.exists(render_path):
        img = Image.open(render_path)
        ax1.imshow(img)
        ax1.set_title("Panel A: Continuous MuJoCo 3D Physics Simulation\n(mj_ray Optical LOS + Rigid Contact Dynamics)", color="#111827", pad=10)
        ax1.axis("off")
        # Overlay indicator tags
        ax1.text(0.04, 0.94, "Arena: 8.0m x 6.0m (Continuous)", transform=ax1.transAxes,
                 fontsize=9, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75, edgecolor="none"))
        ax1.text(0.04, 0.06, "Blue: Pursuer Hunters (m=1.0kg)\nRed: Deceptive Evader (m=0.85kg)", transform=ax1.transAxes,
                 fontsize=8.5, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", alpha=0.85, edgecolor="#0ea5e9"))
    else:
        ax1.text(0.5, 0.5, "Render image not found", ha="center", va="center")
        ax1.set_title("Panel A: MuJoCo 3D Simulation")

    # -------------------------------------------------------------------------
    # Panel B: Capture Win Rate Comparison in MuJoCo Continuous Dynamics
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    colors = ["#64748b", "#0284c7", "#3b82f6", "#10b981"]
    bars = ax2.bar(short_labels, win_rates, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)
    ax2.set_ylabel("Capture Win Rate (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Panel B: MuJoCo Capture Win Rate (%)\n(vs. Deceptive Feinting Evader)", color="#111827", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.45, axis="y")

    for bar, val in zip(bars, win_rates):
        y_val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, y_val + 2.5, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                 color="#0f172a" if val < 80 else "#047857")

    # -------------------------------------------------------------------------
    # Panel C: Mean Time-to-Capture (TTC) & Wall Collision Contacts
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    x_pos = np.arange(len(short_labels))
    width = 0.35

    bar_ttc = ax3.bar(x_pos - width/2, mean_ttcs, width, label="Mean TTC (steps)", color="#38bdf8", edgecolor="#0284c7", linewidth=1.1)
    ax3_twin = ax3.twinx()
    bar_wall = ax3_twin.bar(x_pos + width/2, mean_walls, width, label="Wall Collisions", color="#f87171", edgecolor="#dc2626", linewidth=1.1)

    ax3.set_ylabel("Mean Time-to-Capture (Steps, lower is better)", color="#0369a1")
    ax3_twin.set_ylabel("Mean Wall Collision Impacts (lower is better)", color="#b91c1c")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(short_labels, fontsize=9.5)
    ax3.set_ylim(0, 850)
    ax3_twin.set_ylim(0, 8.0)
    ax3.set_title("Panel C: Time-to-Capture & Collision Safety\n(Continuous Multi-Body Contact Dynamics)", color="#111827", pad=10)
    ax3.grid(True, linestyle="--", alpha=0.35, axis="y")

    # Combined legend
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    output_png = os.path.join(parent_dir, "mujoco_continuous_physics_benchmark.png")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Generated MuJoCo Benchmark Plot] -> {output_png}", flush=True)


if __name__ == "__main__":
    generate_plots()

