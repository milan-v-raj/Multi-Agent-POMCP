"""
Plotting Script: Publication-Grade Multi-Panel Visualization of
Belief Sharing vs. Intention Sharing in Deep-POMCP Pursuit-Evasion.

Generates:
  `project/intention_sharing_benchmark.png`
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

csv_path = os.path.join(parent_dir, "intention_sharing_benchmark_results.csv")
output_png = os.path.join(parent_dir, "intention_sharing_benchmark.png")


def load_data(csv_file):
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "policy": r["policy"],
                "env_type": r["environment_type"],
                "preset": r["preset"],
                "win_rate": float(r["win_rate_pct"]),
                "mean_ttc": float(r["mean_ttc"]),
                "overlap_steps": float(r["mean_corridor_overlap_steps"]),
                "separation": float(r["mean_hunter_separation_px"]),
                "flank_angle": float(r["mean_flank_angle_deg"]),
                "clashes": int(r["mcts_clashes_prevented"]),
                "flank_bonuses": int(r["mcts_flank_bonuses"]),
            })
    return rows


def plot_benchmark():
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark first.")
        return

    data = load_data(csv_path)

    # Filter Dec-POMDP rows
    dec_pomdp_rows = [d for d in data if d["env_type"] == "Dec-POMDP (Pygame)"]
    mujoco_rows = [d for d in data if "MuJoCo" in d["env_type"]]

    policies_order = [
        "Tree Deep-POMCP (Belief Only)",
        "ET-DAG-Deep-POMCP (Belief Only)",
        "CD-ET-DAG-Deep-POMCP (Belief Only)",
        "IS-CD-ET-DAG-Deep-POMCP (Belief + Intention)"
    ]
    policy_labels = [
        "Tree Deep-POMCP\n(Belief Only)",
        "ET-DAG-Deep-POMCP\n(Belief Only)",
        "CD-ET-DAG-POMCP\n(Belief Only)",
        "IS-CD-ET-DAG-POMCP\n(Belief + Intention, Ours)"
    ]
    colors = ["#7f8c8d", "#3498db", "#e67e22", "#2ecc71"]

    presets = ["u_trap", "dense_maze", "bimodal_fork", "figure_8"]
    preset_labels = ["U-Trap", "Dense Maze", "Bimodal Fork", "Figure 8"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # -------------------------------------------------------------------------
    # Panel A: Diagram - Symmetric Greed Deadlock vs. Intention Flanking
    # -------------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")
    ax0.set_title("(A) Tactical Mechanism: Belief vs. Intention Sharing", fontsize=12, fontweight="bold", pad=10)

    # Sub-box 1: Belief Sharing Only (Symmetric Greed)
    rect1 = patches.FancyBboxPatch((0.4, 5.2), 9.2, 4.4, boxstyle="round,pad=0.2",
                                  facecolor="#fcf0f0", edgecolor="#e74c3c", linewidth=1.5)
    ax0.add_patch(rect1)
    ax0.text(0.6, 9.1, "1. Belief Sharing Only: 'Symmetric Greed' & Deadlock", fontsize=9.5, fontweight="bold", color="#c0392b")
    ax0.text(0.6, 8.4, "• Both agents receive identical posterior belief B(t) behind choke.\n• Agent 0 & 1 choose the same corridor -> Corridor Congestion!\n• Evader slips out opposite corridor untouched (Capture Failure).",
             fontsize=8.2, color="#2c3e50")

    # Hunters colliding in single corridor
    ax0.add_patch(patches.Circle((2.2, 6.2), 0.35, color="#2980b9", ec="black"))
    ax0.text(2.2, 6.2, "H0", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax0.add_patch(patches.Circle((3.2, 6.2), 0.35, color="#16a085", ec="black"))
    ax0.text(3.2, 6.2, "H1", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax0.annotate("", xy=(5.2, 6.2), xytext=(3.6, 6.2),
                 arrowprops=dict(arrowstyle="->", lw=2.5, color="#c0392b"))
    ax0.text(4.4, 6.6, "Conflict!", color="#c0392b", fontsize=8.5, fontweight="bold")

    # Obstacle divider
    ax0.add_patch(patches.Rectangle((5.8, 5.8), 0.8, 2.5, facecolor="#7f8c8d", edgecolor="black"))
    ax0.text(6.2, 7.0, "Wall", color="white", ha="center", va="center", fontsize=7.5, rotation=90)

    # Evader escaping
    ax0.add_patch(patches.Circle((8.2, 7.8), 0.35, color="#e74c3c", ec="black"))
    ax0.text(8.2, 7.8, "E", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    ax0.annotate("", xy=(9.2, 6.0), xytext=(8.2, 7.4),
                 arrowprops=dict(arrowstyle="->", lw=2.0, color="#e67e22", linestyle="--"))
    ax0.text(8.8, 6.8, "Escape", color="#d35400", fontsize=7.5, fontweight="bold")

    # Sub-box 2: Intention Sharing (Pincer Flanking)
    rect2 = patches.FancyBboxPatch((0.4, 0.4), 9.2, 4.4, boxstyle="round,pad=0.2",
                                  facecolor="#f0fbf4", edgecolor="#27ae60", linewidth=1.5)
    ax0.add_patch(rect2)
    ax0.text(0.6, 4.3, "2. Belief + Intention Sharing: Anti-Clustering & Pincer", fontsize=9.5, fontweight="bold", color="#1e8449")
    ax0.text(0.6, 3.6, "• Agent 0 commits to waypoint w0*; broadcasts intention I0.\n• Agent 1 MCTS repelled from corridor 0 & incentivized by pincer bonus.\n• Hunters seal both corridor exits simultaneously -> Guaranteed Capture!",
             fontsize=8.2, color="#2c3e50")

    # Hunters taking opposite corridors
    ax0.add_patch(patches.Circle((2.2, 2.4), 0.35, color="#2980b9", ec="black"))
    ax0.text(2.2, 2.4, "H0", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax0.add_patch(patches.Circle((2.2, 1.2), 0.35, color="#27ae60", ec="black"))
    ax0.text(2.2, 1.2, "H1", color="white", ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax0.annotate("", xy=(7.2, 2.6), xytext=(2.6, 2.4),
                 arrowprops=dict(arrowstyle="->", lw=2.2, color="#2980b9"))
    ax0.annotate("", xy=(7.2, 1.0), xytext=(2.6, 1.2),
                 arrowprops=dict(arrowstyle="->", lw=2.2, color="#27ae60"))

    # Wall divider
    ax0.add_patch(patches.Rectangle((4.8, 1.4), 1.6, 0.8, facecolor="#7f8c8d", edgecolor="black"))

    # Trapped evader
    ax0.add_patch(patches.Circle((7.8, 1.8), 0.35, color="#e74c3c", ec="black"))
    ax0.text(7.8, 1.8, "E", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    ax0.text(8.2, 2.4, "PINCER!", color="#1e8449", fontsize=8.5, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel B: Capture Win Rate (%) Comparison across Presets
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(presets))
    width = 0.20

    for i, pol in enumerate(policies_order):
        pol_data = [d for d in dec_pomdp_rows if d["policy"] == pol]
        wr_vals = [next((d["win_rate"] for d in pol_data if d["preset"] == pr), 0.0) for pr in presets]
        rects = ax1.bar(x + (i - 1.5) * width, wr_vals, width, label=policy_labels[i].replace("\n", " "),
                        color=colors[i], edgecolor="black", alpha=0.9)
        for r, val in zip(rects, wr_vals):
            if val > 0:
                ax1.text(r.get_x() + r.get_width() / 2.0, r.get_height() + 1.5, f"{val:.0f}%",
                         ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax1.set_ylabel("Capture Win Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("(B) Capture Win Rate vs. Deceptive Feinting Evader", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(preset_labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=7.5, loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # Panel C: Mean Time to Capture (TTC, steps)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    for i, pol in enumerate(policies_order):
        pol_data = [d for d in dec_pomdp_rows if d["policy"] == pol]
        ttc_vals = [next((d["mean_ttc"] for d in pol_data if d["preset"] == pr), 1000.0) for pr in presets]
        rects = ax2.bar(x + (i - 1.5) * width, ttc_vals, width, label=policy_labels[i].replace("\n", " "),
                        color=colors[i], edgecolor="black", alpha=0.9)
        for r, val in zip(rects, ttc_vals):
            ax2.text(r.get_x() + r.get_width() / 2.0, r.get_height() + 15.0, f"{val:.0f}",
                     ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax2.set_ylabel("Mean Time to Capture (steps)", fontsize=11, fontweight="bold")
    ax2.set_title("(C) Pursuit Efficiency (Lower TTC is Faster)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(preset_labels, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1150)

    # -------------------------------------------------------------------------
    # Panel D: Corridor Conflict / Clutter Reduction (Overlap Steps)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    for i, pol in enumerate(policies_order):
        pol_data = [d for d in dec_pomdp_rows if d["policy"] == pol]
        overlap_vals = [next((d["overlap_steps"] for d in pol_data if d["preset"] == pr), 0.0) for pr in presets]
        rects = ax3.bar(x + (i - 1.5) * width, overlap_vals, width, label=policy_labels[i].replace("\n", " "),
                        color=colors[i], edgecolor="black", alpha=0.9)
        for r, val in zip(rects, overlap_vals):
            ax3.text(r.get_x() + r.get_width() / 2.0, r.get_height() + 2.0, f"{val:.1f}",
                     ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax3.set_ylabel("Corridor Overlap Steps (< 50 px)", fontsize=11, fontweight="bold")
    ax3.set_title("(D) Corridor Clutter & Congestion Suppression", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(preset_labels, fontsize=10, fontweight="bold")
    max_ov = max(d["overlap_steps"] for d in dec_pomdp_rows) if dec_pomdp_rows else 100
    ax3.set_ylim(0, max_ov * 1.25 + 5)

    # -------------------------------------------------------------------------
    # Panel E: Encirclement Flanking Angle at Capture (Degrees)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    for i, pol in enumerate(policies_order):
        pol_data = [d for d in dec_pomdp_rows if d["policy"] == pol]
        flank_vals = [next((d["flank_angle"] for d in pol_data if d["preset"] == pr), 0.0) for pr in presets]
        rects = ax4.bar(x + (i - 1.5) * width, flank_vals, width, label=policy_labels[i].replace("\n", " "),
                        color=colors[i], edgecolor="black", alpha=0.9)
        for r, val in zip(rects, flank_vals):
            if val > 0:
                ax4.text(r.get_x() + r.get_width() / 2.0, r.get_height() + 2.0, f"{val:.0f}°",
                         ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax4.axhline(90.0, color="gray", linestyle="--", linewidth=1.2, label="Orthogonal (90°)")
    ax4.axhline(180.0, color="#8e44ad", linestyle=":", linewidth=1.2, label="Head-on Pincer (180°)")
    ax4.set_ylabel("Flanking Encirclement Angle (°)", fontsize=11, fontweight="bold")
    ax4.set_title("(E) Cooperative Pincer Geometry at Terminal Capture", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(preset_labels, fontsize=10, fontweight="bold")
    ax4.set_ylim(0, 205)
    ax4.legend(fontsize=7.5, loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # Panel F: MuJoCo Continuous Unicycle Transfer Validation
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    if mujoco_rows:
        m_policies = [
            "CD-ET-DAG-Deep-POMCP (MuJoCo, Belief Only)",
            "IS-CD-ET-DAG-Deep-POMCP (MuJoCo, Belief + Intention)"
        ]
        m_labels = ["CD-ET-DAG\n(Belief Only)", "IS-CD-ET-DAG\n(Belief + Intention, Ours)"]
        m_colors = ["#e67e22", "#2ecc71"]
        m_presets = ["u_trap", "bimodal_fork"]
        m_preset_lbls = ["U-Trap", "Bimodal Fork"]

        x_m = np.arange(len(m_presets))
        w_m = 0.32
        for i, m_pol in enumerate(m_policies):
            m_data = [d for d in mujoco_rows if d["policy"] == m_pol]
            wr_m = [next((d["win_rate"] for d in m_data if d["preset"] == pr), 0.0) for pr in m_presets]
            rects = ax5.bar(x_m + (i - 0.5) * w_m, wr_m, w_m, label=m_labels[i].replace("\n", " "),
                            color=m_colors[i], edgecolor="black", alpha=0.9)
            for r, val in zip(rects, wr_m):
                ax5.text(r.get_x() + r.get_width() / 2.0, r.get_height() + 2.0, f"{val:.0f}%",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax5.set_ylabel("Capture Win Rate (%)", fontsize=11, fontweight="bold")
        ax5.set_title("(F) High-Fidelity Physics Transfer (MuJoCo Unicycle)", fontsize=12, fontweight="bold")
        ax5.set_xticks(x_m)
        ax5.set_xticklabels(m_preset_lbls, fontsize=10, fontweight="bold")
        ax5.set_ylim(0, 115)
        ax5.legend(fontsize=8, loc="upper left", frameon=True)

    plt.suptitle("Multi-Agent Deep-POMCP Intention Sharing: Theoretical Proof & Empirical Benchmark",
                 fontsize=15, fontweight="bold", y=0.98)

    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"Publication-grade plot saved to: {output_png}")


if __name__ == "__main__":
    plot_benchmark()

