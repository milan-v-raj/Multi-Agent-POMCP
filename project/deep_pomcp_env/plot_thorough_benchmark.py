"""
Plotting Script: Publication-Grade Multi-Panel Visualization of the
Statistically Rigorous Multi-Agent Pursuit-Evasion Benchmark.

Generates:
  `project/thorough_benchmark_results.png`
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

summary_csv = os.path.join(parent_dir, "thorough_benchmark_results.csv")
raw_csv = os.path.join(parent_dir, "thorough_benchmark_raw_episodes.csv")
output_png = os.path.join(parent_dir, "thorough_benchmark_results.png")


def load_summary():
    rows = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "condition": r["condition"],
                "policy": r["policy"],
                "preset": r["preset"],
                "episodes": int(r["episodes"]),
                "wins": int(r["wins"]),
                "win_rate": float(r["win_rate_pct"]),
                "ci_low": float(r["ci_95_low"]),
                "ci_high": float(r["ci_95_high"]),
                "mean_ttc": float(r["mean_ttc"]),
                "overlap_steps": float(r["mean_corridor_overlap_steps"]),
                "flank_angle": float(r["mean_flanking_angle_deg"]),
            })
    return rows


def plot_thorough_benchmark():
    if not os.path.exists(summary_csv):
        print(f"Error: {summary_csv} not found.")
        return

    data = load_summary()

    presets = ["u_trap", "dense_maze", "bimodal_fork", "figure_8"]
    preset_labels = ["U-Trap", "Dense Maze", "Bimodal Fork", "Figure 8"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(19, 12), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

    col_belief = "#e67e22"      # Orange for Belief Sharing Only
    col_intention = "#27ae60"   # Green for Belief + Intention Sharing

    x = np.arange(len(presets))
    w = 0.35

    # -------------------------------------------------------------------------
    # Panel 1: Standard Dec-POMDP Win Rate with 95% Wilson CI Error Bars
    # -------------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    std_rows = [d for d in data if d["condition"] == "Standard Dec-POMDP"]

    b_std = [next(d for d in std_rows if d["preset"] == pr and "Belief Only" in d["policy"]) for pr in presets]
    i_std = [next(d for d in std_rows if d["preset"] == pr and "Belief + Intention" in d["policy"]) for pr in presets]

    b_wr = [d["win_rate"] for d in b_std]
    b_err_low = [d["win_rate"] - d["ci_low"] for d in b_std]
    b_err_high = [d["ci_high"] - d["win_rate"] for d in b_std]

    i_wr = [d["win_rate"] for d in i_std]
    i_err_low = [d["win_rate"] - d["ci_low"] for d in i_std]
    i_err_high = [d["ci_high"] - d["win_rate"] for d in i_std]

    r0 = ax0.bar(x - w / 2, b_wr, w, label="CD-ET-DAG (Belief Only)", color=col_belief, edgecolor="black", alpha=0.9,
                 yerr=[b_err_low, b_err_high], capsize=4)
    r1 = ax0.bar(x + w / 2, i_wr, w, label="IS-CD-ET-DAG (Belief + Intention, Ours)", color=col_intention, edgecolor="black", alpha=0.9,
                 yerr=[i_err_low, i_err_high], capsize=4)

    for r, val in zip(r0, b_wr):
        ax0.text(r.get_x() + r.get_width() / 2, val + 5.0, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r, val in zip(r1, i_wr):
        ax0.text(r.get_x() + r.get_width() / 2, val + 5.0, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax0.set_ylabel("Capture Win Rate (%)", fontsize=11, fontweight="bold")
    ax0.set_title("(1) Capture Win Rate (N=10/map, 95% Wilson CI)", fontsize=11.5, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(preset_labels, fontsize=9.5, fontweight="bold")
    ax0.set_ylim(0, 120)
    ax0.legend(fontsize=8, loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # Panel 2: Mean Time to Capture (TTC, steps)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    b_ttc = [d["mean_ttc"] for d in b_std]
    i_ttc = [d["mean_ttc"] for d in i_std]

    r0 = ax1.bar(x - w / 2, b_ttc, w, label="Belief Only", color=col_belief, edgecolor="black", alpha=0.9)
    r1 = ax1.bar(x + w / 2, i_ttc, w, label="Belief + Intention", color=col_intention, edgecolor="black", alpha=0.9)

    for r, val in zip(r0, b_ttc):
        ax1.text(r.get_x() + r.get_width() / 2, val + 15.0, f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r, val in zip(r1, i_ttc):
        ax1.text(r.get_x() + r.get_width() / 2, val + 15.0, f"{val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_ylabel("Mean Time to Capture (steps)", fontsize=11, fontweight="bold")
    ax1.set_title("(2) Pursuit Speed & TTC (Lower is Faster)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(preset_labels, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 1150)

    # -------------------------------------------------------------------------
    # Panel 3: Corridor Clutter & Overlap Steps (< 50 px)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    b_ov = [d["overlap_steps"] for d in b_std]
    i_ov = [d["overlap_steps"] for d in i_std]

    r0 = ax2.bar(x - w / 2, b_ov, w, label="Belief Only", color=col_belief, edgecolor="black", alpha=0.9)
    r1 = ax2.bar(x + w / 2, i_ov, w, label="Belief + Intention", color=col_intention, edgecolor="black", alpha=0.9)

    for r, val in zip(r0, b_ov):
        ax2.text(r.get_x() + r.get_width() / 2, val + 2.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r, val in zip(r1, i_ov):
        ax2.text(r.get_x() + r.get_width() / 2, val + 2.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_ylabel("Corridor Overlap Steps (< 50 px)", fontsize=11, fontweight="bold")
    ax2.set_title("(3) Corridor Congestion Suppression", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(preset_labels, fontsize=9.5, fontweight="bold")
    max_ov = max(max(b_ov), max(i_ov), 10)
    ax2.set_ylim(0, max_ov * 1.35 + 5)

    # -------------------------------------------------------------------------
    # Panel 4: Terminal Flanking Encirclement Angle (Degrees)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    b_fl = [d["flank_angle"] for d in b_std]
    i_fl = [d["flank_angle"] for d in i_std]

    r0 = ax3.bar(x - w / 2, b_fl, w, label="Belief Only", color=col_belief, edgecolor="black", alpha=0.9)
    r1 = ax3.bar(x + w / 2, i_fl, w, label="Belief + Intention", color=col_intention, edgecolor="black", alpha=0.9)

    for r, val in zip(r0, b_fl):
        if val > 0:
            ax3.text(r.get_x() + r.get_width() / 2, val + 2.0, f"{val:.0f}°", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r, val in zip(r1, i_fl):
        if val > 0:
            ax3.text(r.get_x() + r.get_width() / 2, val + 2.0, f"{val:.0f}°", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax3.axhline(90.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Orthogonal (90°)")
    ax3.axhline(180.0, color="#8e44ad", linestyle=":", linewidth=1.2, label="Head-on Pincer (180°)")
    ax3.set_ylabel("Flanking Encirclement Angle (°)", fontsize=11, fontweight="bold")
    ax3.set_title("(4) Terminal Flanking Angle (Higher is Superior Encirclement)", fontsize=11.5, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(preset_labels, fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0, 205)
    ax3.legend(fontsize=7.5, loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # Panel 5: Speed Asymmetry Stress Test (alpha = 1.15, Faster Evader)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    asym_rows = [d for d in data if "Speed Asymmetry" in d["condition"]]
    b_asym = [next(d for d in asym_rows if d["preset"] == pr and "Belief Only" in d["policy"]) for pr in presets]
    i_asym = [next(d for d in asym_rows if d["preset"] == pr and "Belief + Intention" in d["policy"]) for pr in presets]

    b_asym_wr = [d["win_rate"] for d in b_asym]
    i_asym_wr = [d["win_rate"] for d in i_asym]

    r0 = ax4.bar(x - w / 2, b_asym_wr, w, label="Belief Only (alpha=1.15)", color=col_belief, edgecolor="black", alpha=0.9)
    r1 = ax4.bar(x + w / 2, i_asym_wr, w, label="Belief + Intention (alpha=1.15)", color=col_intention, edgecolor="black", alpha=0.9)

    for r, val in zip(r0, b_asym_wr):
        ax4.text(r.get_x() + r.get_width() / 2, val + 2.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r, val in zip(r1, i_asym_wr):
        ax4.text(r.get_x() + r.get_width() / 2, val + 2.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax4.set_ylabel("Capture Win Rate (%)", fontsize=11, fontweight="bold")
    ax4.set_title("(5) Speed Asymmetry Stress Test (Evader +15% Speed)", fontsize=11.5, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(preset_labels, fontsize=9.5, fontweight="bold")
    ax4.set_ylim(0, 115)
    ax4.legend(fontsize=8, loc="upper left", frameon=True)

    # -------------------------------------------------------------------------
    # Panel 6: MuJoCo Continuous Physics Transfer (ALL 4 MAPS)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    muj_rows = [d for d in data if "MuJoCo" in d["condition"]]
    b_muj = [next(d for d in muj_rows if d["preset"] == pr and "Belief Only" in d["policy"]) for pr in presets]
    i_muj = [next(d for d in muj_rows if d["preset"] == pr and "Belief + Intention" in d["policy"]) for pr in presets]

    b_muj_wr = [d["win_rate"] for d in b_muj]
    i_muj_wr = [d["win_rate"] for d in i_muj]

    r0 = ax5.bar(x - w / 2, b_muj_wr, w, label="Belief Only (MuJoCo)", color=col_belief, edgecolor="black", alpha=0.9)
    r1 = ax5.bar(x + w / 2, i_muj_wr, w, label="Belief + Intention (MuJoCo, Ours)", color=col_intention, edgecolor="black", alpha=0.9)

    for r, val in zip(r0, b_muj_wr):
        ax5.text(r.get_x() + r.get_width() / 2, val + 2.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for r, val in zip(r1, i_muj_wr):
        ax5.text(r.get_x() + r.get_width() / 2, val + 2.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax5.set_ylabel("Capture Win Rate (%)", fontsize=11, fontweight="bold")
    ax5.set_title("(6) MuJoCo Continuous Physics Transfer (ALL 4 Maps)", fontsize=11.5, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(preset_labels, fontsize=9.5, fontweight="bold")
    ax5.set_ylim(0, 115)
    ax5.legend(fontsize=8, loc="upper left", frameon=True)

    plt.suptitle("Statistically Rigorous Head-to-Head Multi-Agent Pursuit Benchmark (160 Audited Episodes)\nBelief Sharing Only (CD-ET-DAG) vs. Belief + Intention Sharing (IS-CD-ET-DAG)",
                 fontsize=14.5, fontweight="bold", y=0.99)

    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"Publication plot successfully saved to: {output_png}")


if __name__ == "__main__":
    plot_thorough_benchmark()

