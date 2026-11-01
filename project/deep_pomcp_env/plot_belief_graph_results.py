"""
Generates the Publication-Grade 3-Panel Visualization for Latent Belief-Graph MCTS (DAG-POMCP):
  Panel A: Effective Search Depth Reached vs. Simulation Budget (Tree MCTS vs. Belief-Graph DAG-MCTS)
  Panel B: Transposition Merge Rate (%) & Search Graph Compression across Simulation Budgets
  Panel C: Head-to-Head Capture Win Rate & Time-to-Capture (TTC) against Strategic Adversarial Evader
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def generate_plots():
    scaling_csv = os.path.join(parent_dir, "belief_graph_scaling_results.csv")
    adv_csv = os.path.join(parent_dir, "belief_graph_adversarial_results.csv")

    if not os.path.exists(scaling_csv) or not os.path.exists(adv_csv):
        print("CSV files not found. Run run_belief_graph_benchmark.py first.")
        return

    scaling_data = []
    with open(scaling_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scaling_data.append(row)

    adv_data = []
    with open(adv_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            adv_data.append(row)

    n_sims = [int(r["n_sims"]) for r in scaling_data]
    tree_max_d = [float(r["tree_max_depth"]) for r in scaling_data]
    tree_mean_d = [float(r["tree_mean_depth"]) for r in scaling_data]
    dag_max_d = [float(r["dag_max_depth"]) for r in scaling_data]
    dag_mean_d = [float(r["dag_mean_depth"]) for r in scaling_data]
    merge_rates = [float(r["dag_merge_rate_pct"]) for r in scaling_data]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "figure.dpi": 300
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # -------------------------------------------------------------------------
    # Panel A: Effective Search Depth Reached vs Simulation Budget
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(n_sims, dag_max_d, marker="o", linewidth=2.6, markersize=8,
             color="#1b7837", label="DAG-Deep-POMCP (Max Depth)")
    ax1.plot(n_sims, tree_max_d, marker="s", linewidth=2.4, markersize=7,
             linestyle="--", color="#d73027", label="Standard Tree MCTS (Max Depth)")
    ax1.plot(n_sims, dag_mean_d, marker="^", linewidth=2.0, markersize=6,
             color="#5aae61", linestyle="-.", label="DAG-Deep-POMCP (Mean Depth)")
    ax1.plot(n_sims, tree_mean_d, marker="v", linewidth=2.0, markersize=6,
             color="#f46d43", linestyle=":", label="Standard Tree MCTS (Mean Depth)")

    ax1.axhline(6.0, color="#555555", linestyle=":", alpha=0.6, label="Target Horizon (D=6)")
    ax1.set_title("(a) Search Horizon Depth vs. Compute Budget")
    ax1.set_xlabel("Simulation Budget ($N_{\\mathrm{sims}}$)")
    ax1.set_ylabel("Effective Search Depth Reached ($D$)")
    ax1.set_ylim(1.0, 6.6)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8.8, frameon=True)

    # -------------------------------------------------------------------------
    # Panel B: PointNet Transposition Merge Rate (%)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    bars = ax2.bar([str(n) for n in n_sims], merge_rates, color="#2b5c8f", width=0.55, edgecolor="black", alpha=0.88)
    for bar, val in zip(bars, merge_rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1a3a5c"
        )
    ax2.set_title("(b) PointNet Latent Transposition Merge Rate")
    ax2.set_xlabel("Simulation Budget ($N_{\\mathrm{sims}}$)")
    ax2.set_ylabel("Belief-Graph Merge Rate (%)")
    ax2.set_ylim(0, max(merge_rates) * 1.22 + 5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    # -------------------------------------------------------------------------
    # Panel C: 2x2 Factorial + Budget Ablation Win Rate & Total Sims/Ep
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    policies_order = [
        "Tree Deep-POMCP (N=20)",
        "ET-Tree Deep-POMCP (N=20)",
        "DAG-Deep-POMCP (N=20, Ours)",
        "ET-DAG-Deep-POMCP (N=20, Unified)",
        "Tree Deep-POMCP (N=60)",
        "DAG-Deep-POMCP (N=60, Ours)"
    ]
    short_labels = [
        "Tree\n(N=20)",
        "ET-Tree\n(N=20)",
        "DAG\n(N=20)",
        "ET-DAG\n(N=20)",
        "Tree\n(N=60)",
        "DAG\n(N=60)"
    ]

    win_rates = []
    mean_sims = []
    for p_name in policies_order:
        p_rows = [r for r in adv_data if r["policy"] == p_name]
        if p_rows:
            wins = sum(1 for r in p_rows if r["captured"].lower() == "true")
            wr = (wins / len(p_rows)) * 100.0
            sims = [float(r["total_sims"]) for r in p_rows]
            avg_s = float(np.mean(sims)) if sims else 0.0
        else:
            wr, avg_s = 0.0, 0.0
        win_rates.append(wr)
        mean_sims.append(avg_s)

    x_pos = np.arange(len(policies_order))
    colors = ["#e08214", "#fdae61", "#1b7837", "#00441b", "#d73027", "#4d9221"]
    b3 = ax3.bar(x_pos, win_rates, width=0.58, color=colors, edgecolor="black", alpha=0.9)
    for bar, wr, s_ep in zip(b3, win_rates, mean_sims):
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.0,
            f"{wr:.1f}%\n({int(s_ep)} sims)",
            ha="center", va="bottom", fontsize=8.0, fontweight="bold"
        )

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(short_labels, fontsize=8.8, fontweight="bold")
    ax3.set_title("(c) Factorial Ablation: Win Rate & Sims/Ep")
    ax3.set_ylabel("Capture Win Rate (%)")
    ax3.set_ylim(0, 115)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_path_1 = os.path.join(parent_dir, "belief_graph_mcts_benchmark.png")
    out_path_2 = os.path.join(os.path.dirname(parent_dir), "belief_graph_mcts_benchmark.png")
    plt.savefig(out_path_1, bbox_inches="tight")
    plt.savefig(out_path_2, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_path_1}")


if __name__ == "__main__":
    generate_plots()

