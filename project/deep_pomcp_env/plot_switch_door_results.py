"""
Publication Figure Generator: Cooperative Sacrifice / Switch-Door Benchmark.
Generates cooperative_sacrifice_benchmark.png (3-panel plot):
Panel 1: Win Rate & Switch Trigger Comparison (Bar Chart)
Panel 2: Cumulative Step-to-Breach / Time Timeline
Panel 3: Emergent Cooperative Role Division & Path Schematic
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

csv_path = os.path.join(parent_dir, "switch_door_benchmark_results.csv")
output_png = os.path.join(parent_dir, "cooperative_sacrifice_benchmark.png")

if not os.path.exists(csv_path):
    print("CSV not found:", csv_path)
    sys.exit(1)

with open(csv_path, mode="r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

policies = ["Reactive A*", "Vanilla POMCP", "Heuristic POMCP", "Deep-POMCP (Ours)"]
policy_colors = {
    "Reactive A*": "#ef4444",
    "Vanilla POMCP": "#f59e0b",
    "Heuristic POMCP": "#3b82f6",
    "Deep-POMCP (Ours)": "#10b981"
}

# Aggregate metrics
win_rates = []
sw_rates = []
br_rates = []
ttc_means = []

for pol in policies:
    eps = [r for r in rows if r["policy"] == pol]
    n = len(eps)
    wins = sum(1 for r in eps if r["captured"] == "True")
    sws = sum(1 for r in eps if r["switch_triggered"] == "True")
    brs = sum(1 for r in eps if r["gate_breached"] == "True")
    ttcs = [float(r["steps"]) for r in eps if r["captured"] == "True"]

    win_rates.append((wins / n) * 100.0 if n else 0)
    sw_rates.append((sws / n) * 100.0 if n else 0)
    br_rates.append((brs / n) * 100.0 if n else 0)
    ttc_means.append(np.mean(ttcs) if ttcs else 1500.0)

# Create 3-Panel Figure
fig = plt.figure(figsize=(16, 5), dpi=300)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Panel 1: Win Rate & Switch Trigger Rate
ax1 = fig.add_subplot(1, 3, 1)
x = np.arange(len(policies))
w = 0.35
b1 = ax1.bar(x - w/2, win_rates, width=w, label="Capture Win Rate (%)", color="#10b981", alpha=0.9, edgecolor="black")
b2 = ax1.bar(x + w/2, sw_rates, width=w, label="Switch Trigger Rate (%)", color="#6366f1", alpha=0.9, edgecolor="black")

ax1.set_title("(A) Cooperative Sacrifice Success Rates", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([p.replace(" (Ours)", "\n(Ours)") for p in policies], fontsize=10, fontweight="bold")
ax1.set_ylim(0, 105)
ax1.legend(loc="upper left", frameon=True, fontsize=9)

for bar in b1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 2, f"{y:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar in b2:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 2, f"{y:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 2: Mean Time to Capture
ax2 = fig.add_subplot(1, 3, 2)
bar_cols = [policy_colors[p] for p in policies]
bars_ttc = ax2.bar(policies, ttc_means, color=bar_cols, alpha=0.85, edgecolor="black", width=0.55)
ax2.set_title("(B) Mean Time-to-Capture (Steps)", fontsize=13, fontweight="bold", pad=10)
ax2.set_ylabel("Steps (Lower is Better)", fontsize=11, fontweight="bold")
ax2.set_xticklabels([p.replace(" (Ours)", "\n(Ours)") for p in policies], fontsize=10, fontweight="bold")
ax2.set_ylim(0, 1600)
ax2.axhline(1200, color="gray", linestyle="--", alpha=0.7, label="Timeout Threshold")

for bar, pol, ttc in zip(bars_ttc, policies, ttc_means):
    y = bar.get_height()
    label = f"{ttc:.0f} steps" if ttc < 1499 else "FAILED (100%)"
    ax2.text(bar.get_x() + bar.get_width()/2.0, min(1450, y + 25), label, ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.legend(loc="upper left", frameon=True, fontsize=9)

# Panel 3: Schematic of Cooperative Role Division
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_facecolor("#0f172a")
ax3.set_xlim(0, 800)
ax3.set_ylim(0, 600)
ax3.invert_yaxis()

# Draw Boundary & Walls
ax3.add_patch(patches.Rectangle((0, 0), 800, 600, fill=False, edgecolor="#475569", linewidth=2))
# Barrier Wall at x = 580
ax3.add_patch(patches.Rectangle((580, 10), 15, 230, facecolor="#64748b", edgecolor="white", linewidth=1))
ax3.add_patch(patches.Rectangle((580, 360), 15, 230, facecolor="#64748b", edgecolor="white", linewidth=1))
# Open Gate Zone
ax3.add_patch(patches.Rectangle((580, 240), 15, 120, fill=False, edgecolor="#10b981", linestyle="--", linewidth=2))
ax3.text(605, 300, "GATE\n(Dynamic)", color="#10b981", fontsize=9, fontweight="bold", va="center")

# Draw Switch at x = 80, y = 300
ax3.add_patch(patches.Circle((80, 300), 35, fill=True, facecolor="#064e3b", edgecolor="#10b981", linewidth=2))
ax3.plot(80, 300, marker="o", color="#10b981", markersize=8)
ax3.text(80, 350, "SWITCH\n(x=80)", color="#10b981", fontsize=9, fontweight="bold", ha="center")

# Draw Clutter
ax3.add_patch(patches.Rectangle((250, 120), 50, 60, facecolor="#334155", edgecolor="#475569"))
ax3.add_patch(patches.Rectangle((350, 400), 60, 50, facecolor="#334155", edgecolor="#475569"))
ax3.add_patch(patches.Rectangle((420, 200), 50, 70, facecolor="#334155", edgecolor="#475569"))

# Hunter 1 Sacrifice Path (to switch)
ax3.annotate("", xy=(85, 300), xytext=(320, 260), arrowprops=dict(arrowstyle="->", color="#818cf8", lw=2.5, linestyle="--"))
ax3.plot(320, 260, marker="^", color="#818cf8", markersize=10, label="Hunter 1 (Operator)")
ax3.text(200, 240, "Sacrifice Trajectory", color="#818cf8", fontsize=8, fontweight="bold")

# Hunter 2 Breach Path (to gate -> evader)
ax3.annotate("", xy=(575, 300), xytext=(340, 340), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))
ax3.annotate("", xy=(710, 300), xytext=(595, 300), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))
ax3.plot(340, 340, marker="^", color="#38bdf8", markersize=10, label="Hunter 2 (Breacher)")

# Evader
ax3.plot(720, 300, marker="*", color="#ef4444", markersize=14, label="Enclosed Target")
ax3.text(720, 340, "Evader\n(x=720)", color="#ef4444", fontsize=9, fontweight="bold", ha="center")

ax3.set_title("(C) Deep-POMCP Emergent Role Specialization", fontsize=13, fontweight="bold", pad=10)
ax3.legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=8)
ax3.set_xticks([])
ax3.set_yticks([])

plt.tight_layout()
plt.savefig(output_png, dpi=300)
print(f"Cooperative Sacrifice figure saved to: {output_png}")
