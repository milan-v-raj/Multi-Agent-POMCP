"""Plot saved offered-load and outage results without rerunning training."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "runs" / "comparison_plots"
POLICIES = (
    ("dos_rl", "DOS Q-learning", "#187fa3"),
    ("shortest_path", "Re-planned shortest path", "#bc6228"),
)


def read_rows(filename):
    with (RESULTS / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_load():
    rows = [row for row in read_rows("e2_e3_raw.csv") if row["sweep"] == "load"]
    loads = sorted({float(row["value"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for policy, label, colour in POLICIES:
        for ax, metric in zip(axes, ("pdr_pct", "e2e_delay_ms")):
            values = []
            for load in loads:
                selected = [float(row[metric]) for row in rows
                            if row["policy"] == policy and float(row["value"]) == load]
                if not selected:
                    raise ValueError(f"No {policy} records for load {load}")
                values.append(mean(selected))
            ax.plot(loads, values, marker="o", label=label, color=colour, linewidth=2)
    for ax, ylabel in zip(axes, ("Packet delivery ratio (%)", "Estimated end-to-end delay (ms)")):
        ax.set_xlabel("Packets/s per source")
        ax.set_ylabel(ylabel)
        ax.set_xticks(loads)
        ax.grid(alpha=0.24)
    axes[0].legend(fontsize=8)
    fig.suptitle("Offered load: five sources, 10 deployments per setting", fontsize=11)
    fig.savefig(OUTPUT / "offered_load.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_outages():
    groups = defaultdict(list)
    for row in read_rows("e5b_link_failure.csv"):
        groups[(float(row["value"]), row["policy"])].append(float(row["pdr_post"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    for policy, label, colour in POLICIES:
        fractions = sorted(fraction for fraction, name in groups if name == policy)
        values = [mean(groups[(fraction, policy)]) for fraction in fractions]
        ax.plot([100 * fraction for fraction in fractions], values, marker="o",
                linewidth=2.2, label=label, color=colour)
    ax.set_xticks([0, 20, 40])
    ax.set_xlabel("Links temporarily unavailable (%)")
    ax.set_ylabel("Mean post-onset PDR (%)")
    ax.set_ylim(60, 100)  # Matches the saved figure; the axis does not start at zero.
    ax.set_title("Temporary link outages: eight paired deployments")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(OUTPUT / "link_outage.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_load()
    plot_outages()
    print(f"Saved comparison figures to {OUTPUT}")


if __name__ == "__main__":
    main()
