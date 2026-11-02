"""E1 - 5x5 dynamic GridWorld replication of the framework paper (Fig. 4).

Trains the paper's three variants on the paper's own gridworld:

* **DOS-RL** with potential-based shaping (Neural-Net-free, exactly Alg. 1-2);
* **DOS-RL** without shaping;
* **weighted-sum RL** (single scalarised objective, no Dynamic Objective
  Selection) -- the paper's "traditional RL" baseline.

It also compares the four readings of the DOS confidence metric, because the
paper's Eq. 3 and Algorithm 1 disagree about the sign, and the disagreement is
large enough to be visible in the learning curve.

Outputs
-------
results/e1_gridworld.csv
figures/e1_gridworld_replication.png
figures/e1_dos_variants.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl.config import GRID_RL
from dos_rl.gridworld import run_grid_experiment

from experiments.common import fig_path, save_csv, setup_mpl, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--test-episodes", type=int, default=30,
                    help="held-out episodes per seed, averaged after training")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.seeds, args.test_episodes = 600, 3, 10

    variants = {
        "DOS-RL (shaped)": dict(dos=True, shaping=True, dos_select="value"),
        "DOS-RL (unshaped)": dict(dos=True, shaping=False, dos_select="value"),
        "Weighted-sum RL": dict(dos=False, shaping=True, dos_select="value"),
    }
    rows, curves = [], {}
    with timer("E1 gridworld"):
        for label_, kw in variants.items():
            recs = []
            for seed in range(args.seeds):
                rl = GRID_RL.with_(dos_select=kw["dos_select"])
                rec = run_grid_experiment(rl, episodes=args.episodes, seed=seed,
                                          dos=kw["dos"], shaping=kw["shaping"],
                                          test_episodes=args.test_episodes)
                recs.append(rec)
                rows.append({
                    "variant": label_, "seed": seed,
                    # held-out: greedy policy, fresh obstacle draws (rubric item 4)
                    "test_return": float(np.mean(rec["test_return"])),
                    "test_success_pct": float(np.mean(rec["test_success"]) * 100),
                    "test_length": float(np.mean(rec["test_length"])),
                    "test_n_episodes": args.test_episodes,
                    # training-curve tail, kept for comparison
                    "final_return": float(rec["return"][-100:].mean()),
                    "final_success_pct": float(rec["success"][-100:].mean() * 100),
                    "final_episode_length": float(rec["length"][-100:].mean()),
                    "final_obstacle_hits": float(rec["hits"][-100:].mean()),
                    "final_energy_used": float(rec["energy"][-100:].mean()),
                })
            n = min(len(r["return"]) for r in recs)
            curves[label_] = {k: np.mean([r[k][:n] for r in recs], axis=0)
                              for k in ("return", "success", "length", "hits", "energy")}
            tr = np.mean([r["test_success_pct"] for r in rows if r["variant"] == label_])
            tn = np.mean([r["test_return"] for r in rows if r["variant"] == label_])
            print(f"  {label_:18s} held-out success={tr:5.1f}%  return={tn:+6.2f}  "
                  f"(train tail {np.mean([r['final_success_pct'] for r in rows if r['variant']==label_]):5.1f}%)",
                  flush=True)

        # DOS confidence-metric variants
        dos_rows, dos_curves = [], {}
        for mode in ("value", "argmax", "argmin"):
            succ, rr = [], []
            for seed in range(args.seeds):
                rl = GRID_RL.with_(dos_select=mode)
                rec = run_grid_experiment(rl, episodes=args.episodes, seed=seed, dos=True,
                                          test_episodes=args.test_episodes)
                succ.append(rec["test_success"].mean() * 100)
                rr.append(rec["return"])
                dos_rows.append({"variant": f"DOS[{mode}]", "seed": seed,
                                 "test_return": float(np.mean(rec["test_return"])),
                                 "test_success_pct": float(np.mean(rec["test_success"]) * 100),
                                 "test_n_episodes": args.test_episodes,
                                 "final_return": float(rec["return"][-100:].mean()),
                                 "final_success_pct": float(rec["success"][-100:].mean() * 100)})
            dos_curves[mode] = np.mean([x[: min(len(y) for y in rr)] for x in rr], axis=0)
            a = np.array(succ)
            print(f"  DOS[{mode:6s}] held-out success={a.mean():5.1f}% "
                  f"+/-{1.96*a.std(ddof=1)/np.sqrt(len(a)):4.1f}",
                  flush=True)

    save_csv(rows + dos_rows, "e1_gridworld.csv")

    plt = setup_mpl()
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.3))
    colors = {"DOS-RL (shaped)": "#1f77b4", "DOS-RL (unshaped)": "#2ca02c",
              "Weighted-sum RL": "#ff7f0e"}
    for name, c in curves.items():
        w = 50
        for ax, key, ttl, ylab in zip(
            axes,
            ("return", "energy", "length", "hits"),
            ("(a) Reward per episode", "(b) Energy consumed", "(c) Episode length",
             "(d) Frequency of hitting obstacles"),
            ("episode return", "energy used (units)", "steps", "obstacles / episode"),
        ):
            y = c[key]
            if len(y) >= w:
                y = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(len(y)), y, lw=1.3, color=colors[name], label=name)
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel("episode")
            ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("E1: replication of the paper's dynamic 5x5 GridWorld (Fig. 4)", fontsize=11)
    fig.savefig(fig_path("e1_gridworld_replication.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    cmap = {"value": "#1f77b4", "argmax": "#d62728", "argmin": "#7f7f7f"}
    for mode, c in dos_curves.items():
        w = 50
        y = c if len(c) < w else np.convolve(c, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(y)), y, lw=1.4, color=cmap[mode], label=f"DOS [{mode}]")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return (50-ep moving average)")
    ax.set_title("E1b: the three readings of the DOS confidence metric", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(fig_path("e1_dos_variants.png"), bbox_inches="tight")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
