"""Sweep learning rate, discount factor, exploration and exploration decay."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.policy import LearnedPolicy, GreedyHopPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import train

from experiments.common import colour, fig_path, label, save_csv, setup_mpl, timer

SWEEPS = {
    "alpha": [0.02, 0.05, 0.1, 0.3, 0.6, 0.9],
    "gamma": [0.5, 0.7, 0.85, 0.95, 0.99],
    "epsilon": [0.02, 0.05, 0.1, 0.3, 0.5, 0.8],
    "epsilon_decay": [0.9, 0.995, 1.0],
}


def convergence_episode(curve: np.ndarray, tol: float = 0.05, window: int = 50) -> float:
    """First episode where the smoothed return is within ``tol`` of the final level."""
    if curve.size < window * 2:
        return float("nan")
    k = np.ones(window) / window
    sm = np.convolve(curve, k, mode="valid")
    final = float(np.mean(sm[-max(1, window // 2):]))
    scale = max(abs(final), 1e-6)
    ok = np.abs(sm - final) <= tol * scale
    # require it to *stay* within tolerance for the next window
    for i in range(len(ok) - window):
        if np.all(ok[i:i + window]):
            return float(i + window)
    return float("nan")


def learning_auc(curve: np.ndarray, frac: float = 0.3) -> float:
    n = max(1, int(len(curve) * frac))
    return float(np.trapezoid(curve[:n]) / n)


def evaluate_setting(name, value, cfg, seeds, episodes, lam, mode="network"):
    curves, finals, succ, pdr_vs, qsizes = [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(3000 + seed)
        topo = deploy_topology(cfg.deployment, rng)
        srcs = select_sources(topo, cfg.traffic.n_sources, rng)
        rl = NET_RL
        if name == "alpha":
            rl = rl.with_(alpha=value)
        elif name == "gamma":
            rl = rl.with_(gamma=value)
        elif name == "epsilon":
            rl = rl.with_(epsilon=value, decay_epsilon=False)
        elif name == "epsilon_decay":
            rl = rl.with_(epsilon_decay=value, decay_epsilon=value < 1.0)
        if mode == "congestion":
            # Make congestion part of the *episode* so that the learning curve
            # responds to the hyperparameters. In the default (uncongested)
            # setting the task saturates after ~300 episodes for every
            # hyperparameter, which makes the sweep uninformative - that
            # insensitivity is itself reported as a finding.
            mdp = RoutingMDP(
                NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo),
                state_mode="belief",
            )
        else:
            mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
        agent = QLearningAgent(rl)
        hist = train(agent, mdp, srcs, episodes=episodes, seed=seed)
        ret = np.array([h.return_ for h in hist], float)
        suc = np.array([h.success for h in hist], float)
        curves.append(ret)
        finals.append(ret[-200:].mean())
        succ.append(suc[-200:].mean() * 100)
        qsizes.append(agent.q_size())
        res = NetworkSimulator(topo, cfg).run(
            LearnedPolicy(agent, rl, cfg, state_mode=("belief" if mode == "congestion" else "node")),
            srcs, rates=np.full(len(srcs), lam),
        )
        base = NetworkSimulator(topo, cfg).run(GreedyHopPolicy(cfg), srcs,
                                               rates=np.full(len(srcs), lam))
        pdr_vs.append(res.pdr * 100 - base.pdr * 100)
    conv = [convergence_episode(c) for c in curves]
    return {
        "sweep": name,
        "value": value,
        "final_return": float(np.mean(finals)),
        "final_return_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "final_success_pct": float(np.mean(succ)),
        "learning_auc": float(np.mean([learning_auc(c) for c in curves])),
        "convergence_episode": float(np.nanmean(conv)) if np.any(np.isfinite(conv)) else float("nan"),
        "pdr_delta_vs_baseline_pp": float(np.mean(pdr_vs)),
        "pdr_delta_ci95": float(1.96 * np.std(pdr_vs, ddof=1) / np.sqrt(len(pdr_vs))) if len(pdr_vs) > 1 else 0.0,
        "q_table_entries": float(np.mean(qsizes)),
        "n_seeds": len(list(seeds)),
    }, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--load", type=float, default=60.0)
    ap.add_argument("--mode", default="congestion", choices=["network", "congestion"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = range(args.seeds if not args.quick else 3)
    episodes = args.episodes if not args.quick else 600
    cfg = SimConfig()

    rows, curve_store = [], {}
    for name, values in SWEEPS.items():
        with timer(f"E4 {name}"):
            for v in values:
                row, curves = evaluate_setting(name, v, cfg, seeds, episodes, args.load, mode=args.mode)
                rows.append(row)
                curve_store[(name, v)] = np.mean(
                    [c[: min(len(c) for c in curves)] for c in curves], axis=0
                )
                print(
                    f"   {name}={v}: final={row['final_return']:+7.2f} "
                    f"succ={row['final_success_pct']:5.1f}% conv={row['convergence_episode']:6.0f} "
                    f"PDR gain={row['pdr_delta_vs_baseline_pp']:+5.2f}pp",
                    flush=True,
                )
    save_csv(rows, "e4_parameter_sweep.csv")

    # ---------------- figures ----------------
    plt = setup_mpl()
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.2))
    for j, (name, values) in enumerate(SWEEPS.items()):
        ax = axes[0, j]
        cmap = plt.get_cmap("viridis")
        for k, v in enumerate(values):
            c = curve_store[(name, v)]
            # a 50-episode moving average: the raw per-episode return is so noisy
            # on this task (single-packet episodes) that the raw traces overlap
            # into an unreadable band and hide the effect being measured
            w = min(50, max(1, len(c) // 8))
            if len(c) > w:
                c = np.convolve(c, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(len(c)), c, lw=1.3, color=cmap(k / max(1, len(values) - 1)),
                    label=f"{name}={v}")
        ax.set_title(f"(a{j+1}) learning curve vs {name}", fontsize=9)
        ax.set_xlabel("training episode")
        ax.set_ylabel("episode return (50-ep MA)")
        ax.legend(frameon=False, fontsize=6.5)

        ax = axes[1, j]
        sel = [r for r in rows if r["sweep"] == name]
        xs = np.array([r["value"] for r in sel], float)
        ax.plot(xs, [r["pdr_delta_vs_baseline_pp"] for r in sel], marker="o", ms=3.5, lw=1.3,
                color=colour("dos_rl"), label="closed-loop PDR gain")
        ax.errorbar(xs, [r["pdr_delta_vs_baseline_pp"] for r in sel],
                    yerr=[r["pdr_delta_ci95"] for r in sel], fmt="none", capsize=2.2,
                    color=colour("dos_rl"))
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel(name)
        ax.set_ylabel("PDR gain vs baseline (pp)")
        ax2 = ax.twinx()
        ax2.plot(xs, [r["convergence_episode"] for r in sel], marker="s", ms=3, lw=1.1,
                 color=colour("weighted_sum"), label="convergence episode")
        ax2.set_ylabel("convergence episode", color=colour("weighted_sum"))
        ax2.grid(False)
        ax.set_title(f"(b{j+1}) sample efficiency vs {name}", fontsize=9)
    fig.suptitle("E4: sensitivity to RL hyperparameters (closed-loop load = "
                 f"{args.load:.0f} pkt/s)", fontsize=11)
    fig.savefig(fig_path("e4_parameter_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    print("figure written")


if __name__ == "__main__":
    main()
