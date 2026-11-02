"""E8 - Partial observability ladder (the PO-MDP half of the formulation).

The framework paper assumes the controller has a perfect, instantaneous global
view. Real SDWSN controllers do not, so this experiment grades the observation
channel from perfect to severe:

    full       perfect, zero-delay telemetry (the paper's assumption)
    delay      reports are 2 routing epochs (6 s) stale
    quantised  energy rounded to 5 %, +-2 % noise
    noisy      +-5 % noise on energy, load and PRR
    severe     delay + quantisation + noise + 15 % report loss

For each profile the same trained policy is deployed. A policy that does not
carry a belief (pure node-state Q-table) degrades because it acts on stale or
wrong numbers; the belief-tracking linear approximator is expected to degrade
more gracefully. That difference is the justification for the PO-MDP layer.

Run:  python -m experiments.e8_partial_observability [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.linear_fa import LinearQAgent
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.observer import OBS_PROFILES
from dos_rl.policy import GreedyHopPolicy, LearnedPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import train

from experiments.common import colour, fig_path, label, save_csv, setup_mpl, summarise_flow, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--load", type=float, default=60.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.episodes = 4, 600
    cfg = SimConfig()
    profiles = ["full", "delay", "quantised", "noisy", "severe"]

    rows = []
    with timer("E8 partial observability"):
        for seed in range(args.seeds):
            rng = np.random.default_rng(3000 + seed)
            topo = deploy_topology(cfg.deployment, rng)
            srcs = select_sources(topo, cfg.traffic.n_sources, rng)
            rl = NET_RL
            # one tabular agent and one belief-based linear agent per deployment
            mdp_t = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
            ag_t = QLearningAgent(rl)
            train(ag_t, mdp_t, srcs, episodes=args.episodes, seed=seed)

            mdp_l = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
            ag_l = LinearQAgent(rl)
            train(ag_l, mdp_l, srcs, episodes=args.episodes, seed=seed)

            for prof_name in profiles:
                prof = OBS_PROFILES[prof_name]
                pols = {
                    "dos_rl": LearnedPolicy(ag_t, rl, cfg, obs_profile=prof, seed=seed),
                    "linear_fa": LearnedPolicy(ag_l, rl, cfg, obs_profile=prof, seed=seed),
                    "shortest_path": GreedyHopPolicy(cfg, obs_profile=prof, seed=seed),
                }
                for name, pol in pols.items():
                    res = NetworkSimulator(topo, cfg).run(
                        pol, srcs, rates=np.full(len(srcs), args.load)
                    )
                    rows.append({"profile": prof_name, "seed": seed, "policy": name,
                                 **summarise_flow(res)})
            print(f"  seed {seed} done", flush=True)
    save_csv(rows, "e8_partial_observability.csv")

    plt = setup_mpl()
    pols = ["dos_rl", "linear_fa", "shortest_path"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3))
    for ax, key, ylab, ttl in zip(
        axes,
        ("pdr_pct", "energy_efficiency", "load_imbalance"),
        ("PDR (%)", "PDR per watt", "load imbalance (std/mean)"),
        ("(a) Delivery under degraded telemetry", "(b) Energy efficiency", "(c) Load imbalance"),
    ):
        for p in pols:
            ms, cis = [], []
            for prof in profiles:
                a = np.array([r[key] for r in rows if r["profile"] == prof and r["policy"] == p])
                ms.append(a.mean() if a.size else np.nan)
                cis.append(1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)
            ax.errorbar(range(len(profiles)), ms, yerr=cis, marker="o", ms=3.5, lw=1.3,
                        capsize=2.2, color=colour(p), label=label(p))
        ax.set_xticks(range(len(profiles)))
        ax.set_xticklabels(profiles, rotation=15, fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("E8: partial observability ladder (identical policies, degraded telemetry)",
                 fontsize=11)
    fig.savefig(fig_path("e8_partial_observability.png"), bbox_inches="tight")
    plt.close(fig)
    print("figure written")


if __name__ == "__main__":
    main()
