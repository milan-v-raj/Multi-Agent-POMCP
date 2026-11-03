"""Evaluate pretrained agents on new deployment seeds and topology families."""

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
from dos_rl.policy import GreedyHopPolicy, LearnedPolicy, OSPFStaticPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import train

from experiments.common import colour, fig_path, label, save_csv, setup_mpl, summarise_flow, timer

FAMILIES = {
    "A_train_30": dict(n_nodes=30, comm_range_m=450.0),
    "B_unseen_45": dict(n_nodes=45, comm_range_m=450.0),
    "C_unseen_60": dict(n_nodes=60, comm_range_m=450.0),
    "D_unseen_clutter": dict(n_nodes=30, comm_range_m=340.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=2500)
    ap.add_argument("--test-seeds", type=int, default=20)
    ap.add_argument("--load", type=float, default=60.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.train_episodes, args.test_seeds = 800, 8

    base = SimConfig()

    # ---- train on family A ----
    cfg_a = base.with_(**FAMILIES["A_train_30"])
    with timer("E7 training on family A"):
        models = {}
        for name, nobj, cls in (("tabular", 3, QLearningAgent), ("linear_fa", 3, LinearQAgent)):
            agents = []
            for seed in range(3):
                rng = np.random.default_rng(3000 + seed)
                topo = deploy_topology(cfg_a.deployment, rng)
                srcs = select_sources(topo, cfg_a.traffic.n_sources, rng)
                rl = NET_RL
                mdp = RoutingMDP(NetworkSimulator(topo, cfg_a), cfg_a, rl, BeliefTracker(topo))
                agent = cls(rl, n_objectives=nobj) if cls is QLearningAgent else cls(rl)
                train(agent, mdp, srcs, episodes=args.train_episodes, seed=seed)
                agents.append(agent)
            models[name] = agents
            print(f"  {name} trained ({len(agents)} seeds)", flush=True)

    # ---- zero-shot evaluation on every family ----
    rows = []
    with timer("E7 zero-shot evaluation"):
        for fam, kw in FAMILIES.items():
            cfg = base.with_(**kw)
            for seed in range(args.test_seeds):
                rng = np.random.default_rng(5000 + seed)
                topo = deploy_topology(cfg.deployment, rng)
                srcs = select_sources(topo, cfg.traffic.n_sources, rng)
                pols = {
                    "ospf": OSPFStaticPolicy(cfg),
                    "shortest_path": GreedyHopPolicy(cfg),
                }
                for name, agents in models.items():
                    ag = agents[seed % len(agents)]
                    pols[name] = LearnedPolicy(ag, NET_RL, cfg)
                for name, pol in pols.items():
                    res = NetworkSimulator(topo, cfg).run(
                        pol, srcs, rates=np.full(len(srcs), args.load)
                    )
                    rows.append({"family": fam, "seed": seed, "policy": name, **summarise_flow(res)})
            print(f"  {fam} done", flush=True)
    save_csv(rows, "e7_generalisation.csv")

    # ---- figure ----
    plt = setup_mpl()
    pols = ["tabular", "linear_fa", "shortest_path", "ospf"]
    fams = list(FAMILIES)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.3))
    width = 0.8 / len(pols)
    for ax, key, ylab, ttl in zip(
        axes,
        ("pdr_pct", "energy_efficiency", "mean_hops"),
        ("PDR (%)", "PDR per watt", "mean hops"),
        ("(a) Delivery on unseen deployments", "(b) Energy efficiency", "(c) Path length"),
    ):
        for k, p in enumerate(pols):
            ms, cis = [], []
            for fam in fams:
                a = np.array([r[key] for r in rows if r["family"] == fam and r["policy"] == p])
                ms.append(a.mean() if a.size else np.nan)
                cis.append(1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)
            ax.bar(np.arange(len(fams)) + (k - len(pols) / 2 + 0.5) * width, ms, width,
                   yerr=cis, color=colour(p), label=label(p), error_kw=dict(lw=0.7, capsize=2))
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([f.replace("_", "\n", 1) for f in fams], fontsize=6.5)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("E7: zero-shot transfer from a 30-relay training family to unseen deployments",
                 fontsize=11)
    fig.savefig(fig_path("e7_generalisation.png"), bbox_inches="tight")
    plt.close(fig)
    print("figure written")


if __name__ == "__main__":
    main()
