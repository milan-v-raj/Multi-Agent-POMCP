"""Train DOS and weighted-sum Q-learning on the harder cold-start task."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.policy import GreedyHopPolicy, LearnedPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import evaluate, train

from experiments.common import colour, fig_path, label, save_csv, setup_mpl, timer


def run(cfg, seeds, episodes, max_hops, mode="node", loop_free="dag", test_episodes=30):
    curves = {}
    rows = []
    for name, nobj in (("dos_rl", 3), ("scalar", 1)):
        per_seed = []
        for seed in range(seeds):
            rng = np.random.default_rng(3000 + seed)
            topo = deploy_topology(cfg.deployment, rng)
            srcs = select_sources(topo, cfg.traffic.n_sources, rng)
            rl = NET_RL.with_(max_hops=max_hops, loop_free=loop_free)
            mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo),
                             state_mode=mode)
            agent = QLearningAgent(rl, n_objectives=nobj)
            hist = train(agent, mdp, srcs, episodes=episodes, seed=seed)
            ret = np.array([h.return_ for h in hist], float)
            suc = np.array([h.success for h in hist], float)
            # Evaluate greedily with a separate random stream.
            ev = evaluate(agent, mdp, srcs, episodes=test_episodes, seed=9000 + seed)
            per_seed.append((ret, suc))
            rows.append({
                "policy": name, "seed": seed,
                "early_return": float(ret[:50].mean()),
                "late_return": float(ret[-100:].mean()),
                "early_success": float(suc[:50].mean() * 100),
                "late_success": float(suc[-100:].mean() * 100),
                "test_success_pct": ev["success_rate"],
                "test_return_mean": ev["return_mean"],
                "test_return_ci95": ev["return_ci95"],
                "test_length_mean": ev["length_mean"],
                "test_n_episodes": test_episodes,
            })
        n = min(len(r) for r, _ in per_seed)
        curves[name] = {
            "return": np.mean([r[:n] for r, _ in per_seed], axis=0),
            "success": np.mean([s[:n] for _, s in per_seed], axis=0) * 100,
        }
        # average across seeds, not just the last one - reporting rows[-1]
        # printed a single seed's trajectory and read like the mean
        r = [x for x in rows if x["policy"] == name]
        e = float(np.mean([x["early_return"] for x in r]))
        l = float(np.mean([x["late_return"] for x in r]))
        el = float(np.mean([x["early_success"] for x in r]))
        ll = float(np.mean([x["late_success"] for x in r]))
        n_up = int(sum(1 for x in r if x["late_return"] > x["early_return"]))
        ts = float(np.mean([x["test_success_pct"] for x in r]))
        tr = float(np.mean([x["test_return_mean"] for x in r]))
        print(f"  {name:10s} early {e:+.2f} -> late {l:+.2f}   "
              f"success {el:.1f}% -> {ll:.1f}%   improved on {n_up}/{len(r)} seeds",
              flush=True)
        print(f"  {'':10s} HELD-OUT ({test_episodes} eps): success {ts:.1f}%  "
              f"return {tr:+.2f}", flush=True)
    return curves, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.episodes = 4, 500

    # A deliberately hard instance.
    #  * widely spaced relays (330 m range over 1500 m) so the route really is
    #    multi-hop and the hop budget binds;
    #  * every link is a marginal link (PRR slope widened), so an untrained
    #    choice of next hop really can fail;
    #  * ``loop_free="stp"`` widens the action set past the strict DAG, so the
    #    agent must choose between a short risky branch and a longer reliable
    #    one instead of being handed the answer by the action mask.
    cfg = SimConfig().with_(
        comm_range_m=330.0, n_sources=10, n_nodes=30, prr_slope_db=14.0
    )
    with timer("E0b cold start"):
        curves, rows = run(cfg, args.seeds, args.episodes, max_hops=12, mode="node",
                           loop_free="stp")
    save_csv(rows, "e0b_coldstart.csv")

    plt = setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    for name, c in curves.items():
        w = 25
        y = c["return"] if len(c["return"]) < w else np.convolve(
            c["return"], np.ones(w) / w, mode="valid")
        axes[0].plot(np.arange(len(y)), y, lw=1.5, color=colour(name), label=label(name))
        s = c["success"]
        if len(s) >= w:
            s = np.convolve(s, np.ones(w) / w, mode="valid")
        axes[1].plot(np.arange(len(s)), s, lw=1.4, color=colour(name), label=label(name))
    axes[0].set_xlabel("training episode")
    axes[0].set_ylabel("episode return (25-ep moving average)")
    axes[0].set_title("(a) Cold-start learning: reward vs. episode", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("training episode")
    axes[1].set_ylabel("success rate (%)")
    axes[1].set_title("(b) Route-computation success", fontsize=10)
    fig.suptitle("E0b: learning from scratch on a hard instance "
                 f"({cfg.deployment.comm_range_m:.0f} m range, 12-hop budget, widened PRR slope)",
                 fontsize=11)
    fig.savefig(fig_path("e0b_coldstart.png"), bbox_inches="tight")
    plt.close(fig)
    print("figure written")


if __name__ == "__main__":
    main()
