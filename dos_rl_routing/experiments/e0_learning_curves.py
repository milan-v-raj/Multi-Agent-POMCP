"""E0 - Learning curves on the network task (rubric: episode reward vs episode).

Trains six agents on identical deployments and reports
  * episode return vs. training episode (mean +/- 95 % CI over seeds),
  * success rate vs. training episode,
  * episode length,
  * wall-clock cost of learning,

plus a greedy 30-episode test battery per agent per seed.

Outputs
-------
results/e0_curves_raw.csv      one row per (agent, seed, episode)
results/e0_summary.csv         per (agent, seed) final metrics
results/e0_tests.csv           per (agent) test-battery statistics
figures/e0_learning_curves.png
figures/e0_success_and_length.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.linear_fa import LinearQAgent
from dos_rl.agents.tabular import MonteCarloAgent, QLearningAgent, SarsaAgent
from dos_rl.config import NET_RL
from dos_rl.topology import select_sources
from dos_rl.trainer import evaluate, summarise, train

from experiments.common import (
    RESULTS,
    LABELS,
    colour,
    fig_path,
    label,
    save_csv,
    setup_mpl,
    smooth,
    timer,
)

AGENTS = {
    "dos_rl": (QLearningAgent, {}),
    "sarsa": (SarsaAgent, {}),
    "mc_control": (MonteCarloAgent, {}),
    "linear_fa": (LinearQAgent, {}),
}


def build_agent(name: str, rl):
    """Construct the agent for a named policy.

    ``weighted_sum`` uses one Q table with normalized reward weights. Results
    generated before the correction used only the first component.
    """
    if name == "weighted_sum":
        return QLearningAgent(rl, n_objectives=1)
    cls, kw = AGENTS[name]
    return cls(rl, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--n-nodes", type=int, default=30)
    ap.add_argument("--n-sources", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.seeds = 600, 3

    rl = NET_RL
    agents = ["dos_rl", "weighted_sum", "sarsa", "mc_control", "linear_fa"]
    raw_rows, summary_rows, test_rows = [], [], []
    curves = {a: {"return": [], "success": [], "length": [], "epsilon": []} for a in agents}
    x_axis = None

    for seed in range(args.seeds):
        cfg = SimConfig().with_(n_nodes=args.n_nodes, n_sources=args.n_sources)
        rng = np.random.default_rng(2000 + seed)
        topo = deploy_topology(cfg.deployment, rng)
        srcs = select_sources(topo, args.n_sources, rng)

        for name in agents:
            mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
            agent = build_agent(name, rl)
            t0 = os.times().elapsed
            hist = train(agent, mdp, srcs, episodes=args.episodes, seed=seed, log=False)
            wall = os.times().elapsed - t0

            sm = summarise(hist, window=20)
            x_axis = sm["x"]
            curves[name]["return"].append(sm["return"])
            curves[name]["success"].append(sm["success"])
            curves[name]["length"].append(sm["length"])
            curves[name]["epsilon"].append(sm["epsilon"][: len(sm["x"])])

            for k in range(len(sm["x"])):
                raw_rows.append(
                    {
                        "agent": name,
                        "seed": seed,
                        "episode": int(sm["x"][k]),
                        "return": float(sm["return"][k]),
                        "success_pct": float(sm["success"][k]),
                        "length": float(sm["length"][k]),
                        "epsilon": float(sm["epsilon"][min(k, len(sm["epsilon"]) - 1)]),
                    }
                )

            ev = evaluate(agent, mdp, srcs, episodes=args.test_episodes, seed=9000 + seed)
            summary_rows.append(
                {
                    "agent": name,
                    "seed": seed,
                    "train_seconds": wall,
                    "q_entries": agent.q_size() if hasattr(agent, "q_size") else np.nan,
                    "final_train_return": float(np.mean([h.return_ for h in hist[-100:]])),
                    "final_train_success_pct": float(np.mean([h.success for h in hist[-100:]]) * 100),
                    **{f"test_{k}": v for k, v in ev.items()},
                }
            )
            test_rows.append({"agent": name, "seed": seed, **ev})
            print(
                f"seed {seed} {name:12s} test success {ev['success_rate']:5.1f}% "
                f"return {ev['return_mean']:+6.2f} | {wall:4.1f}s",
                flush=True,
            )

    save_csv(raw_rows, "e0_curves_raw.csv")
    # keep the per-seed curves for the report's plotting module
    np.savez(
        os.path.join(RESULTS, "e0_curves.npz"),
        x=x_axis,
        **{f"{a}_{k}": np.array(v, dtype=float) for a in agents for k, v in curves[a].items() if k != "success"},
        **{f"{a}_success": np.array(curves[a]["success"], dtype=float) for a in agents},
    )
    save_csv(summary_rows, "e0_summary.csv")
    save_csv(test_rows, "e0_tests.csv")

    # ---------------- figure ----------------
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    for name in agents:
        r = np.array(curves[name]["return"])
        s = np.array(curves[name]["success"])
        l = np.array(curves[name]["length"])
        m, ci = _mean_ci(r)
        axes[0].plot(x_axis, m, color=colour(name), label=label(name), lw=1.4)
        axes[0].fill_between(x_axis, m - ci, m + ci, color=colour(name), alpha=0.18)
        ms, cis = _mean_ci(s)
        axes[1].plot(x_axis, ms, color=colour(name), lw=1.4)
        axes[1].fill_between(x_axis, ms - cis, ms + cis, color=colour(name), alpha=0.18)
        ml, cil = _mean_ci(l)
        axes[2].plot(x_axis, ml, color=colour(name), lw=1.4)
        axes[2].fill_between(x_axis, ml - cil, ml + cil, color=colour(name), alpha=0.18)
    axes[0].set_xlabel("training episode")
    axes[0].set_ylabel("episode return (20-episode moving average)")
    axes[0].set_ylim(bottom=min(0.0, float(np.nanmin([np.min(np.array(curves[a]['return'])) for a in agents])) - 1))
    axes[0].set_title("(a) Episode reward vs. training episode")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].set_xlabel("training episode")
    axes[1].set_ylabel("success rate (%)")
    axes[1].set_title("(b) Route-computation success")
    axes[2].set_xlabel("training episode")
    axes[2].set_ylabel("episode length (hops)")
    axes[2].set_title("(c) Path length")
    fig.savefig(fig_path("e0_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)

    # test battery bar chart
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    names = agents
    succ = [[r["success_rate"] for r in test_rows if r["agent"] == n] for n in names]
    rets = [[r["return_mean"] for r in test_rows if r["agent"] == n] for n in names]
    axes[0].bar(range(len(names)), [np.mean(v) for v in succ], color=[colour(n) for n in names])
    axes[0].errorbar(range(len(names)), [np.mean(v) for v in succ],
                     yerr=[1.96 * np.std(v, ddof=1) / max(np.sqrt(len(v)), 1) for v in succ],
                     fmt="none", ecolor="k", capsize=3, lw=0.9)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([label(n) for n in names], rotation=20, ha="right")
    axes[0].set_ylabel("test success rate (%)")
    axes[0].set_title("(d) Greedy evaluation, 30 test episodes")
    axes[1].bar(range(len(names)), [np.mean(v) for v in rets], color=[colour(n) for n in names])
    axes[1].errorbar(range(len(names)), [np.mean(v) for v in rets],
                     yerr=[1.96 * np.std(v, ddof=1) / max(np.sqrt(len(v)), 1) for v in rets],
                     fmt="none", ecolor="k", capsize=3, lw=0.9)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([label(n) for n in names], rotation=20, ha="right")
    axes[1].set_ylabel("mean test return")
    axes[1].set_title("(e) Test return")
    fig.savefig(fig_path("e0_success_and_length.png"), bbox_inches="tight")
    plt.close(fig)
    print("figures written")


def _mean_ci(runs: np.ndarray):
    m = runs.mean(axis=0)
    if runs.shape[0] < 2:
        return m, np.zeros_like(m)
    sd = runs.std(axis=0, ddof=1)
    return m, 1.96 * sd / np.sqrt(runs.shape[0])


if __name__ == "__main__":
    main()
