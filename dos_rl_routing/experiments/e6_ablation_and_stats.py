"""E6 - Component ablation + the headline test battery with statistics.

Ablations (each removes exactly one design decision):

===========================  ==================================================
``full``                     complete DOS-RL: 3 objectives, DOS selection,
                             potential shaping, loop-free DAG action set
``no_shaping``               remove the potential-based shaping term F_t
``no_dos``                   scalarise the reward instead of selecting an
                             objective per state (single Q-table)
``argmax_confidence``        use the paper's Eq. 3 verbatim (argmax of C_k)
``argmin_confidence``        use Algorithm 1 verbatim (argmin of C_k)
``no_loop_free``             unrestricted actions -> Q-routing loops
``visited_loop_free``        episode-local visited-set masking (no STP pre-pass)
``corrected_reward``         per-hop energy-cost variant instead of the
                             default rescaled paper energy-difference reward
===========================  ==================================================

The ablations use matched seeds and are evaluated on those same training
deployments. The separate deployment battery uses new seeds relative to the
ablation, but trains a fresh RL agent on each evaluated deployment; it is not
an unseen-topology test. The script reports paired statistics for that battery.

Run:  python -m experiments.e6_ablation_and_stats [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.policy import GreedyHopPolicy, LearnedPolicy, OSPFStaticPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import train

from experiments.common import (
    cliffs_delta,
    colour,
    fig_path,
    label,
    save_csv,
    setup_mpl,
    summarise_flow,
    timer,
    wilcoxon,
)

ABLATIONS = {
    "full": {},
    "no_shaping": dict(use_shaping=False),
    "no_dos": dict(_nobj=1),
    "argmax_confidence": dict(dos_select="argmax"),
    "argmin_confidence": dict(dos_select="argmin"),
    "no_loop_free": dict(loop_free="none"),
    "visited_loop_free": dict(loop_free="visited"),
    "corrected_reward": dict(reward_mode="corrected"),
}

TEST_POLICIES = ("full", "no_dos", "ospf", "shortest_path", "energy_aware_greedy")


def train_ablation(cfg, topo, srcs, seed, episodes, kw):
    kw = dict(kw)
    nobj = kw.pop("_nobj", 3)
    rl = NET_RL.with_(**kw)
    mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
    agent = QLearningAgent(rl, n_objectives=nobj)
    train(agent, mdp, srcs, episodes=episodes, seed=seed)
    return LearnedPolicy(agent, rl, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--test-seeds", type=int, default=30)
    ap.add_argument("--load", type=float, default=60.0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.episodes, args.test_seeds = 4, 600, 10
    cfg = SimConfig()

    # ---------------- ablations ----------------
    rows = []
    with timer("E6 ablations"):
        for name, kw in ABLATIONS.items():
            for seed in range(args.seeds):
                rng = np.random.default_rng(3000 + seed)
                topo = deploy_topology(cfg.deployment, rng)
                srcs = select_sources(topo, cfg.traffic.n_sources, rng)
                pol = train_ablation(cfg, topo, srcs, seed, args.episodes, kw)
                res = NetworkSimulator(topo, cfg).run(
                    pol, srcs, rates=np.full(len(srcs), args.load)
                )
                rows.append({"ablation": name, "seed": seed, **summarise_flow(res)})
            sel = [r for r in rows if r["ablation"] == name]
            print(f"  {name:20s} PDR={np.mean([r['pdr_pct'] for r in sel]):6.2f}% "
                  f"EE={np.mean([r['energy_efficiency'] for r in sel]):7.2f}", flush=True)
    save_csv(rows, "e6_ablation.csv")

    # ---------------- head-to-head battery ----------------
    test_rows = []
    with timer("E6 head-to-head test battery"):
        for seed in range(args.test_seeds):
            rng = np.random.default_rng(5000 + seed)     # unseen test deployments
            topo = deploy_topology(cfg.deployment, rng)
            srcs = select_sources(topo, cfg.traffic.n_sources, rng)
            pols = {
                "ospf": OSPFStaticPolicy(cfg),
                "shortest_path": GreedyHopPolicy(cfg),
            }
            from dos_rl.policy import EnergyAwareThresholdPolicy

            pols["energy_aware_greedy"] = EnergyAwareThresholdPolicy(cfg)
            from dos_rl.agents.linear_fa import LinearQAgent

            for name, nobj, cls in (
                ("full", 3, QLearningAgent),
                ("no_dos", 1, QLearningAgent),
                ("linear_fa", 3, LinearQAgent),
            ):
                rl = NET_RL.with_(loop_free="dag")
                mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
                agent = cls(rl, n_objectives=nobj) if cls is QLearningAgent else cls(rl)
                train(agent, mdp, srcs, episodes=args.episodes, seed=seed)
                pols[name] = LearnedPolicy(agent, rl, cfg)
            for name, pol in pols.items():
                res = NetworkSimulator(topo, cfg).run(
                    pol, srcs, rates=np.full(len(srcs), args.load)
                )
                test_rows.append({"policy": name, "seed": seed, **summarise_flow(res)})
        print(f"  {len(test_rows)} test runs", flush=True)
    save_csv(test_rows, "e6_test_battery.csv")

    # ---------------- statistics ----------------
    names = sorted({r["policy"] for r in test_rows})
    stats = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in ("pdr_pct", "energy_efficiency", "e2e_delay_ms", "fairness"):
                da = {r["seed"]: r[key] for r in test_rows if r["policy"] == a}
                db = {r["seed"]: r[key] for r in test_rows if r["policy"] == b}
                seeds = sorted(set(da) & set(db))
                x = np.array([da[s] for s in seeds])
                y = np.array([db[s] for s in seeds])
                w = wilcoxon(x, y)
                stats.append({
                    "policy_a": a, "policy_b": b, "metric": key,
                    "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                    "delta": float((x - y).mean()),
                    "delta_ci95": float(1.96 * (x - y).std(ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else np.nan,
                    "p_value": w["p"], "rank_biserial": w["effect"],
                    "cliffs_delta": cliffs_delta(x, y), "n_pairs": w["n"],
                })
    save_csv(stats, "e6_statistics.csv")

    # ---------------- figures ----------------
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))
    abl = list(ABLATIONS)
    for ax, key, ylab, ttl in zip(
        axes,
        ("pdr_pct", "energy_efficiency", "fairness"),
        ("PDR (%)", "PDR per watt", "Jain fairness"),
        ("(a) Delivery ratio", "(b) Energy efficiency", "(c) Fairness"),
    ):
        ms, cis = [], []
        for name in abl:
            a = np.array([r[key] for r in rows if r["ablation"] == name])
            ms.append(a.mean())
            cis.append(1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)
        cs = [colour("dos_rl") if n == "full" else "#9aa7b4" for n in abl]
        ax.barh(range(len(abl)), ms, xerr=cis, color=cs, error_kw=dict(lw=0.9, capsize=2.5))
        ax.set_yticks(range(len(abl)))
        ax.set_yticklabels([n.replace("_", " ") for n in abl], fontsize=7)
        ax.set_xlabel(ylab)
        ax.set_title(ttl, fontsize=9)
        ax.invert_yaxis()
    fig.suptitle(f"E6: component ablation (closed-loop load {args.load:.0f} pkt/s)", fontsize=11)
    fig.savefig(fig_path("e6_ablation.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, key, ylab, ttl in zip(
        axes,
        ("pdr_pct", "energy_efficiency"),
        ("PDR (%)", "PDR per watt"),
        ("(a) Deployment battery: PDR", "(b) Deployment battery: energy efficiency"),
    ):
        ms, cis = [], []
        for n in names:
            a = np.array([r[key] for r in test_rows if r["policy"] == n])
            ms.append(a.mean())
            cis.append(1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)
        ax.bar(range(len(names)), ms, yerr=cis, color=[colour(n) for n in names],
               error_kw=dict(lw=0.9, capsize=2.5))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([label(n) for n in names], rotation=18, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9)
    fig.suptitle(f"E6: {args.test_seeds} deployments, trained per deployment", fontsize=11)
    fig.savefig(fig_path("e6_test_battery.png"), bbox_inches="tight")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
