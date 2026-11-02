"""E9 - Qualitative / explanatory figures for the report.

These are the plots that answer rubric item 4 ("interpretation of results
showing how RL design helped"), as opposed to the raw benchmark curves:

* **fig_topology_route** - one deployment with the routes each policy installs,
  so the reader can see *what* the RL controller does differently (it spreads
  load across relays instead of funnelling everything onto the shortest tree);
* **fig_objective_selection** - which objective DOS selects as a function of
  node state, i.e. evidence that objective selection is state-dependent rather
  than a fixed scalarisation;
* **fig_energy_pareto** - energy consumed vs. PDR: the trade-off frontier;
* **fig_lifetime** - nodes alive over time for each policy (network lifetime);
* **fig_hotspot** - per-relay load distribution (the load-balancing story).

Run:  python -m experiments.make_figures
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.policy import EnergyAwareThresholdPolicy, GreedyHopPolicy, LearnedPolicy, OSPFStaticPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import train

from experiments.common import colour, fig_path, label, save_csv, setup_mpl, timer


def _fit(cfg, topo, srcs, seed, episodes=2000, nobj=3):
    rl = NET_RL
    mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
    agent = QLearningAgent(rl, n_objectives=nobj)
    train(agent, mdp, srcs, episodes=episodes, seed=seed)
    return agent, rl


def fig_topology(cfg, topo, srcs, agent, rl, lam, fname="fig_topology_route.png"):
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    sim = NetworkSimulator(topo, cfg)
    panels = [
        ("shortest_path", GreedyHopPolicy(cfg), "(a) Shortest path (OSPF-style)"),
        ("dos_rl", LearnedPolicy(agent, rl, cfg), "(b) DOS-RL (ours)"),
    ]
    for ax, (name, pol, ttl) in zip(axes[:2], panels):
        sim.reset()
        mask = sim.eligible()
        nxt, hops = pol.next_hops(sim, srcs, np.full(len(srcs), lam), mask)
        ev = sim.evaluate(nxt, hops, srcs, np.full(len(srcs), lam), mask)
        # links
        for i in range(sim.n):
            for j in topo.neighbors[i]:
                if j > i and sim.link_up[i, j]:
                    ax.plot([topo.positions[i, 0], topo.positions[j, 0]],
                            [topo.positions[i, 1], topo.positions[j, 1]],
                            color="#dddddd", lw=0.4, zorder=1)
        # chosen routes
        for s in srcs:
            cur, path = s, [s]
            for _ in range(sim.n):
                nx = nxt[cur]
                if nx < 0:
                    break
                path.append(int(nx))
                cur = int(nx)
            path.append("BS")
            xs = [topo.positions[p, 0] if p != "BS" else topo.bs_position[0] for p in path]
            ys = [topo.positions[p, 1] if p != "BS" else topo.bs_position[1] for p in path]
            ax.plot(xs, ys, lw=1.3, zorder=3)
        load = ev.lambda_in
        sc = ax.scatter(topo.positions[:, 0], topo.positions[:, 1], c=load, cmap="viridis",
                        s=26, zorder=4, vmin=0, vmax=max(float(load.max()), 1.0))
        ax.scatter([topo.bs_position[0]], [topo.bs_position[1]], marker="*", s=210,
                   color="#d62728", zorder=5, label="gateway")
        ax.scatter(topo.positions[srcs, 0], topo.positions[srcs, 1], marker="s", s=34,
                   facecolors="none", edgecolors="k", lw=0.9, zorder=6, label="sources")
        # reported PDR must come from the simulator's own end-to-end metric
        pdr = sim.run(pol, srcs, rates=np.full(len(srcs), lam)).pdr * 100
        ax.set_title(f"{ttl}\nPDR {pdr:.1f} %", fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(-40, cfg.deployment.area_m[0] + 40)
        ax.set_ylim(-40, cfg.deployment.area_m[1] + 40)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.035, label="offered load (pkt/s)")

    # (c) per-relay load distribution
    ax = axes[2]
    for name, pol in (("shortest_path", GreedyHopPolicy(cfg)), ("dos_rl", LearnedPolicy(agent, rl, cfg))):
        sim.reset()
        mask = sim.eligible()
        nxt, hops = pol.next_hops(sim, srcs, np.full(len(srcs), lam), mask)
        ev = sim.evaluate(nxt, hops, srcs, np.full(len(srcs), lam), mask)
        rel = ev.lambda_in[ev.lambda_in > 0]
        if rel.size:
            ax.hist(rel, bins=np.arange(0, max(rel.max(), lam) + lam / 4, lam / 4),
                    alpha=0.55, color=colour(name), label=label(name), density=True)
    ax.set_xlabel("offered load at a relay (pkt/s)")
    ax.set_ylabel("density")
    ax.set_title("(c) Load spread across relays", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Installed routing trees and per-relay load (offered load {lam:.0f} pkt/s)", fontsize=11)
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)


def fig_objective_selection(cfg, topo, srcs, agent, rl, fname="fig_objective_selection.png"):
    """Which objective does DOS pick, and does it depend on the state?"""
    plt = setup_mpl()
    rng = np.random.default_rng(7)
    counts = Counter()
    by_energy: dict = {}
    # Build the MDP once. Rebuilding it per sample used to re-randomise the
    # episode state for every decision, so the counts were accumulated over 400
    # *different* hypothetical networks rather than over trajectories in one.
    mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
    for _ in range(400):
        src = int(rng.choice(srcs))
        mdp.sample_episode(src, rng)
        node = src
        for _ in range(12):
            acts = mdp.actions(node)
            if not acts:
                break
            st = mdp.encode(node)
            obj = agent.select_objective(st, acts)
            e = float(mdp.ep.belief.energy[node])
            bucket = "low" if e < 0.35 else ("mid" if e < 0.7 else "high")
            counts[obj] += 1
            by_energy.setdefault(bucket, Counter())[obj] += 1
            idx = agent.greedy(st, acts)
            node = int(acts[idx])
            if node < 0:
                break
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    names = ["o1 energy", "o2 load", "o3 link quality"]
    vals = [counts.get(k, 0) for k in range(3)]
    axes[0].bar(names, vals, color=["#1f77b4", "#2ca02c", "#d62728"])
    total = max(sum(counts.values()), 1)
    axes[0].set_ylabel("decisions where objective selected")
    axes[0].set_title(f"(a) Objective chosen by DOS (n={total})", fontsize=9)
    for i, v in enumerate(vals):
        axes[0].annotate(f"{100*v/total:.0f}%", (i, v), ha="center", va="bottom", fontsize=8)
    axes[0].set_title("(a) Objective chosen by DOS", fontsize=9)
    order = ["low", "mid", "high"]
    bottom = np.zeros(len(order))
    for k in range(3):
        v = np.array([by_energy.get(b, Counter()).get(k, 0) for b in order], float)
        tot = np.maximum(v.sum() * 0 + np.array([sum(by_energy.get(b, Counter()).values()) for b in order]), 1)
        frac = v / tot
        axes[1].bar(order, frac, bottom=bottom, color=["#1f77b4", "#2ca02c", "#d62728"][k],
                    label=names[k])
        bottom += frac
    axes[1].set_ylabel("fraction of decisions")
    axes[1].set_xlabel("residual energy of the current relay")
    axes[1].set_title("(b) Selection vs. node state", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("DOS is dynamic, but variance dominance makes it ~a link-quality policy",
                 fontsize=10)
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)


def fig_pareto_and_lifetime(cfg, seeds=6, loads=(10, 30, 60, 90), episodes=1500):
    """Energy-vs-PDR frontier and network lifetime."""
    plt = setup_mpl()
    pts = {}
    life = {}
    for load in loads:
        for seed in range(seeds):
            rng = np.random.default_rng(3000 + seed)
            topo = deploy_topology(cfg.deployment, rng)
            srcs = select_sources(topo, cfg.traffic.n_sources, rng)
            agent, rl = _fit(cfg, topo, srcs, seed, episodes)
            for name, pol in (
                ("dos_rl", LearnedPolicy(agent, rl, cfg)),
                ("shortest_path", GreedyHopPolicy(cfg)),
                ("ospf", OSPFStaticPolicy(cfg)),
            ):
                sim = NetworkSimulator(topo, cfg)
                res = sim.run(pol, srcs, rates=np.full(len(srcs), float(load)))
                pts.setdefault((name, load), []).append((res.avg_power_w, res.pdr * 100))
                life.setdefault(name, []).append(
                    (sim.energy <= sim.energy_thr).sum() / sim.n
                )
        print(f"  pareto load {load} done", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax = axes[0]
    for name in ("dos_rl", "shortest_path", "ospf"):
        xs, ys, ls = [], [], []
        for load in loads:
            v = np.array(pts.get((name, load), []))
            if v.size:
                xs.append(v[:, 0].mean())
                ys.append(v[:, 1].mean())
                ls.append(str(load))
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.3, color=colour(name), label=label(name))
            for x, y, t in zip(xs, ys, ls):
                ax.annotate(t, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("average power drain (W)")
    ax.set_ylabel("PDR (%)")
    ax.set_title("(a) Energy-vs-delivery frontier (labels = offered load)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    names = ["dos_rl", "shortest_path", "ospf"]
    vals = [np.mean(life.get(n, [0])) * 100 for n in names]
    ax.bar(range(len(names)), vals, color=[colour(n) for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([label(n) for n in names], rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("relays depleted after 200 s (%)")
    ax.set_title("(b) Node depletion", fontsize=9)
    fig.suptitle("Energy/delivery trade-off of the closed-loop controllers", fontsize=11)
    fig.savefig(fig_path("fig_energy_pareto.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = SimConfig()
    seed = 0
    rng = np.random.default_rng(3000 + seed)
    topo = deploy_topology(cfg.deployment, rng)
    srcs = select_sources(topo, cfg.traffic.n_sources, rng)
    with timer("fit DOS-RL for qualitative figures"):
        agent, rl = _fit(cfg, topo, srcs, seed)
    with timer("fig_topology_route"):
        fig_topology(cfg, topo, srcs, agent, rl, lam=60.0)
    with timer("fig_objective_selection"):
        fig_objective_selection(cfg, topo, srcs, agent, rl)
    with timer("fig_energy_pareto"):
        fig_pareto_and_lifetime(cfg)
    print("figures written")


if __name__ == "__main__":
    main()
