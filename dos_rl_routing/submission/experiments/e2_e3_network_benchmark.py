"""Compare routing policies across offered load, source-count and density sweeps.

Policies share the same deployments within each seed for paired comparisons."""

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
    colour,
    fig_path,
    label,
    save_csv,
    setup_mpl,
    summarise_flow,
    timer,
    wilcoxon,
)

POLICIES = ("dos_rl", "scalar", "ospf", "shortest_path")


def build(cfg, topo, srcs, seed, episodes):
    out = {"ospf": OSPFStaticPolicy(cfg), "shortest_path": GreedyHopPolicy(cfg)}
    for name, nobj in (("dos_rl", 3), ("scalar", 1)):
        rl = NET_RL
        mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
        agent = QLearningAgent(rl, n_objectives=nobj)
        train(agent, mdp, srcs, episodes=episodes, seed=seed)
        out[name] = LearnedPolicy(agent, rl, cfg)
    return out


def run_sweep(base, values, key, seeds, episodes, rates_fn, tag):
    rows = []
    for v in values:
        cfg = base.with_(**{key: v})
        for seed in seeds:
            rng = np.random.default_rng(3000 + seed)
            topo = deploy_topology(cfg.deployment, rng)
            srcs = select_sources(topo, cfg.traffic.n_sources, rng)
            pols = build(cfg, topo, srcs, seed, episodes)
            lam = rates_fn(v)
            rates = np.full(len(srcs), lam)
            for name, pol in pols.items():
                res = NetworkSimulator(topo, cfg).run(pol, srcs, rates=rates)
                rows.append(
                    {
                        "sweep": tag,
                        "value": v,
                        "seed": seed,
                        "policy": name,
                        "lambda_pps": lam,
                        **summarise_flow(res),
                    }
                )
        print(f"  {tag}: {key}={v} done", flush=True)
    return rows


def _series(rows, sweep, key, policy):
    vals = sorted({r["value"] for r in rows if r["sweep"] == sweep})
    xs, ms, cis = [], [], []
    for v in vals:
        a = np.array([r[key] for r in rows if r["sweep"] == sweep and r["value"] == v and r["policy"] == policy])
        if a.size == 0:
            continue
        xs.append(v)
        ms.append(a.mean())
        cis.append(1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)
    return np.array(xs), np.array(ms), np.array(cis)


def _paired(rows, sweep, key, policy, ref="shortest_path"):
    """Paired per-seed differences (policy - reference) for one sweep."""
    vals = sorted({r["value"] for r in rows if r["sweep"] == sweep})
    out = {}
    for v in vals:
        d = {}
        for r in rows:
            if r["sweep"] == sweep and r["value"] == v:
                d.setdefault(r["seed"], {})[r["policy"]] = r[key]
        diffs = [x[policy] - x[ref] for x in d.values() if policy in x and ref in x]
        out[v] = np.array(diffs)
    return out


def plot_sweep(rows, sweep, xlabel, fname, title, delta_key="pdr_pct"):
    plt = setup_mpl()
    metrics = [
        ("pdr_pct", "PDR (%)", "(a) Packet delivery ratio"),
        ("e2e_delay_ms", "E2E delay (ms)", "(b) End-to-end delay"),
        ("energy_efficiency", "PDR per watt", "(c) Energy efficiency"),
        ("fairness", "Jain fairness index", "(d) Fairness of energy drain"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.0))
    for j, (key, ylab, ttl) in enumerate(metrics):
        ax = axes[0, j]
        for p in POLICIES:
            xs, ms, cis = _series(rows, sweep, key, p)
            if xs.size == 0:
                continue
            ax.errorbar(xs, ms, yerr=cis, marker="o", ms=3.2, lw=1.25, capsize=2.2,
                        color=colour(p), label=label(p))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9)
        if j == 0:
            ax.legend(frameon=False, fontsize=7)

    # incremental benefit vs the adaptive shortest-path baseline
    ax = axes[1, 0]
    for p in ("dos_rl", "scalar", "ospf"):
        d = _paired(rows, sweep, delta_key, p)
        xs = np.array(sorted(d))
        ms = np.array([d[v].mean() if d[v].size else np.nan for v in xs])
        cis = np.array([
            1.96 * d[v].std(ddof=1) / np.sqrt(d[v].size) if d[v].size > 1 else 0.0 for v in xs
        ])
        ax.errorbar(xs, ms, yerr=cis, marker="s", ms=3.2, lw=1.25, capsize=2.2,
                    color=colour(p), label=label(p))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDR gain vs shortest path (pp)")
    ax.set_title("(e) Paired PDR gain vs baseline", fontsize=9)

    ax = axes[1, 1]
    for p in ("dos_rl", "scalar"):
        d = _paired(rows, sweep, "energy_efficiency", p)
        xs = np.array(sorted(d))
        ms = np.array([d[v].mean() if d[v].size else np.nan for v in xs])
        ax.errorbar(xs, ms, marker="s", ms=3.2, lw=1.25, color=colour(p), label=label(p))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EE gain vs shortest path")
    ax.set_title("(f) Paired energy-efficiency gain", fontsize=9)

    ax = axes[1, 2]
    for p in ("dos_rl", "scalar", "ospf", "shortest_path"):
        xs, ms, cis = _series(rows, sweep, "mean_hops", p)
        if xs.size:
            ax.errorbar(xs, ms, yerr=cis, marker="o", ms=3.2, lw=1.25, color=colour(p))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean hops per packet")
    ax.set_title("(g) Path length", fontsize=9)

    ax = axes[1, 3]
    for p in ("dos_rl", "scalar", "ospf", "shortest_path"):
        xs, ms, cis = _series(rows, sweep, "mean_attempts", p)
        if xs.size:
            ax.errorbar(xs, ms, yerr=cis, marker="o", ms=3.2, lw=1.25, color=colour(p))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAC attempts per packet")
    ax.set_title("(h) Retransmission effort", fontsize=9)

    fig.suptitle(title, fontsize=11)
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = range(args.seeds if not args.quick else 3)
    episodes = args.episodes if not args.quick else 500
    base = SimConfig()

    rows = []
    with timer("E2a: offered-load sweep"):
        rows += run_sweep(base, [10, 20, 30, 45, 60, 80], "pkt_rate_pps", seeds,
                          episodes, lambda v: float(v), "load")
    with timer("E2b: traffic-source sweep"):
        rows += run_sweep(base.with_(pkt_rate_pps=10.0), [2, 4, 6, 8, 10], "n_sources",
                          seeds, episodes, lambda v: 10.0, "sources")
    with timer("E3: relay-density sweep"):
        rows += run_sweep(base.with_(pkt_rate_pps=10.0), [30, 45, 60], "n_nodes",
                          seeds, episodes, lambda v: 10.0, "nodes")

    save_csv(rows, "e2_e3_raw.csv")

    # paired significance for the headline load comparison
    stats = []
    for v in (30, 45, 60, 80):
        for p in ("dos_rl", "scalar"):
            d = _paired(rows, "load", "pdr_pct", p)[v]
            w = wilcoxon(d + 0.0, np.zeros_like(d))
            stats.append(
                {
                    "sweep": "load",
                    "value": v,
                    "policy": p,
                    "mean_delta_pp": float(d.mean()) if d.size else np.nan,
                    "ci95_pp": float(1.96 * d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else np.nan,
                    "p_value": w["p"],
                    "effect_size": w["effect"],
                    "n_pairs": w["n"],
                }
            )
    save_csv(stats, "e2_e3_paired_tests.csv")

    plot_sweep(rows, "load", "offered load per source (pkt/s)", "e2a_offered_load.png",
               f"Effect of offered load (30 relays, {base.traffic.n_sources} sources)")
    plot_sweep(rows, "sources", "number of traffic sources", "e2b_traffic_sources.png",
               "Effect of traffic-source count (30 relays, 10 pkt/s)")
    plot_sweep(rows, "nodes", "number of relay nodes", "e3_node_density.png",
               f"Effect of relay-node density ({base.traffic.n_sources} sources, 10 pkt/s)")
    print("figures written")


if __name__ == "__main__":
    main()
