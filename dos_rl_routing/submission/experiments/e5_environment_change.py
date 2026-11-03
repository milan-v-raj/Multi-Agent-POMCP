"""Evaluate traffic changes, temporary link outages and relay failures.

Post-onset measurements include the outage and subsequent recovery period."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.agents.tabular import QLearningAgent
from dos_rl.config import NET_RL
from dos_rl.network_sim import FailureEvent
from dos_rl.policy import GreedyHopPolicy, LearnedPolicy, OSPFStaticPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import train

from experiments.common import (
    adaptation_metrics,
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


def sweep_arrival(base, values, seeds, episodes, tag="arrival"):
    rows = []
    for lam in values:
        cfg = base.with_(pkt_rate_pps=float(lam))
        for seed in seeds:
            rng = np.random.default_rng(3000 + seed)
            topo = deploy_topology(cfg.deployment, rng)
            srcs = select_sources(topo, cfg.traffic.n_sources, rng)
            pols = build(cfg, topo, srcs, seed, episodes)
            for name, pol in pols.items():
                res = NetworkSimulator(topo, cfg).run(pol, srcs, rates=np.full(len(srcs), float(lam)))
                rows.append({"sweep": tag, "value": lam, "seed": seed, "policy": name,
                             **summarise_flow(res)})
        print(f"  arrival lambda={lam} done", flush=True)
    return rows


def sweep_failure(base, fractions, kind, seeds, episodes, t_fail=100.0, tag="link_failure",
                  duration_s: float = 40.0):
    rows, traces = [], {}
    for frac in fractions:
        for seed in seeds:
            rng = np.random.default_rng(3000 + seed)
            topo = deploy_topology(base.deployment, rng)
            srcs = select_sources(topo, base.traffic.n_sources, rng)
            pols = build(base, topo, srcs, seed, episodes)
            for name, pol in pols.items():
                sim = NetworkSimulator(topo, base)
                ev = [
                    FailureEvent(
                        time_s=t_fail, kind=kind, fraction=float(frac),
                        # seed mixes in the fraction, otherwise a 20 % outage
                        # is a strict subset of the 40 % outage (same RNG
                        # stream, same first draws) and the two conditions are
                        # not independent experiments
                        rng=np.random.default_rng([4242 + seed, int(frac * 1000)]),
                        duration_s=duration_s,     # transient outage
                    )
                ]
                res = sim.run(pol, srcs, rates=np.full(len(srcs), base.traffic.pkt_rate_pps), failures=ev)
                m = adaptation_metrics(res.time_t, res.pdr_t, t_fail) if res.time_t else {
                    "pdr_pre": np.nan, "pdr_post": np.nan, "dip_area": np.nan,
                    "recovery_s": np.nan, "pdr_min": np.nan,
                }
                rows.append(
                    {
                        "sweep": tag, "value": frac, "seed": seed, "policy": name,
                        "pdr_pre": m["pdr_pre"] * 100 if np.isfinite(m["pdr_pre"]) else np.nan,
                        "pdr_post": m["pdr_post"] * 100 if np.isfinite(m["pdr_post"]) else np.nan,
                        "pdr_min": m["pdr_min"] * 100 if np.isfinite(m["pdr_min"]) else np.nan,
                        "dip_area": m["dip_area"],
                        "recovery_s": m["recovery_s"],
                        **summarise_flow(res),
                    }
                )
                traces.setdefault((frac, name), []).append(
                    (np.array(res.time_t), np.array(res.pdr_t))
                )
        print(f"  {tag} fraction={frac} done", flush=True)
    return rows, traces


def sweep_repeated_outages(base, n_events, seeds, episodes, fraction=0.3, tag="repeated"):
    """A burst of transient outages - the realistic 'rain fade' stress test."""
    rows = []
    for n_ev in n_events:
        for seed in seeds:
            rng = np.random.default_rng(3000 + seed)
            topo = deploy_topology(base.deployment, rng)
            srcs = select_sources(topo, base.traffic.n_sources, rng)
            pols = build(base, topo, srcs, seed, episodes)
            for name, pol in pols.items():
                sim = NetworkSimulator(topo, base)
                evs = []
                for k in range(int(n_ev)):
                    evs.append(
                        FailureEvent(
                            time_s=20.0 + 20.0 * k,
                            kind="link" if k % 2 == 0 else "node",
                            fraction=fraction,
                            rng=np.random.default_rng(7000 + 17 * seed + k),
                            duration_s=10.0,
                        )
                    )
                res = sim.run(pol, srcs,
                              rates=np.full(len(srcs), base.traffic.pkt_rate_pps), failures=evs)
                rows.append({"sweep": tag, "value": int(n_ev), "seed": seed,
                             "policy": name, **summarise_flow(res)})
        print(f"  repeated_outages n={n_ev} done", flush=True)
    return rows


def plot_arrival(rows, fname="e5a_arrival_rate.png"):
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.3))
    metrics = [("pdr_pct", "PDR (%)", "(a) PDR vs arrival rate"),
               ("e2e_delay_ms", "E2E delay (ms)", "(b) Delay vs arrival rate"),
               ("energy_efficiency", "PDR per watt", "(c) Energy efficiency"),
               ("throughput_pps", "delivered (pkt/s)", "(d) Goodput")]
    for ax, (key, ylab, ttl) in zip(axes, metrics):
        for p in POLICIES:
            xs, ms, cis = [], [], []
            for v in sorted({r["value"] for r in rows}):
                a = np.array([r[key] for r in rows if r["value"] == v and r["policy"] == p])
                if a.size:
                    xs.append(v); ms.append(a.mean())
                    cis.append(1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)
            if xs:
                ax.errorbar(xs, ms, yerr=cis, marker="o", ms=3.2, lw=1.25, capsize=2.2,
                            color=colour(p), label=label(p))
        ax.set_xlabel("arrival rate per source (pkt/s)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("E5a: environmental change - arrival rate", fontsize=11)
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)


def plot_failure(rows, tag, fname, title):
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    for p in POLICIES:
        xs, pre, post, rec = [], [], [], []
        for v in sorted({r["value"] for r in rows if r["sweep"] == tag}):
            sel = [r for r in rows if r["sweep"] == tag and r["value"] == v and r["policy"] == p]
            if not sel:
                continue
            xs.append(v)
            pre.append(np.nanmean([r["pdr_pre"] for r in sel]))
            post.append(np.nanmean([r["pdr_post"] for r in sel]))
            rec.append(np.nanmean([r["recovery_s"] for r in sel]))
        if not xs:
            continue
        axes[0].plot(xs, pre, marker="o", ms=3.2, lw=1.2, color=colour(p), label=label(p))
        axes[1].plot(xs, post, marker="o", ms=3.2, lw=1.2, color=colour(p), label=label(p))
        axes[2].plot(xs, rec, marker="o", ms=3.2, lw=1.2, color=colour(p), label=label(p))
    axes[0].set_ylabel("PDR before failure (%)")
    axes[0].set_title("(a) Pre-failure delivery", fontsize=9)
    axes[1].set_ylabel("PDR after failure (%)")
    axes[1].set_title("(b) Post-failure delivery", fontsize=9)
    axes[2].set_ylabel("recovery time (s)")
    axes[2].set_title("(c) Time to recover", fontsize=9)
    for ax in axes:
        ax.set_xlabel("fraction of affected " + ("links" if "link" in tag else "relays"))
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)


def plot_traces(traces, frac, fname, title, t_fail=100.0):
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for p in POLICIES:
        key = (frac, p)
        if key not in traces:
            continue
        n = min(len(t) for t, _ in traces[key])
        tt = traces[key][0][0][:n]
        pp = np.mean([y[:n] for _, y in traces[key]], axis=0)
        ax.plot(tt, pp * 100, lw=1.4, color=colour(p), label=label(p))
    ax.axvline(t_fail, color="k", ls="--", lw=0.8)
    ax.text(t_fail + 2, ax.get_ylim()[1] * 0.95, "failure injected", fontsize=7)
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("PDR (%)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = range(args.seeds if not args.quick else 3)
    episodes = args.episodes if not args.quick else 500
    base = SimConfig()

    with timer("E5a arrival rate"):
        rows_a = sweep_arrival(base, [5, 10, 20, 40, 60, 90, 120], seeds, episodes)
    save_csv(rows_a, "e5a_arrival_rate.csv")
    plot_arrival(rows_a)

    with timer("E5b link failure (transient 40 s outage)"):
        rows_l, traces_l = sweep_failure(base, [0.0, 0.2, 0.4], "link", seeds, episodes,
                                         tag="link_failure", duration_s=40.0)
    save_csv(rows_l, "e5b_link_failure.csv")
    plot_failure(rows_l, "link_failure", "e5b_link_failure.png",
                 "E5b: 20 %/40 % of radio links lost for 40 s from t = 100 s")
    plot_traces(traces_l, 0.4, "e5b_link_failure_trace.png",
                "PDR trace under a 40 % link outage (t = 100-140 s)")

    with timer("E5c relay failure (transient 40 s outage)"):
        rows_n, traces_n = sweep_failure(base, [0.0, 0.1, 0.2, 0.3], "node", seeds, episodes,
                                         tag="node_failure", duration_s=40.0)
    save_csv(rows_n, "e5c_node_failure.csv")
    plot_failure(rows_n, "node_failure", "e5c_node_failure.png",
                 "E5c: 10 %/20 %/30 % of relays lost for 40 s from t = 100 s")
    plot_traces(traces_n, 0.3, "e5c_node_failure_trace.png",
                "PDR trace under a 30 % relay outage (t = 100-140 s)")

    # paired significance: RL vs shortest path on post-failure PDR
    stats = []
    for tag, rows in (("link_failure", rows_l), ("node_failure", rows_n)):
        for v in sorted({r["value"] for r in rows}):
            if v == 0.0:
                continue
            d = {}
            for r in rows:
                if r["value"] == v:
                    d.setdefault(r["seed"], {})[r["policy"]] = r["pdr_post"]
            diffs = np.array([x["dos_rl"] - x["shortest_path"] for x in d.values()
                              if "dos_rl" in x and "shortest_path" in x])
            diffs = diffs[np.isfinite(diffs)]
            w = wilcoxon(diffs, np.zeros_like(diffs))
            stats.append({"sweep": tag, "value": v, "comparison": "dos_rl - shortest_path",
                          "mean_delta_pp": float(diffs.mean()) if diffs.size else np.nan,
                          "p_value": w["p"], "effect_size": w["effect"], "n": w["n"]})
    with timer("E5d repeated transient outages"):
        rows_r = sweep_repeated_outages(base, [0, 2, 4, 6, 8], seeds, episodes)
    save_csv(rows_r, "e5d_repeated_outages.csv")

    # robustness under repeated churn: paired RL-vs-baseline PDR
    for v in sorted({r["value"] for r in rows_r}):
        d = {}
        for r in rows_r:
            if r["value"] == v:
                d.setdefault(r["seed"], {})[r["policy"]] = r["pdr_pct"]
        diffs = np.array([x["dos_rl"] - x["shortest_path"] for x in d.values()
                          if "dos_rl" in x and "shortest_path" in x])
        diffs = diffs[np.isfinite(diffs)]
        w = wilcoxon(diffs, np.zeros_like(diffs))
        stats.append({"sweep": "repeated_outages", "value": v,
                      "comparison": "dos_rl - shortest_path",
                      "mean_delta_pp": float(diffs.mean()) if diffs.size else np.nan,
                      "p_value": w["p"], "effect_size": w["effect"], "n": w["n"]})
    save_csv(stats, "e5_paired_tests.csv")

    plot_repeated(rows_r)
    print("figures written")


def plot_repeated(rows, fname="e5d_repeated_outages.png"):
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
    for key, ylab, ttl in (
        ("pdr_pct", "PDR (%)", "(a) Delivery under repeated outages"),
        ("fairness", "Jain fairness", "(b) Energy fairness"),
        ("mean_hops", "mean hops", "(c) Path length"),
    ):
        for p in POLICIES:
            xs, ms, cis = [], [], []
            for v in sorted({r["value"] for r in rows}):
                a = np.array([r[key] for r in rows if r["value"] == v and r["policy"] == p])
                if a.size:
                    xs.append(v); ms.append(a.mean())
                    cis.append(1.96 * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0)
            if xs:
                axes[0 if key == "pdr_pct" else (1 if key == "fairness" else 2)].errorbar(
                    xs, ms, yerr=cis, marker="o", ms=3.2, lw=1.25, capsize=2.2,
                    color=colour(p), label=label(p))
        (axes[0] if key == "pdr_pct" else (axes[1] if key == "fairness" else axes[2])).set_ylabel(ylab)
    for ax, ttl in zip(axes, ["(a) Delivery under repeated outages",
                              "(b) Energy fairness", "(c) Path length"]):
        ax.set_xlabel("number of outage events")
        ax.set_title(ttl, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("E5d: stress test - repeated 30 % transient outages", fontsize=11)
    fig.savefig(fig_path(fname), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
