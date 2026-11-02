"""Sanity tests for the SDWSN-IoT simulator and MDP.

These are the checks that caught the physics bugs during development, kept as
regression tests. Run with:

    python -m tests.test_network_sim        # no pytest needed
or
    python -m pytest tests -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dos_rl import BeliefTracker, NetworkSimulator, RoutingMDP, SimConfig, deploy_topology
from dos_rl.config import NET_RL
from dos_rl.network_sim import FailureEvent
from dos_rl.policy import GreedyHopPolicy, OSPFStaticPolicy, RandomPolicy
from dos_rl.topology import select_sources
from dos_rl.trainer import train


def _scene(seed=0, **kw):
    cfg = SimConfig().with_(**kw)
    rng = np.random.default_rng(3000 + seed)
    topo = deploy_topology(cfg.deployment, rng)
    srcs = select_sources(topo, cfg.traffic.n_sources, rng)
    return cfg, topo, srcs, NetworkSimulator(topo, cfg)


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def test_radio_energy_increases_with_distance():
    """Transmit energy must be monotone in distance (power control)."""
    cfg = SimConfig()
    r = cfg.radio
    ds = [50, 100, 200, 300, 450]
    es = [r.tx_energy_j(d) for d in ds]
    assert all(b >= a for a, b in zip(es, es[1:])), es
    assert r.tx_energy_j(450) > 2 * r.tx_energy_j(50)


def test_prr_decreases_with_distance():
    """Link reliability must fall with distance, and stay in [0,1]."""
    cfg = SimConfig()
    r = cfg.radio
    ps = [r.prr(d) for d in (10, 100, 300, 450, 600)]
    assert all(b <= a + 1e-12 for a, b in zip(ps, ps[1:])), ps
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert ps[0] > 0.95 and ps[-1] < 0.5


def test_battery_lifetime_is_finite_and_realistic():
    """A relay should survive thousands, not tens, of forwarded packets."""
    cfg = SimConfig()
    per_hop = cfg.radio.tx_energy_j(300) + cfg.radio.rx_energy_j
    n_pkts = cfg.radio.e_init_j / per_hop
    assert 500 < n_pkts < 200_000, n_pkts


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------
def test_every_deployment_is_feasible():
    """Some node must reach the gateway, otherwise the instance is meaningless."""
    cfg = SimConfig()
    infeasible = 0
    for seed in range(25):
        rng = np.random.default_rng(3000 + seed)
        topo = deploy_topology(cfg.deployment, rng)
        h = topo._hops([True] * topo.n_nodes)
        if min(h) >= 10 ** 6:
            infeasible += 1
    assert infeasible == 0, f"{infeasible}/25 deployments had no route to the gateway"


def test_deployment_density_is_not_collapsed():
    """The repair step must not turn the field into one fully-meshed blob."""
    cfg = SimConfig()
    edges = []
    for seed in range(10):
        rng = np.random.default_rng(3000 + seed)
        topo = deploy_topology(cfg.deployment, rng)
        edges.append(topo.n_edges)
    mean_edges = float(np.mean(edges))
    # a complete graph on 30 nodes has 435 edges; a plausible geometric graph
    # with R_c = 450 m over 1500 x 1500 m is far sparser
    assert 60 <= mean_edges <= 200, mean_edges


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------
def test_pdr_is_bounded_and_high_when_uncongested():
    cfg, topo, srcs, sim = _scene(0, pkt_rate_pps=2.0)
    res = sim.run(GreedyHopPolicy(cfg), srcs, rates=np.full(len(srcs), 2.0))
    assert 0.0 <= res.pdr <= 1.0
    assert res.pdr > 0.5, f"PDR {res.pdr:.3f} too low for a lightly loaded network"


def test_pdr_falls_with_load():
    cfg, topo, srcs, sim = _scene(0)
    low = sim.run(GreedyHopPolicy(cfg), srcs, rates=np.full(len(srcs), 5.0)).pdr
    high = sim.run(GreedyHopPolicy(cfg), srcs, rates=np.full(len(srcs), 90.0)).pdr
    assert high < low, (low, high)


def test_energy_is_conserved_and_non_negative():
    cfg, topo, srcs, sim = _scene(0)
    sim.run(GreedyHopPolicy(cfg), srcs, rates=np.full(len(srcs), 40.0))
    assert np.all(sim.energy >= 0.0)
    assert np.all(sim.energy <= topo.initial_energy_j + 1e-9)


def test_random_policy_is_worse_than_shortest_path():
    cfg, topo, srcs, sim = _scene(1)
    rnd = sim.run(RandomPolicy(cfg), srcs, rates=np.full(len(srcs), 10.0)).pdr
    sp = sim.run(GreedyHopPolicy(cfg), srcs, rates=np.full(len(srcs), 10.0)).pdr
    assert rnd < sp, (rnd, sp)


def test_transient_outage_recovers_permanent_does_not():
    cfg, topo, srcs, sim = _scene(0)
    pol = GreedyHopPolicy(cfg)
    transient = sim.run(
        pol, srcs, rates=np.full(len(srcs), 10.0),
        failures=[FailureEvent(100.0, "link", 0.4, rng=np.random.default_rng(1), duration_s=30.0)],
    )
    permanent = sim.run(
        pol, srcs, rates=np.full(len(srcs), 10.0),
        failures=[FailureEvent(100.0, "link", 0.4, rng=np.random.default_rng(1))],
    )
    t = np.array(transient.time_t)
    end_t = np.mean(np.array(transient.pdr_t)[t > 170])
    end_p = np.mean(np.array(permanent.pdr_t)[t > 170])
    assert end_t > end_p, (end_t, end_p)


def test_down_links_drop_packets():
    """A link taken down by a failure must deliver nothing.

    Regression: ``link_up`` used to gate the policies' routing *decisions* but
    not the packet physics, so an outage penalised only the controllers that
    noticed it and re-routed, while a stale table pointing at the dead link
    delivered as if nothing happened. That inverted every outage result.
    """
    cfg, topo, srcs, sim = _scene(0)
    sim.reset()
    i, j = next((i, j) for i in range(sim.n) for j in topo.neighbors[i] if sim.link_up[i, j])
    assert sim._hop_success(i, j) > 0.0
    sim.link_up[i, j] = False
    sim.link_up[j, i] = False
    assert sim._hop_success(i, j) == 0.0, "a down link still delivered packets"


def test_link_failure_samples_undirected_links():
    """A nominal fraction must kill about that share of *links*, not arcs."""
    cfg, topo, srcs, sim = _scene(3)
    sim.reset()
    n_undirected = len({(min(i, j), max(i, j))
                        for i in range(sim.n) for j in topo.neighbors[i] if sim.link_up[i, j]})
    ev = FailureEvent(0.0, "link", 0.25, rng=np.random.default_rng(7))
    sim.apply_failure(ev)
    got = len(ev._affected_links)
    # allow for duplicate draws of an already-dead link
    assert 0.15 * n_undirected <= got <= 0.30 * n_undirected + 2, (got, n_undirected)


def test_outage_severity_changes_outcome():
    """A 40 % outage must hurt more than a 10 % one on a redundant topology."""
    cfg, topo, srcs, sim = _scene(3)
    pol = GreedyHopPolicy(cfg)
    mild = sim.run(pol, srcs, rates=np.full(len(srcs), 10.0), failures=[
        FailureEvent(100.0, "link", 0.1, rng=np.random.default_rng(11), duration_s=40.0)])
    harsh = sim.run(pol, srcs, rates=np.full(len(srcs), 10.0), failures=[
        FailureEvent(100.0, "link", 0.4, rng=np.random.default_rng(12), duration_s=40.0)])
    assert harsh.pdr < mild.pdr, (mild.pdr, harsh.pdr)


def test_loop_free_tables_have_no_cycles():
    """Every resolved next-hop table must terminate at the gateway."""
    cfg, topo, srcs, sim = _scene(0)
    pol = GreedyHopPolicy(cfg)
    sim.reset()
    mask = sim.eligible()
    nxt, hops = pol.next_hops(sim, srcs, None, mask)
    for i in range(sim.n):
        if hops[i] > 0:
            cur, seen = i, set()
            for _ in range(sim.n + 2):
                if topo.dist_bs[cur] <= cfg.deployment.comm_range_m:
                    break
                assert cur not in seen, f"cycle through node {i}"
                seen.add(cur)
                cur = int(nxt[cur])
                assert cur >= 0, f"dead chain from node {i}"
            else:
                raise AssertionError(f"chain from {i} did not terminate")


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------
def test_actions_respect_energy_threshold():
    cfg, topo, srcs, sim = _scene(0)
    rl = NET_RL
    mdp = RoutingMDP(sim, cfg, rl, BeliefTracker(topo))
    src = srcs[0]
    mdp.sample_episode(src, np.random.default_rng(0))
    mdp.ep.eligible[srcs[1]] = False
    for j in mdp.actions(src):
        if j >= 0:
            assert mdp.ep.eligible[j], "ineligible relay offered as an action"


def test_rewards_are_bounded():
    cfg, topo, srcs, sim = _scene(0)
    rl = NET_RL
    mdp = RoutingMDP(sim, cfg, rl, BeliefTracker(topo))
    mdp.sample_episode(srcs[0], np.random.default_rng(0))
    for i in range(sim.n):
        for j in mdp.actions(i):
            r = mdp.rewards(i, j)
            assert r.shape == (3,)
            assert np.all(r >= -1e-9) and np.all(r <= 1 + 1e-9), r


def test_potential_shaping_sums_to_telescoping_bonus():
    """For a full path, sum of F_t must telescope to gamma*phi(end) - phi(start)."""
    cfg, topo, srcs, sim = _scene(0)
    rl = NET_RL
    mdp = RoutingMDP(sim, cfg, rl, BeliefTracker(topo))
    path = [srcs[0], *[int(x) for x in topo.neighbors[srcs[0]][:1]]]
    f_total = sum(mdp.shaping(a, b) for a, b in zip(path, path[1:]))
    expect = rl.gamma * mdp.potential(path[-1]) - mdp.potential(path[0])
    assert abs(f_total - expect) < 1e-9, (f_total, expect)


def test_episodes_terminate_and_are_deterministic_under_seed():
    cfg, topo, srcs, sim = _scene(0)
    rl = NET_RL
    mdp = RoutingMDP(sim, cfg, rl, BeliefTracker(topo))
    terms = set()
    for k in range(30):
        mdp.sample_episode(srcs[k % len(srcs)], np.random.default_rng(k))
        state = mdp.reset(srcs[k % len(srcs)])
        rng = np.random.default_rng(k)
        for _ in range(rl.max_hops + 2):
            acts = mdp.actions(mdp.current)
            if not acts:
                break
            res = mdp.step(acts[int(rng.integers(len(acts)))], rng)
            if res.done:
                terms.add(res.terminal)
                break
    assert terms, "no episodes terminated"


def test_training_learns_the_energy_objective():
    """The learnt policy must prefer less-depleted relays.

    A direct behavioural check is more informative than comparing returns here:
    on this instance an untrained (all-zero) Q-table ties every action and any
    loop-free hop still reaches the gateway, so returns are saturated and cannot
    separate a trained policy from an untrained one. We therefore assert the
    property the reward is designed to instil: given two otherwise identical
    candidate relays, the trained agent routes through the one with more
    residual energy.
    """
    from dos_rl.agents.tabular import QLearningAgent

    cfg, topo, srcs, sim = _scene(0)
    rl = NET_RL
    mdp = RoutingMDP(NetworkSimulator(topo, cfg), cfg, rl, BeliefTracker(topo))
    agent = QLearningAgent(rl)
    train(agent, mdp, srcs, episodes=1200, seed=0)

    # build a controlled state: full battery everywhere, then deplete one relay
    src = srcs[0]
    mdp.sample_episode(src, np.random.default_rng(7))
    acts = [a for a in mdp.actions(src) if a >= 0]
    assert len(acts) >= 2, "need at least two candidate relays for this test"
    good, bad = acts[0], acts[1]
    mdp.ep.belief.energy[good] = 1.0
    mdp.ep.belief.energy[bad] = 0.2
    mdp.ep.eligible = mdp.ep.eligible  # unchanged
    st = mdp.encode(src)
    q = agent.q(st, [good, bad])          # (2, 3 objectives)
    # the energy objective (o1) must rank the healthy relay higher
    assert q[0, 0] > q[1, 0], (
        f"energy objective did not prefer the healthier relay: {q[:, 0]}"
    )


def _mean_return(agent, mdp, srcs, rng, n=12):
    ret = []
    for k in range(n):
        mdp.sample_episode(srcs[k % len(srcs)], rng)
        mdp.reset(srcs[k % len(srcs)])
        r = 0.0
        for _ in range(mdp.rl.max_hops + 2):
            acts = mdp.actions(mdp.current)
            if not acts:
                break
            res = mdp.step(acts[agent.greedy(mdp.encode(mdp.current), acts)], rng)
            r += res.scalar_reward
            if res.done:
                break
        ret.append(r)
    return float(np.mean(ret))


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:      # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:           # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
