"""Routing policies used during network evaluation.

Learned policies use controller telemetry; classical baselines use the
topology and, where applicable, relay energy and link status."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .belief import BeliefTracker
from .config import RLConfig, SimConfig
from .mdp import RoutingMDP
from .network_sim import NetworkSimulator, RouteEval
from .observer import ControllerObserver, LinkMetrics


def _telemetry(
    sim: NetworkSimulator,
    observer: ControllerObserver,
    mask: np.ndarray,
    queue_override: Optional[np.ndarray] = None,
) -> LinkMetrics:
    """Sample the controller's view of the network."""
    ev = sim.last_eval
    n = sim.n
    if ev is not None:
        load = np.clip(ev.utilisation, 0.0, 1.0)
        qf = np.clip(ev.queue_mean / max(sim.queue_capacity, 1), 0.0, 1.0)
    else:
        load = np.zeros(n)
        qf = np.ones(n) * 0.5
    if queue_override is not None:
        qf = np.clip(queue_override, 0.0, 1.0)
    energy = np.clip(sim.energy / max(sim.cfg.radio.e_init_j, 1e-9), 0.0, 1.0)
    return observer.observe(energy, load, qf, sim.prr, mask)


# ---------------------------------------------------------------------------
class _BasePolicy:
    name = "policy"

    def __init__(self, sim_cfg: SimConfig, obs_profile=None, seed: int = 0):
        self.sim_cfg = sim_cfg
        self.obs_profile = obs_profile
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def next_hops(self, sim: NetworkSimulator, sources, rates, mask) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
class OSPFStaticPolicy(_BasePolicy):
    """Static minimum-hop routing table, computed once at the start of a run.
    
    The table is not recomputed after failures; this is an OSPF-like baseline,
    not a full implementation of the OSPF protocol."""

    name = "ospf_static"

    def __init__(self, sim_cfg, obs_profile=None, seed: int = 0):
        super().__init__(sim_cfg, obs_profile, seed)
        self._table: Optional[np.ndarray] = None

    def reset(self):
        self._table = None

    def next_hops(self, sim, sources, rates, mask):
        n = sim.n
        if self._table is None:
            nxt = np.full(n, -1, dtype=np.int64)
            d_bs = sim.topo.dist_bs
            hops = sim.topo.hop_map(mask)
            for i in range(n):
                if not mask[i]:
                    continue
                cands = [
                    j
                    for j in sim.topo.neighbors[i]
                    if mask[j] and sim.link_up[i, j] and 0 < hops[j] < hops[i]
                ]
                if cands:
                    nxt[i] = min(cands, key=lambda j: (int(hops[j]), d_bs[j]))
            self._table = nxt
        # static table, but a dead next hop strands the traffic (no re-route)
        nxt = np.where(mask, self._table, -1)
        return sim.resolve(nxt, mask)


class GreedyHopPolicy(_BasePolicy):
    """Recompute minimum-hop next hops from the currently eligible relays."""

    name = "shortest_path"

    def next_hops(self, sim, sources, rates, mask):
        """Build next hops using BFS hop counts and gateway distance as a tie-breaker."""
        n = sim.n
        hops = sim.topo.hop_map(mask)
        d_bs = sim.topo.dist_bs
        nxt = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if not mask[i] or hops[i] <= 0:
                continue
            if d_bs[i] <= sim.cfg.deployment.comm_range_m:
                nxt[i] = -1                     # directly reaches the gateway
                continue
            cands = [
                j
                for j in sim.topo.neighbors[i]
                if mask[j] and sim.link_up[i, j] and 0 < hops[j] < hops[i]
            ]
            if cands:
                nxt[i] = min(cands, key=lambda j: (int(hops[j]), d_bs[j]))
        return sim.resolve(nxt, mask)


class EnergyAwareThresholdPolicy(_BasePolicy):
    """Prefer closer relays above a soft energy threshold.
    
    If none meet the threshold, fall back to other closer eligible relays."""

    name = "energy_aware_greedy"

    def __init__(self, sim_cfg, obs_profile=None, seed: int = 0, soft_threshold: float = 0.25):
        super().__init__(sim_cfg, obs_profile, seed)
        self.soft_threshold = soft_threshold

    def next_hops(self, sim, sources, rates, mask):
        n = sim.n
        frac = np.clip(sim.energy / max(sim.cfg.radio.e_init_j, 1e-9), 0.0, 1.0)
        healthy = frac > self.soft_threshold
        nxt = np.full(n, -1, dtype=np.int64)
        d_bs = sim.topo.dist_bs
        for i in range(n):
            if not mask[i]:
                continue
            cands = [j for j in sim.topo.neighbors[i] if mask[j] and sim.link_up[i, j]]
            closer = [j for j in cands if d_bs[j] < d_bs[i]]
            if not closer:
                continue          # nowhere strictly closer: this node cannot forward
            good = [j for j in closer if healthy[j]]
            pool = good if good else closer
            nxt[i] = min(pool, key=lambda j: d_bs[j])
        return sim.resolve(nxt, mask)


# ---------------------------------------------------------------------------
class LearnedPolicy(_BasePolicy):
    """Build a next-hop table using a trained agent and current telemetry."""

    name = "learned"

    def __init__(
        self,
        agent,
        rl: RLConfig,
        sim_cfg: SimConfig,
        state_mode: str = "node",
        obs_profile=None,
        seed: int = 0,
        online: bool = False,
        online_alpha: float = 0.25,
    ):
        super().__init__(sim_cfg, obs_profile, seed)
        self.agent = agent
        self.rl = rl
        self.state_mode = state_mode
        # The reported experiments use online=False: routes are replanned,
        # but the trained values are not updated during deployment.
        self.online = online
        self.online_alpha = online_alpha
        self._prev: Optional[dict] = None
        self._pdr_prev: Optional[float] = None
        self.tracker: Optional[BeliefTracker] = None
        self.observer: Optional[ControllerObserver] = None
        self.mdp: Optional[RoutingMDP] = None
        self.hops_cache: Optional[np.ndarray] = None
        self.greedy = True

    def bind(self, sim: NetworkSimulator) -> RoutingMDP:
        from .observer import PartialObsProfile

        if self.mdp is None:
            self.tracker = BeliefTracker(sim.topo)
            prof = self.obs_profile or PartialObsProfile("full")
            self.observer = prof.build(sim.topo, seed=self.seed)
            self.mdp = RoutingMDP(
                sim, self.sim_cfg, self.rl, self.tracker, state_mode=self.state_mode
            )
        return self.mdp

    def reset(self):
        if self.observer is not None:
            self.observer.reset()
        if self.tracker is not None:
            self.tracker.reset()

    def next_hops(self, sim, sources, rates, mask):
        mdp = self.bind(sim)
        m = _telemetry(sim, self.observer, mask)
        mdp.from_metrics(m)
        mdp.ep.hops = mdp.topo.hop_map(m.alive)
        n = sim.n
        nxt = np.full(n, -1, dtype=np.int64)
        chosen: dict = {}

        for i in range(n):
            if not mask[i] or not mdp.ep.eligible[i]:
                continue
            acts = mdp.actions(i)
            if not acts:
                continue
            state = mdp.encode(i)
            if hasattr(self.agent, "q_values"):        # linear FA
                feats = np.stack([mdp.ep.belief.node_features(i, a, gateway_prr=mdp._bs_prr(i)) for a in acts])
                idx = self.agent.act(feats, state, acts, self._rng, greedy=True)
            else:                                       # tabular
                idx = self.agent.greedy(state, acts)
            nxt[i] = acts[idx]
            chosen[i] = (state, acts, int(idx))

        nxt, hops = sim.resolve(nxt, mask)
        if self.online:
            self._online_update(mdp, chosen, hops, mask)
        self._prev = chosen if self.online else None
        return nxt, hops

    # ------------------------------------------------------------------
    def _online_update(self, mdp, chosen, hops, mask):
        """Tabular online Q-routing update from the realised outage.

        The reward a controller can actually compute online is the *outcome* of
        the route it just installed. We use the estimated delivery probability
        of the chosen path (from the simulator's path metrics) as a dense
        proxy: improving it raises the value of every chosen hop, and the
        terminal bonus fires when the delivery probability is high. This is the
        deployed-side analogue of the offline per-hop reward and lets the
        controller chase congestion and node death *after* deployment.
        """
        ev = getattr(mdp.sim, "last_eval", None)
        if ev is None or self._prev is None:
            return
        # realised success proxy: mean 1 - blocking over the chosen relays
        hot = np.array([ev.block_prob[i] if i < ev.block_prob.size else 0.0 for i in range(mdp.n)])
        p_now = float(np.mean(1.0 - hot[mask])) if mask.any() else 0.0
        if self._pdr_prev is None:       # first evaluated epoch: nothing to learn from yet
            self._pdr_prev = p_now
            return
        novelty = p_now - self._pdr_prev
        target_vec = np.array(
            [
                1.0 - float(np.mean(hot)),
                float(np.mean(1.0 - ev.utilisation)),
                float(np.mean(mdp.ep.belief.prr[np.arange(mdp.n), np.arange(mdp.n)] > 0)),
            ]
        )
        for i, (state, acts, idx) in self._prev.items():
            action = acts[idx]
            q = self.agent.q(state, [action])[0]
            tgt = target_vec + self.rl.gamma * np.maximum(q, 0.0) + novelty
            err = tgt - q
            for o in range(self.agent.n_obj):
                if hasattr(self.agent, "_write"):
                    self.agent._write(
                        state, action, o, q[o] + self.online_alpha * err[o], err[o] ** 2
                    )
        self._pdr_prev = p_now



# ---------------------------------------------------------------------------
class RandomPolicy(_BasePolicy):
    """Uniform random next hop -- sanity baseline."""

    name = "random"

    def next_hops(self, sim, sources, rates, mask):
        n = sim.n
        nxt = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if not mask[i]:
                continue
            cands = [j for j in sim.topo.neighbors[i] if mask[j] and sim.link_up[i, j]]
            if cands:
                nxt[i] = int(self._rng.choice(cands))
        return sim.resolve(nxt, mask)
