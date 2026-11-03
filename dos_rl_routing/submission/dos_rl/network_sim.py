"""Analytical flow simulator for routing policies.

The model estimates queue blocking, link success, energy and delay. It is not
a packet-level simulation or a measurement from a physical deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SimConfig
from .energy import RadioModel
from .topology import Topology


# ---------------------------------------------------------------------------
# Results containers
# ---------------------------------------------------------------------------
@dataclass
class RouteEval:
    """Per-node outcome of one evaluation window."""

    lambda_in: np.ndarray          # offered packets/s
    lambda_out: np.ndarray         # accepted (non-dropped) packets/s
    block_prob: np.ndarray         # M/M/1/K blocking probability
    utilisation: np.ndarray        # rho
    queue_mean: np.ndarray         # E[queue length]
    queue_delay_s: np.ndarray      # E[sojourn time]
    attempts: np.ndarray           # expected MAC attempts per departure
    tx_rate: np.ndarray            # transmissions/s
    rx_rate: np.ndarray            # receptions/s
    power_w: np.ndarray            # energy drain (W)
    hops_to_bs: np.ndarray         # hop count through chosen next hops
    next_hop: np.ndarray           # chosen next hop (-1 = none)
    dead: np.ndarray               # loops / unreachable / depleted
    drop_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class FlowResult:
    """Aggregate network metrics for one simulation run."""

    pdr: float
    e2e_delay_s: float
    energy_efficiency: float       # PDR fraction / total average network power (W)
    avg_power_w: float
    total_energy_j: float
    fairness: float                # Jain index over per-node energy drain
    load_imbalance: float          # std/mean of node utilisation
    mean_hops: float
    mean_attempts: float
    throughput_pps: float
    # time series, for lifetime / adaptation plots
    pdr_t: List[float] = field(default_factory=list)
    time_t: List[float] = field(default_factory=list)
    energy_t: List[np.ndarray] = field(default_factory=list)
    replan_pdr: List[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pdr": self.pdr,
            "e2e_delay_s": self.e2e_delay_s,
            "energy_efficiency": self.energy_efficiency,
            "avg_power_w": self.avg_power_w,
            "total_energy_j": self.total_energy_j,
            "fairness": self.fairness,
            "load_imbalance": self.load_imbalance,
            "mean_hops": self.mean_hops,
            "mean_attempts": self.mean_attempts,
            "throughput_pps": self.throughput_pps,
        }


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------
@dataclass
class FailureEvent:
    """A link or node failure, either temporary or permanent."""

    time_s: float
    kind: str                       # "link" | "node" | "restore"
    fraction: float = 0.3           # fraction of links/nodes affected
    rng: Optional[np.random.Generator] = None
    duration_s: Optional[float] = None   # None -> permanent
    # internal bookkeeping (filled by the simulator)
    _affected_links: Optional[list] = None
    _affected_nodes: Optional[list] = None


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class NetworkSimulator:
    """Closed-loop evaluator of routing policies."""

    def __init__(self, topo: Topology, cfg: SimConfig):
        self.topo = topo
        self.cfg = cfg
        self.radio = RadioModel(cfg.radio)

        n = topo.n_nodes
        self.n = n
        # nominal (failure-free, initial-energy) state
        self.energy = np.asarray(topo.initial_energy_j, dtype=float).copy()
        self.energy_thr = float(topo.energy_threshold_j)
        self.link_up = np.zeros((n, n), dtype=bool)
        self.prr = np.zeros((n, n), dtype=float)
        r = cfg.deployment.comm_range_m
        for i in range(n):
            for j in topo.neighbors[i]:
                if topo.dist[i, j] <= r:
                    self.link_up[i, j] = True
                    self.prr[i, j] = self.radio.prr(float(topo.dist[i, j]))
        self.node_up = np.ones(n, dtype=bool)
        self.queue_capacity = cfg.traffic.buffer_pkts
        # service time = frame airtime + turnaround + inter-frame gap
        self.serve_time_s = cfg.radio.frame_time_s + 2.5e-3
        self.traffic_mult = np.ones(n)
        # last evaluation, used by controllers that need an occupancy estimate
        self.last_eval: Optional[RouteEval] = None

    # -- public API --------------------------------------------------------
    def reset(self):
        n = self.n
        self.energy = np.asarray(self.topo.initial_energy_j, dtype=float).copy()
        self.node_up[:] = True
        self.last_eval = None
        r = self.cfg.deployment.comm_range_m
        self.link_up[:] = False
        self.prr[:] = 0.0
        for i in range(n):
            for j in self.topo.neighbors[i]:
                if self.topo.dist[i, j] <= r:
                    self.link_up[i, j] = True
                    self.prr[i, j] = self.radio.prr(float(self.topo.dist[i, j]))

    def eligible(self) -> np.ndarray:
        return self.node_up & (self.energy > self.energy_thr)

    def available_neighbors(self, i: int, mask: np.ndarray) -> List[int]:
        return [j for j in self.topo.neighbors[i] if self.link_up[i, j] and mask[j]]

    def apply_failure(self, ev: FailureEvent):
        """Kill a random fraction of links or nodes."""
        rng = ev.rng if ev.rng is not None else np.random.default_rng(0)
        if ev.kind == "link":
            pairs = [
                (i, j)
                for i in range(self.n)
                for j in self.topo.neighbors[i]
                if self.link_up[i, j]
            ]
            if not pairs:
                return
            # Draw whole links (unordered pairs), not directed arcs. Choosing
            # from the directed list and then clearing both directions means a
            # nominal ``fraction`` kills roughly half that many links and, worse,
            # biases the sample towards pairs that were listed twice.
            undirected = sorted({(min(i, j), max(i, j)) for i, j in pairs})
            k = max(1, int(round(ev.fraction * len(undirected))))
            killed = []
            for idx in rng.choice(len(undirected), size=min(k, len(undirected)), replace=False):
                i, j = undirected[int(idx)]
                if not self.link_up[i, j]:
                    continue
                self.link_up[i, j] = False
                self.link_up[j, i] = False
                killed.append((i, j))
            ev._affected_links = killed
        elif ev.kind == "node":
            cand = np.where(self.node_up)[0]
            k = max(1, int(round(ev.fraction * cand.size)))
            dead = rng.choice(cand, size=min(k, cand.size), replace=False)
            self.node_up[dead] = False
            ev._affected_nodes = [int(x) for x in dead]
        elif ev.kind == "restore":
            self.node_up[:] = True

    def restore_failure(self, ev: FailureEvent):
        """Undo a transient outage (links come back, nodes rejoin)."""
        if ev._affected_links:
            for i, j in ev._affected_links:
                self.link_up[i, j] = True
                self.link_up[j, i] = True
        if ev._affected_nodes:
            for i in ev._affected_nodes:
                self.node_up[i] = True

    # -- routing -----------------------------------------------------------
    def next_hop_table(
        self,
        action_fn: Callable[[int, np.ndarray, np.ndarray], int],
        mask: np.ndarray,
        energy_est: Optional[np.ndarray] = None,
        queue_est: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ask a policy for one next hop per node, then resolve loops.

        ``action_fn(node, eligible_mask, info) -> next_hop`` must return -1 when
        it has no action. Returned tables are loop-checked against the gateway:
        any node whose chain does not terminate at the BS within ``n`` steps is
        marked dead (its traffic is dropped) -- this is the classic
        Q-routing loop pathology and it is charged to the policy, not hidden.
        """
        n = self.n
        nxt = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if not mask[i]:
                continue
            a = action_fn(i, mask, None)
            if a is not None and a >= 0 and self.link_up[i, a] and mask[a]:
                nxt[i] = int(a)
        return self.resolve(nxt, mask)

    def resolve(self, nxt: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Loop-check a next-hop table and compute hop distance to the gateway.

        Any node whose forwarding chain does not terminate at the base station
        is declared dead (traffic dropped) and its next hop cleared. This is the
        standard Q-routing loop pathology; charging it to the policy is more
        honest than silently repairing it.
        """
        n = self.n
        nxt = np.array(nxt, dtype=np.int64, copy=True)
        reach_bs = self.topo.dist_bs <= self.cfg.deployment.comm_range_m
        hops = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if mask[i] and reach_bs[i]:
                hops[i] = 1

        for i in range(n):
            if not mask[i] or hops[i] > 0:
                continue
            seen = {i}
            cur = i
            chain = [i]
            ok = False
            base = 0
            for _ in range(n):
                nx = nxt[cur]
                if nx < 0:
                    break
                if hops[nx] > 0:
                    ok = True
                    base = int(hops[nx])
                    break
                if nx in seen:
                    break
                seen.add(nx)
                chain.append(nx)
                cur = nx
            if ok:
                for depth, node in enumerate(reversed(chain), start=1):
                    hops[node] = base + depth
            else:
                for node in chain:
                    nxt[node] = -1
                    hops[node] = -1
        return nxt, hops

    # -- analytic flow evaluation -----------------------------------------
    def evaluate(
        self,
        nxt: np.ndarray,
        hops: np.ndarray,
        sources: Sequence[int],
        rates: np.ndarray,
        mask: np.ndarray,
    ) -> RouteEval:
        n = self.n
        agg = max(1.0, float(self.cfg.traffic.agg_factor))
        # node traffic intensity scales with aggregation: a relay closer to the
        # sink forwards fused data from many leaves, so it sees the load of
        # several sources rather than one.
        y = 2.0 * self.topo.dist_bs / max(self.cfg.deployment.comm_range_m, 1e-9)
        self.traffic_mult = 1.0 + (agg - 1.0) * np.exp(-np.maximum(y, 0.0))
        lam = np.zeros(n, dtype=float)
        for s, r in zip(sources, rates):
            if mask[s] and hops[s] > 0:
                lam[s] += r

        # propagate offered load downstream in reverse hop order
        order = np.argsort(-hops, kind="stable")
        lam_in = np.zeros(n, dtype=float)
        for i in order:
            if hops[i] <= 0:
                continue
            lam_in[i] += lam[i]
            # what fraction of i's traffic actually departs is decided by its own
            # queue; do a two-pass solve below, so here we accumulate offered load
        # resolve queue blocking iteratively (blocking shrinks upstream load)
        block = np.zeros(n, dtype=float)
        accepted = lam.copy()
        for _ in range(6):
            offered = np.zeros(n, dtype=float)
            for s, r in zip(sources, rates):
                if mask[s] and hops[s] > 0:
                    offered[s] += r
            for i in order:
                if hops[i] <= 0:
                    continue
                acc = offered[i] * (1.0 - block[i])
                nx = nxt[i]
                if nx >= 0:
                    offered[nx] += acc * self.traffic_mult[nx]
            mu = 1.0 / self.serve_time_s
            rho = offered / mu
            rho = np.clip(rho, 0.0, 0.9995)
            k = self.queue_capacity
            p0 = np.where(
                np.abs(rho - 1.0) < 1e-9,
                1.0 / (k + 1.0),
                (1.0 - rho) / (1.0 - rho ** (k + 1)),
            )
            block = np.where(hops > 0, np.clip(p0 * rho ** k, 0.0, 1.0), 0.0)
            block[~mask] = 1.0
            accepted = offered * (1.0 - block)

        mu = 1.0 / self.serve_time_s
        # ``offered`` is the aggregated downstream load actually seen by each
        # relay; report it as lambda_in. (An earlier version reported only each
        # node's own generated traffic, which made every relay look identically
        # loaded and hid the whole load-balancing story in the figures.)
        lam_in = offered.copy()
        rho = np.clip(offered / mu, 0.0, 0.9995)
        k = self.queue_capacity
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.where(
                np.abs(rho - 1.0) < 1e-9,
                k / 2.0,
                rho / (1.0 - rho) - (k + 1.0) * rho ** (k + 1) / (1.0 - rho ** (k + 1)),
            )
            # Little's law over the accepted stream
            q_delay = np.where(accepted > 1e-9, L / np.maximum(accepted, 1e-9), 0.0)
        q_delay = np.nan_to_num(q_delay, nan=0.0, posinf=k * self.serve_time_s)

        # Expected attempts for independent transmissions that stop after the
        # first success: sum of failure probabilities for attempts 0..R-1.
        R = self.radio.max_retx
        attempts = np.ones(n, dtype=float)
        hop_dist = np.zeros(n, dtype=float)
        for i in range(n):
            if hops[i] <= 0:
                continue
            if nxt[i] >= 0:
                hop_dist[i] = float(self.topo.dist[i, nxt[i]])
                p = float(np.clip(self.prr[i, nxt[i]], 1e-6, 1.0))
            else:                              # direct link to the base station
                hop_dist[i] = float(self.topo.dist_bs[i])
                p = float(np.clip(self.radio.prr(hop_dist[i]), 1e-6, 1.0))
            # Attempt k is reached only if the preceding k attempts failed.
            attempts[i] = sum((1.0 - p) ** k for k in range(R))

        # --- energy accounting ------------------------------------------
        tx_rate = np.zeros(n)
        rx_rate = np.zeros(n)
        power = np.zeros(n)
        e_rx = self.radio.e_rx
        for i in range(n):
            if hops[i] <= 0 or accepted[i] <= 0:
                continue
            tx_rate[i] = accepted[i] * attempts[i]
            power[i] += tx_rate[i] * self.radio.tx_energy(hop_dist[i])
            if nxt[i] >= 0:
                rx_rate[nxt[i]] += tx_rate[i] * self.prr[i, nxt[i]]
            else:
                pass  # base station is mains-powered
        power += rx_rate * e_rx

        dead = (~mask) | (hops <= 0)
        return RouteEval(
            lambda_in=lam_in,
            lambda_out=accepted,
            block_prob=block,
            utilisation=rho,
            queue_mean=L,
            queue_delay_s=q_delay,
            attempts=attempts,
            tx_rate=tx_rate,
            rx_rate=rx_rate,
            power_w=power,
            hops_to_bs=hops,
            next_hop=nxt,
            dead=dead,
        )

    # -- end-to-end path metrics ------------------------------------------
    def path_metrics(
        self, nxt: np.ndarray, hops: np.ndarray, ev: RouteEval, sources: Sequence[int]
    ) -> Tuple[float, float, float]:
        """(mean delivery probability, mean E2E delay, mean attempts) over sources."""
        probs, delays, atts = [], [], []
        for s in sources:
            if hops[s] <= 0:
                probs.append(0.0)
                delays.append(np.nan)
                atts.append(self.radio.max_retx)
                continue
            p = 1.0
            d = 0.0
            a_acc = []
            cur = s
            guard = 0
            while cur >= 0 and guard < 4 * self.n:
                guard += 1
                # was the packet accepted at `cur`?
                p *= max(1e-9, 1.0 - ev.block_prob[cur])
                d += ev.queue_delay_s[cur] + self.serve_time_s
                if self.topo.dist_bs[cur] <= self.cfg.deployment.comm_range_m:
                    p *= max(1e-9, self._hop_success(cur, -1))
                    d += self.serve_time_s
                    break
                nx = nxt[cur]
                if nx < 0:
                    p = 0.0
                    break
                p *= self._hop_success(cur, nx)
                d += (ev.attempts[cur] - 1.0) * self.serve_time_s
                a_acc.append(ev.attempts[cur])
                cur = nx
            probs.append(float(p))
            delays.append(float(d))
            atts.append(float(np.mean(a_acc)) if a_acc else self.radio.max_retx)
        finite = [x for x in delays if np.isfinite(x)]
        mean_delay = float(np.mean(finite)) if finite else float(self.serve_time_s * self.n)
        return float(np.mean(probs)), mean_delay, float(np.mean(atts))

    def _hop_success(self, i: int, j: int) -> float:
        """Probability a packet traverses hop (i -> j) with ARQ.

        A link that has been taken *down* by a failure event must transmit
        nothing, so its success probability is zero. This check was missing:
        ``link_up`` gated the policies' *routing decisions* but not the packet
        physics, so an outage penalised only the controllers that correctly
        noticed it and re-routed around it, while a controller whose stale
        table still pointed at the dead link delivered as if nothing happened.
        That inverted every outage result (a static OSPF table was reported as
        immune to 80 % link loss). One-way array assignment avoids allocating
        a full index tuple per hop.
        """
        if j < 0:
            p = float(self.prr[i, i]) if self.prr[i, i] > 0 else 0.0
            if p <= 0.0:
                p = self.radio.prr(float(self.topo.dist_bs[i]))
        else:
            if not self.link_up[i, j]:
                return 0.0
            p = float(self.prr[i, j])
        p = float(np.clip(p, 1e-6, 1.0))
        R = self.radio.max_retx
        return 1.0 - (1.0 - p) ** R

    # -- full closed-loop run ---------------------------------------------
    def run(
        self,
        policy=None,
        sources: Sequence[int] = (),
        rates: Optional[np.ndarray] = None,
        route_lifetime_s: float = 3.0,
        failures: Sequence[FailureEvent] = (),
        charge_energy: bool = True,
        record_series: bool = True,
        action_fn: Optional[Callable[[int, np.ndarray, dict], int]] = None,
    ) -> FlowResult:
        """Simulate ``sim_time_s`` seconds with periodic route recomputation.

        Two call styles are supported:

        * ``run(policy, sources, ...)`` where ``policy`` is a deployable routing
          policy exposing ``reset()`` and ``next_hops(sim, sources, rates,
          eligible, mask)`` -- this is what the RL controllers use, since they
          need the occupancy feedback from the previous evaluation;
        * ``run(action_fn=lambda i, mask, info: hop, ...)`` for stateless
          baselines and unit tests.
        """
        if policy is None and action_fn is None:
            raise ValueError("run() needs either `policy` or `action_fn`")
        if policy is not None and hasattr(policy, "reset"):
            policy.reset()

        self.reset()
        cfg = self.cfg
        n = self.n
        if rates is None:
            rates = np.full(len(sources), cfg.traffic.pkt_rate_pps)
        rates = np.asarray(rates, float)

        pending = sorted(failures, key=lambda e: e.time_s)
        active: list = []          # transient outages currently in force
        n_windows = max(1, int(round(cfg.traffic.sim_time_s / route_lifetime_s)))
        dt = cfg.traffic.sim_time_s / n_windows

        pdr_t, time_t, replan_pdr = [], [], []
        delivered = 0.0
        offered_total = 0.0
        e2e_w = 0.0
        att_w = 0.0
        hop_w = 0.0
        energy_used = np.zeros(n)
        util_acc = np.zeros(n)
        util_max = np.zeros(n)
        win_count = 0
        w_fair = 0.0

        for w in range(n_windows):
            t = w * dt
            while pending and pending[0].time_s <= t:
                ev = pending.pop(0)
                self.apply_failure(ev)
                if ev.duration_s is not None:
                    active.append((t + float(ev.duration_s), ev))
            for item in list(active):
                end_t, ev = item
                if t >= end_t:
                    self.restore_failure(ev)
                    active.remove(item)
            mask = self.eligible()

            if policy is not None and hasattr(policy, "next_hops"):
                nxt, hops = policy.next_hops(self, sources, rates, mask)
            else:
                nxt, hops = self.next_hop_table(action_fn, mask)
            ev = self.evaluate(nxt, hops, sources, rates, mask)
            self.last_eval = ev
            p, d, a = self.path_metrics(nxt, hops, ev, sources)

            src_rate = float(rates.sum())
            delivered_w = p * src_rate
            offered_total += src_rate * dt
            delivered += delivered_w * dt
            reach = [hops[s] for s in sources if hops[s] > 0]

            if charge_energy:
                energy_used += ev.power_w * dt
                self.energy = np.maximum(0.0, self.energy - ev.power_w * dt)
            util_acc += ev.utilisation
            win_count += 1
            w_fair += _jain(ev.power_w)

            pdr_t.append(p)
            time_t.append(t + dt)
            replan_pdr.append(p)
            e2e_w += d * delivered_w * dt if np.isfinite(d) else 0.0
            att_w += a * delivered_w * dt
            hop_w += (float(np.mean(reach)) if reach else 0.0) * src_rate * dt

        total_energy = float(energy_used.sum())
        avg_power = total_energy / max(cfg.traffic.sim_time_s, 1e-9)
        pdr = delivered / max(offered_total, 1e-9)
        ee = pdr / max(avg_power, 1e-9)
        util_mean = util_acc / max(win_count, 1)
        imbalance = float(np.std(util_mean) / max(np.mean(util_mean), 1e-9))
        fairness = float(np.sum(energy_used) ** 2 / (n * max((energy_used ** 2).sum(), 1e-12)))
        denom = max(delivered, 1e-9)

        return FlowResult(
            pdr=float(pdr),
            e2e_delay_s=float(e2e_w / denom),
            energy_efficiency=float(ee),
            avg_power_w=float(avg_power),
            total_energy_j=total_energy,
            fairness=fairness,
            load_imbalance=imbalance,
            mean_hops=float(hop_w / max(offered_total, 1e-9)),
            mean_attempts=float(att_w / denom),
            throughput_pps=float(delivered / max(cfg.traffic.sim_time_s, 1e-9)),
            pdr_t=pdr_t if record_series else [],
            time_t=time_t if record_series else [],
            energy_t=[],
            replan_pdr=replan_pdr if record_series else [],
        )


def _jain(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.sum()
    if s <= 0:
        return 0.0
    return float(s * s / (len(x) * (x * x).sum()))
