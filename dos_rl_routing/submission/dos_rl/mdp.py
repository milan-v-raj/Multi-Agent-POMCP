"""Route episodes, available actions, rewards and stochastic transitions.

A route episode starts at a source and terminates at the gateway or on failure.
The default Q-table key contains only the current relay ID."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .belief import BeliefState, BeliefTracker
from .config import RLConfig, SimConfig
from .network_sim import NetworkSimulator
from .observer import LinkMetrics

# -- terminal codes -----------------------------------------------------------
TERM_NONE = "running"
TERM_GOAL = "goal"
TERM_HOP_LIMIT = "hop_limit"
TERM_LOOP = "loop"
TERM_DEAD_END = "dead_end"
TERM_EXHAUSTED = "energy_exhausted"


@dataclass
class ActionResult:
    next_state: tuple
    next_node: int
    rewards: np.ndarray          # per-objective, normalised to [0, 1]
    shaping: float               # F_t = gamma * phi(s') - phi(s)
    scalar_reward: float         # normalized weighted reward + shaping + terminal bonus
    done: bool
    terminal: str = TERM_NONE
    info: dict = field(default_factory=dict)

    @property
    def total_for_objective(self) -> np.ndarray:
        """Per-objective reward including shaping (paper Eq. 8)."""
        return self.rewards + self.shaping


@dataclass
class EpisodeSnapshot:
    """Belief + eligibility for one episode (a 'network state' sample)."""

    belief: BeliefState
    eligible: np.ndarray
    hops: np.ndarray


class RoutingMDP:
    """The routing task as a (PO-)MDP."""

    def __init__(
        self,
        sim: NetworkSimulator,
        cfg: SimConfig,
        rl: RLConfig,
        belief_tracker: BeliefTracker,
        grid_bins: Tuple[int, int] = (3, 3),
        state_mode: str = "node",
    ):
        self.state_mode = state_mode
        self.sim = sim
        self.cfg = cfg
        self.rl = rl
        self.tracker = belief_tracker
        self.topo = sim.topo
        self.n = sim.n
        self.goal = -1                      # node index for the base station
        self._bs_reach = self.topo.dist_bs <= cfg.deployment.comm_range_m
        self._d_max = float(self.topo.dist_bs.max()) + 1e-9
        self._e_bins, self._q_bins = grid_bins
        self.ep = EpisodeSnapshot(
            belief=BeliefState(
                energy=np.ones(self.n),
                energy_var=np.zeros(self.n),
                load=np.zeros(self.n),
                load_var=np.zeros(self.n),
                queue_frac=np.zeros(self.n),
                prr=np.full((self.n, self.n), 0.5),
                prr_var=np.zeros((self.n, self.n)),
                alive_prob=np.ones(self.n),
            ),
            eligible=np.ones(self.n, dtype=bool),
            hops=self.topo.hop_map(np.ones(self.n, dtype=bool)),
        )
        self._visited: set = set()
        self._t = 0
        self._cur = -1
        self._source = -1
        self._energy_at_start: Dict[int, float] = {}

    # ==================================================================
    # Episode construction
    # ==================================================================
    def sample_episode(
        self,
        source: int,
        rng: np.random.Generator,
        energy_spread: float = 0.35,
        prr_jitter: float = 0.0,
        queue_spread: float = 0.5,
    ) -> EpisodeSnapshot:
        """Draw a network state for one training episode.

        ``energy_spread`` randomises the residual-energy profile around the
        simulator's current state so that the learnt policy generalises over
        depletion levels instead of memorising one battery profile.
        """
        e = np.clip(self.sim.energy / max(self.cfg.radio.e_init_j, 1e-9), 0.0, 1.0)
        e = np.clip(e + rng.uniform(-energy_spread, energy_spread, size=self.n), 0.02, 1.0)
        eligible = self.sim.node_up & (e * self.cfg.radio.e_init_j > self.topo.energy_threshold_j)
        prr = self.sim.prr.copy()
        if prr_jitter > 0:
            prr = np.clip(prr + rng.normal(0, prr_jitter, prr.shape), 0.0, 1.0)
        # Sample queue occupancy so training covers a range of load conditions.
        if queue_spread > 0:
            q = np.clip(rng.beta(2.0, 2.0, size=self.n), 0.0, 1.0)
            q = q * queue_spread * 2.0 if queue_spread > 0.5 else q
            q = np.clip(q, 0.0, 1.0)
        else:
            q = np.full(self.n, 0.5)
        belief = BeliefState(
            energy=e,
            energy_var=np.full(self.n, 0.02),
            load=q.copy(),
            load_var=np.full(self.n, 0.02),
            queue_frac=q,
            prr=prr,
            prr_var=np.full((self.n, self.n), 0.02),
            alive_prob=eligible.astype(float),
        )
        hops = self.topo.hop_map(eligible)
        self.ep = EpisodeSnapshot(belief=belief, eligible=eligible, hops=hops)
        self._visited = set()
        self._t = 0
        self._source = int(source)
        self._cur = int(source)
        self._energy_at_start = {i: float(e[i]) for i in range(self.n)}
        return self.ep

    def from_metrics(self, m: LinkMetrics, initial_energy: Optional[np.ndarray] = None) -> EpisodeSnapshot:
        """Build an episode snapshot from controller telemetry only."""
        self.tracker.update(m, initial_energy)
        b = self.tracker.state
        eligible = m.alive & (b.energy > 0.02)
        prr = b.prr.copy()
        prr[np.arange(self.n), np.arange(self.n)] = 1.0
        belief = b.copy()
        belief.prr = prr
        hops = self.topo.hop_map(eligible)
        self.ep = EpisodeSnapshot(belief=belief, eligible=eligible, hops=hops)
        return self.ep

    def reset(self, source: int) -> tuple:
        self._visited = {source}
        self._t = 0
        self._cur = int(source)
        return self.encode(source, self.state_mode)

    # ==================================================================
    # State space
    # ==================================================================
    def encode(self, node: int, mode: Optional[str] = None) -> tuple:
        """State key.

        ``node``  : s = i, exactly the paper's (state <-> relay node) mapping.
        ``belief``: (i, energy bucket of i, buffer bucket of i). This is a
                    compact observation, not a sufficient Markov state: other
                    relays, links and episode history can affect outcomes.
        """
        mode = mode or self.state_mode
        if mode == "node":
            return (int(node),)
        b = self.ep.belief
        eb = int(min(self._e_bins - 1, max(0, b.energy[node] * self._e_bins)))
        qb = int(min(self._q_bins - 1, max(0, b.queue_frac[node] * self._q_bins)))
        return (int(node), eb, qb)

    # ==================================================================
    # Action space
    # ==================================================================
    def actions(self, node: int) -> List[int]:
        """Eligible next hops.

        The default ``dag`` mode normally restricts neighbours to lower BFS
        hop distance, with a geographic fallback if no such candidate remains.
        ``stp`` permits slack and is not an implementation of STP. Loop freedom
        is therefore not guaranteed by every mode or fallback.
        """
        if self.topo.dist_bs[node] <= self.cfg.deployment.comm_range_m:
            return [self.goal]           # gateway is directly reachable
        elig = self.ep.eligible
        acts = [j for j in self.topo.neighbors[node] if self.sim.link_up[node, j] and elig[j]]
        mode = self.rl.loop_free
        if mode in ("stp", "dag"):
            h_i = int(self.ep.hops[node])
            if h_i <= 0:                 # this node currently has no route
                return []
            slack = 0 if mode == "dag" else float(self.rl.dag_slack)
            acts = [j for j in acts if 0 < int(self.ep.hops[j]) <= h_i - 1 + slack]
            # Retain lower-hop neighbours before applying the geographic fallback.
            closer = [j for j in acts if int(self.ep.hops[j]) < h_i]
            for j in closer:
                if j not in acts:
                    acts.append(j)
            if not acts:                 # safety net: anything strictly closer
                d_i = self.topo.dist_bs[node]
                acts = [j for j in acts] or [
                    j
                    for j in self.topo.neighbors[node]
                    if elig[j] and self.sim.link_up[node, j] and self.topo.dist_bs[j] < d_i
                ]
        elif mode == "visited":
            acts = [j for j in acts if j not in self._visited]
        return acts

    def action_matrix(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        """(next hops, hop distances) for vectorised reward evaluation."""
        acts = self.actions(node)
        nodes = [a for a in acts if a >= 0]
        d = np.array([self.topo.dist[node, a] if a >= 0 else self.topo.dist_bs[node] for a in acts])
        return np.array(nodes, dtype=int), d

    # ==================================================================
    # Reward design
    # ==================================================================
    def rewards(self, i: int, j: int) -> np.ndarray:
        """Per-objective reward, normalised to [0, 1] (see REPORT.md Eq. R1-R3)."""
        b = self.ep.belief
        e_i, e_j = float(b.energy[i]), float(b.energy[j] if j >= 0 else 1.0)
        if self.rl.reward_mode == "paper":
            # Eq. 4 verbatim, rescaled from [-1, 1] to [0, 1]
            r_e = 0.5 * (1.0 + (e_j - e_i))
        else:
            # Corrected: pay the actual per-hop consumption, penalise weak relays
            dist = self.topo.dist[i, j] if j >= 0 else float(self.topo.dist_bs[i])
            cost = (self.sim.radio.tx_energy(dist) + self.sim.radio.e_rx) / max(
                self.cfg.energy_budget_j, 1e-12
            )
            r_e = float(np.clip(1.0 - cost - (1.0 - e_j), 0.0, 1.0))

        # Paper Eq. 5: R_Q = AvailableBuffer(j)/MaxBuffer - Load(i)/MaxLoad.
        # ``queue_frac`` is *occupancy*, so the available buffer is its
        # complement. An earlier version fed the occupancy directly into this
        # term, which rewarded choosing relays whose buffers were already full -
        # the exact opposite of load balancing, and the reason the load objective
        # appeared to have no effect. Fixed and regression-tested.
        avail_j = 1.0 - float(b.queue_frac[j]) if j >= 0 else 1.0
        load_i = float(b.load[i])
        r_q = 0.5 * (1.0 + (avail_j - load_i))

        # Eq. 6: EWMA packet reception ratio on the chosen link
        r_lq = float(b.prr[i, j]) if j >= 0 else float(self._bs_prr(i))
        return np.array([r_e, r_q, r_lq], dtype=float)

    def _bs_prr(self, i: int) -> float:
        return float(self.sim.radio.prr(float(self.topo.dist_bs[i])))

    def potential(self, node: int) -> float:
        """phi(s) = 1 - d(s, G) / d_max   (paper Eq. 9)."""
        d = float(self.topo.dist_bs[node]) if node >= 0 else 0.0
        return 1.0 - d / self._d_max

    def shaping(self, i: int, j: int) -> float:
        if not self.rl.use_shaping:
            return 0.0
        return self.rl.gamma * self.potential(j) - self.potential(i)

    # ==================================================================
    # Transition + terminal conditions
    # ==================================================================
    def _mk(
        self,
        next_node: int,
        r_vec: np.ndarray,
        F: float,
        terminal: str,
        terminal_reward: float,
        info: Optional[dict] = None,
    ) -> ActionResult:
        """Assemble an ActionResult with a consistent scalar reward."""
        w = np.asarray(self.rl.reward_weights, dtype=float)
        scalar = float(np.dot(w, r_vec) / max(w.sum(), 1e-12)) + F + terminal_reward
        return ActionResult(
            next_state=self.encode(next_node, self.state_mode) if next_node >= 0 else (-1,),
            next_node=next_node,
            rewards=r_vec,
            shaping=F,
            scalar_reward=scalar,
            done=terminal != TERM_NONE,
            terminal=terminal,
            info=dict(info or {}, terminal_reward=terminal_reward),
        )

    def step(self, action: int, rng: Optional[np.random.Generator] = None) -> ActionResult:
        """Apply action ``action`` from the current node.

        Transition rule (P): the packet moves to ``action`` only if at least one
        MAC attempt succeeds, with per-attempt success probability PRR(i, j) and
        up to ``max_retx`` attempts; otherwise the packet is discarded
        (terminal DEAD_END). Loops, hop budget and unreachable states are
        terminal and are penalised.
        """
        rng = rng if rng is not None else np.random.default_rng(0)
        i = self._cur
        carried = self.rl.goal_reward
        dead = self.rl.dead_reward

        # --- illegal / terminal-constructor actions -----------------------
        acts = self.actions(i)
        r_self = self.rewards(i, i)
        # Use zero potential for every terminal outcome.
        terminal_F = -self.potential(i) if self.rl.use_shaping else 0.0
        if not acts:
            return self._mk(i, r_self, terminal_F, TERM_DEAD_END, dead, {"reason": "no_eligible_neighbour"})
        j = int(action)
        if j not in acts:
            return self._mk(i, r_self, terminal_F, TERM_DEAD_END, dead, {"reason": "invalid_action"})

        r_vec = self.rewards(i, j)
        F = self.shaping(i, j)

        # --- energy feasibility (terminal EXHAUSTED) ----------------------
        dist = self.topo.dist[i, j] if j >= 0 else float(self.topo.dist_bs[i])
        hop_cost = self.sim.radio.tx_energy(dist) + self.sim.radio.e_rx
        e_i_j = self.ep.belief.energy[i] * self.cfg.radio.e_init_j
        if e_i_j - hop_cost <= self.topo.energy_threshold_j:
            return self._mk(i, r_vec, terminal_F, TERM_EXHAUSTED, dead, {"hop_cost_j": hop_cost})

        # --- stochastic link transition -----------------------------------
        p = float(np.clip(self.ep.belief.prr[i, j] if j >= 0 else self._bs_prr(i), 1e-6, 1.0))
        success = rng.random() < (1.0 - (1.0 - p) ** self.sim.radio.max_retx)
        if not success:
            return self._mk(i, r_vec, terminal_F, TERM_DEAD_END, dead, {"reason": "link_failure"})

        # --- bookkeeping ---------------------------------------------------
        self._t += 1
        self.ep.belief.energy[i] = max(
            0.0, self.ep.belief.energy[i] - hop_cost / self.cfg.radio.e_init_j
        )
        self.ep.belief.queue_frac[j if j >= 0 else i] = min(
            1.0, self.ep.belief.queue_frac[j if j >= 0 else i] + 0.05
        )
        self.ep.belief.load[i] = min(1.0, self.ep.belief.load[i] + 0.05)

        if j < 0:                                  # reached the gateway
            return self._mk(-1, r_vec, terminal_F, TERM_GOAL, carried, {"hops": self._t})

        self._cur = j
        if j in self._visited:
            return self._mk(j, r_vec, terminal_F, TERM_LOOP, dead)
        self._visited.add(j)

        if self._t >= self.rl.max_hops or self.ep.hops[j] <= 0:
            return self._mk(j, r_vec, terminal_F, TERM_HOP_LIMIT, dead)

        if not self.actions(j):
            return self._mk(j, r_vec, terminal_F, TERM_DEAD_END, dead, {"reason": "dead_end"})
        return self._mk(j, r_vec, F, TERM_NONE, 0.0)

    # ==================================================================
    # Helpers
    # ==================================================================
    @property
    def source(self) -> int:
        return self._source

    @property
    def current(self) -> int:
        return self._cur

    def optimal_hops(self, source: int) -> float:
        """Lower bound on hops via BFS over the believed-eligible set (oracle hint)."""
        h = self.ep.hops[source]
        return float(h) if h > 0 else float("inf")
