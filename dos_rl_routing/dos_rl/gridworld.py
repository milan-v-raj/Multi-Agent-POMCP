"""5x5 dynamic GridWorld replication of the framework paper (Sec. 4.2, Fig. 4).

Reproduces the paper's *learning* experiment as described in the text:

* 5 x 5 grid, start (0, 0), goal (4, 4);
* 5-unit energy reservoir, -0.1 per state transition;
* up to 3 obstacles, re-drawn every 100 episodes, -20 on collision;
* recharge points give +5 units; running out of energy costs -20;
* three correlated objectives: energy, load and reliability analogues
  (o1 progress, o2 obstacle avoidance, o3 recharge/energy conservation).

Purpose: validate the DOS-RL implementation against the paper's own published
learning curves *before* trusting it on the network task.

Reward-design note
------------------
An earlier version of these objectives paid a constant positive reward every
step for "being safe" and "having battery". Because such a term is collected at
every step, its infinite-horizon value is ``c / (1 - gamma)`` -- about 100 at
gamma = 0.99 -- which swamped the +10 goal bonus and made *wandering forever*
optimal. The objectives below are therefore event-based costs in {-1, 0}, so
the terminal rewards dominate. This is documented in the report as a reward
design finding rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .agents.tabular import QLearningAgent, TabularAgent
from .config import RLConfig

START = (0, 0)
GOAL = (4, 4)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@dataclass
class GridEpisode:
    return_: float
    length: int
    energy_used: float
    obstacle_hits: int
    success: int
    terminal: str


class DynamicGridWorld:
    """5x5 grid with a periodically re-drawn obstacle field."""

    def __init__(
        self,
        size: int = 5,
        obstacle_period: int = 100,
        max_obstacles: int = 3,
        n_recharge: int = 2,
    ):
        self.size = size
        self.obstacle_period = obstacle_period
        self.max_obstacles = max_obstacles
        self.n_recharge = n_recharge
        self.cells = [(r, c) for r in range(size) for c in range(size)]
        self.obstacles: set = set()
        self.recharge: set = set()
        self.episode = 0
        self._initialised = False

    def reset_world(self, rng: np.random.Generator):
        free = [c for c in self.cells if c != START and c != GOAL]
        k = min(self.max_obstacles, len(free))
        idx = rng.choice(len(free), size=k, replace=False)
        self.obstacles = {free[int(i)] for i in np.atleast_1d(idx)}
        rem = [c for c in free if c not in self.obstacles]
        r_idx = rng.choice(len(rem), size=min(self.n_recharge, len(rem)), replace=False)
        self.recharge = {rem[int(i)] for i in np.atleast_1d(r_idx)}
        self._initialised = True

    def ensure_world(self, rng: np.random.Generator):
        if not self._initialised:
            self.reset_world(rng)

    def tick(self, rng: np.random.Generator):
        """Called once per episode; re-draws obstacles every ``obstacle_period``."""
        if not self._initialised:
            self.reset_world(rng)
        elif self.episode > 0 and self.episode % self.obstacle_period == 0:
            self.reset_world(rng)
        self.episode += 1

    def actions(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = cell
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                out.append((nr, nc))
        return out


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def grid_objectives(world: DynamicGridWorld, pos, a, energy: float) -> np.ndarray:
    """Three correlated objectives (o1 progress, o2 obstacle, o3 energy).

    Reward-design notes -- both traps below were found empirically and are
    documented in the report rather than hidden:

    1. **Constant per-step positives make wandering optimal.** Paying +1 every
       step for "safe" or "has battery" gives an infinite-horizon value of
       c/(1-gamma) ~ 100 at gamma = 0.99, which swamps the +10 goal bonus.
    2. **Zero-cost neutral moves make oscillation optimal.** With purely
       event-based costs, a 2-cycle between two safe cells costs 0 per step,
       so the agent prefers looping to taking any risk. The paper charges
       -0.1 energy per state transition for exactly this reason, and that
       step cost is what makes the shortest path optimal.

    Final design (all terms in [-1, 1], normalised by the 5-unit reservoir):

        o3 energy      : -(0.1 / 5.0) every transition  (+0.98 on recharge,
                         -1.0 if the battery is critical)
        o1 progress    : +1 if the move reduces Manhattan distance to the goal,
                         -1 if it increases it, else 0
        o2 obstacle    : -1 if the move enters an obstacle cell, else 0
    """
    step_cost = -0.1 / 5.0
    if a in world.recharge:
        o3 = (5.0 - 0.1) / 5.0
    elif energy - 0.1 < 1.0:
        o3 = -1.0
    else:
        o3 = step_cost
    d = abs(a[0] - GOAL[0]) + abs(a[1] - GOAL[1])
    d0 = abs(pos[0] - GOAL[0]) + abs(pos[1] - GOAL[1])
    o1 = 1.0 if d < d0 else (-1.0 if d > d0 else 0.0)
    o2 = -1.0 if a in world.obstacles else 0.0
    return np.array([o1, o2, o3], dtype=float)


def grid_potential(cell) -> float:
    """Paper Eq. 9 with Manhattan distance (Fig. 4's caption uses Manhattan)."""
    return 1.0 - (abs(cell[0] - GOAL[0]) + abs(cell[1] - GOAL[1])) / 8.0


def _grid_state(pos, energy: float) -> tuple:
    """State = (row, col, coarse energy bucket), 3 battery levels."""
    e = int(energy > 2.0) + int(energy > 3.5)
    return (pos[0], pos[1], min(2, max(0, e)))


def scalarise(rl: RLConfig, rvec: np.ndarray) -> float:
    w = np.asarray(rl.reward_weights, dtype=float)
    return float(np.dot(w, rvec) / max(w.sum(), 1e-12))


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
def run_grid_episode(
    world: DynamicGridWorld,
    agent: TabularAgent,
    rl: RLConfig,
    rng: np.random.Generator,
    train: bool = True,
    dos: bool = True,
    max_steps: int = 40,
) -> GridEpisode:
    """One gridworld episode (one navigation attempt)."""
    energy = 5.0
    pos = START
    total = 0.0
    steps = 0
    hits = 0
    terminal = "hop_limit"

    while steps <= max_steps:
        acts = world.actions(pos)
        if not acts:
            terminal = "dead_end"
            break
        state = _grid_state(pos, energy)
        n_act = len(acts)

        if dos:
            idx = agent.epsilon_greedy(state, list(range(n_act)), rng) if train else agent.greedy(
                state, list(range(n_act))
            )
        else:
            idx = _weighted_act(agent, state, n_act, rl, rng, train)
        a = acts[int(idx)]

        rvec = grid_objectives(world, pos, a, energy)
        F = rl.gamma * grid_potential(a) - grid_potential(pos) if rl.use_shaping else 0.0

        # --- collision / depletion: terminal, -20 -------------------------
        if a in world.obstacles:
            hits += 1
            total += scalarise(rl, rvec) + F + rl.collision_reward
            if train:
                agent.update_transition(state, int(idx), rvec, None, (), True, rl.collision_reward)
                if not dos:
                    _weighted_update(agent, state, int(idx), rvec, None, [], rl)
            terminal = "dead_end"
            break

        energy -= 0.1
        if energy <= 0.0:
            total += scalarise(rl, rvec) + F + rl.collision_reward
            if train:
                agent.update_transition(state, int(idx), rvec, None, (), True, rl.collision_reward)
                if not dos:
                    _weighted_update(agent, state, int(idx), rvec, None, [], rl)
            terminal = "energy_exhausted"
            break

        if a in world.recharge:
            energy = min(5.0, energy + 5.0)

        steps += 1
        nxt = a
        # --- goal ---------------------------------------------------------
        if nxt == GOAL:
            total += scalarise(rl, rvec) + F + rl.goal_reward
            if train:
                agent.update_transition(state, int(idx), rvec, None, (), True, rl.goal_reward)
                if not dos:
                    _weighted_update(agent, state, int(idx), rvec, None, [], rl)
            terminal = "goal"
            break

        # --- ordinary transition -----------------------------------------
        total += scalarise(rl, rvec) + F
        nxt_state = _grid_state(nxt, energy)
        n_nxt = len(world.actions(nxt))
        if train:
            if dos:
                agent.update_transition(
                    state, int(idx), rvec, nxt_state, list(range(n_nxt)), False, 0.0
                )
            else:
                _weighted_update(agent, state, int(idx), rvec, nxt_state, list(range(n_nxt)), rl)
        pos = nxt

    if train:
        agent.decay_epsilon()
    return GridEpisode(
        return_=float(total),
        length=steps,
        energy_used=float(5.0 - max(energy, 0.0)),
        obstacle_hits=hits,
        success=int(terminal == "goal"),
        terminal=terminal,
    )


# ---------------------------------------------------------------------------
# Weighted-sum ablation (single scalarised objective, no DOS)
# ---------------------------------------------------------------------------
def _weighted_act(agent, state, n_actions, rl, rng, train):
    q = agent.q(state, list(range(n_actions)))
    w = np.asarray(rl.reward_weights, float)
    if q.shape[1] == 1:                 # single-objective ablation: already scalar
        v = q[:, 0]
    else:
        v = q @ (w / max(w.sum(), 1e-12))
    if train:
        exploit_p = rl.epsilon if rl.epsilon_convention == "paper" else 1.0 - rl.epsilon
        if rng.random() >= exploit_p:
            return int(rng.integers(n_actions))
    return int(np.argmax(v))


def _weighted_update(agent, state, idx, rvec, next_state, next_acts, rl):
    """Bellman update on the scalarised reward (the non-DOS comparison)."""
    w = np.asarray(rl.reward_weights, float)
    n_obj = getattr(agent, "n_obj", 3)
    if n_obj == 1:
        w = np.ones(1)
    else:
        w = w / max(w.sum(), 1e-12)
    row = agent.Q.setdefault(state, {}).get(int(idx))
    if row is None:
        row = np.zeros(agent.n_obj)
        agent.Q[state][int(idx)] = row
    rscalar = float(scalarise(rl, rvec))
    cur = float(row @ w)
    target = rscalar
    if next_state is not None and len(next_acts):
        nq_all = agent.q(next_state, list(range(len(next_acts))))
        nq = nq_all[:, 0] if n_obj == 1 else nq_all @ w
        target += rl.gamma * float(np.max(nq))
    err = target - cur
    row[...] = row + rl.alpha * err * w
    vtbl = agent.VAR.setdefault(state, {})
    v = vtbl.get(int(idx))
    if v is None:
        v = np.zeros(agent.n_obj)
        vtbl[int(idx)] = v
    v[...] = (1 - rl.confidence_psi) * v + rl.confidence_psi * (err ** 2)
    agent.n_updates += 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_grid_experiment(
    rl: RLConfig,
    episodes: int = 1500,
    seed: int = 0,
    dos: bool = True,
    shaping: bool = True,
    obstacle_period: int = 100,
    test_episodes: int = 0,
) -> Dict[str, np.ndarray]:
    """Full learning curve for one gridworld configuration.

    If ``test_episodes`` > 0, the agent is additionally evaluated **after
    training has finished**, greedily, for that many episodes with fresh
    obstacle draws, and the results are returned under the ``test_*`` keys. That
    is what a held-out evaluation is supposed to mean; earlier versions of E1
    reported ``mean(success[-100:])`` of the *training* curve, which measures
    the exploration-polluted policy while it is still learning, not the policy
    that would be deployed.
    """
    rng = np.random.default_rng(seed)
    world = DynamicGridWorld(obstacle_period=obstacle_period)
    world.reset_world(rng)
    rl = rl.with_(use_shaping=shaping)
    agent = QLearningAgent(rl, n_objectives=3)
    rec = {k: [] for k in ("return", "length", "energy", "hits", "success")}
    for _ in range(episodes):
        world.tick(rng)
        e = run_grid_episode(world, agent, rl, rng, train=True, dos=dos)
        rec["return"].append(e.return_)
        rec["length"].append(e.length)
        rec["energy"].append(e.energy_used)
        rec["hits"].append(e.obstacle_hits)
        rec["success"].append(e.success)

    out = {k: np.asarray(v, float) for k, v in rec.items()}
    if test_episodes > 0:
        env = np.random.default_rng(seed + 90_000)      # independent test stream
        eps = {"return": [], "length": [], "energy": [], "hits": [], "success": []}
        for _ in range(test_episodes):
            world.tick(env)
            e = run_grid_episode(world, agent, rl, env, train=False, dos=dos)
            eps["return"].append(e.return_)
            eps["length"].append(e.length)
            eps["energy"].append(e.energy_used)
            eps["hits"].append(e.obstacle_hits)
            eps["success"].append(e.success)
        for k, v in eps.items():
            out[f"test_{k}"] = np.asarray(v, float)
    return out
