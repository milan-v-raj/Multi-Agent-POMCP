"""Tabular multi-objective agents: Q-learning, SARSA, first-visit MC control.

All three keep one Q-table **per objective**, exactly as the framework paper
prescribes (Eq. 7), plus a variance tracker used by the Dynamic Objective
Selection rule (Eq. 2 / Eq. 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RLConfig


class TabularAgent:
    """Shared Q-table machinery and the DOS confidence metric."""

    name = "tabular"

    def __init__(self, rl: RLConfig, n_objectives: int = 3):
        self.rl = rl
        self.n_obj = n_objectives
        self.Q: Dict[tuple, Dict[int, np.ndarray]] = {}
        self.VAR: Dict[tuple, Dict[int, np.ndarray]] = {}
        self.n_updates = 0
        self.epsilon = float(rl.epsilon)
        self.alpha = float(rl.alpha)

    # -- table access ------------------------------------------------------
    def q(self, state: tuple, actions: Sequence[int]) -> np.ndarray:
        """(n_actions, n_objectives) view of the Q values, initialised to zero."""
        tbl = self.Q.setdefault(state, {})
        out = np.zeros((len(actions), self.n_obj))
        for k, a in enumerate(actions):
            v = tbl.get(int(a))
            if v is None:
                v = np.zeros(self.n_obj)
                tbl[int(a)] = v
            out[k] = v
        return out

    def var(self, state: tuple, actions: Sequence[int]) -> np.ndarray:
        tbl = self.VAR.setdefault(state, {})
        out = np.zeros((len(actions), self.n_obj))
        for k, a in enumerate(actions):
            v = tbl.get(int(a))
            if v is None:
                v = np.zeros(self.n_obj)
                tbl[int(a)] = v
            out[k] = v
        return out

    # -- DOS ---------------------------------------------------------------
    def objective_statistics(self, state: tuple, actions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Per-objective (mean of best Q, mean tracked variance) at ``state``."""
        q = self.q(state, actions)                     # (A, O)
        v = self.var(state, actions)                   # (A, O)
        mu = q.max(axis=0)                             # most promising value per objective
        sigma2 = v.mean(axis=0)                        # average uncertainty per objective
        return mu, sigma2

    def select_objective(self, state: tuple, actions: Sequence[int]) -> int:
        """Dynamic Objective Selection.

        Four selection rules are supported via ``RLConfig.dos_select``;
        the default selects by value and does not use the confidence metric:

        ``argmax``  (paper Eq. 3, verbatim)  C = (mu + e^{sigma^2}) / sum(...),
                    pick argmax. Dominated by the exponential: the objective
                    term can favour high variance when it dominates, although
                    the value term also affects the result.
        ``argmin``  (paper Algorithm 1, literal) pick argmin C. Still uses the
                    same exponential term, so it is steered by variance rather
                    than by estimated value.
        ``var``     most stable objective (smallest variance only).
        ``value``   (used by default) objective with the highest Q value among
                    the available actions, without using the variance estimate.
        """
        if self.n_obj == 1:
            return 0
        mu, sigma2 = self.objective_statistics(state, actions)
        mode = self.rl.dos_select
        if mode == "var":
            return int(np.argmin(sigma2))
        if mode == "value":
            return int(np.argmax(mu))
        num = mu + np.exp(np.clip(sigma2, -20.0, 20.0))
        C = num / max(float(num.sum()), 1e-12)
        if mode == "argmin":
            return int(np.argmin(C))
        return int(np.argmax(C))

    # -- exploration -------------------------------------------------------
    def epsilon_greedy(self, state: tuple, actions: Sequence[int], rng: np.random.Generator) -> int:
        """epsilon-greedy over the objective chosen by DOS.

        ``RLConfig.epsilon_convention``:
          * ``standard`` - explore with probability epsilon (Sutton & Barto);
          * ``paper``    - the paper's literal Eq. 10, i.e. *exploit* with
                           probability epsilon and explore with 1 - epsilon.
        """
        obj = self.select_objective(state, actions)
        q = self.q(state, actions)[:, obj]
        if self.rl.epsilon_convention == "paper":
            exploit_p = self.epsilon
        else:
            exploit_p = 1.0 - self.epsilon
        if rng.random() < exploit_p:
            return int(np.argmax(q))
        return int(rng.integers(len(actions)))

    def greedy(self, state: tuple, actions: Sequence[int]) -> int:
        obj = self.select_objective(state, actions)
        return int(np.argmax(self.q(state, actions)[:, obj]))

    # -- updates -----------------------------------------------------------
    def _update_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Match reward dimensions; a single-objective agent scalarises vectors."""
        rewards = np.asarray(rewards, dtype=float).reshape(-1)
        if self.n_obj == 1 and rewards.size != 1:
            weights = np.asarray(self.rl.reward_weights, dtype=float)
            if weights.shape != rewards.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("reward_weights must be finite, nonnegative, and match rewards")
            return np.array([float(np.dot(weights, rewards) / weights.sum())])
        if rewards.size != self.n_obj:
            raise ValueError(f"expected {self.n_obj} rewards, got {rewards.size}")
        return rewards

    def _write(self, state: tuple, action: int, idx: int, value: float, err_sq: float):
        tbl = self.Q.setdefault(state, {})
        vec = tbl.get(int(action))
        if vec is None:
            vec = np.zeros(self.n_obj)
            tbl[int(action)] = vec
        vec[idx] = value
        vtbl = self.VAR.setdefault(state, {})
        vvec = vtbl.get(int(action))
        if vvec is None:
            vvec = np.zeros(self.n_obj)
            vtbl[int(action)] = vvec
        psi = self.rl.confidence_psi
        vvec[idx] = (1.0 - psi) * vvec[idx] + psi * float(err_sq)

    def decay_epsilon(self):
        if self.rl.decay_epsilon:
            self.epsilon = max(self.rl.epsilon_min, self.epsilon * self.rl.epsilon_decay)

    def q_size(self) -> int:
        return sum(len(v) for v in self.Q.values())


# ---------------------------------------------------------------------------
class QLearningAgent(TabularAgent):
    """Off-policy TD(0) with per-objective updates (paper Eq. 7-8)."""

    name = "q_learning"

    def update_transition(
        self,
        state: tuple,
        action: int,
        rewards: np.ndarray,
        next_state: Optional[tuple],
        next_actions: Sequence[int],
        done: bool,
        terminal_reward: float,
    ):
        a = self.alpha
        gamma = self.rl.gamma
        rewards = self._update_rewards(rewards)
        cur = self.q(state, [action])[0]
        if done or not next_actions:
            target = rewards + terminal_reward
        else:
            nq = self.q(next_state, next_actions).max(axis=0)
            target = rewards + terminal_reward + gamma * nq
        err = target - cur
        for o in range(self.n_obj):
            self._write(state, action, o, cur[o] + a * err[o], err[o] ** 2)
        self.n_updates += 1

    # -- off-policy rollout used during training --------------------------
    def act(self, state: tuple, actions: Sequence[int], rng, greedy: bool = False) -> int:
        if greedy:
            return self.greedy(state, actions)
        return self.epsilon_greedy(state, actions, rng)


# ---------------------------------------------------------------------------
class SarsaAgent(TabularAgent):
    """On-policy TD(0): learns the value of the *epsilon-greedy* policy it runs.

    Unlike Q-learning, its target uses the next action drawn from its current
    exploration policy. A safety advantage is not assumed.
    """

    name = "sarsa"

    def update_transition(
        self,
        state: tuple,
        action: int,
        rewards: np.ndarray,
        next_state: Optional[tuple],
        next_action: Optional[int],
        done: bool,
        terminal_reward: float,
    ):
        a = self.alpha
        gamma = self.rl.gamma
        rewards = self._update_rewards(rewards)
        cur = self.q(state, [action])[0]
        if done or next_state is None or next_action is None:
            target = rewards + terminal_reward
        else:
            nxt = self.q(next_state, [next_action])[0]
            target = rewards + terminal_reward + gamma * nxt
        err = target - cur
        for o in range(self.n_obj):
            self._write(state, action, o, cur[o] + a * err[o], err[o] ** 2)
        self.n_updates += 1


# ---------------------------------------------------------------------------
class MonteCarloAgent(TabularAgent):
    """Every-visit Monte Carlo control with a multi-objective return vector."""

    name = "mc_control"

    def update_episode(
        self,
        trajectory: List[Tuple[tuple, int, np.ndarray, float]],
        terminal_reward: float,
    ):
        """``trajectory`` = [(state, action, reward_vec, shaping)] in time order."""
        G = np.zeros(self.n_obj)
        returns: List[Tuple[tuple, int, np.ndarray]] = []
        for state, action, rvec, F in reversed(trajectory):
            G = rvec + F + terminal_reward + self.rl.gamma * G
            terminal_reward = 0.0
            returns.append((state, action, G.copy()))
        for state, action, G in returns:
            cur = self.q(state, [action])[0]
            err = G - cur
            for o in range(self.n_obj):
                self._write(state, action, o, cur[o] + self.alpha * err[o], err[o] ** 2)
        self.n_updates += len(returns)

    def act(self, state: tuple, actions: Sequence[int], rng, greedy: bool = False) -> int:
        if greedy:
            return self.greedy(state, actions)
        return self.epsilon_greedy(state, actions, rng)
