"""Linear function approximation over belief features.

``Q(b, a) = theta_o^T phi(b, a)`` with semi-gradient TD(0) updates, one weight
vector per objective. The feature map is the belief-state vector from
:meth:`BeliefState.node_features`, so the same network can evaluate actions in a
topology it has never seen - which is precisely what a tabular Q-table cannot
do. This is the rubric's "linear function approximator" option and the answer
to the paper's scalability weakness (table size = O(N^2 * k)).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..belief import BeliefState
from ..config import RLConfig


class LinearQAgent:
    name = "linear_fa"

    def __init__(self, rl: RLConfig, n_features: int = 10, n_objectives: int = 3):
        self.rl = rl
        self.n_feat = n_features
        self.n_obj = n_objectives
        self.theta = np.zeros((n_objectives, n_features))
        self.epsilon = float(rl.epsilon)
        self.alpha = float(rl.alpha)
        self.n_updates = 0
        # per-(state, action) variance tracker, keyed identically to the tabular
        # agents so the DOS rule can be reused unchanged
        self.VAR: Dict[tuple, Dict[int, np.ndarray]] = {}

    # -- features ----------------------------------------------------------
    def features(self, belief: BeliefState, i: int, j: int) -> np.ndarray:
        return belief.node_features(i, j)

    # -- value access ------------------------------------------------------
    def q_values(self, feats: np.ndarray) -> np.ndarray:
        return feats @ self.theta.T          # (n_actions, n_objectives)

    def _var(self, state: tuple, actions: Sequence[int]) -> np.ndarray:
        tbl = self.VAR.setdefault(state, {})
        out = np.zeros((len(actions), self.n_obj))
        for k, a in enumerate(actions):
            v = tbl.get(int(a))
            if v is None:
                v = np.zeros(self.n_obj)
                tbl[int(a)] = v
            out[k] = v
        return out

    def select_objective(self, q: np.ndarray, var: np.ndarray) -> int:
        if self.n_obj == 1:
            return 0
        mu = q.max(axis=0)
        sigma2 = var.mean(axis=0)
        mode = self.rl.dos_select
        if mode == "var":
            return int(np.argmin(sigma2))
        if mode == "value":
            return int(np.argmax(mu))
        num = mu + np.exp(np.clip(sigma2, -20.0, 20.0))
        C = num / max(float(num.sum()), 1e-12)
        return int(np.argmin(C)) if mode == "argmin" else int(np.argmax(C))

    def act(
        self,
        feats: np.ndarray,
        state: tuple,
        actions: Sequence[int],
        rng: np.random.Generator,
        greedy: bool = False,
    ) -> int:
        q = self.q_values(feats)
        var = self._var(state, actions)
        obj = self.select_objective(q, var)
        col = q[:, obj]
        if greedy:
            return int(np.argmax(col))
        exploit_p = self.epsilon if self.rl.epsilon_convention == "paper" else 1.0 - self.epsilon
        if rng.random() < exploit_p:
            return int(np.argmax(col))
        return int(rng.integers(len(actions)))

    # -- learning ----------------------------------------------------------
    def update(
        self,
        feats: np.ndarray,
        state: tuple,
        action_idx: int,
        action: int,
        rewards: np.ndarray,
        next_feats: Optional[np.ndarray],
        done: bool,
        terminal_reward: float,
    ):
        a = self.alpha
        gamma = self.rl.gamma
        feats = np.asarray(feats, dtype=float).reshape(-1)
        if feats.shape != (self.n_feat,):
            raise ValueError("unexpected feature vector shape")
        cur = feats @ self.theta.T
        target = np.asarray(rewards, dtype=float) + terminal_reward
        if not done and next_feats is not None and len(next_feats):
            next_feats = np.atleast_2d(next_feats)
            target = target + gamma * self.q_values(next_feats).max(axis=0)
        err = target - cur
        for o in range(self.n_obj):
            # semi-gradient step on theta_o
            self.theta[o] += a * err[o] * feats
        psi = self.rl.confidence_psi
        tbl = self.VAR.setdefault(state, {})
        v = tbl.get(int(action))
        if v is None:
            v = np.zeros(self.n_obj)
            tbl[int(action)] = v
        v[:] = (1.0 - psi) * v + psi * (err ** 2)
        self.n_updates += 1

    def decay_epsilon(self):
        if self.rl.decay_epsilon:
            self.epsilon = max(self.rl.epsilon_min, self.epsilon * self.rl.epsilon_decay)
