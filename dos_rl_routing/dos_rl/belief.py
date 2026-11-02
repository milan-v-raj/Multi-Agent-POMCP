"""Belief state for the PO-MDP agent.

The paper tracks a per-(state, action, objective) *variance* with

    VAR(s, a, o) <- (1 - psi) VAR(s, a, o) + psi * delta_o^2          (Eq. 2)

which is an EWMA of the *squared TD error*, not of the value distribution: it
grows whenever the reward scale is large, whether or not the policy is stable.
We therefore implement two variants side by side

* ``paper``  : EWMA of delta^2 exactly as Eq. 2;
* ``true``   : Welford/EWMA of the observed objective reward, i.e. a genuine
               estimate of Var[R_o | s, a],

and E4/E8 in the report quantify what the difference does to objective
selection. The confidence metric of Eq. 3 is applied to whichever is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .observer import LinkMetrics


@dataclass
class BeliefState:
    """Running estimator of the hidden network state."""

    energy: np.ndarray          # E[residual energy fraction]
    energy_var: np.ndarray      # Var[residual energy fraction]
    load: np.ndarray            # E[utilisation]
    load_var: np.ndarray
    queue_frac: np.ndarray
    prr: np.ndarray             # (N, N) E[PRR]
    prr_var: np.ndarray
    alive_prob: np.ndarray      # P(node eligible)
    age: float = 0.0

    def copy(self) -> "BeliefState":
        return BeliefState(
            energy=self.energy.copy(),
            energy_var=self.energy_var.copy(),
            load=self.load.copy(),
            load_var=self.load_var.copy(),
            queue_frac=self.queue_frac.copy(),
            prr=self.prr.copy(),
            prr_var=self.prr_var.copy(),
            alive_prob=self.alive_prob.copy(),
            age=self.age,
        )

    # -- features used by the linear function approximator -----------------
    def node_features(self, i: int, j: int, gateway_prr: Optional[float] = None) -> np.ndarray:
        """Action features; the gateway is not a relay array index."""
        if j < 0:
            if gateway_prr is None:
                raise ValueError("gateway_prr is required for a gateway action")
            return np.array([1.0, 1.0, self.energy[i], 0.0, 0.0,
                             self.load[i], gateway_prr, 1.0, 0.0, self.age], dtype=float)
        return np.array(
            [
                1.0,
                self.energy[j],
                self.energy[i],
                self.queue_frac[j],
                self.load[j],
                self.load[i],
                self.prr[i, j],
                self.alive_prob[j],
                self.energy_var[j],
                self.age,
            ],
            dtype=float,
        )

    FEATURE_NAMES: Tuple[str, ...] = (
        "bias",
        "E_j",
        "E_i",
        "Q_j",
        "load_j",
        "load_i",
        "prr_ij",
        "alive_j",
        "var_E_j",
        "age",
    )


class BeliefTracker:
    """Turns telemetry into a belief, with tunable smoothing and variance style."""

    def __init__(self, topo, beta: float = 0.35, variance_mode: str = "true"):
        n = topo.n_nodes
        self.n = n
        self.beta = float(beta)
        self.variance_mode = variance_mode
        self.state: Optional[BeliefState] = None
        self._init_arrays = (
            np.ones(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            None,
            None,
            np.ones(n),
        )

    # ------------------------------------------------------------------
    def reset(self, initial_energy: Optional[np.ndarray] = None):
        n = self.n
        e0 = np.ones(n) if initial_energy is None else np.asarray(initial_energy, float)
        self.state = BeliefState(
            energy=e0.copy(),
            energy_var=np.full(n, 1e-3),
            load=np.zeros(n),
            load_var=np.zeros(n),
            queue_frac=np.zeros(n),
            prr=np.full((n, n), 0.5),
            prr_var=np.full((n, n), 1e-3),
            alive_prob=np.ones(n),
            age=0.0,
        )

    # ------------------------------------------------------------------
    def update(self, m: LinkMetrics, initial_energy: Optional[np.ndarray] = None):
        if self.state is None:
            self.reset(initial_energy)
        b = self.state
        beta = self.beta

        # ---- energy: EWMA mean + variance ------------------------------
        err = m.energy - b.energy
        b.energy = b.energy + beta * err
        if self.variance_mode == "true":
            b.energy_var = (1 - beta) * b.energy_var + beta * err * err
        else:  # paper Eq. 2 style on the (normalised) TD error
            b.energy_var = (1 - 0.1) * b.energy_var + 0.1 * (err ** 2)

        # ---- load ------------------------------------------------------
        b.load = b.load + beta * (m.load - b.load)
        b.queue_frac = b.queue_frac + beta * (m.queue_frac - b.queue_frac)
        b.load_var = (1 - beta) * b.load_var + beta * (m.load - b.load) ** 2

        # ---- link quality ---------------------------------------------
        perr = m.prr - b.prr
        b.prr = b.prr + beta * perr
        b.prr_var = (1 - beta) * b.prr_var + beta * perr * perr

        # ---- liveness ---------------------------------------------------
        b.alive_prob = b.alive_prob + beta * (m.alive.astype(float) - b.alive_prob)
        b.age = float(m.age_epochs)

    # ------------------------------------------------------------------
    def confidence(self, var: np.ndarray, mean: np.ndarray, k: int = 1) -> np.ndarray:
        """Eq. 3 confidence, computed over one state's action set.

        ``C_k = K (mu_k + e^{sigma^2_k}) / sum_k (mu_k + e^{sigma^2_k})``
        Unstable objectives carry a large variance and therefore a large
        numerator, so *lower* C means *more* trustworthy -- hence Algorithm 1's
        ``argmin``. The paper's prose says "highest certainty"; both readings are
        supported by ``RLConfig.dos_select``.
        """
        num = mean + np.exp(np.clip(var, -20.0, 20.0))
        denom = np.sum(num, axis=0, keepdims=True)
        return num / np.maximum(denom, 1e-12)
