"""Controller observation channel -- the ``O`` and ``Omega`` of the PO-MDP.

The framework paper assumes the SDN controller holds a perfect, instantaneous
global view. Real SDWSN controllers do not: cluster-head reports are periodic,
carried over the same lossy radios, quantised to save bytes, and arrive after a
delay. This module models that channel with four knobs, so partial
observability can be dialled from *none* (paper setting) to *severe*:

    delay_epochs   report age in route-recomputation epochs
    noise_*        additive Gaussian noise on the reported quantity
    quant_energy   energy quantisation step (fraction of full battery)
    loss           probability that a report is dropped entirely

``ControllerObserver.observe`` therefore returns a :class:`LinkMetrics`
object that may differ from the true state. Learned policies use the telemetry
for the belief tracker, but action masking and some baselines also access
simulator ground truth; see MDP.md for this observability limitation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class LinkMetrics:
    """One round of controller telemetry (an out-of-date, noisy snapshot)."""

    energy: np.ndarray          # residual energy fraction in [0, 1]
    load: np.ndarray            # utilisation estimate (rho), clipped to [0, 1]
    queue_frac: np.ndarray      # buffer occupancy fraction in [0, 1]
    prr: np.ndarray             # (N, N) packet reception-ratio estimate
    alive: np.ndarray           # node-eligibility report (bool)
    age_epochs: int = 0

    def copy(self) -> "LinkMetrics":
        return LinkMetrics(
            energy=self.energy.copy(),
            load=self.load.copy(),
            queue_frac=self.queue_frac.copy(),
            prr=self.prr.copy(),
            alive=self.alive.copy(),
            age_epochs=self.age_epochs,
        )


class ControllerObserver:
    """Turns the ground truth into what the controller is allowed to know."""

    def __init__(
        self,
        topo,
        delay_epochs: int = 0,
        noise_energy: float = 0.0,
        noise_load: float = 0.0,
        noise_prr: float = 0.0,
        quant_energy: float = 0.0,
        loss: float = 0.0,
        prr_ema: float = 0.7,
        seed: int = 0,
    ):
        self.topo = topo
        self.delay_epochs = int(delay_epochs)
        self.noise_energy = float(noise_energy)
        self.noise_load = float(noise_load)
        self.noise_prr = float(noise_prr)
        self.quant_energy = float(quant_energy)
        self.loss = float(loss)
        self.prr_ema = float(prr_ema)
        self.rng = np.random.default_rng(seed)
        self._history: deque = deque(maxlen=max(1, self.delay_epochs + 1))
        self._last: Optional[LinkMetrics] = None
        self._prr_smooth: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def observe(
        self,
        energy: np.ndarray,
        load: np.ndarray,
        queue_frac: np.ndarray,
        prr: np.ndarray,
        alive: np.ndarray,
    ) -> LinkMetrics:
        n = energy.size
        # --- link quality is measured, so it is always EWMA-smoothed -------
        if self._prr_smooth is None:
            self._prr_smooth = prr.copy()
        else:
            self._prr_smooth = self.prr_ema * self._prr_smooth + (1 - self.prr_ema) * prr
        m = LinkMetrics(
            energy=np.asarray(energy, float).copy(),
            load=np.asarray(load, float).copy(),
            queue_frac=np.asarray(queue_frac, float).copy(),
            prr=self._prr_smooth.copy(),
            alive=np.asarray(alive, bool).copy(),
        )
        # --- quantise / noise / drop --------------------------------------
        if self.quant_energy > 0:
            m.energy = np.round(m.energy / self.quant_energy) * self.quant_energy
        if self.noise_energy > 0:
            m.energy = m.energy + self.rng.normal(0.0, self.noise_energy, size=n)
        m.energy = np.clip(m.energy, 0.0, 1.0)
        if self.noise_load > 0:
            m.load = np.clip(m.load + self.rng.normal(0.0, self.noise_load, size=n), 0.0, 1.0)
            m.queue_frac = np.clip(
                m.queue_frac + self.rng.normal(0.0, self.noise_load, size=n), 0.0, 1.0
            )
        if self.noise_prr > 0:
            m.prr = np.clip(
                m.prr + self.rng.normal(0.0, self.noise_prr, size=m.prr.shape), 0.0, 1.0
            )
        if self.loss > 0:
            drop = self.rng.random(n) < self.loss
            if drop.any() and self._last is not None:
                m.energy[drop] = self._last.energy[drop]
                m.load[drop] = self._last.load[drop]
                m.queue_frac[drop] = self._last.queue_frac[drop]
                m.alive[drop] = self._last.alive[drop]

        # --- propagation delay --------------------------------------------
        self._history.append(m)
        if self.delay_epochs > 0 and len(self._history) > self.delay_epochs:
            out = self._history[-(self.delay_epochs + 1)].copy()
            out.age_epochs = self.delay_epochs
        else:
            out = m.copy()
            out.age_epochs = min(self.delay_epochs, len(self._history) - 1)
        out.alive = out.alive & (out.energy > 0.0)
        self._last = m
        return out

    def reset(self):
        self._history.clear()
        self._last = None
        self._prr_smooth = None


# ---------------------------------------------------------------------------
class OracleObserver(ControllerObserver):
    """Perfect, instantaneous observation (the paper's assumption)."""

    def __init__(self, topo):
        super().__init__(topo, delay_epochs=0)


# ---------------------------------------------------------------------------
@dataclass
class PartialObsProfile:
    """Named severities of the observation channel."""

    name: str
    delay_epochs: int = 0
    noise_energy: float = 0.0
    noise_load: float = 0.0
    noise_prr: float = 0.0
    quant_energy: float = 0.0
    loss: float = 0.0

    def build(self, topo, seed: int = 0) -> ControllerObserver:
        return ControllerObserver(
            topo,
            delay_epochs=self.delay_epochs,
            noise_energy=self.noise_energy,
            noise_load=self.noise_load,
            noise_prr=self.noise_prr,
            quant_energy=self.quant_energy,
            loss=self.loss,
            seed=seed,
        )


OBS_PROFILES: Dict[str, PartialObsProfile] = {
    "full": PartialObsProfile("full"),
    "delay": PartialObsProfile("delay", delay_epochs=2),
    "quantised": PartialObsProfile("quantised", quant_energy=0.05, noise_energy=0.02),
    "noisy": PartialObsProfile("noisy", noise_energy=0.05, noise_load=0.05, noise_prr=0.05),
    "severe": PartialObsProfile(
        "severe", delay_epochs=2, noise_energy=0.05, noise_load=0.05, noise_prr=0.08,
        quant_energy=0.05, loss=0.15,
    ),
}
