"""Radio energy and link-quality calculations for individual transmissions."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import RadioConfig


class RadioModel:
    """Vectorisable energy/PRR model for a fixed frame size."""

    def __init__(self, cfg: RadioConfig):
        self.cfg = cfg
        self.packet_bits = cfg.packet_bits
        self.frame_time_s = cfg.frame_time_s
        self.e_rx = cfg.rx_energy_j
        self.e_tx_max = cfg.tx_energy_j(1e9)          # worst case (full power)
        self.max_retx = cfg.max_retx
        self._d_grid = np.linspace(1.0, max(cfg.max_range_m() * 1.5, 50.0), 1024)
        self._prr_grid = np.array([cfg.prr(d) for d in self._d_grid])

    # -- point evaluations -------------------------------------------------
    def tx_energy(self, dist_m: float) -> float:
        return self.cfg.tx_energy_j(dist_m)

    def prr(self, dist_m: float) -> float:
        return self.cfg.prr(dist_m)

    def hop_energy(self, dist_m: float) -> float:
        """Energy a sender spends (transmit + its own receive overhead)."""
        return self.cfg.tx_energy_j(dist_m) + self.e_rx

    def hop_success(self, dist_m: float) -> float:
        """Probability that at least one of ``max_retx`` attempts gets through."""
        p = float(np.clip(self.cfg.prr(dist_m), 1e-9, 1.0))
        return float(1.0 - (1.0 - p) ** self.max_retx)

    # -- vectorised helpers ------------------------------------------------
    def tx_energy_vec(self, dist_m: np.ndarray) -> np.ndarray:
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        need = self.cfg.p_sens_dbm + (self.cfg.pathloss_1m_db + 10.0 * self.cfg.pathloss_exp * np.log10(d)) + self.cfg.fade_margin_db
        p_dbm = np.clip(need, self.cfg.p_tx_min_dbm, self.cfg.p_tx_max_dbm)
        p_w = 10.0 ** (p_dbm / 10.0) * 1e-3
        return self.cfg.energy_scale * p_w * self.frame_time_s

    def prr_vec(self, dist_m: np.ndarray) -> np.ndarray:
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        pl = self.cfg.pathloss_1m_db + 10.0 * self.cfg.pathloss_exp * np.log10(d)
        margin = self.cfg.p_tx_max_dbm - pl - self.cfg.p_sens_dbm
        return 1.0 / (1.0 + np.exp(-margin / self.cfg.prr_slope_db))

    @property
    def energy_per_hop_budget(self) -> float:
        """Nominal cost of one worst-case forwarding step."""
        return self.e_tx_max + self.e_rx

    def hop_cost_normalised(self, dist_m: float, budget_j: float) -> float:
        return self.hop_energy(dist_m) / max(budget_j, 1e-12)

    # -- fleet statistics --------------------------------------------------
    def packets_per_battery(self, dist_m: float) -> int:
        e = self.hop_energy(dist_m)
        return int(max(0.0, self.cfg.e_init_j / e)) if e > 0 else 0
