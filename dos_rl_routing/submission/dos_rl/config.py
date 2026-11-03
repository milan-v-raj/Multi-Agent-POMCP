"""Configuration for the deployment, radio model, traffic and learning agents."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class DeploymentConfig:
    """Physical deployment of relay nodes (RNs) and the base station (BS)."""

    n_nodes: int = 30                      # paper Table 2: 30 / 45 / 60 RNs
    area_m: Tuple[float, float] = (1500.0, 1500.0)  # paper Table 2: 1500 x 1500 m
    bs_position: Tuple[float, float] = (1500.0, 1500.0)  # sink at a corner
    comm_range_m: float = 450.0            # nominal radio range
    min_degree: int = 2                    # unused; no minimum-degree guarantee
    energy_threshold: float = 0.10         # project choice; paper gives no numeric threshold
    energy_jitter: float = 0.05            # initial residual-energy spread (fraction)


@dataclass(frozen=True)
class RadioConfig:
    """Radio link budget, reception probability and transmission energy settings."""

    e_init_j: float = 5.0                  # paper Table 2
    packet_bits: int = 512                 # paper Table 2
    bitrate_bps: float = 1_000_000.0        # 802.15.4g-class PHY (512 b = 0.51 ms airtime)
    p_tx_max_dbm: float = 14.0             # long-range sub-GHz class
    p_tx_min_dbm: float = -20.0
    p_sens_dbm: float = -101.0             # CC2420 sensitivity
    p_rx_mw: float = 59.1                  # CC2420 receive power
    pathloss_1m_db: float = 40.0
    pathloss_exp: float = 2.7
    fade_margin_db: float = 3.0
    prr_slope_db: float = 6.0
    energy_scale: float = 1.0              # global knob for lifetime sweeps
    max_retx: int = 3                      # MAC-layer ARQ attempts

    # -- derived ----------------------------------------------------------
    @property
    def frame_time_s(self) -> float:
        return self.packet_bits / self.bitrate_bps

    @property
    def p_rx_w(self) -> float:
        return self.p_rx_mw * 1e-3

    def pathloss_db(self, dist_m: float) -> float:
        import math

        d = max(float(dist_m), 1.0)
        return self.pathloss_1m_db + 10.0 * self.pathloss_exp * math.log10(d)

    def p_tx_dbm(self, dist_m: float) -> float:
        raw = self.p_sens_dbm + self.pathloss_db(dist_m) + self.fade_margin_db
        return float(min(self.p_tx_max_dbm, max(self.p_tx_min_dbm, raw)))

    def p_tx_w(self, dist_m: float) -> float:
        return 10.0 ** (self.p_tx_dbm(dist_m) / 10.0) * 1e-3

    def tx_energy_j(self, dist_m: float) -> float:
        """Transmitter energy for one frame at distance ``dist_m``."""
        return self.energy_scale * self.p_tx_w(dist_m) * self.frame_time_s

    @property
    def rx_energy_j(self) -> float:
        return self.energy_scale * self.p_rx_w * self.frame_time_s

    @property
    def e_rx(self) -> float:
        return self.rx_energy_j

    def prr(self, dist_m: float) -> float:
        """Per-frame packet reception ratio (logistic in SNR margin)."""
        import math

        margin_db = self.p_tx_max_dbm - self.pathloss_db(dist_m) - self.p_sens_dbm
        return float(1.0 / (1.0 + math.exp(-margin_db / self.prr_slope_db)))

    def max_range_m(self) -> float:
        """Distance at which the link budget is exhausted."""
        need_db = self.p_tx_max_dbm - self.p_sens_dbm - self.pathloss_1m_db
        return float(10.0 ** (need_db / (10.0 * self.pathloss_exp)))


@dataclass(frozen=True)
class TrafficConfig:
    """UDP CBR traffic + M/M/1/K forwarding queues."""

    n_sources: int = 5                     # paper Table 2: 2,4,6,8,10 traffic sources
    pkt_rate_pps: float = 10.0             # paper Table 2: 10 packets/s per source
    buffer_pkts: int = 20                  # RN buffer capacity (K in M/M/1/K)
    sim_time_s: float = 200.0              # paper Table 2: 200 s
    service_cv: float = 1.0                # service-time variability (1.0 = exponential)
    agg_factor: float = 4.0                # heuristic near-sink traffic multiplier


@dataclass(frozen=True)
class SimConfig:
    """Full simulator configuration."""

    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)

    robust_routing: bool = True            # True -> rerun the SAME policy at t=100 s
    warmup_frac: float = 0.05              # excluded from queue statistics
    route_lifetime_s: float = 3.0          # paper Sec. 3.2.2: "route lifetime = 3 s"

    def with_(self, **overrides) -> "SimConfig":
        """Return a copy with nested overrides, e.g. ``cfg.with_(n_nodes=45)``."""
        dep_names = set(DeploymentConfig.__dataclass_fields__)
        radio_names = set(RadioConfig.__dataclass_fields__)
        traf_names = set(TrafficConfig.__dataclass_fields__)

        dep = {k: v for k, v in overrides.items() if k in dep_names}
        rad = {k: v for k, v in overrides.items() if k in radio_names}
        tra = {k: v for k, v in overrides.items() if k in traf_names}
        top = {
            k: v
            for k, v in overrides.items()
            if k not in dep_names and k not in radio_names and k not in traf_names
        }
        return SimConfig(
            deployment=replace(self.deployment, **dep),
            radio=replace(self.radio, **rad),
            traffic=replace(self.traffic, **tra),
            **top,
        )

    @property
    def energy_budget_j(self) -> float:
        """Per-packet routing energy cap used to normalise the energy reward."""
        cap = self.deployment.comm_range_m
        return 10.0 * (self.radio.tx_energy_j(cap) + self.radio.rx_energy_j)


@dataclass(frozen=True)
class RLConfig:
    """Learning rates, exploration, rewards and action restrictions."""

    alpha: float = 0.3                     # paper Table 2
    gamma: float = 0.85                    # paper Table 2
    epsilon: float = 0.3                   # paper Table 1
    epsilon_min: float = 0.01              # paper Table 1: phi
    epsilon_decay: float = 0.995           # paper Table 1: beta
    decay_epsilon: bool = True

    confidence_psi: float = 0.1            # Eq. 2 mixing weight
    # "argmax" (paper Eq. 3 literal) | "argmin" (Algorithm 1 literal) |
    # "var" (most stable) | "value" (best estimated value; used by default)
    dos_select: str = "value"

    reward_mode: str = "paper"             # "paper" | "corrected"
    reward_weights: Tuple[float, ...] = (1.0, 1.0, 1.0)  # reporting; grid scalarisation
    goal_reward: float = 10.0              # terminal bonus for reaching the gateway
    dead_reward: float = -10.0             # terminal penalty on the network task
    collision_reward: float = -20.0        # paper Sec. 4.2: obstacle / depletion hit
    use_shaping: bool = True

    epsilon_convention: str = "standard"   # "standard" | "paper"
    alpha_schedule: str = "constant"       # "constant" | "decay"
    alpha_decay: float = 0.9995

    # Action-set restriction. "dag" normally favours lower BFS hop counts
    # but has a geographic fallback; "stp" allows hop slack. Neither should be
    # treated as a literal STP implementation or an unconditional guarantee.
    # "visited" masks episode-local revisits; "none" allows all candidates.
    loop_free: str = "dag"
    dag_slack: float = 2.5
    max_hops: int = 20                     # safety cap on episode length

    def with_(self, **overrides) -> "RLConfig":
        return replace(self, **overrides)


def default_config() -> SimConfig:
    return SimConfig()


# Convenience: the paper's exact Table 1 / Table 2 settings.
GRID_RL = RLConfig(alpha=0.1, gamma=0.99, epsilon=0.3, epsilon_decay=0.995)
NET_RL = RLConfig(alpha=0.3, gamma=0.85, epsilon=0.3, decay_epsilon=False)
