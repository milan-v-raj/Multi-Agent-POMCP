"""Experimental fully observed state for a single routing episode.

This subclass changes only the state key, not the original task's actions,
rewards or transition logic. Each instance represents one fixed topology and
fixed simulator/RL configuration. A Q-table must not be shared across different
instances with different topology or configuration.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dos_rl.mdp import RoutingMDP, TERM_NONE


class FullRouteState(NamedTuple):
    """Exact, hashable snapshot of the inputs to ``actions/rewards/step``.

    Array fields contain little-endian float64, int64 or uint8 bytes in C order,
    with fixed shapes determined by the instance's fixed topology. Packing into
    immutable bytes both preserves exact values and makes Q-table keys hashable.
    These quantities are the *episode model's* values, which are synthetic
    snapshots or estimates, not necessarily the true physical network state.
    """
    relay: int
    steps: int
    visited: tuple[int, ...]
    energy: bytes
    queue: bytes
    load: bytes
    prr: bytes
    eligible: bytes
    hops: bytes
    link_up: bytes


def _bytes(values, dtype):
    return np.ascontiguousarray(values, dtype=dtype).tobytes()


class FullStateRoutingMDP(RoutingMDP):
    """Full-snapshot route MDP; deliberately not a closed-loop network model.

    The topology, radio, configuration and transition rules are fixed for the
    lifetime of an episode/MDP; link up/down status is dynamic and encoded.
    For independent fresh transition randomness, ``FullRouteState`` includes all
    mutable quantities currently read by ``actions``, ``rewards`` and ``step``.
    If those implementations gain new dependencies, this encoding and its tests
    must be audited. Do not treat the key as a Bayesian belief or use this class
    to claim that delayed real-world telemetry is fully observable.
    """

    MODE = "full_route_snapshot"

    def __init__(self, sim, cfg, rl, belief_tracker):
        super().__init__(sim, cfg, rl, belief_tracker, state_mode=self.MODE)

    def encode(self, node: int, mode: str | None = None) -> FullRouteState:
        if mode is not None and mode != self.MODE:
            raise ValueError(f"FullStateRoutingMDP only supports {self.MODE!r}")
        if self.state_mode != self.MODE:
            raise ValueError("Changing state_mode on FullStateRoutingMDP is unsupported")
        if node < 0 or node != self._cur:
            raise ValueError("Encode the current relay of an active routing episode")
        b = self.ep.belief
        return FullRouteState(
            relay=int(node),
            steps=int(self._t),
            visited=tuple(sorted(self._visited)),
            energy=_bytes(b.energy, "<f8"),
            queue=_bytes(b.queue_frac, "<f8"),
            load=_bytes(b.load, "<f8"),
            prr=_bytes(b.prr, "<f8"),
            eligible=_bytes(self.ep.eligible, "u1"),
            hops=_bytes(self.ep.hops, "<i8"),
            link_up=_bytes(self.sim.link_up, "u1"),
        )

    def _mk(self, next_node, r_vec, F, terminal, terminal_reward, info=None):
        # The base class assembles the same action result as before. Use an
        # explicit absorbing key for *every* terminal, including failure at a
        # relay: an identical live snapshot is not a terminal state.
        result = super()._mk(next_node, r_vec, F, terminal, terminal_reward, info)
        if terminal != TERM_NONE:
            result.next_state = ("terminal", terminal)
        return result
