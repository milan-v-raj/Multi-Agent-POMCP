"""Generate relay positions, neighbour lists and hop distances.

Deployment may move disconnected components to obtain gateway connectivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DeploymentConfig


@dataclass
class Topology:
    """Static geometry of one deployment."""

    positions: np.ndarray            # (N, 2) node coordinates in metres
    bs_position: np.ndarray          # (2,) base-station coordinate
    comm_range_m: float
    dist: np.ndarray                 # (N, N) pairwise node distances
    dist_bs: np.ndarray              # (N,) node -> base-station distances
    neighbors: List[List[int]]       # adjacency lists (static, radio-range based)
    initial_energy_j: np.ndarray     # (N,) initial residual energy
    energy_threshold_j: float        # absolute threshold below which a node is out

    @property
    def n_nodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_edges(self) -> int:
        return sum(len(nb) for nb in self.neighbors) // 2

    def hops_via_bfs(self, sources: Sequence[int]) -> Dict[int, int]:
        """Hop count from every node to the BS through *eligible* nodes only."""
        return self._hop_map(set(range(self.n_nodes)))

    def hop_map(self, eligible: Sequence[bool]) -> np.ndarray:
        """Vectorised multihop distance (in hops) for a given eligibility mask."""
        mask = np.zeros(self.n_nodes, dtype=bool) if eligible is None else np.asarray(eligible, bool)
        return np.asarray(self._hops(mask), dtype=np.int32)

    # -- internals ---------------------------------------------------------
    def _hops(self, eligible) -> List[int]:
        n = self.n_nodes
        INF = 10 ** 6
        hops = [INF] * n
        frontier: List[int] = []
        for i in range(n):
            if eligible[i] and self.dist_bs[i] <= self.comm_range_m:
                hops[i] = 1
                frontier.append(i)
        depth = 1
        while frontier:
            depth += 1
            nxt: List[int] = []
            for i in frontier:
                for j in self.neighbors[i]:
                    if eligible[j] and hops[j] == INF:
                        hops[j] = depth
                        nxt.append(j)
            frontier = nxt
        return hops

    def _hop_map(self, eligible) -> Dict[int, int]:
        return {i: h for i, h in enumerate(self._hops(eligible))}

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "positions": self.positions.tolist(),
            "bs_position": self.bs_position.tolist(),
            "comm_range_m": self.comm_range_m,
        }


def deploy_topology(
    cfg: DeploymentConfig,
    rng: np.random.Generator,
    jitter: bool = True,
) -> Topology:
    """Sample one random deployment according to ``cfg``."""
    n = cfg.n_nodes
    w, h = cfg.area_m
    pos = rng.uniform(0.0, 1.0, size=(n, 2)) * np.array([w, h])
    bs = np.asarray(cfg.bs_position, dtype=float)

    # --- connectivity repair (component merging, minimally invasive) ---------
    # A random draw can be unsolvable in two ways:
    #   (a) no relay within radio range of the gateway -> PDR = 0 for *every*
    #       policy, so the instance carries no information about routing;
    #   (b) the relay graph splits into disconnected components, so sources in
    #       a component without a gateway-adjacent node can never deliver.
    # We deliberately do **not** mass-relocate nodes: an earlier version pulled
    # every sparse node toward the centroid and collapsed the field into a fully
    # meshed cluster (435 edges, 1-hop diameter, i.e. no routing problem left).
    # Instead each component is shifted as a rigid body until one of its nodes
    # sits within radio range of the nearest node of the component that already
    # contains the gateway.
    lo, hi = 0.0, np.array([w, h])
    rng_m = cfg.comm_range_m

    def components(p: np.ndarray) -> list:
        d = _pairwise(p)
        adj = d <= rng_m
        seen = np.zeros(n, dtype=bool)
        comps = []
        for s in range(n):
            if seen[s]:
                continue
            stack, comp = [s], []
            seen[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in np.where(adj[u] & ~seen)[0]:
                    seen[v] = True
                    stack.append(int(v))
            comps.append(comp)
        return comps

    # (a) gateway must have at least one neighbour
    d_bs = np.linalg.norm(pos - bs, axis=1)
    if d_bs.min() > rng_m * 0.8:
        k = int(np.argmin(d_bs))
        direction = (bs - pos[k]) / max(np.linalg.norm(bs - pos[k]), 1e-9)
        pos[k] = np.clip(bs - direction * rng_m * 0.7, lo, hi)

    # (b) merge every component into the gateway-connected one
    for _ in range(n):
        comps = components(pos)
        if len(comps) <= 1:
            break
        near_bs = np.linalg.norm(pos - bs, axis=1) <= rng_m
        main = next((c for c in comps if near_bs[c].any()), None)
        if main is None:              # pragma: no cover - (a) guarantees one
            main = max(comps, key=len)
        others = [c for c in comps if c is not main]
        if not others:
            break
        # move the component whose closest approach to `main` is smallest
        best_move, best_gap = None, np.inf
        for c in others:
            d = _pairwise(pos)[np.ix_(c, main)]
            i_loc, j_loc = np.unravel_index(np.argmin(d), d.shape)
            if d[i_loc, j_loc] < best_gap:
                best_gap = d[i_loc, j_loc]
                best_move = (c, c[i_loc], main[j_loc])
        c, i_glob, j_glob = best_move
        shift = pos[j_glob] + (pos[j_glob] - pos[i_glob]) / max(best_gap, 1e-9) * (0.6 * rng_m)
        delta = shift - pos[i_glob]
        pos[c] = pos[c] + delta
        # keep the moved component inside the field
        pos[c] = np.clip(pos[c], lo, hi)
        if np.allclose(delta, 0):
            break


    d = _pairwise(pos)
    dbs = np.linalg.norm(pos - bs, axis=1)
    neighbors: List[List[int]] = []
    for i in range(n):
        nb = [j for j in range(n) if j != i and d[i, j] <= cfg.comm_range_m]
        neighbors.append(nb)

    if jitter:
        lo, hi = 1.0 - cfg.energy_jitter, 1.0
        frac = rng.uniform(lo, hi, size=n)
    else:
        frac = np.ones(n)

    return Topology(
        positions=pos,
        bs_position=bs,
        comm_range_m=cfg.comm_range_m,
        dist=d,
        dist_bs=dbs,
        neighbors=neighbors,
        initial_energy_j=frac * 5.0,     # scaled by RadioConfig.e_init_j downstream
        energy_threshold_j=cfg.energy_threshold * 5.0,
    )


def _pairwise(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def select_sources(
    topo: Topology,
    n_sources: int,
    rng: np.random.Generator,
    min_hops: int = 2,
) -> List[int]:
    """Pick traffic sources that are genuinely multi-hop from the gateway."""
    hops = topo._hops([True] * topo.n_nodes)
    cand = [i for i in range(topo.n_nodes) if hops[i] >= min_hops and hops[i] < 10 ** 6]
    if len(cand) < n_sources:
        cand = [i for i in range(topo.n_nodes) if hops[i] < 10 ** 6]
    if not cand:
        cand = list(range(topo.n_nodes))
    pick = rng.choice(np.array(cand), size=min(n_sources, len(cand)), replace=False)
    return sorted(int(p) for p in pick)
