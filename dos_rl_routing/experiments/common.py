"""Shared utilities for the DOS-RL experiment suite.

Design rules
------------
* Every experiment is **seeded** and writes both raw per-run CSV and an
  aggregated summary CSV into ``results/``.
* Figures go to ``figures/`` at 200 dpi with a consistent house style, so the
  report can be rebuilt from scratch with one command.
* Nothing here mutates global state: each function returns what it computed.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.abspath(os.environ.get("DOS_RL_OUTPUT_BASE", BASE))
RESULTS = os.path.join(OUTPUT_BASE, "results")
FIGURES = os.path.join(OUTPUT_BASE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


# ---------------------------------------------------------------------------
# plotting style
# ---------------------------------------------------------------------------
PALETTE = {
    "dos_rl": "#1f77b4",
    "q_learning": "#1f77b4",
    "sarsa": "#2ca02c",
    "mc_control": "#9467bd",
    "linear_fa": "#d62728",
    "weighted_sum": "#ff7f0e",
    "shortest_path": "#7f7f7f",
    "ospf": "#7f7f7f",
    "sdn_q": "#8c564b",
    "energy_aware_greedy": "#bcbd22",
    "random": "#c7c7c7",
}

LABELS = {
    "dos_rl": "DOS-RL (ours)",
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
    "mc_control": "Monte-Carlo control",
    "linear_fa": "Linear FA (belief)",
    "weighted_sum": "Weighted-sum RL",
    "shortest_path": "Shortest path (OSPF)",
    "energy_aware_greedy": "Energy-aware greedy",
    "random": "Random",
}


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )
    return plt


def colour(name: str) -> str:
    return PALETTE.get(name, "#333333")


def label(name: str) -> str:
    return LABELS.get(name, name)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def smooth(x: np.ndarray, window: int = 20) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size < window:
        return x
    k = np.ones(window) / window
    return np.convolve(x, k, mode="valid")


def mean_ci(runs: np.ndarray, axis: int = 0) -> tuple:
    """Mean and 95 % CI half-width across independent seeds."""
    runs = np.asarray(runs, float)
    m = runs.mean(axis=axis)
    if runs.shape[axis] < 2:
        return m, np.zeros_like(m)
    sd = runs.std(axis=axis, ddof=1)
    return m, 1.96 * sd / np.sqrt(runs.shape[axis])


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean (robust for small N)."""
    x = np.asarray(x, float)
    if x.size < 2:
        return float(x.mean()) if x.size else float("nan"), (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), (float(lo), float(hi))


def wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test with a rank-biserial effect size."""
    from scipy.stats import wilcoxon as _w

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    if d.size < 5 or np.allclose(d, 0):
        return {"stat": float("nan"), "p": float("nan"), "effect": 0.0, "n": int(d.size)}
    try:
        stat, p = _w(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return {"stat": float("nan"), "p": float("nan"), "effect": 0.0, "n": int(d.size)}
    # rank-biserial correlation
    ranks = _rankdata(np.abs(d))
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    effect = float((pos - neg) / (pos + neg)) if (pos + neg) > 0 else 0.0
    return {"stat": float(stat), "p": float(p), "effect": effect, "n": int(d.size)}


def _rankdata(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1] (>0.33 large, >0.474 very large)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def save_csv(rows: Iterable[dict], name: str) -> str:
    import csv

    rows = list(rows)
    path = os.path.join(RESULTS, name)
    if not rows:
        return path
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})
    return path


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return int(v)
    return v


def save_json(obj, name: str) -> str:
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def fig_path(name: str) -> str:
    return os.path.join(FIGURES, name)


@contextmanager
def timer(msg: str):
    t0 = time.time()
    print(f"[start] {msg}", flush=True)
    try:
        yield
    finally:
        print(f"[done ] {msg}  ({time.time() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# topology / scenario helpers
# ---------------------------------------------------------------------------
def make_scene(seed: int, n_nodes: int = 30, n_sources: int = 5, cfg=None):
    """Build a reproducible (topology, sources, simulator) triple."""
    from dos_rl import NetworkSimulator, SimConfig, deploy_topology
    from dos_rl.topology import select_sources

    cfg = cfg or SimConfig()
    cfg = cfg.with_(n_nodes=n_nodes, n_sources=n_sources)
    rng = np.random.default_rng(1000 + seed)
    topo = deploy_topology(cfg.deployment, rng)
    srcs = select_sources(topo, n_sources, rng)
    sim = NetworkSimulator(topo, cfg)
    return topo, srcs, sim, cfg


def summarise_flow(res) -> Dict[str, float]:
    return {
        "pdr_pct": res.pdr * 100.0,
        "e2e_delay_ms": res.e2e_delay_s * 1e3,
        "energy_efficiency": res.energy_efficiency,
        "avg_power_w": res.avg_power_w,
        "total_energy_j": res.total_energy_j,
        "fairness": res.fairness,
        "load_imbalance": res.load_imbalance,
        "mean_hops": res.mean_hops,
        "mean_attempts": res.mean_attempts,
        "throughput_pps": res.throughput_pps,
    }


def failure_events(t_fail: float, fraction: float, kind: str, seed: int):
    from dos_rl.network_sim import FailureEvent

    return [FailureEvent(time_s=t_fail, kind=kind, fraction=fraction, rng=np.random.default_rng(seed + 777))]


def adaptation_metrics(times: Sequence[float], pdr: Sequence[float], t_fail: float) -> Dict[str, float]:
    """Quantify the dip and the recovery after an injected failure."""
    t = np.asarray(times, float)
    p = np.asarray(pdr, float)
    pre = p[(t > t_fail - 30) & (t <= t_fail)]
    post = p[t > t_fail]
    if pre.size == 0 or post.size == 0:
        return {"pdr_pre": float("nan"), "pdr_post": float("nan"), "dip_area": float("nan"),
                "recovery_s": float("nan"), "pdr_min": float("nan")}
    base = float(pre.mean())
    dip = float(np.trapezoid(np.clip(base - post, 0, None), t[t > t_fail]) / 1e3) if post.size > 1 else 0.0
    thresh = base - 0.05 * max(base, 1e-9)          # within 5 % of pre-failure level
    recovery = float("nan")
    above = np.where(post >= thresh)[0]
    if above.size:
        # need to stay above threshold for the rest of the run
        for i in above:
            if np.all(post[i:] >= thresh):
                recovery = float(t[t > t_fail][i] - t_fail)
                break
    return {
        "pdr_pre": base,
        "pdr_post": float(post.mean()),
        "pdr_min": float(post.min()),
        "dip_area": dip,
        "recovery_s": recovery,
    }
