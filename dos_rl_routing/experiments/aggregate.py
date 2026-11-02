"""Aggregate every experiment CSV into one traceable summary.

Every number quoted in REPORT.md can be re-derived by running::

    python -m experiments.aggregate            # to stdout
    python -m experiments.aggregate --csv      # tidy long-format CSV as well

The output is intentionally plain: each block names the source CSV, then the
mean (+/- 95% CI across seeds) of the relevant columns.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import RESULTS, save_csv

# (policy order for display, metric columns to print)
METRICS = [("pdr_pct", "PDR%"), ("e2e_delay_ms", "delay_ms"),
           ("energy_efficiency", "EE"), ("mean_hops", "hops")]


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r]


def num(v):
    try:
        out = float(v)
        return out if np.isfinite(out) else np.nan
    except (TypeError, ValueError):
        return np.nan


def agg(rows, keys, col):
    """mean and 95% CI of ``col`` grouped by the tuple ``keys``."""
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(num(r[col]))
    out = {}
    for k, vals in buckets.items():
        v = np.array([x for x in vals if np.isfinite(x)])
        if not len(v):
            out[k] = (np.nan, np.nan, 0)
            continue
        half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out[k] = (float(v.mean()), float(half), len(v))
    return out


def table(rows, keys, metric="pdr_pct", width=11):
    """Print a matrix: first key = row, second key = column."""
    a = agg(rows, keys, metric)
    rowvals = sorted({k[0] for k in a}, key=lambda x: (len(x), x))
    colvals = sorted({k[1] for k in a})
    print("    " + "".join(f"{str(c)[:width]:>{width}s}" for c in colvals))
    for rv in rowvals:
        cells = []
        for cv in colvals:
            m, h, n = a.get((rv, cv), (np.nan, np.nan, 0))
            cells.append(f"{m:{width-2}.1f}" if np.isfinite(m) else f"{'-':>{width}s}")
        print(f"    {str(rv)[:width-2]:<{width}s}" + " ".join(cells))


def by_policy(rows, title, group_keys=(), metric="pdr_pct"):
    """Print one row per (group..., policy) with mean +/- CI."""
    if not rows:
        print(f"    {title}: (no data)\n")
        return
    keys = (*group_keys, "policy")
    a = agg(rows, keys, metric)
    print(f"    {title}  [{metric}]")
    for k in sorted(a, key=lambda x: tuple(str(i) for i in x)):
        m, h, n = a[k]
        label = " ".join(str(i) for i in k)
        print(f"      {label:36s} {m:7.2f} +/-{h:5.2f}   n={n}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write results/SUMMARY.csv")
    args = ap.parse_args()
    flat = []

    def emit(section, keys, metric, value, ci, n):
        flat.append({"section": section, "group": "|".join(map(str, keys)),
                     "metric": metric, "mean": round(value, 4),
                     "ci95": round(ci, 4), "n": n})

    print("=" * 80)
    print("AGGREGATED RESULTS   (mean +/- 95% CI across seeds)")
    print("=" * 80)

    # -- E0 ---------------------------------------------------------------
    rows = load("e0_summary.csv")
    print("\n[E0] default network task (saturated regime)")
    a = agg(rows, ("agent",), "test_success_rate")
    print("    agent                success%      return      hops")
    for k in sorted(a):
        ag = k[0]
        s, sh, n = a[k]
        r = agg(rows, ("agent",), "test_return_mean")[k]
        h = agg(rows, ("agent",), "test_length_mean")[k]
        print(f"    {ag:18s} {s:7.2f}+/-{sh:4.2f}  {r[0]:7.2f}  {h[0]:6.2f}")
        emit("E0", k, "success_pct", s, sh, n)

    rows = load("e0b_coldstart.csv")
    print("\n[E0b] cold start on a hard instance")
    for k, (m, h, n) in sorted(agg(rows, ("policy",), "early_return").items()):
        late = agg(rows, ("policy",), "late_return")[k]
        print(f"    {k[0]:18s} early {m:+7.2f} -> late {late[0]:+7.2f}")
        emit("E0b", k, "early_return", m, h, n)
        emit("E0b", k, "late_return", late[0], late[1], late[2])

    # -- E1 ---------------------------------------------------------------
    rows = load("e1_gridworld.csv")
    print("\n[E1] paper gridworld replication")
    a = agg(rows, ("variant",), "final_success_pct")
    print("    variant                        success%")
    for k in sorted(a):
        s, sh, n = a[k]
        print(f"    {k[0]:30s} {s:6.2f} +/-{sh:5.2f}  n={n}")
        emit("E1", k, "success_pct", s, sh, n)

    # -- E2/E3 ------------------------------------------------------------
    rows = load("e2_e3_raw.csv")
    print("\n[E2/E3] network benchmark - PDR% by policy")
    for sweep in ("load", "sources", "nodes"):
        sel = [r for r in rows if r["sweep"] == sweep]
        if not sel:
            continue
        print(f"  -- {sweep}")
        table(sel, ("value", "policy"), "pdr_pct")
        for k, (m, h, n) in sorted(agg(sel, ("value", "policy"), "pdr_pct").items()):
            emit(f"E2.{sweep}", k, "pdr_pct", m, h, n)

    # -- E4 ---------------------------------------------------------------
    rows = load("e4_parameter_sweep.csv")
    print("\n[E4] hyperparameter sensitivity")
    if rows:
        print("    sweep     value   final_return  success%   conv_ep   PDRgain(pp)")
        for r in sorted(rows, key=lambda r: (r["sweep"], num(r["value"]))):
            print(f"    {r['sweep']:9s} {num(r['value']):5.2f}   "
                  f"{num(r['final_return']):+8.3f}  {num(r['final_success_pct']):7.2f}  "
                  f"{num(r['convergence_episode']):8.0f}  {num(r['pdr_delta_vs_baseline_pp']):+8.2f}")
            emit("E4", (r["sweep"], r["value"]), "final_return",
                 num(r["final_return"]), num(r["final_return_std"]), int(num(r["n_seeds"])))

    # -- E5 ---------------------------------------------------------------
    print("\n[E5] environmental change")
    rows = load("e5a_arrival_rate.csv")
    if rows:
        for k, (m, h, n) in sorted(agg(rows, ("policy",), "pdr_pct").items()):
            emit("E5a", k, "pdr_pct", m, h, n)
        print(f"    arrival-rate rows: {len(rows)}")
    for name, tag in (("e5b_link_failure.csv", "E5b"), ("e5c_node_failure.csv", "E5c"),
                      ("e5d_repeated_outages.csv", "E5d")):
        rows = load(name)
        if rows:
            print(f"  -- {tag}")
            table(rows, ("value", "policy"), "pdr_pct")
            for k, (m, h, n) in sorted(agg(rows, ("value", "policy"), "pdr_pct").items()):
                emit(tag, k, "pdr_pct", m, h, n)
    rows = load("e5_paired_tests.csv")
    if rows:
        print("  -- paired tests vs shortest path")
        for r in rows:
            print(f"    {r['sweep']:10s} {num(r['value']):6.2f}  "
                  f"delta={num(r['mean_delta_pp']):+7.2f}pp  p={num(r['p_value']):.4g}  "
                  f"r={num(r['effect_size']):+.2f}  n={r['n']}")

    # -- E6 ---------------------------------------------------------------
    rows = load("e6_ablation.csv")
    print("\n[E6] component ablation")
    if rows:
        print("    ablation                       PDR%      EE     hops")
        for k, (m, h, n) in sorted(agg(rows, ("ablation",), "pdr_pct").items()):
            ee = agg(rows, ("ablation",), "energy_efficiency")[k]
            hp = agg(rows, ("ablation",), "mean_hops")[k]
            print(f"    {k[0]:30s} {m:6.2f}+/-{h:4.2f} {ee[0]:6.2f}  {hp[0]:5.2f}")
            emit("E6.ablation", k, "pdr_pct", m, h, n)
    rows = load("e6_statistics.csv")
    if rows:
        print("  -- significance (Wilcoxon signed-rank, paired by deployment)")
        for r in rows:
            p = num(r["p_value"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {r['metric']:18s} {r['policy_a']:20s} vs {r['policy_b']:20s} "
                  f"delta={num(r['delta']):+7.2f}  p={p:.4g} {sig:3s} "
                  f"cliff={num(r['cliffs_delta']):+.2f} n={r['n_pairs']}")
    rows = load("e6_test_battery.csv")
    if rows:
        a = agg(rows, ("policy",), "pdr_pct")
        print("  -- held-out deployment battery")
        for k in sorted(a):
            m, h, n = a[k]
            print(f"    {k[0]:26s} {m:6.2f}+/-{h:4.2f} n={n}")
            emit("E6.battery", k, "pdr_pct", m, h, n)

    # -- E7 ---------------------------------------------------------------
    rows = load("e7_generalisation.csv")
    print("\n[E7] zero-shot generalisation (PDR% by family x policy)")
    if rows:
        table(rows, ("family", "policy"), "pdr_pct", width=14)
        for k, (m, h, n) in sorted(agg(rows, ("family", "policy"), "pdr_pct").items()):
            emit("E7", k, "pdr_pct", m, h, n)

    # -- E8 ---------------------------------------------------------------
    rows = load("e8_partial_observability.csv")
    print("\n[E8] partial observability ladder (PDR% by profile x policy)")
    if rows:
        table(rows, ("profile", "policy"), "pdr_pct", width=14)
        for k, (m, h, n) in sorted(agg(rows, ("profile", "policy"), "pdr_pct").items()):
            emit("E8", k, "pdr_pct", m, h, n)

    if args.csv and flat:
        save_csv(flat, "SUMMARY.csv")
        print(f"\nwrote results/SUMMARY.csv ({len(flat)} rows)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
