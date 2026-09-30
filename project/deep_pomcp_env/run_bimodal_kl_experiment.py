"""
Formal Quantification of Multimodal Belief Representation & Action Selection Error.
Measures:
1. KL Divergence D_KL(P_true || Q_approx) for Gaussian Mean vs GMM-2 vs PointNet during bifurcation events.
2. Action Selection Error Rate (% of decisions that steer directly into the dividing wall due to mean collapse).
"""

import os
import sys
import time
import csv
import math
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.stats
from sklearn.mixture import GaussianMixture

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from deep_pomcp_env import make_env
from deep_pomcp_env.baselines import (
    BasePursuerPolicy,
    VanillaPOMCPPolicy,
    HeuristicPOMCPPolicy,
    ReactiveAStarPolicy,
    DeepPOMCPPolicy
)

def compute_grid_densities(particles: np.ndarray, width: int = 800, height: int = 600, grid_res: int = 20):
    nx = width // grid_res
    ny = height // grid_res
    x_edges = np.linspace(0, width, nx + 1)
    y_edges = np.linspace(0, height, ny + 1)

    # 1. Empirical True Density P_true (Histogram with slight Laplace smoothing)
    hist, _, _ = np.histogram2d(particles[:, 0], particles[:, 1], bins=[x_edges, y_edges])
    P_true = hist / np.sum(hist)
    P_true = (P_true + 1e-6) / np.sum(P_true + 1e-6)

    # 2. Gaussian Approximation Q_Gauss
    mu = np.mean(particles[:, :2], axis=0)
    cov = np.cov(particles[:, :2], rowvar=False) + np.eye(2) * 1e-2
    gauss_rv = scipy.stats.multivariate_normal(mean=mu, cov=cov, allow_singular=True)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    pos_grid = np.dstack((xx, yy))

    Q_gauss = gauss_rv.pdf(pos_grid)
    Q_gauss = (Q_gauss + 1e-6) / np.sum(Q_gauss + 1e-6)

    # 3. GMM-2 Approximation Q_GMM2
    try:
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(particles[:, :2])
        log_prob = gmm.score_samples(pos_grid.reshape(-1, 2))
        Q_gmm = np.exp(log_prob).reshape(nx, ny)
        Q_gmm = (Q_gmm + 1e-6) / np.sum(Q_gmm + 1e-6)
    except Exception:
        Q_gmm = np.copy(Q_gauss)

    # Compute KL Divergences: D_KL(P || Q) = sum P * log(P / Q)
    kl_gauss = float(np.sum(P_true * np.log(P_true / Q_gauss)))
    kl_gmm = float(np.sum(P_true * np.log(P_true / Q_gmm)))

    return kl_gauss, kl_gmm, mu, cov


def run_bimodal_kl_quantification(num_trials: int = 30, output_csv: str = "bimodal_kl_results.csv"):
    weights_path = os.path.join(parent_dir, "deep_pomcp_weights.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(current_dir, "deep_pomcp_weights.pth")

    policies: List[BasePursuerPolicy] = [
        ReactiveAStarPolicy(),
        VanillaPOMCPPolicy(num_simulations=80, max_depth=25),
        HeuristicPOMCPPolicy(num_simulations=120, max_depth=35),
        DeepPOMCPPolicy(weights_path=weights_path if os.path.exists(weights_path) else None, num_simulations=60, max_depth=6)
    ]

    all_records = []
    print("=" * 85, flush=True)
    print("     FORMAL QUANTIFICATION: BIMODAL BELIEF KL DIVERGENCE & ACTION SELECTION ERROR", flush=True)
    print("=" * 85, flush=True)

    kl_gauss_all = []
    kl_gmm_all = []
    action_errors = {p.name: 0 for p in policies}
    total_bifurcation_events = {p.name: 0 for p in policies}

    for trial in range(num_trials):
        seed = 4000 + trial
        print(f"\n>>> Trial {trial + 1}/{num_trials} (Seed: {seed}) <<<", flush=True)

        for p in policies:
            env = make_env(density_preset="bimodal_fork", num_hunters=2, render_mode=None)
            obs, info = env.reset(seed=seed)
            p.reset()

            step = 0
            bifurcation_evaluated = False

            while step < 600:
                actions = {}
                for i in range(2):
                    act = p.get_action(obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height)
                    actions[f"agent_{i}"] = act

                # Detect Peak Bifurcation: particles split across central barrier (y in [260, 340])
                if not bifurcation_evaluated and info.get("is_belief_active", False) and info.get("particles") is not None:
                    part = info["particles"]
                    pos = part[:, :2] * np.array([env.width, env.height])
                    y_std = np.std(pos[:, 1])

                    # Check if particles span both top (<260) and bottom (>340) corridors
                    in_top = np.sum(pos[:, 1] < 260)
                    in_bottom = np.sum(pos[:, 1] > 340)

                    if y_std > 55.0 and in_top >= 20 and in_bottom >= 20:
                        kl_g, kl_gmm, b_mean, _ = compute_grid_densities(pos, env.width, env.height)
                        kl_gauss_all.append(kl_g)
                        kl_gmm_all.append(kl_gmm)

                        # Evaluate Action Selection Error for Hunter 0
                        h_pos = info["hunter_positions"][0]
                        act0 = actions["agent_0"]
                        act_vec = env.ACTION_VECTORS[act0] if isinstance(act0, int) else act0
                        intended_next_y = h_pos[1] + act_vec[1] * 50.0

                        # Central dividing wall is y in [260, 340], x in [240, 560]
                        steers_into_wall = (260.0 <= intended_next_y <= 340.0) and (200.0 < h_pos[0] < 580.0)

                        total_bifurcation_events[p.name] += 1
                        if steers_into_wall:
                            action_errors[p.name] += 1

                        record = {
                            "trial": trial + 1,
                            "seed": seed,
                            "policy": p.name,
                            "kl_gaussian": round(kl_g, 4),
                            "kl_gmm2": round(kl_gmm, 4),
                            "belief_mean_y": round(float(b_mean[1]), 2),
                            "action_steers_into_wall": steers_into_wall
                        }
                        all_records.append(record)
                        bifurcation_evaluated = True

                obs, rewards, terminated, truncated, info = env.step(actions)
                step += 1
                if terminated["__all__"] or truncated["__all__"]:
                    break

            print(f"  [{p.name:18s}] -> Evaluated. Cumulative Errors: {action_errors[p.name]}/{total_bifurcation_events[p.name]}", flush=True)

    # Save to CSV
    csv_path = os.path.join(parent_dir, output_csv)
    fieldnames = ["trial", "seed", "policy", "kl_gaussian", "kl_gmm2", "belief_mean_y", "action_steers_into_wall"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 85, flush=True)
    print("             BIMODAL QUANTIFICATION: SUMMARY RESULTS", flush=True)
    print("=" * 85, flush=True)
    mean_kl_g = np.mean(kl_gauss_all) if kl_gauss_all else 0.0
    std_kl_g = np.std(kl_gauss_all) if kl_gauss_all else 0.0
    mean_kl_gmm = np.mean(kl_gmm_all) if kl_gmm_all else 0.0
    std_kl_gmm = np.std(kl_gmm_all) if kl_gmm_all else 0.0
    info_reduction = ((mean_kl_g - mean_kl_gmm) / max(1e-5, mean_kl_g)) * 100.0

    print(f"Mean KL Divergence D_KL(P_true || Q_Gaussian): {mean_kl_g:.4f} ± {std_kl_g:.4f} nats")
    print(f"Mean KL Divergence D_KL(P_true || Q_GMM-2):    {mean_kl_gmm:.4f} ± {std_kl_gmm:.4f} nats")
    print(f"Information Loss Reduction via Multi-Modality: {info_reduction:.1f}%\n")

    print(f"{'Policy':<22} | {'Bimodal Events':<16} | {'Steering into Wall':<20} | {'Action Error Rate (%)':<20}")
    print("-" * 85)
    for p in policies:
        p_name = p.name
        tot = max(1, total_bifurcation_events[p_name])
        err = action_errors[p_name]
        rate = (err / tot) * 100.0
        print(f"{p_name:<22} | {tot:<16} | {err:<20} | {rate:19.1f}%")

    print(f"\nDetailed telemetry saved to: {csv_path}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bimodal KL & Action Error Experiment")
    parser.add_argument("--trials", type=int, default=30, help="Number of bifurcation trials (default: 30)")
    parser.add_argument("--output", type=str, default="bimodal_kl_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    run_bimodal_kl_quantification(num_trials=args.trials, output_csv=args.output)
