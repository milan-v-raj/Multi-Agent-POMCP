"""
Pilot Benchmark: Level-2 Deceptive Feinting Evader (`DeceptiveFeintingEvader`)
vs. Standard Unimodal Planners & Counter-Deceptive Bimodal Belief-Graph MCTS (`CD-ET-DAG-Deep-POMCP`).

Evaluates:
  1. Evader Intelligence Upgrade: How much harder `DeceptiveFeintingEvader` is compared to
     `StrategicAdversarialEvader` (measuring J-hook occlusion feints and hunter win-rate drop).
  2. Counter-Deceptive Recovery: How Bimodal Occlusion Splitting + Bayesian Negative-Sighting
     Culling (`enable_counter_deception=True`) defeats the deceptive evader with zero ground-truth leaks.
"""

import os
import sys
import csv
import random
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from deep_pomcp_env import make_env, StrategicAdversarialEvader, DeceptiveFeintingEvader
from deep_pomcp_env.baselines import ReactiveAStarPolicy, BeliefGraphDeepPOMCPPolicy


def run_pilot(episodes_per_preset: int = 3):
    weights_path = os.path.join(parent_dir, "deep_pomcp_weights.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(current_dir, "deep_pomcp_weights.pth")

    presets = ["u_trap", "dense_maze", "bimodal_fork", "figure_8"]
    deceptive_evader = DeceptiveFeintingEvader(max_force=0.50, feint_duration=26)

    contenders = [
        ReactiveAStarPolicy(),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=False, enable_event_trigger=False, enable_counter_deception=False,
            name="Tree Deep-POMCP (N=20, Unimodal)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True, enable_counter_deception=False,
            name="ET-DAG-Deep-POMCP (N=20, Unimodal)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True, enable_counter_deception=True,
            name="CD-ET-DAG-Deep-POMCP (N=20, Counter-Deceptive)"
        ),
    ]

    sep = "=" * 115
    print(sep, flush=True)
    print("  PILOT TEST: DECEPTIVE FEINTING EVADER (J-HOOK + VORONOI) VS. COUNTER-DECEPTIVE POMCP", flush=True)
    print(sep, flush=True)

    rows = []
    for policy in contenders:
        wins = 0
        total_eps = 0
        ttc_list = []
        feints_list = []
        prunes_list = []
        splits_list = []

        for preset in presets:
            for ep in range(episodes_per_preset):
                seed = 6100 + ep * 17
                np.random.seed(seed)
                random.seed(seed)
                deceptive_evader.reset_state()

                env = make_env(
                    density_preset=preset,
                    evader_policy=deceptive_evader,
                    num_hunters=2,
                    render_mode=None
                )
                obs, info = env.reset(seed=seed)
                policy.reset()

                step = 0
                captured = False
                while step < 1200:
                    actions = {}
                    for i in range(2):
                        actions[f"agent_{i}"] = policy.get_action(
                            obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height
                        )
                    obs, rewards, terminated, truncated, info = env.step(actions)
                    step += 1
                    if terminated["__all__"] or truncated["__all__"]:
                        captured = terminated["__all__"]
                        break

                total_eps += 1
                if captured:
                    wins += 1
                    ttc_list.append(step)

                ep_feints = int(deceptive_evader.total_feints_triggered)
                ep_prunes = int(getattr(policy, "negative_evidence_prunes", 0))
                ep_splits = int(getattr(policy, "bimodal_splits_triggered", 0))

                feints_list.append(ep_feints)
                prunes_list.append(ep_prunes)
                splits_list.append(ep_splits)

                rows.append({
                    "policy": policy.name,
                    "preset": preset,
                    "episode": ep + 1,
                    "seed": seed,
                    "captured": captured,
                    "steps": step,
                    "evader_feints": ep_feints,
                    "bimodal_splits": ep_splits,
                    "neg_evidence_prunes": ep_prunes
                })

        win_rate = (wins / total_eps) * 100.0 if total_eps > 0 else 0.0
        mean_ttc = float(np.mean(ttc_list)) if ttc_list else 1200.0
        mean_feints = float(np.mean(feints_list)) if feints_list else 0.0
        mean_splits = float(np.mean(splits_list)) if splits_list else 0.0
        mean_prunes = float(np.mean(prunes_list)) if prunes_list else 0.0

        print(
            f"  [{policy.name:46s}] -> Win: {win_rate:5.1f}% | Mean TTC: {mean_ttc:6.1f} | "
            f"Evader Feints/Ep: {mean_feints:4.1f} | Bimodal Splits: {mean_splits:4.1f} | Neg-LOS Prunes: {mean_prunes:4.1f}",
            flush=True
        )

    out_csv = os.path.join(parent_dir, "deceptive_evader_pilot_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved pilot results to:", out_csv, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    run_pilot(episodes_per_preset=args.episodes)

