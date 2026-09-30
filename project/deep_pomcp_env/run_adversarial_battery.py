"""
Comprehensive Adversarial Evader Benchmark.
Evaluates Reactive A*, Vanilla POMCP, Heuristic POMCP, and Deep-POMCP against:
1. Standard Raycast Evader (Baseline)
2. Strategic Adversarial Evader (Active Occlusion Seeking + Anti-Pincer)
"""

import os
import sys
import time
import csv
import argparse
from typing import Dict, List, Any
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from deep_pomcp_env import make_env
from deep_pomcp_env.evaders import SmartRaycastEvader, StrategicAdversarialEvader
from deep_pomcp_env.baselines import (
    BasePursuerPolicy,
    VanillaPOMCPPolicy,
    HeuristicPOMCPPolicy,
    ReactiveAStarPolicy,
    DeepPOMCPPolicy
)

def run_adversarial_benchmark(
    episodes_per_preset: int = 5,
    presets: List[str] = ["open", "moderate", "dense_maze", "u_trap", "figure_8", "bimodal_fork"],
    output_csv: str = "adversarial_benchmark_results.csv"
):
    weights_path = os.path.join(parent_dir, "deep_pomcp_weights.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(current_dir, "deep_pomcp_weights.pth")

    policies: List[BasePursuerPolicy] = [
        ReactiveAStarPolicy(),
        VanillaPOMCPPolicy(num_simulations=80, max_depth=25),
        HeuristicPOMCPPolicy(num_simulations=120, max_depth=35),
        DeepPOMCPPolicy(weights_path=weights_path if os.path.exists(weights_path) else None, num_simulations=60, max_depth=6)
    ]

    evader_types = {
        "Standard Raycast": SmartRaycastEvader(max_force=0.45),
        "Strategic Adversarial": StrategicAdversarialEvader(max_force=0.50)
    }

    all_records = []
    print("=" * 90, flush=True)
    print("         SMART ADVERSARIAL EVADER BENCHMARK: ACTIVE OCCLUSION & ANTI-PINCER", flush=True)
    print("=" * 90, flush=True)

    summary_matrix = {}

    for ev_name, ev_policy in evader_types.items():
        print(f"\n#################################################################", flush=True)
        print(f"       EVALUATING AGAINST EVADER: [{ev_name.upper()}]", flush=True)
        print(f"#################################################################", flush=True)

        for p in policies:
            p_name = p.name
            wins = 0
            total_eps = 0
            ttc_list = []
            wall_list = []
            los_breaks_all = []

            for preset in presets:
                for ep in range(episodes_per_preset):
                    seed = 5000 + (0 if ev_name == "Standard Raycast" else 500) + ep * 10
                    env = make_env(density_preset=preset, evader_policy=ev_policy, num_hunters=2, render_mode=None)
                    obs, info = env.reset(seed=seed)
                    p.reset()

                    step = 0
                    captured = False
                    wall_hits = 0
                    blind_steps = 0

                    while step < 1500:
                        actions = {}
                        for i in range(2):
                            act = p.get_action(obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height)
                            actions[f"agent_{i}"] = act

                        obs, rewards, terminated, truncated, info = env.step(actions)
                        step += 1

                        if not info.get("can_see_global", False):
                            blind_steps += 1

                        if info.get("wall_hits", 0) > 0:
                            wall_hits += info["wall_hits"]

                        if terminated["__all__"] or truncated["__all__"]:
                            captured = terminated["__all__"]
                            break

                    total_eps += 1
                    if captured:
                        wins += 1
                        ttc_list.append(step)
                    wall_list.append(wall_hits)
                    los_break_ratio = (blind_steps / step) * 100.0 if step > 0 else 0.0
                    los_breaks_all.append(los_break_ratio)

                    record = {
                        "evader_type": ev_name,
                        "policy": p_name,
                        "preset": preset,
                        "episode": ep + 1,
                        "seed": seed,
                        "captured": captured,
                        "steps": step,
                        "wall_hits": wall_hits,
                        "los_break_pct": round(los_break_ratio, 1)
                    }
                    all_records.append(record)

            win_rate = (wins / total_eps) * 100.0 if total_eps > 0 else 0.0
            mean_ttc = np.mean(ttc_list) if ttc_list else float('nan')
            mean_walls = np.mean(wall_list) if wall_list else 0.0
            mean_los_break = np.mean(los_breaks_all) if los_breaks_all else 0.0

            summary_matrix.setdefault(p_name, {})[ev_name] = {
                "win_rate": win_rate,
                "mean_ttc": mean_ttc,
                "mean_walls": mean_walls,
                "mean_los_break": mean_los_break
            }

            print(f"  [{p_name:18s}] vs [{ev_name:21s}] -> Win Rate: {win_rate:5.1f}% | TTC: {mean_ttc:5.1f} steps | Blind: {mean_los_break:4.1f}% | Walls: {mean_walls:4.1f}", flush=True)

    # Save to CSV
    csv_path = os.path.join(parent_dir, output_csv)
    fieldnames = ["evader_type", "policy", "preset", "episode", "seed", "captured", "steps", "wall_hits", "los_break_pct"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 90, flush=True)
    print("                     ADVERSARIAL EVADER COMPARISON MATRIX", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Policy':<20} | {'Standard Evader Win%':<22} | {'Adversarial Evader Win%':<25} | {'Degradation Delta (Δ)':<22}")
    print("-" * 95)

    for p in policies:
        p_name = p.name
        std_wr = summary_matrix[p_name]["Standard Raycast"]["win_rate"]
        adv_wr = summary_matrix[p_name]["Strategic Adversarial"]["win_rate"]
        delta = adv_wr - std_wr
        print(f"{p_name:<20} | {std_wr:20.1f}% | {adv_wr:23.1f}% | {delta:+20.1f}%")

    print(f"\nDetailed telemetry saved to: {csv_path}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adversarial Evader Benchmark")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per preset (default: 5)")
    parser.add_argument("--output", type=str, default="adversarial_benchmark_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    run_adversarial_benchmark(episodes_per_preset=args.episodes, output_csv=args.output)
