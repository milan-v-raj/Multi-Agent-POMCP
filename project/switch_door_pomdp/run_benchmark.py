"""
Benchmark Runner for Switch-Door Dec-POMDP.
Evaluates:
- Greedy A* (Greedy baseline)
- Primitive POMCP (Low-level force tree search)
- Mac-POMCP (Macro-action POMDP tree search)
- Deep Mac-POMCP (Mac-POMCP with FROZEN neural weights)

Zero neural retraining, zero metadata leaks.
"""

import os
import sys
import time
import csv
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from switch_door_pomdp.env import make_switch_door_pomdp_env
from switch_door_pomdp.policies import (
    GreedyAStarPolicy,
    PrimitivePOMCPPolicy,
    MacroPOMCPPolicy,
    DeepMacPOMCPPolicy
)
from deep_pomcp_env.evaders import StrategicAdversarialEvader

def run_pomdp_benchmark(episodes: int = 10, output_csv: str = "pomdp_benchmark_results.csv"):
    weights_path = os.path.join(project_dir, "deep_pomcp_weights.pth")

    policies = [
        GreedyAStarPolicy(),
        PrimitivePOMCPPolicy(num_simulations=20, max_depth=10),
        MacroPOMCPPolicy(num_simulations=25, max_depth=3, sim_steps_per_option=120),
        DeepMacPOMCPPolicy(weights_path=weights_path, num_simulations=25, max_depth=3, sim_steps_per_option=120)
    ]

    evader = StrategicAdversarialEvader(max_force=0.45)
    all_records = []
    summary = {}

    sep = "=" * 95
    print(sep, flush=True)
    print("      FORMAL SWITCH-DOOR Dec-POMDP BENCHMARK", flush=True)
    print(f"      Evaluates N = {episodes} Episodes per Policy across Randomized Environments", flush=True)
    print("      Generative Transition Model with Dynamic Gate Causal Physics", flush=True)
    print("      FROZEN Neural Weights — Zero Retraining", flush=True)
    print(sep, flush=True)

    for p in policies:
        p_name = p.name
        wins = 0
        switch_hits = 0
        breaches = 0
        ttc_list = []
        t_start = time.time()

        print(f"\nEvaluating Policy: [{p_name}] ...", flush=True)

        for ep in range(episodes):
            seed = 7000 + ep * 15
            env = make_switch_door_pomdp_env(evader_policy=evader, render_mode=None)
            obs, info = env.reset(seed=seed)
            p.reset()

            step = 0
            captured = False

            while step < env.max_steps:
                actions = {}
                clean_info = {k: v for k, v in info.items() if k != "evader_ground_truth"}
                for i in range(env.num_hunters):
                    act = p.get_action(obs[f"agent_{i}"], clean_info, i, env.obstacles, env.width, env.height)
                    actions[f"agent_{i}"] = act

                obs, rewards, terminated, truncated, info = env.step(actions)
                step += 1

                if terminated["__all__"] or truncated["__all__"]:
                    captured = terminated["__all__"]
                    break

            sw_hit = info.get("switch_triggered", False)
            breached = info.get("gate_breach_step", None) is not None

            if captured:
                wins += 1
                ttc_list.append(step)
            if sw_hit:
                switch_hits += 1
            if breached:
                breaches += 1

            all_records.append({
                "policy": p_name,
                "episode": ep + 1,
                "seed": seed,
                "captured": captured,
                "steps": step,
                "switch_triggered": sw_hit,
                "gate_breached": breached
            })

            print(f"  Ep {ep+1:02d} (seed {seed}): Steps={step:4d} | Captured={str(captured):<5} | Switch={str(sw_hit):<5} | Breach={str(breached)}", flush=True)

        elapsed = time.time() - t_start
        win_rate = (wins / episodes) * 100.0
        sw_rate = (switch_hits / episodes) * 100.0
        br_rate = (breaches / episodes) * 100.0
        mean_ttc = float(np.mean(ttc_list)) if ttc_list else float("nan")

        summary[p_name] = {
            "win_rate": win_rate,
            "switch_rate": sw_rate,
            "breach_rate": br_rate,
            "mean_ttc": mean_ttc,
            "elapsed": elapsed
        }

    # Save to CSV
    csv_file = output_csv if os.path.isabs(output_csv) else os.path.join(current_dir, output_csv)
    fieldnames = ["policy", "episode", "seed", "captured", "steps", "switch_triggered", "gate_breached"]
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + sep, flush=True)
    print("                     POMDP BENCHMARK FINAL SUMMARY TABLE", flush=True)
    print(sep, flush=True)
    print(f"{'Policy':<28} | {'Win Rate (%)':<14} | {'Switch Trigger (%)':<20} | {'Gate Breach (%)':<16} | {'Mean TTC (Steps)':<16}")
    print("-" * 95)
    for p_name, st in summary.items():
        ttc_str = f"{st['mean_ttc']:5.1f}" if not np.isnan(st['mean_ttc']) else "   N/A"
        print(f"{p_name:<28} | {st['win_rate']:12.1f}% | {st['switch_rate']:18.1f}% | {st['breach_rate']:14.1f}% | {ttc_str:<16}")
    print(sep, flush=True)
    print(f"Results saved to: {csv_file}\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="pomdp_benchmark_results.csv")
    args = parser.parse_args()
    run_pomdp_benchmark(episodes=args.episodes, output_csv=args.output)
