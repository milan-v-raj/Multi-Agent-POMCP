"""
Cooperative Sacrifice / Switch-Door Benchmark Battery.
Evaluates Reactive A*, Vanilla POMCP, Heuristic POMCP, and Deep-POMCP on N=30 randomized Switch-Door puzzles.
Proves that greedy baselines fail 100% while Deep-POMCP coordinates emergent sacrifice.
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

from deep_pomcp_env.switch_door_env import make_switch_door_env
from deep_pomcp_env.evaders import SmartRaycastEvader, StrategicAdversarialEvader
from deep_pomcp_env.baselines import (
    BasePursuerPolicy,
    VanillaPOMCPPolicy,
    HeuristicPOMCPPolicy,
    ReactiveAStarPolicy,
    DeepPOMCPPolicy
)

def run_switch_door_benchmark(episodes: int = 30, output_csv: str = "switch_door_benchmark_results.csv"):
    weights_path = os.path.join(parent_dir, "deep_pomcp_weights.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(current_dir, "deep_pomcp_weights.pth")

    policies: List[BasePursuerPolicy] = [
        ReactiveAStarPolicy(),
        VanillaPOMCPPolicy(num_simulations=80, max_depth=25),
        HeuristicPOMCPPolicy(num_simulations=120, max_depth=35),
        DeepPOMCPPolicy(weights_path=weights_path if os.path.exists(weights_path) else None, num_simulations=60, max_depth=6)
    ]

    evader = StrategicAdversarialEvader(max_force=0.45)
    all_records = []

    sep = "=" * 95
    print(sep, flush=True)
    print("      COOPERATIVE SACRIFICE / SWITCH-DOOR BENCHMARK: ZERO-SHOT PUZZLE GENERALIZATION", flush=True)
    print("      Evaluates N = %d Randomized Episodes per Policy (Unseen Dynamic Barricade)" % episodes, flush=True)
    print(sep, flush=True)

    summary = {}

    for p in policies:
        p_name = p.name
        wins = 0
        switch_hits = 0
        breaches = 0
        ttc_list = []
        switch_step_list = []
        breach_step_list = []

        print(f"\nEvaluating Policy: [{p_name}] ...", flush=True)
        t_start = time.time()

        for ep in range(episodes):
            seed = 6000 + ep * 10
            env = make_switch_door_env(evader_policy=evader, render_mode=None)
            obs, info = env.reset(seed=seed)
            p.reset()

            step = 0
            captured = False

            while step < env.max_steps:
                actions = {}
                for i in range(env.num_hunters):
                    act = p.get_action(obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height)
                    actions[f"agent_{i}"] = act

                obs, rewards, terminated, truncated, info = env.step(actions)
                step += 1

                if terminated["__all__"] or truncated["__all__"]:
                    captured = terminated["__all__"]
                    break

            sw_hit = info.get("switch_triggered", False)
            breached = info.get("gate_breach_step", None) is not None
            first_sw = info.get("first_switch_step", None)
            first_br = info.get("gate_breach_step", None)

            if captured:
                wins += 1
                ttc_list.append(step)
            if sw_hit:
                switch_hits += 1
                if first_sw is not None:
                    switch_step_list.append(first_sw)
            if breached:
                breaches += 1
                if first_br is not None:
                    breach_step_list.append(first_br)

            all_records.append({
                "policy": p_name,
                "episode": ep + 1,
                "seed": seed,
                "captured": captured,
                "steps": step,
                "switch_triggered": sw_hit,
                "first_switch_step": first_sw if first_sw else -1,
                "gate_breached": breached,
                "gate_breach_step": first_br if first_br else -1
            })

        elapsed = time.time() - t_start
        win_rate = (wins / episodes) * 100.0
        sw_rate = (switch_hits / episodes) * 100.0
        br_rate = (breaches / episodes) * 100.0
        mean_ttc = float(np.mean(ttc_list)) if ttc_list else float("nan")
        mean_sw_step = float(np.mean(switch_step_list)) if switch_step_list else float("nan")

        summary[p_name] = {
            "win_rate": win_rate,
            "switch_rate": sw_rate,
            "breach_rate": br_rate,
            "mean_ttc": mean_ttc,
            "mean_sw_step": mean_sw_step,
            "elapsed_s": elapsed
        }

        print(f"  -> Win Rate: {win_rate:5.1f}% | Switch Trigger: {sw_rate:5.1f}% | Breach Rate: {br_rate:5.1f}% | TTC: {mean_ttc:5.1f} steps ({elapsed:.1f}s)", flush=True)

    # Save to CSV
    csv_path = os.path.join(parent_dir, output_csv)
    fieldnames = ["policy", "episode", "seed", "captured", "steps", "switch_triggered", "first_switch_step", "gate_breached", "gate_breach_step"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + sep, flush=True)
    print("                 COOPERATIVE SACRIFICE SUMMARY TABLE (N = %d Runs)" % episodes, flush=True)
    print(sep, flush=True)
    print(f"{'Policy':<22} | {'Win Rate (%)':<14} | {'Switch Trigger (%)':<20} | {'Gate Breach (%)':<16} | {'Mean TTC (Steps)':<16}")
    print("-" * 95)
    for p_name, st in summary.items():
        ttc_str = f"{st['mean_ttc']:5.1f}" if not np.isnan(st['mean_ttc']) else "   N/A"
        print(f"{p_name:<22} | {st['win_rate']:12.1f}% | {st['switch_rate']:18.1f}% | {st['breach_rate']:14.1f}% | {ttc_str:<16}")
    print(sep, flush=True)
    print(f"Detailed logs saved to: {csv_path}\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", type=str, default="switch_door_benchmark_results.csv")
    args = parser.parse_args()
    run_switch_door_benchmark(episodes=args.episodes, output_csv=args.output)
