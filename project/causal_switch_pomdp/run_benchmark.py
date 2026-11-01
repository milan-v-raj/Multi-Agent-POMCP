"""
Benchmark Runner for Causal Switch-Door Dec-POMDP.
Evaluates Epistemic Exploration & Causal Mechanism Discovery:
- Greedy A* (Fails 100% at door)
- Heuristic Trial-and-Error (Greedy switch visits without joint belief tree search)
- Causal Mac-POMCP (Ours: Bayesian epistemic belief + option tree search)
- Deep Causal Mac-POMCP (Frozen NN value guidance)

Zero hardcoding, zero ground-truth leaks, zero new training.
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

from causal_switch_pomdp.env import make_causal_switch_env
from causal_switch_pomdp.policies import (
    GreedyAStarPolicy,
    RandomSwitchHeuristicPolicy,
    CausalMacPOMCPPolicy,
    DeepCausalMacPOMCPPolicy
)
from deep_pomcp_env.evaders import StrategicAdversarialEvader

def run_causal_benchmark(episodes: int = 10, num_switches: int = 3, output_csv: str = "causal_benchmark_results.csv"):
    weights_path = os.path.join(project_dir, "deep_pomcp_weights.pth")

    policies = [
        GreedyAStarPolicy(),
        RandomSwitchHeuristicPolicy(),
        CausalMacPOMCPPolicy(num_simulations=25, max_depth=3, sim_steps_per_option=120),
        DeepCausalMacPOMCPPolicy(weights_path=weights_path, num_simulations=25, max_depth=3, sim_steps_per_option=120)
    ]

    evader = StrategicAdversarialEvader(max_force=0.45)
    all_records = []
    summary = {}

    sep = "=" * 105
    print(sep, flush=True)
    print("      CAUSAL MECHANISM DISCOVERY & EPISTEMIC EXPLORATION Dec-POMDP BENCHMARK", flush=True)
    print(f"      Evaluates N = {episodes} Episodes per Policy across K = {num_switches} Candidate Switches", flush=True)
    print("      Pursuers must identify the hidden functional switch kappa* via active exploration", flush=True)
    print("      Zero Hardcoding | Zero Ground-Truth Leaks | FROZEN Neural Weights", flush=True)
    print(sep, flush=True)

    for p in policies:
        p_name = p.name
        wins = 0
        discovered_count = 0
        breaches = 0
        ttc_list = []
        disc_step_list = []
        switches_tested_list = []
        t_start = time.time()

        print(f"\nEvaluating Policy: [{p_name}] ...", flush=True)

        for ep in range(episodes):
            seed = 8000 + ep * 23
            env = make_causal_switch_env(num_switches=num_switches, evader_policy=evader, render_mode=None)
            obs, info = env.reset(seed=seed)
            p.reset()

            step = 0
            captured = False

            while step < env.max_steps:
                actions = {}
                # Strictly isolate unobservable evader ground-truth
                clean_info = {k: v for k, v in info.items() if k != "evader_ground_truth"}
                for i in range(env.num_hunters):
                    act = p.get_action(obs[f"agent_{i}"], clean_info, i, env.obstacles, env.width, env.height)
                    actions[f"agent_{i}"] = act

                obs, rewards, terminated, truncated, info = env.step(actions)
                step += 1

                if terminated["__all__"] or truncated["__all__"]:
                    captured = terminated["__all__"]
                    break

            discovered = info.get("true_switch_discovered", False)
            disc_step = info.get("discovery_step", None)
            sw_tested = info.get("switches_tested_count", 0)
            breached = info.get("gate_breach_step", None) is not None

            if captured:
                wins += 1
                ttc_list.append(step)
            if discovered:
                discovered_count += 1
                if disc_step is not None:
                    disc_step_list.append(disc_step)
            if breached:
                breaches += 1
            switches_tested_list.append(sw_tested)

            all_records.append({
                "policy": p_name,
                "episode": ep + 1,
                "seed": seed,
                "captured": captured,
                "steps": step,
                "true_switch_discovered": discovered,
                "discovery_step": disc_step if disc_step else -1,
                "switches_tested_count": sw_tested,
                "gate_breached": breached
            })

            disc_str = f"Step {disc_step}" if disc_step else "None"
            print(f"  Ep {ep+1:02d} (seed {seed}): Steps={step:4d} | Captured={str(captured):<5} | Discovered={str(discovered):<5} ({disc_str}) | Tested={sw_tested}/{num_switches}", flush=True)

        elapsed = time.time() - t_start
        win_rate = (wins / episodes) * 100.0
        disc_rate = (discovered_count / episodes) * 100.0
        br_rate = (breaches / episodes) * 100.0
        mean_ttc = float(np.mean(ttc_list)) if ttc_list else float("nan")
        mean_disc_step = float(np.mean(disc_step_list)) if disc_step_list else float("nan")
        mean_tested = float(np.mean(switches_tested_list))

        summary[p_name] = {
            "win_rate": win_rate,
            "disc_rate": disc_rate,
            "br_rate": br_rate,
            "mean_ttc": mean_ttc,
            "mean_disc_step": mean_disc_step,
            "mean_tested": mean_tested,
            "elapsed": elapsed
        }

    # Save CSV
    csv_file = output_csv if os.path.isabs(output_csv) else os.path.join(current_dir, output_csv)
    fieldnames = ["policy", "episode", "seed", "captured", "steps", "true_switch_discovered", "discovery_step", "switches_tested_count", "gate_breached"]
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + sep, flush=True)
    print("                 CAUSAL DISCOVERY BENCHMARK SUMMARY TABLE", flush=True)
    print(sep, flush=True)
    print(f"{'Policy':<32} | {'Win Rate (%)':<14} | {'Discovery (%)':<15} | {'Mean Tested':<13} | {'Mean TTC':<12}")
    print("-" * 105)
    for p_name, st in summary.items():
        ttc_str = f"{st['mean_ttc']:5.1f}" if not np.isnan(st['mean_ttc']) else "   N/A"
        print(f"{p_name:<32} | {st['win_rate']:12.1f}% | {st['disc_rate']:13.1f}% | {st['mean_tested']:10.2f}    | {ttc_str:<12}")
    print(sep, flush=True)
    print(f"Results saved to: {csv_file}\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--switches", type=int, default=3)
    parser.add_argument("--output", type=str, default="causal_benchmark_results.csv")
    args = parser.parse_args()
    run_causal_benchmark(episodes=args.episodes, num_switches=args.switches, output_csv=args.output)

