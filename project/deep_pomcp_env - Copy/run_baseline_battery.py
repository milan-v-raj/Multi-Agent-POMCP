"""
Automated Multi-Density Benchmark Runner for Baseline Battery.
Evaluates Vanilla POMCP, Heuristic POMCP, Reactive A* Tracker, and Deep-POMCP across deterministic seeds.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any
import numpy as np

# Ensure parent directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from deep_pomcp_env import make_env, StatsLogger
from deep_pomcp_env.baselines import (
    BasePursuerPolicy,
    VanillaPOMCPPolicy,
    HeuristicPOMCPPolicy,
    ReactiveAStarPolicy,
    DeepPOMCPPolicy
)

def evaluate_baseline(policy: BasePursuerPolicy,
                      preset: str,
                      episodes: int,
                      start_seed: int = 100,
                      num_hunters: int = 2,
                      render: bool = False,
                      logger: StatsLogger = None) -> Dict[str, Any]:
    """Runs N evaluation episodes on fixed seeds and returns summary stats."""
    render_mode = "human" if render else None
    env = make_env(density_preset=preset, num_hunters=num_hunters, render_mode=render_mode)

    success_count = 0
    ttc_list = []
    latency_list = []
    rmse_list = []
    inter_dist_list = []

    print(f"\n--- Evaluating '{policy.name}' on Preset '{preset}' ({episodes} episodes) ---", flush=True)

    for ep in range(episodes):
        seed = start_seed + ep
        obs, info = env.reset(seed=seed)
        policy.reset()

        if logger:
            logger.start_episode(ep + 1)

        ep_latencies = []
        ep_running = True
        step = 0

        while ep_running:
            actions = {}
            for i in range(num_hunters):
                t0 = time.perf_counter()
                act = policy.get_action(obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height)
                t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                ep_latencies.append(t_elapsed_ms)
                actions[f"agent_{i}"] = act

            obs, rewards, terminated, truncated, info = env.step(actions)
            step += 1

            if logger:
                logger.log_step(
                    hunter_positions=info["hunter_positions"],
                    evader_pos=info["evader_ground_truth"],
                    belief_mean=info["belief_mean"],
                    planning_latency_ms=np.mean(ep_latencies[-num_hunters:]),
                    wall_hit=(info["wall_hits"] > 0)
                )

            if terminated["__all__"] or truncated["__all__"]:
                captured = terminated["__all__"]
                outcome = "CAPTURED" if captured else "TIMEOUT"
                if logger:
                    stat = logger.end_episode(success=captured, outcome=outcome)
                    if stat.min_inter_agent_dist > 0:
                        inter_dist_list.append(stat.min_inter_agent_dist)
                    if stat.mean_rmse > 0:
                        rmse_list.append(stat.mean_rmse)

                if captured:
                    success_count += 1
                    ttc_list.append(step)

                ep_running = False

        avg_lat = np.mean(ep_latencies) if ep_latencies else 0.0
        latency_list.append(avg_lat)
        outcome_str = "WIN " if terminated["__all__"] else "FAIL"
        print(f"  [Ep {ep+1:02d}/{episodes:02d} | Seed {seed}] Outcome: {outcome_str} in {step:4d} steps | Avg Latency: {avg_lat:5.2f}ms", flush=True)

    env.close()

    success_rate = (success_count / episodes) * 100.0
    mean_ttc = float(np.mean(ttc_list)) if ttc_list else float('nan')
    std_ttc = float(np.std(ttc_list)) if ttc_list else float('nan')
    mean_lat = float(np.mean(latency_list)) if latency_list else 0.0
    mean_rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
    mean_inter_dist = float(np.mean(inter_dist_list)) if inter_dist_list else 0.0

    return {
        "policy": policy.name,
        "preset": preset,
        "episodes": episodes,
        "success_rate_pct": success_rate,
        "mean_ttc": mean_ttc,
        "std_ttc": std_ttc,
        "mean_latency_ms": mean_lat,
        "mean_rmse_px": mean_rmse,
        "mean_min_inter_dist_px": mean_inter_dist
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Density Baseline Battery Benchmark Runner")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per configuration")
    parser.add_argument("--presets", type=str, default="open,moderate,dense_maze", help="Comma-separated density presets")
    parser.add_argument("--baselines", type=str, default="reactive,vanilla,heuristic,deep_pomcp", help="Comma-separated baseline policies")
    parser.add_argument("--output_csv", type=str, default="baseline_battery_results.csv", help="Output CSV file")
    parser.add_argument("--render", action="store_true", help="Render Pygame window during evaluation")
    args = parser.parse_args()

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    baseline_keys = [b.strip() for b in args.baselines.split(",") if b.strip()]

    policy_map = {
        "reactive": ReactiveAStarPolicy(name="Reactive A*"),
        "vanilla": VanillaPOMCPPolicy(num_simulations=100, name="Vanilla POMCP"),
        "heuristic": HeuristicPOMCPPolicy(num_simulations=120, name="Heuristic POMCP"),
        "deep_pomcp": DeepPOMCPPolicy(num_simulations=40, max_depth=5, name="Deep-POMCP (Ours)")
    }

    selected_policies = [policy_map[k] for k in baseline_keys if k in policy_map]

    logger = StatsLogger(output_csv=args.output_csv)
    all_results = []

    print("=" * 80, flush=True)
    print("STARTING DEEP-POMCP MULTI-AGENT BASELINE BATTERY BENCHMARK", flush=True)
    print(f"Episodes per configuration: {args.episodes}", flush=True)
    print(f"Density Presets: {presets}", flush=True)
    print(f"Policies: {[p.name for p in selected_policies]}", flush=True)
    print("=" * 80, flush=True)

    for preset in presets:
        for policy in selected_policies:
            res = evaluate_baseline(
                policy=policy,
                preset=preset,
                episodes=args.episodes,
                start_seed=100,
                num_hunters=2,
                render=args.render,
                logger=logger
            )
            all_results.append(res)

    # --- PRINT FINAL TABLE 1 ---
    print("\n" + "=" * 95, flush=True)
    print("=== TABLE 1: EMPIRICAL PERFORMANCE COMPARISON OVER STANDARDIZED SEEDS ===", flush=True)
    print("=" * 95, flush=True)
    header = f"{'Policy':<22} | {'Preset':<12} | {'Success Rate':<14} | {'Mean TTC (Steps)':<18} | {'Latency (ms)':<14} | {'Belief RMSE':<12}"
    print(header, flush=True)
    print("-" * 95, flush=True)

    for r in all_results:
        ttc_str = f"{r['mean_ttc']:.1f} ± {r['std_ttc']:.1f}" if not np.isnan(r['mean_ttc']) else "N/A (0 wins)"
        print(f"{r['policy']:<22} | {r['preset']:<12} | {r['success_rate_pct']:>6.1f}%        | {ttc_str:<18} | {r['mean_latency_ms']:>6.2f} ms     | {r['mean_rmse_px']:>6.1f} px", flush=True)

    print("=" * 95, flush=True)
    print(f"[INFO] Full telemetry and results saved to '{args.output_csv}'.", flush=True)


if __name__ == "__main__":
    main()

