"""
Empirical Benchmark: Belief Sharing vs. Intention Sharing in Deep-POMCP Multi-Agent Pursuit.

Ablation Structure:
  1. Tree Deep-POMCP (Belief Sharing Only)
     - Pure epistemic belief sharing via particle filter.
     - Standard Tree MCTS, no intention sharing.
  2. ET-DAG-Deep-POMCP (Belief Sharing Only)
     - DAG-MCTS + Event-Triggered Information Gating.
     - Epistemic belief sharing only.
  3. CD-ET-DAG-Deep-POMCP (Belief Sharing Only)
     - Level-2 Counter-Deception + Bimodal Occlusion Filter.
     - Epistemic belief sharing only (vulnerable to corridor congestion / symmetric greed).
  4. IS-CD-ET-DAG-Deep-POMCP (Belief + Intention Sharing, Ours)
     - Full unified architecture: DAG-MCTS + Event-Triggered + Counter-Deception + Intention Sharing.
     - Teleological intention communication bus: broadcasts intended macro waypoints,
       approach headings, and Voronoi commitment.
     - MCTS lookahead enforces anti-clustering repulsive penalty and orthogonal pincer flanking bonus.

Audited Metrics:
  - Capture Win Rate (%)
  - Mean Time to Capture (TTC, steps)
  - Corridor Overlap Steps (steps where hunter-hunter distance < 50 px)
  - Mean Inter-Hunter Separation (px)
  - Flanking Encirclement Angle at Capture (degrees, 0 = parallel, 180 = head-on pincer)
  - MCTS Intention Clashes Prevented & Flank Bonuses Triggered
"""

import os
import sys
import time
import csv
import math
import random
import argparse
from typing import Dict, Any, List, Tuple
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

from deep_pomcp_env import make_env, make_mujoco_env, DeceptiveFeintingEvader
from deep_pomcp_env.baselines.belief_graph_pomcp import BeliefGraphDeepPOMCPPolicy


def compute_flanking_angle(h0_pos: np.ndarray, h1_pos: np.ndarray, evader_pos: np.ndarray) -> float:
    """Computes the angular separation of two hunters around the evader in degrees."""
    v0 = evader_pos - h0_pos
    v1 = evader_pos - h1_pos
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-4 or n1 < 1e-4:
        return 0.0
    cos_angle = float(np.dot(v0, v1) / (n0 * n1))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def run_benchmark(episodes_per_preset: int = 4, max_steps: int = 1000):
    weights_path = os.path.join(parent_dir, "deep_pomcp_weights.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(current_dir, "deep_pomcp_weights.pth")

    presets = ["u_trap", "dense_maze", "bimodal_fork", "figure_8"]
    deceptive_evader = DeceptiveFeintingEvader(max_force=0.50, feint_duration=26)

    contenders = [
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=False, enable_event_trigger=False,
            enable_counter_deception=False, enable_intention_sharing=False,
            name="Tree Deep-POMCP (Belief Only)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True,
            enable_counter_deception=False, enable_intention_sharing=False,
            name="ET-DAG-Deep-POMCP (Belief Only)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True,
            enable_counter_deception=True, enable_intention_sharing=False,
            name="CD-ET-DAG-Deep-POMCP (Belief Only)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True,
            enable_counter_deception=True, enable_intention_sharing=True,
            name="IS-CD-ET-DAG-Deep-POMCP (Belief + Intention)"
        ),
    ]

    sep = "=" * 115
    print(sep, flush=True)
    print("  RIGOROUS ABLATION BENCHMARK: BELIEF SHARING ONLY VS. BELIEF + INTENTION SHARING", flush=True)
    print("  Evaluator: Deceptive Feinting Evader (J-Hook Reversals + Occlusion Feints)", flush=True)
    print(sep, flush=True)

    results_csv = os.path.join(parent_dir, "intention_sharing_benchmark_results.csv")
    csv_rows = []

    for policy in contenders:
        print(f"\n---> Evaluating Policy: [{policy.name}]", flush=True)
        total_wins = 0
        total_episodes = 0
        all_ttc = []
        all_overlaps = []
        all_separations = []
        all_flank_angles = []

        for preset in presets:
            preset_wins = 0
            preset_ttcs = []
            preset_overlaps = []
            preset_flank_angles = []

            for ep in range(episodes_per_preset):
                seed = 8300 + ep * 31
                np.random.seed(seed)
                random.seed(seed)
                deceptive_evader.reset_state()

                env = make_env(
                    density_preset=preset,
                    evader_policy=deceptive_evader,
                    num_hunters=2,
                    render_mode=None,
                    max_steps=max_steps
                )
                obs, info = env.reset(seed=seed)
                policy.reset()

                step = 0
                captured = False
                ep_overlap_steps = 0
                ep_separations = []
                final_flank_angle = 0.0

                while step < max_steps:
                    actions = {}
                    for i in range(2):
                        actions[f"agent_{i}"] = policy.get_action(
                            obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height
                        )

                    h0 = info["hunter_positions"][0]
                    h1 = info["hunter_positions"][1]
                    h_dist = float(np.linalg.norm(h0 - h1))
                    ep_separations.append(h_dist)
                    if h_dist < 50.0:
                        ep_overlap_steps += 1

                    obs, rewards, terminated, truncated, info = env.step(actions)
                    step += 1

                    if terminated["__all__"] or truncated["__all__"]:
                        captured = terminated["__all__"]
                        final_flank_angle = compute_flanking_angle(
                            info["hunter_positions"][0],
                            info["hunter_positions"][1],
                            env.evader_pos
                        )
                        break

                total_episodes += 1
                preset_overlaps.append(ep_overlap_steps)
                all_overlaps.append(ep_overlap_steps)
                all_separations.extend(ep_separations)

                if captured:
                    preset_wins += 1
                    total_wins += 1
                    preset_ttcs.append(step)
                    all_ttc.append(step)
                    preset_flank_angles.append(final_flank_angle)
                    all_flank_angles.append(final_flank_angle)

            p_win_rate = (preset_wins / episodes_per_preset) * 100.0
            p_mean_ttc = float(np.mean(preset_ttcs)) if preset_ttcs else float(max_steps)
            p_mean_overlap = float(np.mean(preset_overlaps))
            p_mean_flank = float(np.mean(preset_flank_angles)) if preset_flank_angles else 0.0

            print(f"  [{preset:<13}] Win: {p_win_rate:5.1f}% | Mean TTC: {p_mean_ttc:5.1f} | Overlap Steps: {p_mean_overlap:4.1f} | Flank Angle: {p_mean_flank:5.1f} deg", flush=True)

            csv_rows.append({
                "policy": policy.name,
                "environment_type": "Dec-POMDP (Pygame)",
                "preset": preset,
                "episodes": episodes_per_preset,
                "win_rate_pct": round(p_win_rate, 1),
                "mean_ttc": round(p_mean_ttc, 1),
                "mean_corridor_overlap_steps": round(p_mean_overlap, 1),
                "mean_hunter_separation_px": round(float(np.mean(ep_separations)), 1),
                "mean_flank_angle_deg": round(p_mean_flank, 1),
                "mcts_clashes_prevented": policy.intention_clash_penalties_applied,
                "mcts_flank_bonuses": policy.intention_flank_bonuses_applied,
            })

        overall_win_rate = (total_wins / total_episodes) * 100.0
        overall_mean_ttc = float(np.mean(all_ttc)) if all_ttc else float(max_steps)
        overall_mean_overlap = float(np.mean(all_overlaps))
        overall_mean_sep = float(np.mean(all_separations))
        overall_mean_flank = float(np.mean(all_flank_angles)) if all_flank_angles else 0.0

        print(f"  >> OVERALL: Win Rate: {overall_win_rate:5.1f}% | TTC: {overall_mean_ttc:5.1f} | Overlap Steps: {overall_mean_overlap:4.1f} | Sep: {overall_mean_sep:5.1f} px | Flank: {overall_mean_flank:5.1f} deg", flush=True)

    # -------------------------------------------------------------------------
    # PART 2: MuJoCo Continuous Unicycle Physics Validation
    # -------------------------------------------------------------------------
    print("\n" + sep, flush=True)
    print("  MUJOCO NON-HOLONOMIC CONTINUOUS UNICYCLE VALIDATION (U-TRAP & BIMODAL FORK)", flush=True)
    print(sep, flush=True)

    mujoco_contenders = [
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True,
            enable_counter_deception=True, enable_intention_sharing=False,
            name="CD-ET-DAG-Deep-POMCP (MuJoCo, Belief Only)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True,
            enable_counter_deception=True, enable_intention_sharing=True,
            name="IS-CD-ET-DAG-Deep-POMCP (MuJoCo, Belief + Intention)"
        ),
    ]

    mujoco_presets = ["u_trap", "bimodal_fork"]
    mujoco_eps = 3

    for policy in mujoco_contenders:
        print(f"\n---> Evaluating in MuJoCo Physics: [{policy.name}]", flush=True)
        for preset in mujoco_presets:
            m_wins = 0
            m_ttcs = []
            m_overlaps = []
            m_flanks = []
            m_seps = []

            for ep in range(mujoco_eps):
                seed = 9400 + ep * 43
                np.random.seed(seed)
                random.seed(seed)
                deceptive_evader.reset_state()

                env = make_mujoco_env(
                    density_preset=preset,
                    evader_policy=deceptive_evader,
                    num_hunters=2,
                    render_mode=None,
                    max_steps=max_steps
                )
                obs, info = env.reset(seed=seed)
                policy.reset()

                step = 0
                captured = False
                ep_overlap = 0
                ep_seps = []
                final_flank = 0.0

                while step < max_steps:
                    actions = {}
                    for i in range(2):
                        actions[f"agent_{i}"] = policy.get_action(
                            obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height
                        )

                    h0 = info["hunter_positions"][0]
                    h1 = info["hunter_positions"][1]
                    h_dist = float(np.linalg.norm(h0 - h1))
                    ep_seps.append(h_dist)
                    if h_dist < 50.0:
                        ep_overlap += 1

                    obs, rewards, terminated, truncated, info = env.step(actions)
                    step += 1

                    if terminated["__all__"] or truncated["__all__"]:
                        captured = terminated["__all__"]
                        final_flank = compute_flanking_angle(
                            info["hunter_positions"][0],
                            info["hunter_positions"][1],
                            env.evader_pos
                        )
                        break

                m_overlaps.append(ep_overlap)
                m_seps.extend(ep_seps)
                if captured:
                    m_wins += 1
                    m_ttcs.append(step)
                    m_flanks.append(final_flank)

            m_win_rate = (m_wins / mujoco_eps) * 100.0
            m_mean_ttc = float(np.mean(m_ttcs)) if m_ttcs else float(max_steps)
            m_mean_overlap = float(np.mean(m_overlaps))
            m_mean_flank = float(np.mean(m_flanks)) if m_flanks else 0.0

            print(f"  [MuJoCo {preset:<12}] Win: {m_win_rate:5.1f}% | Mean TTC: {m_mean_ttc:5.1f} | Overlap Steps: {m_mean_overlap:4.1f} | Flank: {m_mean_flank:5.1f} deg", flush=True)

            csv_rows.append({
                "policy": policy.name,
                "environment_type": "MuJoCo Continuous Unicycle",
                "preset": preset,
                "episodes": mujoco_eps,
                "win_rate_pct": round(m_win_rate, 1),
                "mean_ttc": round(m_mean_ttc, 1),
                "mean_corridor_overlap_steps": round(m_mean_overlap, 1),
                "mean_hunter_separation_px": round(float(np.mean(m_seps)), 1),
                "mean_flank_angle_deg": round(m_mean_flank, 1),
                "mcts_clashes_prevented": policy.intention_clash_penalties_applied,
                "mcts_flank_bonuses": policy.intention_flank_bonuses_applied,
            })

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + sep, flush=True)
    print(f"  BENCHMARK COMPLETE! Results saved to: {results_csv}", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Intention Sharing Benchmark")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per preset (default: 3)")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    args = parser.parse_args()

    run_benchmark(episodes_per_preset=args.episodes, max_steps=args.steps)

