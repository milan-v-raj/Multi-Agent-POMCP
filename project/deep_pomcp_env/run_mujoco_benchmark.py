"""
High-Fidelity MuJoCo Continuous Multi-Body Physics Benchmark.

Evaluates multi-agent pursuit-evasion policies in continuous 3D rigid-body dynamics
with Coulomb friction, contact impulses, and continuous optical raycasting.

Contenders:
  1. Reactive A* (Classical baseline)
  2. Tree Deep-POMCP (N=20, Unimodal)
  3. ET-DAG-Deep-POMCP (N=20, Unimodal)
  4. CD-ET-DAG-Deep-POMCP (N=20, Counter-Deceptive)
"""

import os
import sys
import time
import csv
import random
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from deep_pomcp_env import make_mujoco_env, DeceptiveFeintingEvader
from deep_pomcp_env.baselines import ReactiveAStarPolicy, BeliefGraphDeepPOMCPPolicy


def run_benchmark(episodes_per_preset: int = 3, max_steps: int = 1200):
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
            name="Tree Deep-POMCP (N=20)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True, enable_counter_deception=False,
            name="ET-DAG-Deep-POMCP (N=20)"
        ),
        BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path, num_simulations=20, max_depth=6,
            enable_dag_merging=True, enable_event_trigger=True, enable_counter_deception=True,
            name="CD-ET-DAG-Deep-POMCP (N=20)"
        ),
    ]

    sep = "=" * 115
    print(sep, flush=True)
    print("  MUJOCO CONTINUOUS MULTI-BODY PHYSICS BENCHMARK (DECEPTIVE EVADER, 3D RAYCASTING, RIGID CONTACT)", flush=True)
    print(sep, flush=True)

    results_csv = os.path.join(parent_dir, "mujoco_benchmark_results.csv")
    csv_rows = []
    saved_renders = {}

    for policy in contenders:
        wins = 0
        total_eps = 0
        ttc_list = []
        wall_contacts_list = []
        solver_contacts_list = []
        latency_list = []

        print(f"\nEvaluating Policy: [{policy.name}] ...", flush=True)

        for preset in presets:
            preset_wins = 0
            for ep in range(episodes_per_preset):
                seed = 7200 + ep * 23
                np.random.seed(seed)
                random.seed(seed)
                deceptive_evader.reset_state()

                # Enable render capture on the first episode of u_trap and dense_maze
                capture_frame = (ep == 0 and preset in ["u_trap", "dense_maze"] and "CD-ET-DAG" in policy.name)
                render_mode = "rgb_array" if capture_frame else None

                env = make_mujoco_env(
                    density_preset=preset,
                    evader_policy=deceptive_evader,
                    num_hunters=2,
                    render_mode=render_mode,
                    max_steps=max_steps
                )
                obs, info = env.reset(seed=seed)
                policy.reset()

                step = 0
                captured = False
                total_contacts = 0
                wall_hit_count = 0
                step_latencies = []

                while step < max_steps:
                    t0 = time.perf_counter()
                    actions = {}
                    for i in range(2):
                        actions[f"agent_{i}"] = policy.get_action(
                            obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height
                        )
                    t_act = (time.perf_counter() - t0) * 1000.0
                    step_latencies.append(t_act)

                    obs, rewards, terminated, truncated, info = env.step(actions)
                    step += 1
                    total_contacts += info.get("mujoco_contacts", 0)
                    wall_hit_count += info.get("wall_hits", 0)

                    # Save 3D render at a dramatic mid-chase moment
                    if capture_frame and (step == 35 or terminated["__all__"]):
                        if f"{preset}_perspective" not in saved_renders:
                            img_persp = env.render(camera_name="perspective")
                            img_over = env.render(camera_name="overhead")
                            saved_renders[f"{preset}_perspective"] = img_persp
                            saved_renders[f"{preset}_overhead"] = img_over

                    if terminated["__all__"] or truncated["__all__"]:
                        captured = terminated["__all__"]
                        break

                env.close()

                total_eps += 1
                if captured:
                    wins += 1
                    preset_wins += 1
                    ttc_list.append(step)
                else:
                    ttc_list.append(max_steps)

                wall_contacts_list.append(wall_hit_count)
                solver_contacts_list.append(total_contacts / max(1, step))
                mean_lat = float(np.mean(step_latencies)) if step_latencies else 0.0
                latency_list.append(mean_lat)

                csv_rows.append({
                    "policy": policy.name,
                    "preset": preset,
                    "episode": ep + 1,
                    "seed": seed,
                    "captured": int(captured),
                    "ttc": step,
                    "wall_contacts": wall_hit_count,
                    "mean_solver_contacts": f"{(total_contacts / max(1, step)):.2f}",
                    "mean_latency_ms": f"{mean_lat:.2f}"
                })

            print(f"  -> Preset [{preset:12s}]: {preset_wins}/{episodes_per_preset} wins ({preset_wins/episodes_per_preset*100:.1f}%)", flush=True)

        win_rate = (wins / total_eps) * 100.0
        mean_ttc = float(np.mean(ttc_list))
        mean_wall = float(np.mean(wall_contacts_list))
        mean_solver = float(np.mean(solver_contacts_list))
        mean_overall_lat = float(np.mean(latency_list))

        print(f"  Summary [{policy.name}]: Win Rate: {win_rate:5.1f}% | Mean TTC: {mean_ttc:5.1f} steps | Wall Hits: {mean_wall:4.1f} | Contacts/Step: {mean_solver:4.2f} | Latency: {mean_overall_lat:4.2f} ms", flush=True)

    # Save to CSV
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["policy", "preset", "episode", "seed", "captured", "ttc", "wall_contacts", "mean_solver_contacts", "mean_latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[Saved MuJoCo Benchmark Results] -> {results_csv}", flush=True)

    # Save 3D perspective render images if captured
    for preset, img_arr in saved_renders.items():
        render_path = os.path.join(parent_dir, f"mujoco_render_{preset}.png")
        Image.fromarray(img_arr).save(render_path)
        print(f"[Saved MuJoCo 3D Render Image] -> {render_path}", flush=True)

    print(sep, flush=True)


if __name__ == "__main__":
    run_benchmark(episodes_per_preset=2, max_steps=800)
