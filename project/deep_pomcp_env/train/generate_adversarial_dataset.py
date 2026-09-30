import os
import sys
import time
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from deep_pomcp_env import make_env
from deep_pomcp_env.evaders import StrategicAdversarialEvader
from deep_pomcp_env.baselines.heuristic_pomcp import HeuristicPOMCPPolicy
from deep_pomcp_env.pathfinder import Pathfinder

PRESETS = ["open", "moderate", "dense_maze", "u_trap", "figure_8", "bimodal_fork"]
GAMMA = 0.99
PLANNING_INTERVAL = 30
MAX_STEPS = 1500


def collect_adversarial_dataset(total_episodes=500, output_path=None):
    if output_path is None:
        output_path = os.path.join(current_dir, "adversarial_dataset.npz")

    episodes_per_preset = max(1, total_episodes // len(PRESETS))
    evader = StrategicAdversarialEvader(max_force=0.50)
    policy = HeuristicPOMCPPolicy(num_simulations=120, max_depth=35)

    all_particles, all_kinematics, all_grids = [], [], []
    all_policy_targets, all_value_targets = [], []
    ep_count = 0
    win_count = 0

    sep = "=" * 70
    print(sep, flush=True)
    print("  ADVERSARIAL DATASET GENERATOR", flush=True)
    print("  Expert: Heuristic POMCP (sim=120, depth=35)", flush=True)
    print("  Evader: Strategic Adversarial (max_force=0.50)", flush=True)
    n_ep = episodes_per_preset * len(PRESETS)
    print(f"  Episodes: {n_ep} ({episodes_per_preset} per preset)", flush=True)
    print(sep, flush=True)

    t_start = time.time()

    for preset in PRESETS:
        preset_wins = 0
        preset_transitions = 0

        for ep in range(episodes_per_preset):
            seed = 9000 + ep * 7 + abs(hash(preset)) % 1000
            env = make_env(density_preset=preset, evader_policy=evader,
                           num_hunters=2, render_mode=None)
            obs, info = env.reset(seed=seed)
            policy.reset()
            policy.pathfinder = Pathfinder(env.obstacles, env.width, env.height, grid_size=20)

            episode_transitions = []
            step = 0
            captured = False

            while step < MAX_STEPS:
                actions = {}
                for agent_id in range(2):
                    plan_offset = 0 if agent_id == 0 else 15
                    obs_vec = obs["agent_" + str(agent_id)]
                    h_pos = info["hunter_positions"][agent_id]
                    h_vel = info["hunter_velocities"][agent_id]
                    b_mean = info["belief_mean"]
                    particles = info.get("particles", None)
                    is_active = info.get("is_belief_active", False) and (b_mean is not None)

                    if agent_id not in policy.current_paths:
                        policy.current_paths[agent_id] = []

                    if (step + plan_offset) % PLANNING_INTERVAL == 0 and is_active:
                        other_id = 1 - agent_id
                        ally_tgt = policy.final_targets.get(other_id, None)
                        _, visit_dist = policy._run_mcts_with_distribution(
                            h_pos, h_vel, b_mean, env.width, env.height, ally_tgt,
                            particles=particles
                        )
                        p_arr = particles if particles is not None else np.zeros((200, 4), dtype=np.float32)
                        episode_transitions.append({
                            "particles": p_arr,
                            "kinematics": obs_vec[0:8].copy(),
                            "grid": obs_vec[15:136].copy(),
                            "policy_target": visit_dist.copy(),
                            "step_t": step
                        })

                    act = policy.get_action(obs_vec, info, agent_id, env.obstacles, env.width, env.height)
                    actions["agent_" + str(agent_id)] = act

                obs, rewards, terminated, truncated, info = env.step(actions)
                step += 1
                if terminated["__all__"] or truncated["__all__"]:
                    captured = terminated["__all__"]
                    break

            T = step
            base_reward = 1.0 if captured else -1.0
            if captured:
                preset_wins += 1
                win_count += 1

            for trans in episode_transitions:
                t = trans["step_t"]
                vt = float(base_reward * (GAMMA ** (T - t - 1)))
                all_particles.append(trans["particles"])
                all_kinematics.append(trans["kinematics"])
                all_grids.append(trans["grid"])
                all_policy_targets.append(trans["policy_target"])
                all_value_targets.append([vt])
                preset_transitions += 1

            ep_count += 1

        elapsed = time.time() - t_start
        pwr = (preset_wins / episodes_per_preset) * 100.0
        print(
            "  [" + preset.rjust(12) + "]"
            " WR: " + f"{pwr:5.1f}%" +
            " | Trans: " + str(preset_transitions).rjust(5) +
            " | " + f"{elapsed:.0f}s",
            flush=True
        )

    particles_np = np.array(all_particles, dtype=np.float32)
    kinematics_np = np.array(all_kinematics, dtype=np.float32)
    grids_np = np.array(all_grids, dtype=np.float32)
    policy_targets_np = np.array(all_policy_targets, dtype=np.float32)
    value_targets_np = np.array(all_value_targets, dtype=np.float32)

    overall_wr = (win_count / ep_count) * 100.0 if ep_count > 0 else 0.0

    print("\n" + sep, flush=True)
    print(
        "  COMPLETE: " + str(ep_count) + " episodes"
        " | WR: " + f"{overall_wr:.1f}%" +
        " | Transitions: " + str(len(particles_np)),
        flush=True
    )
    win_n = int((value_targets_np > 0).sum())
    loss_n = int((value_targets_np <= 0).sum())
    print("  Win: " + str(win_n) + " | Loss: " + str(loss_n), flush=True)
    print(sep, flush=True)

    np.savez_compressed(
        output_path,
        particles=particles_np,
        kinematics=kinematics_np,
        local_grids=grids_np,
        policy_targets=policy_targets_np,
        value_targets=value_targets_np
    )
    print("  Saved to: " + output_path, flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial training dataset")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    collect_adversarial_dataset(total_episodes=args.episodes, output_path=args.output)
