"""
Interactive Visual Demonstration of the deep_pomcp_env Gymnasium Benchmark.
Features:
  - A* Obstacle-Aware Pathfinding & Cornering Navigation
  - Dynamic Line-of-Sight (LOS) Vision Raycasting
  - Sequential Monte Carlo Particle Filter Belief Tracking
  - Real-Time Preset Switching (0% Open, 15% Moderate, 30% Dense Maze, 45% Extreme)
Controls:
  [R] - Reset Environment
  [1] - Switch to 0% Open Arena
  [2] - Switch to 15% Moderate Pillars
  [3] - Switch to 30% Symmetric Occlusion Maze
  [4] - Switch to 45% Dense Labyrinth
  [ESC] - Quit
"""

import os
import sys
import math
from typing import List, Dict
import numpy as np
import pygame

# Ensure parent directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from deep_pomcp_env import make_env, ScenarioConfig, Pathfinder, SmartRaycastEvader

def run_demo():
    preset = "dense_maze"
    env = make_env(density_preset=preset, num_hunters=2, render_mode="human")
    obs, info = env.reset(seed=42)

    pathfinder = Pathfinder(env.obstacles, env.width, env.height, grid_size=20)
    current_paths: Dict[int, List[np.ndarray]] = {i: [] for i in range(env.num_hunters)}

    running = True
    clock = pygame.time.Clock()
    step_count = 0

    print("=" * 60)
    print("DEEP-POMCP ENVIRONMENT INTERACTIVE DEMO (A* NAVIGATION)")
    print("=" * 60)
    print("Controls:")
    print("  [R] : Reset Environment")
    print("  [1] : 0% Open Preset")
    print("  [2] : 15% Moderate Preset")
    print("  [3] : 30% Dense Maze Preset")
    print("  [4] : 45% Extreme Preset")
    print("  [ESC] or Close Window : Exit")
    print("=" * 60)

    def update_environment(new_preset: str):
        nonlocal env, pathfinder, current_paths, step_count
        env.close()
        env = make_env(density_preset=new_preset, num_hunters=2, render_mode="human")
        o, inf = env.reset()
        pathfinder = Pathfinder(env.obstacles, env.width, env.height, grid_size=20)
        current_paths = {i: [] for i in range(env.num_hunters)}
        step_count = 0
        return o, inf

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    current_paths = {i: [] for i in range(env.num_hunters)}
                    step_count = 0
                elif event.key == pygame.K_1:
                    obs, info = update_environment("open")
                elif event.key == pygame.K_2:
                    obs, info = update_environment("moderate")
                elif event.key == pygame.K_3:
                    obs, info = update_environment("dense_maze")
                elif event.key == pygame.K_4:
                    obs, info = update_environment("extreme")

        # --- A* Obstacle-Aware Pathfinding & Steering Controller ---
        b_mean = info["belief_mean"]
        actions = {}

        for i in range(env.num_hunters):
            h_pos = info["hunter_positions"][i]
            h_vel = info["hunter_velocities"][i]

            # Re-plan path every 15 steps or when current path is empty
            if step_count % 15 == 0 or len(current_paths[i]) == 0:
                target_point = b_mean
                # Add slight flanking separation between hunter 1 and hunter 2
                if env.num_hunters == 2 and i == 1:
                    target_point = b_mean + np.array([0.0, 30.0 if b_mean[1] < env.height/2 else -30.0])

                walkable_target = pathfinder.get_nearest_walkable(target_point)
                new_path = pathfinder.find_path(h_pos, walkable_target)
                if new_path:
                    current_paths[i] = new_path

            # Follow Waypoints
            steer_force = np.array([0.0, 0.0], dtype=np.float32)
            if current_paths[i]:
                # Pop reached waypoint
                if np.linalg.norm(h_pos - current_paths[i][0]) < 25.0:
                    current_paths[i].pop(0)

                if current_paths[i]:
                    desired_dir = current_paths[i][0] - h_pos
                    norm_dir = np.linalg.norm(desired_dir)
                    if norm_dir > 0:
                        desired_vel = (desired_dir / norm_dir) * 4.0
                        steer_force = desired_vel - h_vel

            # Safety Wall Repulsion
            for obs_item in env.obstacles:
                lookahead_pt = h_pos + h_vel * 8.0
                if obs_item.collides_point(lookahead_pt[0], lookahead_pt[1], buffer=6.0):
                    steer_force += np.array([-h_vel[1], h_vel[0]], dtype=np.float32) * 2.0

            # Inter-Hunter Separation (Boids)
            for j in range(env.num_hunters):
                if i != j:
                    other_pos = info["hunter_positions"][j]
                    dist_to_other = np.linalg.norm(h_pos - other_pos)
                    if dist_to_other < 35.0 and dist_to_other > 0:
                        push_away = (h_pos - other_pos) / dist_to_other
                        steer_force += push_away * 1.5

            # Map continuous steering force to best discrete action (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)
            action_candidates = [
                (np.array([0.0, -0.5]), 0),  # UP
                (np.array([0.0, 0.5]), 1),   # DOWN
                (np.array([-0.5, 0.0]), 2),  # LEFT
                (np.array([0.5, 0.0]), 3),   # RIGHT
            ]
            if np.linalg.norm(steer_force) > 0.01:
                best_act = max(action_candidates, key=lambda pair: np.dot(pair[0], steer_force))[1]
            else:
                best_act = 4  # WAIT
            actions[f"agent_{i}"] = best_act

        obs, rewards, terminated, truncated, info = env.step(actions)
        step_count += 1

        # Draw A* Path Trails in Window
        if env.screen is not None:
            path_colors = [(96, 165, 250), (103, 232, 249)]
            for i in range(env.num_hunters):
                if len(current_paths[i]) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in current_paths[i]]
                    pygame.draw.lines(env.screen, path_colors[i % len(path_colors)], False, pts, 1)
            pygame.display.flip()

        if terminated["__all__"] or truncated["__all__"]:
            outcome = "TARGET CAPTURED!" if terminated["__all__"] else "TIME LIMIT REACHED"
            print(f"Episode Finished in {step_count} steps. Outcome: {outcome}")
            pygame.time.wait(800)
            obs, info = env.reset()
            current_paths = {i: [] for i in range(env.num_hunters)}
            step_count = 0

    env.close()
    print("Demo closed.")

if __name__ == "__main__":
    run_demo()
