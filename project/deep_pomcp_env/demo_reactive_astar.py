"""
Interactive Visualizer: Reactive A* Pathfinding Tracker Showcase
Demonstrates how A* computes shortest paths to the belief mean with ultra-low latency,
and how it compares visually with Deep-POMCP and Heuristic POMCP.

Controls:
  [A]   - Switch to REACTIVE A* POLICY
  [D]   - Switch to DEEP-POMCP POLICY
  [H]   - Switch to HEURISTIC POMCP POLICY
  [1-4] - Density Presets (0% Open, 15% Moderate, 30% Dense Maze, 45% Extreme)
  [5-7] - Pathological Traps (5: U-Trap, 6: Figure-8, 7: Bimodal Fork)
  [R]   - Reset / Randomize Spawns
  [ESC] - Quit
"""

import os
import sys
import time
import math
import numpy as np
import pygame

# Ensure parent directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from deep_pomcp_env import make_env, ScenarioConfig
from deep_pomcp_env.baselines import ReactiveAStarPolicy, DeepPOMCPPolicy, HeuristicPOMCPPolicy

def run_reactive_astar_showcase():
    preset = "dense_maze"
    env = make_env(density_preset=preset, num_hunters=2, render_mode="human")
    obs, info = env.reset(seed=42)

    policy_astar = ReactiveAStarPolicy(name="Reactive A*")
    policy_deep = DeepPOMCPPolicy(weights_path="deep_pomcp_weights.pth", num_simulations=40, max_depth=6, name="Deep-POMCP")
    policy_heuristic = HeuristicPOMCPPolicy(num_simulations=100, max_depth=15, name="Heuristic POMCP")

    active_policy = policy_astar
    mode_name = "REACTIVE A*"
    mode_color = (56, 189, 248) # Sky blue

    clock = pygame.time.Clock()
    running = True
    step_count = 0
    recent_latencies = []

    print("=" * 75)
    print("REACTIVE A* vs. DEEP-POMCP VISUAL DEMONSTRATION")
    print("=" * 75)
    print("Controls:")
    print("  [A]   : Activate REACTIVE A*")
    print("  [D]   : Activate DEEP-POMCP (Neural PUCT)")
    print("  [H]   : Activate HEURISTIC POMCP")
    print("  [1-4] : Switch Obstacle Presets (0%, 15%, 30%, 45%)")
    print("  [5-7] : Switch Pathological Traps (U-Trap, Figure-8, Bimodal Fork)")
    print("  [R]   : Reset Episode & Randomize Spawns")
    print("  [ESC] : Exit")
    print("=" * 75)

    def switch_preset(new_preset: str):
        nonlocal env, obs, info, step_count
        env.close()
        env = make_env(density_preset=new_preset, num_hunters=2, render_mode="human")
        obs, info = env.reset()
        policy_astar.reset()
        policy_deep.reset()
        policy_heuristic.reset()
        step_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    active_policy = policy_astar
                    mode_name = "REACTIVE A*"
                    mode_color = (56, 189, 248)
                    policy_astar.reset()
                    obs, info = env.reset()
                    step_count = 0
                    print("[*] Active AI: REACTIVE A* (Direct Shortest Path Tracker)")
                elif event.key == pygame.K_d:
                    active_policy = policy_deep
                    mode_name = "DEEP-POMCP (Ours)"
                    mode_color = (129, 140, 248)
                    policy_deep.reset()
                    obs, info = env.reset()
                    step_count = 0
                    print("[*] Active AI: DEEP-POMCP (PointNet + Truncated PUCT)")
                elif event.key == pygame.K_h:
                    active_policy = policy_heuristic
                    mode_name = "HEURISTIC POMCP"
                    mode_color = (245, 158, 11)
                    policy_heuristic.reset()
                    obs, info = env.reset()
                    step_count = 0
                    print("[*] Active AI: HEURISTIC POMCP")
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    active_policy.reset()
                    step_count = 0
                elif event.key == pygame.K_1: switch_preset("open")
                elif event.key == pygame.K_2: switch_preset("moderate")
                elif event.key == pygame.K_3: switch_preset("dense_maze")
                elif event.key == pygame.K_4: switch_preset("extreme")
                elif event.key == pygame.K_5: switch_preset("u_trap")
                elif event.key == pygame.K_6: switch_preset("figure_8")
                elif event.key == pygame.K_7: switch_preset("bimodal_fork")

        # Compute Hunter Actions
        actions = {}
        step_latencies = []
        for i in range(env.num_hunters):
            t0 = time.perf_counter()
            act = active_policy.get_action(obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies.append(lat_ms)
            actions[f"agent_{i}"] = act

        avg_lat = np.mean(step_latencies)
        recent_latencies.append(avg_lat)
        if len(recent_latencies) > 30:
            recent_latencies.pop(0)

        obs, rewards, terminated, truncated, info = env.step(actions)
        step_count += 1

        if env.screen is not None:
            # Draw Live Planned Paths
            path_dict = getattr(active_policy, "current_paths", {})
            path_colors = [(56, 189, 248), (244, 114, 182)] # Cyan and Pink
            for i in range(env.num_hunters):
                pts_list = path_dict.get(i, [])
                if len(pts_list) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in pts_list]
                    pygame.draw.lines(env.screen, path_colors[i % len(path_colors)], False, pts, 3)
                    # Draw waypoints as small circles
                    for p in pts:
                        pygame.draw.circle(env.screen, path_colors[i % len(path_colors)], p, 3)

            # Draw Belief Mean Center & Spread Circle
            b_mean = info.get('belief_mean', None)
            b_spread = info.get('belief_spread', 0.0)
            is_active = info.get('is_belief_active', False) and (b_mean is not None)

            if is_active:
                pygame.draw.circle(env.screen, (234, 179, 8), (int(b_mean[0]), int(b_mean[1])), 6)
                pygame.draw.circle(env.screen, (234, 179, 8), (int(b_mean[0]), int(b_mean[1])), max(10, int(b_spread)), 1)

            # HUD Telemetry Card
            hud_rect = pygame.Rect(10, env.height - 95, 520, 85)
            pygame.draw.rect(env.screen, (15, 23, 42), hud_rect, border_radius=6)
            pygame.draw.rect(env.screen, mode_color, hud_rect, 2, border_radius=6)

            font = pygame.font.SysFont("monospace", 14, bold=True)
            font_small = pygame.font.SysFont("monospace", 12)

            state_label = "TRACKING BELIEF CLOUD" if is_active else "SECTOR PATROL (UNOBSERVED)"
            state_color = (34, 197, 94) if is_active else (244, 63, 94) # Green vs Rose
            text_mode = font.render(f"ACTIVE POLICY: {mode_name} | [{state_label}]", True, mode_color)
            mean_rolling_lat = np.mean(recent_latencies) if recent_latencies else 0.0
            
            h0_path_len = len(path_dict.get(0, []))
            h1_path_len = len(path_dict.get(1, []))
            
            text_stats = font_small.render(
                f"LATENCY: {mean_rolling_lat:4.2f}ms | PRESET: {env.config.density_preset.upper()} | NODES: H1={h0_path_len}, H2={h1_path_len}",
                True, (226, 232, 240)
            )
            text_desc = font_small.render(
                "A* computes shortest grid path to belief mean (Patrols sectors if unobserved)" if mode_name == "REACTIVE A*" else
                "Deep-POMCP evaluates leaf states via PointNet neural value function",
                True, (148, 163, 184)
            )
            text_inst = font_small.render(
                "[A] A*  |  [D] Deep-POMCP  |  [H] Heuristic  |  [1-7] Maps  |  [R] Reset",
                True, (203, 213, 225)
            )

            env.screen.blit(text_mode, (22, env.height - 88))
            env.screen.blit(text_stats, (22, env.height - 68))
            env.screen.blit(text_desc, (22, env.height - 48))
            env.screen.blit(text_inst, (22, env.height - 28))

            pygame.display.flip()
            clock.tick(60)

    env.close()

if __name__ == "__main__":
    run_reactive_astar_showcase()

