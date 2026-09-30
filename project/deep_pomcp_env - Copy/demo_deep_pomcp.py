"""
Interactive Live A/B Showcase: Heuristic POMCP vs Deep-POMCP (PointNet + Value Cutoff).
Controls:
  [M]   - Instant Toggle between HEURISTIC POMCP and DEEP-POMCP
  [R]   - Reset Environment
  [1-4] - Switch Map Presets (0% Open, 15% Moderate, 30% Dense Maze, 45% Extreme)
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

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from deep_pomcp_env import make_env, ScenarioConfig
from deep_pomcp_env.baselines import HeuristicPOMCPPolicy, DeepPOMCPPolicy

def run_deep_demo():
    preset = "dense_maze"
    env = make_env(density_preset=preset, num_hunters=2, render_mode="human")
    obs, info = env.reset(seed=42)

    heuristic_policy = HeuristicPOMCPPolicy(num_simulations=120, max_depth=40)
    deep_policy = DeepPOMCPPolicy(num_simulations=50, max_depth=5)

    mode = "DEEP-POMCP"
    active_policy = deep_policy

    running = True
    clock = pygame.time.Clock()
    step_count = 0
    recent_latencies = []

    print("=" * 70)
    print("DEEP-POMCP INTERACTIVE A/B LIVE SHOWCASE")
    print("=" * 70)
    print("Controls:")
    print("  [M]   : Toggle AI Mode (HEURISTIC POMCP <-> DEEP-POMCP)")
    print("  [R]   : Reset Environment")
    print("  [1-4] : Switch Map Presets (0%, 15%, 30%, 45%)")
    print("  [ESC] : Exit")
    print("=" * 70)

    def switch_preset(new_preset: str):
        nonlocal env, obs, info, step_count
        env.close()
        env = make_env(density_preset=new_preset, num_hunters=2, render_mode="human")
        obs, info = env.reset()
        heuristic_policy.reset()
        deep_policy.reset()
        step_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    mode = "HEURISTIC" if mode == "DEEP-POMCP" else "DEEP-POMCP"
                    active_policy = heuristic_policy if mode == "HEURISTIC" else deep_policy
                    heuristic_policy.reset()
                    deep_policy.reset()
                    obs, info = env.reset()
                    step_count = 0
                    print(f"[*] Switched AI Mode to: {mode}")
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    heuristic_policy.reset()
                    deep_policy.reset()
                    step_count = 0
                elif event.key == pygame.K_1:
                    switch_preset("open")
                elif event.key == pygame.K_2:
                    switch_preset("moderate")
                elif event.key == pygame.K_3:
                    switch_preset("dense_maze")
                elif event.key == pygame.K_4:
                    switch_preset("extreme")

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
            hud_rect = pygame.Rect(10, env.height - 80, 420, 68)
            pygame.draw.rect(env.screen, (15, 23, 42), hud_rect, border_radius=6)
            pygame.draw.rect(env.screen, (99, 102, 241), hud_rect, 2, border_radius=6)
            font = pygame.font.SysFont("monospace", 14, bold=True)
            font_small = pygame.font.SysFont("monospace", 12)

            mode_color = (99, 102, 241) if mode == "DEEP-POMCP" else (245, 158, 11)
            text_mode = font.render(f"ACTIVE AI: {mode}", True, mode_color)
            mean_rolling_lat = np.mean(recent_latencies) if recent_latencies else 0.0
            text_stats = font_small.render(f"LATENCY: {mean_rolling_lat:4.2f}ms/step | PRESET: {env.config.density_preset.upper()}", True, (226, 232, 240))
            text_inst = font_small.render("[M] Toggle AI Mode  |  [R] Reset  |  [1-4] Presets", True, (148, 163, 184))

            env.screen.blit(text_mode, (22, env.height - 72))
            env.screen.blit(text_stats, (22, env.height - 52))
            env.screen.blit(text_inst, (22, env.height - 32))

            path_dict = getattr(active_policy, "current_paths", {})
            path_colors = [(129, 140, 248), (56, 189, 248)]
            for i in range(env.num_hunters):
                pts_list = path_dict.get(i, [])
                if len(pts_list) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in pts_list]
                    pygame.draw.lines(env.screen, path_colors[i % len(path_colors)], False, pts, 2)

            pygame.display.flip()

        if terminated["__all__"] or truncated["__all__"]:
            outcome = "TARGET CAPTURED!" if terminated["__all__"] else "TIME LIMIT REACHED"
            print(f"[{mode}] Episode finished in {step_count} steps. Outcome: {outcome}")
            pygame.time.wait(800)
            obs, info = env.reset()
            heuristic_policy.reset()
            deep_policy.reset()
            step_count = 0

    env.close()

if __name__ == "__main__":
    run_deep_demo()

