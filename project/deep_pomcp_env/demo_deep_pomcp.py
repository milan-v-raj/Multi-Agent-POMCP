"""
Interactive Live Showcase: Deep-POMCP vs Baselines in Multi-Agent Pursuit Evasion.
Controls:
  [M]     - Cycle Pursuer AI (DEEP-POMCP -> REACTIVE A* -> HEURISTIC POMCP -> VANILLA POMCP)
  [E]     - Toggle Evader AI (STRATEGIC ADVERSARIAL <-> STANDARD RAYCAST)
  [R]     - Reset Environment with new seed
  [SPACE] - Pause / Resume simulation
  [1-7]   - Switch Map Presets:
            1: Open (0%)         2: Moderate (15%)    3: Dense Maze (30%)   4: Extreme (45%)
            5: U-Trap            6: Figure-8          7: Bimodal Fork
  [ESC]   - Quit
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

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from deep_pomcp_env import make_env
from deep_pomcp_env.evaders import SmartRaycastEvader, StrategicAdversarialEvader
from deep_pomcp_env.baselines import (
    DeepPOMCPPolicy,
    ReactiveAStarPolicy,
    HeuristicPOMCPPolicy,
    VanillaPOMCPPolicy
)

def run_deep_demo():
    preset = "dense_maze"
    evader_type_name = "Strategic Adversarial"
    current_evader = StrategicAdversarialEvader(max_force=0.50)

    weights_path = os.path.join(parent_dir, "deep_pomcp_weights.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(current_dir, "deep_pomcp_weights.pth")

    policies = {
        "DEEP-POMCP (Ours)": DeepPOMCPPolicy(weights_path=weights_path if os.path.exists(weights_path) else None, num_simulations=60, max_depth=6),
        "REACTIVE A*": ReactiveAStarPolicy(),
        "HEURISTIC POMCP": HeuristicPOMCPPolicy(num_simulations=120, max_depth=35),
        "VANILLA POMCP": VanillaPOMCPPolicy(num_simulations=80, max_depth=25)
    }
    policy_names = list(policies.keys())
    policy_idx = 0
    active_policy = policies[policy_names[policy_idx]]

    env = make_env(density_preset=preset, evader_policy=current_evader, num_hunters=2, render_mode="human")
    obs, info = env.reset(seed=42)

    running = True
    paused = False
    step_count = 0
    recent_latencies = []

    print("=" * 80)
    print("        DEEP-POMCP INTERACTIVE MULTI-AGENT PURSUIT-EVASION SHOWCASE")
    print("=" * 80)
    print("Controls:")
    print("  [M]     : Cycle Pursuer AI (DEEP-POMCP -> REACTIVE A* -> HEURISTIC -> VANILLA)")
    print("  [E]     : Toggle Evader AI (STRATEGIC ADVERSARIAL <-> STANDARD RAYCAST)")
    print("  [R]     : Reset Environment (new random seed)")
    print("  [SPACE] : Pause / Resume")
    print("  [1-7]   : Switch Map (1:Open, 2:Mod, 3:Maze, 4:Ext, 5:U-Trap, 6:Fig-8, 7:Fork)")
    print("  [ESC]   : Exit")
    print("=" * 80)

    def recreate_env(new_preset: str = None, new_evader = None):
        nonlocal env, obs, info, step_count, preset, current_evader
        if new_preset is not None:
            preset = new_preset
        if new_evader is not None:
            current_evader = new_evader
        env.close()
        env = make_env(density_preset=preset, evader_policy=current_evader, num_hunters=2, render_mode="human")
        obs, info = env.reset()
        for p in policies.values():
            p.reset()
        step_count = 0

    preset_map = {
        pygame.K_1: "open",
        pygame.K_2: "moderate",
        pygame.K_3: "dense_maze",
        pygame.K_4: "extreme",
        pygame.K_5: "u_trap",
        pygame.K_6: "figure_8",
        pygame.K_7: "bimodal_fork"
    }

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"[*] Simulation {'PAUSED' if paused else 'RESUMED'}")
                elif event.key == pygame.K_m:
                    policy_idx = (policy_idx + 1) % len(policy_names)
                    active_policy = policies[policy_names[policy_idx]]
                    for p in policies.values():
                        p.reset()
                    obs, info = env.reset()
                    step_count = 0
                    print(f"[*] Pursuer AI: {policy_names[policy_idx]}")
                elif event.key == pygame.K_e:
                    if evader_type_name == "Strategic Adversarial":
                        evader_type_name = "Standard Raycast"
                        recreate_env(new_evader=SmartRaycastEvader(max_force=0.45))
                    else:
                        evader_type_name = "Strategic Adversarial"
                        recreate_env(new_evader=StrategicAdversarialEvader(max_force=0.50))
                    print(f"[*] Evader AI: {evader_type_name}")
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    for p in policies.values():
                        p.reset()
                    step_count = 0
                    print("[*] Environment reset.")
                elif event.key in preset_map:
                    recreate_env(new_preset=preset_map[event.key])
                    print(f"[*] Preset switched to: {preset_map[event.key].upper()}")

        if not paused:
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
        else:
            terminated = {"__all__": False}
            truncated = {"__all__": False}

        if env.screen is not None:
            # Draw particle cloud
            particles = info.get("particles", None)
            if particles is not None and len(particles) > 0:
                for pt in particles:
                    px = int(pt[0] * env.width)
                    py = int(pt[1] * env.height)
                    if 0 <= px < env.width and 0 <= py < env.height:
                        pygame.draw.circle(env.screen, (244, 63, 94), (px, py), 2)

            # Draw Planned Paths
            path_dict = getattr(active_policy, "current_paths", {})
            path_colors = [(129, 140, 248), (56, 189, 248)]
            for i in range(env.num_hunters):
                pts_list = path_dict.get(i, [])
                if len(pts_list) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in pts_list]
                    pygame.draw.lines(env.screen, path_colors[i % len(path_colors)], False, pts, 2)

            # Draw HUD
            hud_w, hud_h = 560, 95
            hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
            pygame.draw.rect(hud_surface, (15, 23, 42, 230), (0, 0, hud_w, hud_h), border_radius=8)
            pygame.draw.rect(hud_surface, (99, 102, 241), (0, 0, hud_w, hud_h), 2, border_radius=8)
            env.screen.blit(hud_surface, (10, env.height - hud_h - 10))

            font_bold = pygame.font.SysFont("monospace", 13, bold=True)
            font_small = pygame.font.SysFont("monospace", 11)

            p_color = (99, 102, 241) if "DEEP" in policy_names[policy_idx] else ((52, 211, 153) if "REACT" in policy_names[policy_idx] else (245, 158, 11))
            text_p = font_bold.render(f"PURSUER: {policy_names[policy_idx]}", True, p_color)
            e_color = (239, 68, 68) if "ADVERSARIAL" in evader_type_name.upper() else (148, 163, 184)
            text_e = font_bold.render(f"EVADER: {evader_type_name}", True, e_color)

            mean_rolling_lat = np.mean(recent_latencies) if recent_latencies else 0.0
            los_status = "LOCKED" if info.get("can_see_global", False) else "OCCLUDED (SMC Particle Filter Active)"
            los_color = (34, 197, 94) if info.get("can_see_global", False) else (234, 179, 8)

            text_telemetry = font_small.render(
                f"STEP: {step_count:4d} | LATENCY: {mean_rolling_lat:4.2f}ms | MAP: {preset.upper()}",
                True, (226, 232, 240)
            )
            text_los_label = font_small.render("LOS: ", True, (226, 232, 240))
            text_los = font_small.render(los_status, True, los_color)

            text_controls = font_small.render(
                "[M] AI Mode | [E] Evader | [R] Reset | [SPACE] Pause | [1-7] Presets",
                True, (148, 163, 184)
            )

            env.screen.blit(text_p, (20, env.height - hud_h - 2))
            env.screen.blit(text_e, (300, env.height - hud_h - 2))
            env.screen.blit(text_telemetry, (20, env.height - hud_h + 20))
            env.screen.blit(text_los, (20 + text_telemetry.get_width() + 10, env.height - hud_h + 20))
            env.screen.blit(text_controls, (20, env.height - hud_h + 44))

            pygame.display.flip()

        if terminated["__all__"] or truncated["__all__"]:
            outcome = "TARGET CAPTURED!" if terminated["__all__"] else "TIME LIMIT REACHED"
            print(f"[{policy_names[policy_idx]}] Episode finished in {step_count} steps. Outcome: {outcome}")
            pygame.time.wait(800)
            obs, info = env.reset()
            for p in policies.values():
                p.reset()
            step_count = 0

    env.close()

if __name__ == "__main__":
    run_deep_demo()
