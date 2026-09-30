"""
Interactive Live Showcase: Information-Theoretic Event-Triggered MCTS (ET-MCTS).
Displays live event badges on screen:
  - [TRIGGER: ENTROPY FLUX]
  - [TRIGGER: LOS TRANSITION]
  - [TRIGGER: PATH INVALID]
  - [IDLE: A* WAYPOINT TRACKING]
Controls:
  [E]     - Toggle Evader AI (Strategic Adversarial <-> Standard Raycast)
  [R]     - Reset Environment
  [1-7]   - Switch Map Presets
  [SPACE] - Pause / Resume
  [ESC]   - Quit
"""

import os
import sys
import time
import numpy as np
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from deep_pomcp_env import make_env
from deep_pomcp_env.evaders import SmartRaycastEvader, StrategicAdversarialEvader
from deep_pomcp_env.baselines.et_deep_pomcp import EventTriggeredDeepPOMCPPolicy

def run_et_demo():
    preset = "dense_maze"
    evader = StrategicAdversarialEvader(max_force=0.50)
    evader_name = "Strategic Adversarial"

    weights_path = os.path.join(parent_dir, "deep_pomcp_weights.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(current_dir, "deep_pomcp_weights.pth")

    policy = EventTriggeredDeepPOMCPPolicy(weights_path=weights_path, name="ET-Deep-POMCP (Ours)")
    env = make_env(density_preset=preset, evader_policy=evader, num_hunters=2, render_mode="human")
    obs, info = env.reset(seed=42)

    running = True
    paused = False
    step_count = 0
    recent_latencies = []

    print("=" * 80)
    print("      EVENT-TRIGGERED MCTS (ET-MCTS) LIVE INTERACTIVE SHOWCASE")
    print("=" * 80)
    print("Controls:")
    print("  [E]     : Toggle Evader AI")
    print("  [R]     : Reset Environment")
    print("  [1-7]   : Switch Map Presets")
    print("  [SPACE] : Pause / Resume")
    print("  [ESC]   : Exit")
    print("=" * 80)

    preset_map = {
        pygame.K_1: "open",
        pygame.K_2: "moderate",
        pygame.K_3: "dense_maze",
        pygame.K_4: "extreme",
        pygame.K_5: "u_trap",
        pygame.K_6: "figure_8",
        pygame.K_7: "bimodal_fork"
    }

    def recreate(new_preset=None, new_evader=None):
        nonlocal env, obs, info, step_count, preset, evader
        if new_preset:
            preset = new_preset
        if new_evader:
            evader = new_evader
        env.close()
        env = make_env(density_preset=preset, evader_policy=evader, num_hunters=2, render_mode="human")
        obs, info = env.reset()
        policy.reset()
        step_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    policy.reset()
                    step_count = 0
                elif event.key == pygame.K_e:
                    if evader_name == "Strategic Adversarial":
                        evader_name = "Standard Raycast"
                        recreate(new_evader=SmartRaycastEvader(max_force=0.45))
                    else:
                        evader_name = "Strategic Adversarial"
                        recreate(new_evader=StrategicAdversarialEvader(max_force=0.50))
                elif event.key in preset_map:
                    recreate(new_preset=preset_map[event.key])

        if not paused:
            actions = {}
            step_latencies = []
            for i in range(env.num_hunters):
                t0 = time.perf_counter()
                act = policy.get_action(obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height)
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
            path_dict = getattr(policy, "current_paths", {})
            path_colors = [(129, 140, 248), (56, 189, 248)]
            for i in range(env.num_hunters):
                pts_list = path_dict.get(i, [])
                if len(pts_list) > 1:
                    pts = [(int(p[0]), int(p[1])) for p in pts_list]
                    pygame.draw.lines(env.screen, path_colors[i % len(path_colors)], False, pts, 2)

            # Draw HUD
            hud_w, hud_h = 580, 100
            hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
            pygame.draw.rect(hud_surface, (15, 23, 42, 230), (0, 0, hud_w, hud_h), border_radius=8)
            pygame.draw.rect(hud_surface, (16, 185, 129), (0, 0, hud_w, hud_h), 2, border_radius=8)
            env.screen.blit(hud_surface, (10, env.height - hud_h - 10))

            font_bold = pygame.font.SysFont("monospace", 13, bold=True)
            font_small = pygame.font.SysFont("monospace", 11)

            text_title = font_bold.render("AI ENGINE: INFORMATION-THEORETIC ET-MCTS", True, (16, 185, 129))
            mean_rolling_lat = np.mean(recent_latencies) if recent_latencies else 0.0

            # Event triggers display
            h0_trigger = policy._last_trigger_reason.get(0, "IDLE")
            h1_trigger = policy._last_trigger_reason.get(1, "IDLE")
            t_col0 = (239, 68, 68) if "FLUX" in h0_trigger or "LOS" in h0_trigger else (148, 163, 184)
            t_col1 = (239, 68, 68) if "FLUX" in h1_trigger or "LOS" in h1_trigger else (148, 163, 184)

            text_telemetry = font_small.render(
                f"STEP: {step_count:4d} | LATENCY: {mean_rolling_lat:4.2f}ms | MCTS CALLS: {policy.total_mcts_calls:3d}",
                True, (226, 232, 240)
            )
            text_events = font_small.render(
                f"H0: [{h0_trigger}] | H1: [{h1_trigger}]",
                True, (244, 208, 63)
            )
            text_controls = font_small.render(
                "[E] Evader | [R] Reset | [SPACE] Pause | [1-7] Presets",
                True, (148, 163, 184)
            )

            env.screen.blit(text_title, (20, env.height - hud_h - 2))
            env.screen.blit(text_telemetry, (20, env.height - hud_h + 20))
            env.screen.blit(text_events, (20, env.height - hud_h + 40))
            env.screen.blit(text_controls, (20, env.height - hud_h + 60))

            pygame.display.flip()

        if terminated["__all__"] or truncated["__all__"]:
            outcome = "TARGET CAPTURED!" if terminated["__all__"] else "TIME LIMIT REACHED"
            print(f"[ET-MCTS] Episode finished in {step_count} steps. Total MCTS Calls: {policy.total_mcts_calls}. Outcome: {outcome}")
            pygame.time.wait(800)
            obs, info = env.reset()
            policy.reset()
            step_count = 0

    env.close()

if __name__ == "__main__":
    run_et_demo()
