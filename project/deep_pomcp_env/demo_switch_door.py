"""
Interactive Live Demo: Switch-Door Cooperative Sacrifice Showcase.
Controls:
  [M]     - Cycle Pursuer Policy (DEEP-POMCP -> REACTIVE A* -> HEURISTIC -> VANILLA)
  [R]     - Reset Environment (Randomize switch, gate, clutter, and spawns)
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

from deep_pomcp_env.switch_door_env import make_switch_door_env
from deep_pomcp_env.evaders import StrategicAdversarialEvader
from deep_pomcp_env.baselines import (
    DeepPOMCPPolicy,
    ReactiveAStarPolicy,
    HeuristicPOMCPPolicy,
    VanillaPOMCPPolicy
)

def run_switch_door_demo():
    evader = StrategicAdversarialEvader(max_force=0.45)
    env = make_switch_door_env(evader_policy=evader, render_mode="human")
    obs, info = env.reset(seed=42)

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

    running = True
    paused = False
    step_count = 0

    print("=" * 80)
    print("      SWITCH-DOOR COOPERATIVE SACRIFICE LIVE DEMO")
    print("=" * 80)
    print("Controls:")
    print("  [M]     : Cycle Pursuer AI")
    print("  [R]     : Reset with new random map")
    print("  [SPACE] : Pause / Resume")
    print("  [ESC]   : Exit")
    print("=" * 80)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_m:
                    policy_idx = (policy_idx + 1) % len(policy_names)
                    active_policy = policies[policy_names[policy_idx]]
                    for p in policies.values():
                        p.reset()
                    obs, info = env.reset()
                    step_count = 0
                    print(f"[*] Policy switched to: {policy_names[policy_idx]}")
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    for p in policies.values():
                        p.reset()
                    step_count = 0
                    print("[*] Environment reset with new randomized puzzle.")

        if not paused:
            actions = {}
            for i in range(env.num_hunters):
                act = active_policy.get_action(obs[f"agent_{i}"], info, i, env.obstacles, env.width, env.height)
                actions[f"agent_{i}"] = act

            obs, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1
        else:
            terminated = {"__all__": False}
            truncated = {"__all__": False}

        if env.screen is not None:
            # Draw HUD
            hud_w, hud_h = 560, 95
            hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
            pygame.draw.rect(hud_surface, (15, 23, 42, 230), (0, 0, hud_w, hud_h), border_radius=8)
            pygame.draw.rect(hud_surface, (99, 102, 241), (0, 0, hud_w, hud_h), 2, border_radius=8)
            env.screen.blit(hud_surface, (10, env.height - hud_h - 10))

            font_bold = pygame.font.SysFont("monospace", 13, bold=True)
            font_small = pygame.font.SysFont("monospace", 11)

            p_color = (16, 185, 129) if "DEEP" in policy_names[policy_idx] else ((239, 68, 68) if "REACT" in policy_names[policy_idx] else (245, 158, 11))
            text_p = font_bold.render(f"PURSUER: {policy_names[policy_idx]}", True, p_color)

            gate_status = "GATE: OPEN (Switch Active)" if info.get("gate_open", False) else "GATE: LOCKED (Press Switch)"
            gate_color = (34, 197, 94) if info.get("gate_open", False) else (239, 68, 68)
            text_gate = font_bold.render(gate_status, True, gate_color)

            text_telemetry = font_small.render(
                f"STEP: {step_count:4d} | SWITCH HIT: {info.get('switch_triggered', False)} | BREACH: {info.get('gate_breach_step') is not None}",
                True, (226, 232, 240)
            )
            text_controls = font_small.render(
                "[M] AI Mode | [R] New Random Puzzle | [SPACE] Pause | [ESC] Exit",
                True, (148, 163, 184)
            )

            env.screen.blit(text_p, (20, env.height - hud_h - 2))
            env.screen.blit(text_gate, (270, env.height - hud_h - 2))
            env.screen.blit(text_telemetry, (20, env.height - hud_h + 20))
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
    run_switch_door_demo()
