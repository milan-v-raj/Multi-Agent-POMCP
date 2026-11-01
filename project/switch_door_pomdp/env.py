"""
Switch-Door Dec-POMDP Environment.
Formulates the Switch-Door puzzle with explicit, observable environmental affordances.
The switch location and gate status are part of the formal observation space.
"""

import math
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from deep_pomcp_env.core_env import PursuitEvasionEnv, ParticleFilterInternal
from deep_pomcp_env.scenarios import Obstacle, ScenarioConfig
from deep_pomcp_env.evaders import BaseEvader, StrategicAdversarialEvader

class SwitchDoorPOMDPEnv(PursuitEvasionEnv):
    def __init__(self, config: Optional[ScenarioConfig] = None,
                 evader_policy: Optional[BaseEvader] = None,
                 render_mode: Optional[str] = None):
        super().__init__(config=config, evader_policy=evader_policy, render_mode=render_mode)
        self.switch_pos = np.array([80.0, 300.0], dtype=np.float32)
        self.switch_radius = 35.0
        self.gate_open = False
        self.gate_obstacle: Optional[Obstacle] = None
        self.static_obstacles: List[Obstacle] = []
        self.first_switch_step: Optional[int] = None
        self.gate_breach_step: Optional[int] = None
        self.switch_triggered = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.current_step = 0
        self.last_seen_step = 0
        self.gate_open = False
        self.first_switch_step = None
        self.gate_breach_step = None
        self.switch_triggered = False

        self._generate_switch_door_map()
        self._spawn_agents()

        self.pf.clear()
        self._update_sensing()

        observations = self._get_observations()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, info

    def _generate_switch_door_map(self):
        w, h = self.width, self.height
        self.static_obstacles = []

        self.static_obstacles.append(Obstacle(0, 0, w, 10))
        self.static_obstacles.append(Obstacle(0, h - 10, w, 10))
        self.static_obstacles.append(Obstacle(0, 0, 10, h))
        self.static_obstacles.append(Obstacle(w - 10, 0, 10, h))

        wall_x = 580.0
        gate_h = 100.0
        gate_y = float(self.rng.randint(int(h * 0.35), int(h * 0.55)))

        if gate_y > 15:
            self.static_obstacles.append(Obstacle(wall_x, 10, 15, gate_y - 10))
        bot_y = gate_y + gate_h
        if bot_y < h - 15:
            self.static_obstacles.append(Obstacle(wall_x, bot_y, 15, h - 10 - bot_y))

        self.gate_obstacle = Obstacle(wall_x, gate_y, 15, gate_h)

        sw_x = float(self.rng.randint(60, 110))
        sw_y = float(self.rng.randint(int(h * 0.25), int(h * 0.75)))
        self.switch_pos = np.array([sw_x, sw_y], dtype=np.float32)

        num_clutter = self.rng.randint(3, 5)
        for _ in range(num_clutter):
            cw = float(self.rng.randint(30, 60))
            ch = float(self.rng.randint(30, 70))
            cx = float(self.rng.randint(160, int(wall_x - 100)))
            cy = float(self.rng.randint(40, int(h - 100)))
            self.static_obstacles.append(Obstacle(cx, cy, cw, ch))

        self.obstacles = list(self.static_obstacles) + [self.gate_obstacle]
        self._build_occupancy_map()

    def _spawn_agents(self):
        w, h = self.width, self.height
        for i in range(self.num_hunters):
            for _ in range(100):
                hx = float(self.rng.randint(220, 380))
                hy = float(self.rng.randint(120, h - 120))
                cand = np.array([hx, hy], dtype=np.float32)
                if not any(obs.collides_point(cand[0], cand[1], buffer=20.0) for obs in self.obstacles):
                    self.hunter_pos[i] = cand
                    break
            self.hunter_vel[i] = np.array([0.0, 0.0], dtype=np.float32)

        for _ in range(100):
            ex = float(self.rng.randint(640, int(w - 50)))
            ey = float(self.rng.randint(120, h - 120))
            cand = np.array([ex, ey], dtype=np.float32)
            if not any(obs.collides_point(cand[0], cand[1], buffer=20.0) for obs in self.obstacles):
                self.evader_pos = cand
                break
        self.evader_vel = np.array([0.0, 0.0], dtype=np.float32)

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self.current_step += 1
        wall_hits = [False] * self.num_hunters

        dists_to_sw = [float(np.linalg.norm(self.hunter_pos[i] - self.switch_pos)) for i in range(self.num_hunters)]
        min_sw_dist = min(dists_to_sw)
        self.gate_open = (min_sw_dist < self.switch_radius)

        if self.gate_open:
            self.switch_triggered = True
            if self.first_switch_step is None:
                self.first_switch_step = self.current_step
            self.obstacles = list(self.static_obstacles)
        else:
            self.obstacles = list(self.static_obstacles) + [self.gate_obstacle]

        self._build_occupancy_map()

        for i in range(self.num_hunters):
            if self.hunter_pos[i][0] > 600.0 and self.gate_breach_step is None:
                self.gate_breach_step = self.current_step

        for i in range(self.num_hunters):
            act = actions.get(f"agent_{i}", 4)
            force = self.ACTION_VECTORS[act] if isinstance(act, (int, np.integer)) else np.array(act, dtype=np.float32)

            self.hunter_vel[i] += force
            self.hunter_vel[i] *= self.friction
            spd = np.linalg.norm(self.hunter_vel[i])
            if spd > self.max_speed_hunter:
                self.hunter_vel[i] = (self.hunter_vel[i] / spd) * self.max_speed_hunter

            next_pos = self.hunter_pos[i] + self.hunter_vel[i]
            next_pos, hit = self._resolve_wall_collisions(self.hunter_pos[i], next_pos, radius=10.0)
            if hit:
                wall_hits[i] = True
                self.hunter_vel[i] *= -0.5
            self.hunter_pos[i] = next_pos

        h_pos_list = [self.hunter_pos[i] for i in range(self.num_hunters)]
        e_force = self.evader_policy.get_action(
            self.evader_pos, self.evader_vel, h_pos_list, self.obstacles, self.width, self.height
        )
        self.evader_vel += e_force
        self.evader_vel *= self.friction
        e_spd = np.linalg.norm(self.evader_vel)
        if e_spd > self.max_speed_evader:
            self.evader_vel = (self.evader_vel / e_spd) * self.max_speed_evader

        next_e_pos = self.evader_pos + self.evader_vel
        next_e_pos, _ = self._resolve_wall_collisions(self.evader_pos, next_e_pos, radius=10.0)
        self.evader_pos = next_e_pos

        self._update_sensing()

        dists_to_evader = [float(np.linalg.norm(self.hunter_pos[i] - self.evader_pos)) for i in range(self.num_hunters)]
        min_dist = min(dists_to_evader)
        captured = min_dist < self.capture_radius
        timeout = self.current_step >= self.max_steps

        terminated = {f"agent_{i}": captured for i in range(self.num_hunters)}
        terminated["__all__"] = captured
        truncated = {f"agent_{i}": timeout for i in range(self.num_hunters)}
        truncated["__all__"] = timeout

        rewards = self._compute_rewards(min_dist, captured, wall_hits)
        observations = self._get_observations()
        info = self._get_info()
        info["captured"] = captured
        info["timeout"] = timeout
        info["min_dist_to_evader"] = min_dist

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, truncated, info

    def _get_observations(self) -> Dict[str, np.ndarray]:
        base_obs = super()._get_observations()
        sw_feat = np.array([
            self.switch_pos[0] / self.width,
            self.switch_pos[1] / self.height,
            1.0 if self.gate_open else 0.0
        ], dtype=np.float32)

        augmented_obs = {}
        for k, v in base_obs.items():
            augmented_obs[k] = np.concatenate([v, sw_feat])
        return augmented_obs

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info.update({
            "switch_pos": np.copy(self.switch_pos),
            "switch_radius": self.switch_radius,
            "gate_open": self.gate_open,
            "gate_obstacle": self.gate_obstacle,
            "static_obstacles": list(self.static_obstacles),
            "switch_triggered": self.switch_triggered,
            "first_switch_step": self.first_switch_step,
            "gate_breach_step": self.gate_breach_step
        })
        return info


def make_switch_door_pomdp_env(evader_policy: Optional[BaseEvader] = None, render_mode: Optional[str] = None) -> SwitchDoorPOMDPEnv:
    cfg = ScenarioConfig(width=800, height=600, max_steps=1200, capture_radius=30.0)
    return SwitchDoorPOMDPEnv(config=cfg, evader_policy=evader_policy, render_mode=render_mode)
