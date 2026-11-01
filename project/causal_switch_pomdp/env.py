"""
Causal Switch-Door Dec-POMDP Environment.
Features K candidate pressure plates in the arena, with exactly one hidden
functional switch kappa* in {0, ..., K-1} that opens the security gate.

Pursuers observe the spatial positions of all K switches, but do NOT know
which switch is connected to the gate mechanism. They must engage in epistemic
exploration (trial and causal hypothesis pruning) under partial observability.
"""

import math
from typing import Dict, Any, List, Tuple, Optional, Union, Set
import numpy as np
import pygame

from deep_pomcp_env.core_env import PursuitEvasionEnv
from deep_pomcp_env.scenarios import Obstacle, ScenarioConfig
from deep_pomcp_env.evaders import BaseEvader, StrategicAdversarialEvader

class CausalSwitchDoorEnv(PursuitEvasionEnv):
    def __init__(self,
                 num_switches: int = 3,
                 config: Optional[ScenarioConfig] = None,
                 evader_policy: Optional[BaseEvader] = None,
                 render_mode: Optional[str] = None):
        super().__init__(config=config, evader_policy=evader_policy, render_mode=render_mode)
        self.num_switches = num_switches
        self.switch_positions = np.zeros((num_switches, 2), dtype=np.float32)
        self.switch_radius = 35.0
        self.true_switch_idx = 0
        self.gate_open = False
        self.gate_obstacle: Optional[Obstacle] = None
        self.static_obstacles: List[Obstacle] = []

        # Telemetry
        self.first_switch_step: Optional[int] = None
        self.gate_breach_step: Optional[int] = None
        self.switches_tested: Set[int] = set()
        self.true_switch_discovered = False
        self.discovery_step: Optional[int] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.current_step = 0
        self.last_seen_step = 0
        self.gate_open = False
        self.first_switch_step = None
        self.gate_breach_step = None
        self.switches_tested = set()
        self.true_switch_discovered = False
        self.discovery_step = None

        # 1. Randomly assign which switch is the functional one
        self.true_switch_idx = int(self.rng.randint(0, self.num_switches - 1))

        # 2. Build map and place candidate switches
        self._generate_causal_switch_map()

        # 3. Spawn hunters in central corridor, evader in sealed chamber
        self._spawn_agents()

        # 4. Initialize Particle Filter
        self.pf.clear()
        self._update_sensing()
        if not self.pf.is_initialized:
            # Bayes Filter Prior b0: Target hypothesis initialized in unobserved target wing
            prior_pos = np.array([self.width * 0.85, self.height * 0.5], dtype=np.float32)
            self.pf.initialize(prior_pos, np.array([0.0, 0.0], dtype=np.float32), self.np_rng)

        observations = self._get_observations()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, info

    def _generate_causal_switch_map(self):
        w, h = self.width, self.height
        self.static_obstacles = []

        # Boundaries
        self.static_obstacles.append(Obstacle(0, 0, w, 10))
        self.static_obstacles.append(Obstacle(0, h - 10, w, 10))
        self.static_obstacles.append(Obstacle(0, 0, 10, h))
        self.static_obstacles.append(Obstacle(w - 10, 0, 10, h))

        # Barrier Wall at x = 580 with randomized gate position
        wall_x = 580.0
        gate_h = 100.0
        gate_y = float(self.rng.randint(int(h * 0.35), int(h * 0.55)))

        if gate_y > 15:
            self.static_obstacles.append(Obstacle(wall_x, 10, 15, gate_y - 10))
        bot_y = gate_y + gate_h
        if bot_y < h - 15:
            self.static_obstacles.append(Obstacle(wall_x, bot_y, 15, h - 10 - bot_y))

        # Dynamic Gate Obstacle
        self.gate_obstacle = Obstacle(wall_x, gate_y, 15, gate_h)

        # Place K distinct candidate switches in the left wing
        # Stratified vertically to ensure clear spatial separation
        zone_h = (h - 100) / self.num_switches
        for k in range(self.num_switches):
            y_min = int(50 + k * zone_h)
            y_max = int(50 + (k + 1) * zone_h - 20)
            sw_x = float(self.rng.randint(60, 110))
            sw_y = float(self.rng.randint(y_min, y_max))
            self.switch_positions[k] = np.array([sw_x, sw_y], dtype=np.float32)

        # 3-4 Clutter Blocks in Central Hallway
        num_clutter = self.rng.randint(3, 5)
        for _ in range(num_clutter):
            cw = float(self.rng.randint(30, 60))
            ch = float(self.rng.randint(30, 70))
            cx = float(self.rng.randint(170, int(wall_x - 100)))
            cy = float(self.rng.randint(40, int(h - 100)))
            self.static_obstacles.append(Obstacle(cx, cy, cw, ch))

        self.obstacles = list(self.static_obstacles) + [self.gate_obstacle]
        self._build_occupancy_map()

    def _spawn_agents(self):
        w, h = self.width, self.height
        # Hunters spawn in staging area (x: 220..380)
        for i in range(self.num_hunters):
            for _ in range(100):
                hx = float(self.rng.randint(220, 380))
                hy = float(self.rng.randint(120, h - 120))
                cand = np.array([hx, hy], dtype=np.float32)
                if not any(obs.collides_point(cand[0], cand[1], buffer=20.0) for obs in self.obstacles):
                    self.hunter_pos[i] = cand
                    break
            self.hunter_vel[i] = np.array([0.0, 0.0], dtype=np.float32)

        # Evader spawns behind barricade (x: 640..w-50)
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

        # 1. Apply Hunter Kinematics
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

        # 2. Update Dynamic Gate Causal Mechanism from Current Post-Movement Positions
        active_switch = None
        for k in range(self.num_switches):
            for i in range(self.num_hunters):
                dist = float(np.linalg.norm(self.hunter_pos[i] - self.switch_positions[k]))
                if dist < self.switch_radius:
                    active_switch = k
                    self.switches_tested.add(k)
                    if self.first_switch_step is None:
                        self.first_switch_step = self.current_step
                    break
            if active_switch is not None:
                break

        # Causal Transition: Gate opens IF AND ONLY IF the active switch is the true functional one
        if active_switch is not None and active_switch == self.true_switch_idx:
            self.gate_open = True
            self.true_switch_discovered = True
            if self.discovery_step is None:
                self.discovery_step = self.current_step
            self.obstacles = list(self.static_obstacles)
        else:
            self.gate_open = False
            self.obstacles = list(self.static_obstacles) + [self.gate_obstacle]

        self._build_occupancy_map()

        # Check Gate Breach
        for i in range(self.num_hunters):
            if self.hunter_pos[i][0] > 600.0 and self.gate_breach_step is None:
                self.gate_breach_step = self.current_step

        # 3. Update Evader Kinematics
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

        # 4. Sensing & Particle Filter
        self._update_sensing()

        # 5. Check Termination
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
        """
        Observable features:
        - base pursuit features
        - switch positions (normalized)
        - gate_open (1.0 or 0.0)
        CRITICAL: true_switch_idx is NOT exposed here.
        """
        base_obs = super()._get_observations()
        sw_flat = (self.switch_positions / np.array([self.width, self.height], dtype=np.float32)).flatten()
        gate_feat = np.array([1.0 if self.gate_open else 0.0], dtype=np.float32)
        sw_feat = np.concatenate([sw_flat, gate_feat])

        augmented_obs = {}
        for k, v in base_obs.items():
            augmented_obs[k] = np.concatenate([v, sw_feat])
        return augmented_obs

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info.update({
            "num_switches": self.num_switches,
            "switch_positions": np.copy(self.switch_positions),
            "switch_radius": self.switch_radius,
            "gate_open": self.gate_open,
            "gate_obstacle": self.gate_obstacle,
            "static_obstacles": list(self.static_obstacles),
            "true_switch_discovered": self.true_switch_discovered,
            "discovery_step": self.discovery_step,
            "switches_tested_count": len(self.switches_tested),
            "first_switch_step": self.first_switch_step,
            "gate_breach_step": self.gate_breach_step
        })
        return info


def make_causal_switch_env(num_switches: int = 3,
                           evader_policy: Optional[BaseEvader] = None,
                           render_mode: Optional[str] = None) -> CausalSwitchDoorEnv:
    cfg = ScenarioConfig(width=800, height=600, max_steps=1200, capture_radius=30.0)
    return CausalSwitchDoorEnv(num_switches=num_switches, config=cfg,
                               evader_policy=evader_policy, render_mode=render_mode)

