"""
Explicit Generative Transition Model G(s, a) for the Switch-Door Dec-POMDP.
Accurately models the environmental affordance transition:
P(gate_open | hunter on switch) = 1.0.
"""

import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from deep_pomcp_env.scenarios import Obstacle

class SwitchDoorGenerativeState:
    def __init__(self,
                 h_pos: List[np.ndarray],
                 h_vel: List[np.ndarray],
                 e_pos: np.ndarray,
                 e_vel: np.ndarray,
                 switch_pos: np.ndarray,
                 switch_radius: float,
                 static_obstacles: List[Obstacle],
                 gate_obstacle: Optional[Obstacle],
                 gate_open: bool,
                 width: int = 800,
                 height: int = 600):
        self.num_hunters = len(h_pos)
        self.h_pos = [np.copy(p) for p in h_pos]
        self.h_vel = [np.copy(v) for v in h_vel]
        self.e_pos = np.copy(e_pos)
        self.e_vel = np.copy(e_vel)
        self.switch_pos = np.copy(switch_pos)
        self.switch_radius = float(switch_radius)
        self.static_obstacles = static_obstacles
        self.gate_obstacle = gate_obstacle
        self.gate_open = bool(gate_open)
        self.width = width
        self.height = height

        self.friction = 0.95
        self.max_hunter_spd = 5.0
        self.max_evader_spd = 7.0
        self.capture_radius = 30.0

    def copy(self) -> 'SwitchDoorGenerativeState':
        return SwitchDoorGenerativeState(
            self.h_pos, self.h_vel, self.e_pos, self.e_vel,
            self.switch_pos, self.switch_radius,
            self.static_obstacles, self.gate_obstacle,
            self.gate_open, self.width, self.height
        )

    def get_active_obstacles(self) -> List[Obstacle]:
        if self.gate_open or self.gate_obstacle is None:
            return self.static_obstacles
        return list(self.static_obstacles) + [self.gate_obstacle]

    def _sim_evader_flee(self) -> np.ndarray:
        best_score = -1e9
        best_dir = np.array([1.0, 0.0], dtype=np.float32)
        n_rays = 12
        obs = self.get_active_obstacles()

        for k in range(n_rays):
            ang = (2.0 * math.pi * k) / n_rays
            d = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
            tp = self.e_pos + d * 40.0

            if not (20 < tp[0] < self.width - 20 and 20 < tp[1] < self.height - 20):
                continue
            if any(o.collides_point(tp[0], tp[1], buffer=10.0) for o in obs):
                continue

            min_h_dist = min(np.linalg.norm(tp - self.h_pos[i]) for i in range(self.num_hunters))
            score = min_h_dist
            if score > best_score:
                best_score = score
                best_dir = d

        return best_dir * 0.40

    def step_primitive(self, actions: List[np.ndarray]) -> Tuple[float, bool]:
        wall_hits = False
        active_obs = self.get_active_obstacles()

        for i in range(self.num_hunters):
            self.h_vel[i] += actions[i]
            self.h_vel[i] *= self.friction
            spd = np.linalg.norm(self.h_vel[i])
            if spd > self.max_hunter_spd:
                self.h_vel[i] = (self.h_vel[i] / spd) * self.max_hunter_spd

            np_cand = self.h_pos[i] + self.h_vel[i]
            hit = False
            if not (15 < np_cand[0] < self.width - 15 and 15 < np_cand[1] < self.height - 15):
                hit = True
            elif any(o.collides_point(np_cand[0], np_cand[1], buffer=10.0) for o in active_obs):
                hit = True

            if hit:
                wall_hits = True
                self.h_vel[i] *= -0.5
            else:
                self.h_pos[i] = np_cand

        min_sw_dist = min(np.linalg.norm(self.h_pos[i] - self.switch_pos) for i in range(self.num_hunters))
        self.gate_open = (min_sw_dist < self.switch_radius)
        active_obs = self.get_active_obstacles()

        e_flee = self._sim_evader_flee()
        self.e_vel = self.e_vel * self.friction + e_flee
        e_spd = np.linalg.norm(self.e_vel)
        if e_spd > self.max_evader_spd:
            self.e_vel = (self.e_vel / e_spd) * self.max_evader_spd

        e_cand = self.e_pos + self.e_vel
        e_hit = False
        if not (15 < e_cand[0] < self.width - 15 and 15 < e_cand[1] < self.height - 15):
            e_hit = True
        elif any(o.collides_point(e_cand[0], e_cand[1], buffer=10.0) for o in active_obs):
            e_hit = True

        if not e_hit:
            self.e_pos = e_cand
        else:
            self.e_vel *= -0.5

        dists_to_evader = [np.linalg.norm(self.h_pos[i] - self.e_pos) for i in range(self.num_hunters)]
        min_dist = min(dists_to_evader)
        if min_dist < self.capture_radius:
            return 500.0, True

        reward = -(min_dist / 100.0) - 0.2
        if wall_hits:
            reward -= 5.0

        return reward, False


class SwitchDoorPOMDPModel:
    def __init__(self,
                 static_obstacles: List[Obstacle],
                 gate_obstacle: Optional[Obstacle],
                 switch_pos: np.ndarray,
                 switch_radius: float = 35.0,
                 width: int = 800,
                 height: int = 600):
        self.static_obstacles = static_obstacles
        self.gate_obstacle = gate_obstacle
        self.switch_pos = np.copy(switch_pos)
        self.switch_radius = float(switch_radius)
        self.width = width
        self.height = height

    def create_initial_state(self,
                             h_pos: List[np.ndarray],
                             h_vel: List[np.ndarray],
                             sampled_e_pos: np.ndarray,
                             sampled_e_vel: np.ndarray,
                             gate_open: bool = False) -> SwitchDoorGenerativeState:
        return SwitchDoorGenerativeState(
            h_pos=h_pos,
            h_vel=h_vel,
            e_pos=sampled_e_pos,
            e_vel=sampled_e_vel,
            switch_pos=self.switch_pos,
            switch_radius=self.switch_radius,
            static_obstacles=self.static_obstacles,
            gate_obstacle=self.gate_obstacle,
            gate_open=gate_open,
            width=self.width,
            height=self.height
        )
