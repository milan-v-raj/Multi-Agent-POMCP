import math
from typing import Dict, Any, List, Optional
import numpy as np
import pygame
from .base_policy import BasePursuerPolicy
from ..scenarios import Obstacle
from ..pathfinder import Pathfinder

class ReactiveAStarPolicy(BasePursuerPolicy):
    def __init__(self, name: str = 'Reactive A*'):
        super().__init__(name)
        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}
        self.action_vectors = [
            np.array([0.0, -0.5], dtype=np.float32),
            np.array([0.0, 0.5], dtype=np.float32),
            np.array([-0.5, 0.0], dtype=np.float32),
            np.array([0.5, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32)
        ]

    def reset(self):
        self.current_paths.clear()

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int, obstacles: List[Obstacle], width: int, height: int) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos = info['hunter_positions'][agent_id]
        h_vel = info['hunter_velocities'][agent_id]
        b_mean = info['belief_mean']
        step = info.get('current_step', 0)

        if agent_id not in self.current_paths:
            self.current_paths[agent_id] = []

        is_active = info.get('is_belief_active', False) and (b_mean is not None)

        if step % 15 == 0 or len(self.current_paths[agent_id]) == 0:
            if is_active:
                target_pos = b_mean
                if agent_id == 1:
                    target_pos = b_mean + np.array([0.0, 35.0 if b_mean[1] < height / 2 else -35.0])
                walkable_target = self.pathfinder.get_nearest_walkable(target_pos)
            else:
                # Target not yet discovered: Patrol searching respective map sectors
                if agent_id == 0:
                    patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60), np.random.uniform(60, height * 0.45)])
                else:
                    patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60), np.random.uniform(height * 0.55, height - 60)])
                walkable_target = self.pathfinder.get_nearest_walkable(patrol_pt)

            new_path = self.pathfinder.find_path(h_pos, walkable_target)
            if new_path:
                self.current_paths[agent_id] = new_path

        steer_force = np.array([0.0, 0.0], dtype=np.float32)
        if self.current_paths[agent_id]:
            if np.linalg.norm(h_pos - self.current_paths[agent_id][0]) < 25.0:
                self.current_paths[agent_id].pop(0)

            if self.current_paths[agent_id]:
                desired_dir = self.current_paths[agent_id][0] - h_pos
                norm_dir = np.linalg.norm(desired_dir)
                if norm_dir > 0:
                    desired_vel = (desired_dir / norm_dir) * 4.0
                    steer_force = desired_vel - h_vel

        for obs_item in obstacles:
            lookahead = h_pos + h_vel * 8.0
            if obs_item.collides_point(lookahead[0], lookahead[1], buffer=6.0):
                steer_force += np.array([-h_vel[1], h_vel[0]], dtype=np.float32) * 2.0

        for j in range(len(info['hunter_positions'])):
            if j != agent_id:
                other_pos = info['hunter_positions'][j]
                dist = np.linalg.norm(h_pos - other_pos)
                if 0 < dist < 35.0:
                    steer_force += ((h_pos - other_pos) / dist) * 1.5

        if np.linalg.norm(steer_force) > 0.01:
            return max(range(4), key=lambda a: np.dot(self.action_vectors[a], steer_force))
        return 4
