import math
import random
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pygame
from .scenarios import Obstacle

class BaseEvader(ABC):
    @abstractmethod
    def get_action(self, evader_pos: np.ndarray, evader_vel: np.ndarray,
                   hunter_positions: List[np.ndarray], obstacles: List[Obstacle],
                   width: int, height: int) -> np.ndarray:
        """Returns a 2D acceleration force vector."""
        pass


class SmartRaycastEvader(BaseEvader):
    """
    8-Directional Raycasting Evader.
    Maximizes distance to the nearest active pursuer while avoiding obstacle collisions and dead ends.
    """
    def __init__(self, max_force: float = 0.2, ray_dist: float = 35.0, wall_penalty_cost: float = 1000.0):
        self.max_force = max_force
        self.ray_dist = ray_dist
        self.wall_penalty_cost = wall_penalty_cost

    def get_action(self, evader_pos: np.ndarray, evader_vel: np.ndarray,
                   hunter_positions: List[np.ndarray], obstacles: List[Obstacle],
                   width: int, height: int) -> np.ndarray:
        best_score = -float('inf')
        best_dir = np.array([0.0, 0.0])

        for angle_deg in range(0, 360, 45):
            rad = math.radians(angle_deg)
            dir_vec = np.array([math.cos(rad), math.sin(rad)])
            test_pos = evader_pos + (dir_vec * self.ray_dist)
            wall_penalty = 0.0

            # 1. Check workspace boundaries
            if not (20 < test_pos[0] < width - 20 and 20 < test_pos[1] < height - 20):
                wall_penalty = self.wall_penalty_cost
            else:
                # 2. Check obstacle proximity with safety buffer
                test_rect = pygame.Rect(test_pos[0] - 12, test_pos[1] - 12, 24, 24)
                for obs in obstacles:
                    if test_rect.colliderect(obs.rect.inflate(30, 30)):
                        wall_penalty = self.wall_penalty_cost
                        break

            # 3. Calculate distance to closest hunter
            if len(hunter_positions) > 0:
                nearest_hunter_dist = min(np.linalg.norm(test_pos - h_pos) for h_pos in hunter_positions)
            else:
                nearest_hunter_dist = 0.0

            # Score = Distance to hunters - penalty for walls
            score = nearest_hunter_dist - wall_penalty

            if score > best_score:
                best_score = score
                best_dir = dir_vec

        if best_score > -500.0 and np.linalg.norm(best_dir) > 0:
            return (best_dir / np.linalg.norm(best_dir)) * self.max_force

        # Fallback random exploration
        random_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        norm = np.linalg.norm(random_dir)
        return (random_dir / norm) * self.max_force if norm > 0 else np.array([0.0, 0.0])


class RandomEvader(BaseEvader):
    """Random walking evader with momentum smoothing."""
    def __init__(self, max_force: float = 0.2):
        self.max_force = max_force
        self.current_acc = np.array([0.0, 0.0])

    def get_action(self, evader_pos: np.ndarray, evader_vel: np.ndarray,
                   hunter_positions: List[np.ndarray], obstacles: List[Obstacle],
                   width: int, height: int) -> np.ndarray:
        perturbation = np.array([random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)])
        self.current_acc = (self.current_acc * 0.8) + perturbation
        norm = np.linalg.norm(self.current_acc)
        if norm > self.max_force:
            self.current_acc = (self.current_acc / norm) * self.max_force
        return self.current_acc


class KeyboardEvader(BaseEvader):
    """Interactive human player controlling evader via WASD keys."""
    def __init__(self, max_force: float = 0.2):
        self.max_force = max_force

    def get_action(self, evader_pos: np.ndarray, evader_vel: np.ndarray,
                   hunter_positions: List[np.ndarray], obstacles: List[Obstacle],
                   width: int, height: int) -> np.ndarray:
        keys = pygame.key.get_pressed()
        input_vec = np.array([0.0, 0.0])
        if keys[pygame.K_w] or keys[pygame.K_UP]: input_vec[1] = -1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: input_vec[1] = 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: input_vec[0] = -1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: input_vec[0] = 1.0

        norm = np.linalg.norm(input_vec)
        if norm > 0:
            return (input_vec / norm) * self.max_force
        return np.array([0.0, 0.0])

