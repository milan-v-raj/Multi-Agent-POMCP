import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import pygame

@dataclass
class Obstacle:
    """Rigid rectangular obstacle in the workspace."""
    x: float
    y: float
    width: float
    height: float

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))

    def collides_point(self, px: float, py: float, buffer: float = 0.0) -> bool:
        return (self.x - buffer <= px <= self.x + self.width + buffer and
                self.y - buffer <= py <= self.y + self.height + buffer)

    def collides_circle(self, cx: float, cy: float, radius: float) -> bool:
        closest_x = max(self.x, min(cx, self.x + self.width))
        closest_y = max(self.y, min(cy, self.y + self.height))
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) < (radius * radius)

    def collides_line(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        r = self.rect
        clipped = r.clipline(x1, y1, x2, y2)
        return clipped != ()

    def distance_to_point(self, px: float, py: float) -> float:
        dx = max(self.x - px, 0, px - (self.x + self.width))
        dy = max(self.y - py, 0, py - (self.y + self.height))
        return math.hypot(dx, dy)


@dataclass
class ScenarioConfig:
    width: int = 800
    height: int = 600
    density_preset: str = "dense_maze"  # 'open' (0%), 'moderate' (15%), 'dense_maze' (30%), 'extreme' (45%)
    num_hunters: int = 2
    evader_speed_mult: float = 1.0  # 1.0 = speed 7.0 px/frame
    max_steps: int = 1500
    capture_radius: float = 30.0
    d_safe: float = 40.0
    seed: Optional[int] = None
    custom_obstacles: Optional[List[Obstacle]] = None


class ScenarioGenerator:
    """Generates standard and procedural obstacle layouts and valid spawns."""

    @staticmethod
    def get_preset_obstacles(preset: str, width: int = 800, height: int = 600) -> List[Obstacle]:
        obstacles = []
        # Outer boundary walls
        wall_thick = 10
        obstacles.append(Obstacle(0, 0, width, wall_thick))
        obstacles.append(Obstacle(0, height - wall_thick, width, wall_thick))
        obstacles.append(Obstacle(0, 0, wall_thick, height))
        obstacles.append(Obstacle(width - wall_thick, 0, wall_thick, height))

        if preset == "open" or preset == "0%":
            return obstacles

        elif preset == "moderate" or preset == "15%":
            # Classic 3-pillar layout
            obstacles.append(Obstacle(300, 100, 50, 400))
            obstacles.append(Obstacle(500, 0, 50, 250))
            obstacles.append(Obstacle(500, 350, 50, 250))
            return obstacles

        elif preset == "dense_maze" or preset == "30%":
            # Symmetric Occlusion Maze
            obstacles.append(Obstacle(150, 150, 50, 300))  # Left vertical
            obstacles.append(Obstacle(200, 275, 100, 50))  # Left horizontal cross
            obstacles.append(Obstacle(400, 0, 50, 200))    # Center top pillar
            obstacles.append(Obstacle(400, 400, 50, 200))  # Center bottom pillar
            obstacles.append(Obstacle(600, 250, 50, 200))  # Right vertical
            obstacles.append(Obstacle(500, 100, 150, 50))  # Right top horizontal
            obstacles.append(Obstacle(400, 275, 50, 50))   # Central split pillar
            return obstacles

        elif preset == "extreme" or preset == "45%":
            obstacles.append(Obstacle(150, 100, 50, 400))
            obstacles.append(Obstacle(200, 250, 100, 50))
            obstacles.append(Obstacle(350, 0, 50, 220))
            obstacles.append(Obstacle(350, 380, 50, 220))
            obstacles.append(Obstacle(450, 180, 50, 240))
            obstacles.append(Obstacle(550, 0, 50, 200))
            obstacles.append(Obstacle(550, 350, 50, 250))
            obstacles.append(Obstacle(650, 200, 100, 50))
            obstacles.append(Obstacle(250, 450, 150, 50))
            obstacles.append(Obstacle(400, 280, 50, 50))
            return obstacles

        else:
            return obstacles

    @staticmethod
    def is_valid_position(pos: np.ndarray, obstacles: List[Obstacle], radius: float = 20.0,
                           width: int = 800, height: int = 600, margin: float = 30.0) -> bool:
        x, y = pos[0], pos[1]
        if x < margin or x > width - margin or y < margin or y > height - margin:
            return False
        for obs in obstacles:
            if obs.collides_circle(x, y, radius):
                return False
        return True

    @classmethod
    def generate(cls, config: ScenarioConfig, rng: random.Random) -> Tuple[List[Obstacle], List[np.ndarray], np.ndarray]:
        """Generates obstacles and valid non-overlapping initial spawns."""
        if config.custom_obstacles is not None:
            obstacles = config.custom_obstacles
        else:
            obstacles = cls.get_preset_obstacles(config.density_preset, config.width, config.height)

        # Generate Hunters Spawns (on left half of the map)
        hunter_spawns = []
        for i in range(config.num_hunters):
            spawn = None
            for _ in range(500):
                y_band = config.height / (config.num_hunters + 1)
                base_y = (i + 1) * y_band
                test_pos = np.array([
                    rng.uniform(40, config.width * 0.35),
                    rng.uniform(max(30, base_y - 80), min(config.height - 30, base_y + 80))
                ])
                if cls.is_valid_position(test_pos, obstacles, radius=25.0, width=config.width, height=config.height):
                    if all(np.linalg.norm(test_pos - h_pos) > 40.0 for h_pos in hunter_spawns):
                        spawn = test_pos
                        break
            if spawn is None:
                spawn = np.array([80.0, 150.0 + i * 150.0])
            hunter_spawns.append(spawn)

        # Generate Evader Spawn (on right half of the map, far from hunters)
        evader_spawn = None
        for _ in range(500):
            test_pos = np.array([
                rng.uniform(config.width * 0.65, config.width - 50),
                rng.uniform(50, config.height - 50)
            ])
            if cls.is_valid_position(test_pos, obstacles, radius=25.0, width=config.width, height=config.height):
                if all(np.linalg.norm(test_pos - h_pos) > 250.0 for h_pos in hunter_spawns):
                    evader_spawn = test_pos
                    break

        if evader_spawn is None:
            evader_spawn = np.array([config.width - 100.0, config.height / 2.0])

        return obstacles, hunter_spawns, evader_spawn

