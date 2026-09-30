import heapq
import math
from typing import List, Tuple, Optional
import numpy as np
import pygame
from .scenarios import Obstacle

class Pathfinder:
    """A* Grid Pathfinder for navigating around obstacles."""

    def __init__(self, obstacles: List[Obstacle], width: int = 800, height: int = 600, grid_size: int = 20):
        self.obstacles = obstacles
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cols = width // grid_size
        self.rows = height // grid_size

    def update_obstacles(self, obstacles: List[Obstacle]):
        self.obstacles = obstacles

    def get_grid_pos(self, pos: np.ndarray) -> Tuple[int, int]:
        c = int(np.clip(pos[0] // self.grid_size, 0, self.cols - 1))
        r = int(np.clip(pos[1] // self.grid_size, 0, self.rows - 1))
        return (c, r)

    def get_world_pos(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        return np.array([
            grid_pos[0] * self.grid_size + self.grid_size / 2.0,
            grid_pos[1] * self.grid_size + self.grid_size / 2.0
        ], dtype=np.float32)

    def is_walkable(self, grid_pos: Tuple[int, int], buffer: float = 6.0) -> bool:
        c, r = grid_pos
        if c < 0 or c >= self.cols or r < 0 or r >= self.rows:
            return False
        cell_rect = pygame.Rect(
            c * self.grid_size - int(buffer),
            r * self.grid_size - int(buffer),
            self.grid_size + int(buffer * 2),
            self.grid_size + int(buffer * 2)
        )
        for obs in self.obstacles:
            if cell_rect.colliderect(obs.rect):
                return False
        return True

    def get_nearest_walkable(self, pos: np.ndarray) -> np.ndarray:
        start_node = self.get_grid_pos(pos)
        if self.is_walkable(start_node):
            return pos

        queue = [start_node]
        visited = {start_node}

        while queue and len(visited) < 300:
            current = queue.pop(0)
            if self.is_walkable(current):
                return self.get_world_pos(current)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in visited and 0 <= neighbor[0] < self.cols and 0 <= neighbor[1] < self.rows:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return pos

    def find_path(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        start_node = self.get_grid_pos(self.get_nearest_walkable(start_pos))
        end_node = self.get_grid_pos(self.get_nearest_walkable(end_pos))

        if start_node == end_node:
            return [end_pos]

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: abs(end_node[0] - start_node[0]) + abs(end_node[1] - start_node[1])}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end_node:
                return self._reconstruct_path(came_from, current)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_walkable(neighbor):
                    continue

                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + abs(end_node[0] - neighbor[0]) + abs(end_node[1] - neighbor[1])
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

        return []

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[np.ndarray]:
        total_path = [self.get_world_pos(current)]
        while current in came_from:
            current = came_from[current]
            total_path.append(self.get_world_pos(current))
        return total_path[::-1]

