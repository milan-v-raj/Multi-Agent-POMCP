# environment.py
import pygame
import random
import numpy as np
import heapq
from config import *

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.rect = pygame.Rect(x, y, w, h)

class Arena:
    def __init__(self):
        self.walls = pygame.sprite.Group()
        self.cols = WIDTH // GRID_SIZE
        self.rows = HEIGHT // GRID_SIZE
        self.generate_random_maze()

    def generate_random_maze(self):
        self.walls.empty()
        
        # 1. Outer Boundaries
        self.walls.add(Obstacle(0, 0, WIDTH, 10))
        self.walls.add(Obstacle(0, HEIGHT-10, WIDTH, 10))
        self.walls.add(Obstacle(0, 0, 10, HEIGHT))
        self.walls.add(Obstacle(WIDTH-10, 0, 10, HEIGHT))

        # 2. Random Interior Blocks
        # We spawn 12-15 random blocks of varying sizes
        num_blocks = random.randint(12, 18)
        for _ in range(num_blocks):
            w = random.randint(2, 6) * GRID_SIZE
            h = random.randint(2, 6) * GRID_SIZE
            x = random.randint(2, self.cols - 8) * GRID_SIZE
            y = random.randint(2, self.rows - 8) * GRID_SIZE
            self.walls.add(Obstacle(x, y, w, h))

    def is_walkable(self, pos, padding=10):
        # Checks if a point is inside a wall (with a safety buffer)
        check_rect = pygame.Rect(pos[0] - padding, pos[1] - padding, padding*2, padding*2)
        for wall in self.walls:
            if check_rect.colliderect(wall.rect):
                return False
        return True

    def get_safe_spawn(self, quadrant="any"):
        # Guarantees agents don't spawn inside a wall
        while True:
            if quadrant == "bottom_left":
                x = random.randint(30, WIDTH//3)
                y = random.randint(HEIGHT*2//3, HEIGHT-30)
            elif quadrant == "top_right":
                x = random.randint(WIDTH*2//3, WIDTH-30)
                y = random.randint(30, HEIGHT//3)
            else:
                x = random.randint(30, WIDTH-30)
                y = random.randint(30, HEIGHT-30)
            
            if self.is_walkable((x, y), padding=15):
                return np.array([float(x), float(y)])

    def draw(self, screen):
        for wall in self.walls:
            pygame.draw.rect(screen, WALL_COLOR, wall.rect)
            pygame.draw.rect(screen, WALL_OUTLINE, wall.rect, 1)

class Pathfinder:
    def __init__(self, arena):
        self.arena = arena

    def get_grid_pos(self, pos):
        return (int(pos[0] // GRID_SIZE), int(pos[1] // GRID_SIZE))

    def get_world_pos(self, grid_pos):
        return np.array([grid_pos[0] * GRID_SIZE + GRID_SIZE/2, 
                         grid_pos[1] * GRID_SIZE + GRID_SIZE/2])

    # THE FIX: Find the nearest safe floor tile if the AI points into a wall!
    def get_nearest_walkable(self, pos):
        if self.arena.is_walkable(pos, padding=5): 
            return pos
            
        start_node = self.get_grid_pos(pos)
        queue = [start_node]
        visited = {start_node}
        
        while queue and len(visited) < 300:
            current = queue.pop(0)
            world_pos = self.get_world_pos(current)
            
            if self.arena.is_walkable(world_pos, padding=5):
                return world_pos
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in visited and 0 <= neighbor[0] < self.arena.cols and 0 <= neighbor[1] < self.arena.rows:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return pos

    def find_path(self, start_pos, end_pos):
        start_node = self.get_grid_pos(start_pos)
        end_node = self.get_grid_pos(end_pos)
        
        # Removed the strict "start_pos" check so they don't freeze if they graze a wall!
        if not self.arena.is_walkable(end_pos, padding=2):
            return []

        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end_node:
                path = []
                while current in came_from:
                    path.append(self.get_world_pos(current))
                    current = came_from[current]
                return path[::-1] 
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                world_pos = self.get_world_pos(neighbor)
                
                if not (0 <= world_pos[0] <= WIDTH and 0 <= world_pos[1] <= HEIGHT):
                    continue
                if not self.arena.is_walkable(world_pos, padding=5):
                    continue
                    
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + abs(end_node[0]-neighbor[0]) + abs(end_node[1]-neighbor[1])
                    heapq.heappush(open_set, (f_score, neighbor))
                    
        return []