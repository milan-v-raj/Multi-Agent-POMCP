import pygame
import math
import numpy as np
import random
import pomcp  # Ensure your pomcp.py is the latest version we discussed

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # Intruder
BLUE = (50, 50, 200)    # Hunter
GRAY = (100, 100, 100)  # Walls
YELLOW = (255, 255, 0)  # Vision Ray

# Physics Constants
MAX_SPEED = 5
ACCELERATION = 0.2
FRICTION = 0.95  # Air resistance

import heapq

# --- CLASS: PATHFINDER (THE PILOT) ---
class Pathfinder:
    def __init__(self, walls, grid_size=20): # Grid size 20 for precision
        self.walls = walls
        self.grid_size = grid_size
        self.cols = WIDTH // grid_size
        self.rows = HEIGHT // grid_size

    def get_grid_pos(self, pos):
        return (int(pos[0] // self.grid_size), int(pos[1] // self.grid_size))

    def get_world_pos(self, grid_pos):
        return np.array([grid_pos[0] * self.grid_size + self.grid_size/2, 
                         grid_pos[1] * self.grid_size + self.grid_size/2])

    def is_walkable(self, grid_pos):
        # 1. Check Map Boundaries
        c, r = grid_pos
        if c < 0 or c >= self.cols or r < 0 or r >= self.rows:
            return False
            
        # 2. Check Walls
        cell_rect = pygame.Rect(c * self.grid_size, r * self.grid_size, 
                                self.grid_size, self.grid_size)
        
        # Check collision (inflated slightly for safety)
        for obs in self.walls:
            if cell_rect.colliderect(obs.rect.inflate(5, 5)):
                return False
        return True

    def get_nearest_walkable(self, pos):
        """
        Finds the closest valid grid cell to the given position.
        Used to fix MCTS targets that end up inside walls.
        """
        start_node = self.get_grid_pos(pos)
        
        if self.is_walkable(start_node):
            return pos
            
        # BFS Search for nearest walkable cell
        queue = [start_node]
        visited = set()
        visited.add(start_node)
        
        # Search radius limit
        while queue and len(visited) < 200:
            current = queue.pop(0)
            
            if self.is_walkable(current):
                return self.get_world_pos(current)
            
            # Check neighbors
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in visited and 0 <= neighbor[0] < self.cols and 0 <= neighbor[1] < self.rows:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return pos # Fallback

    def find_path(self, start_pos, end_pos):
        start_node = self.get_grid_pos(start_pos)
        end_node = self.get_grid_pos(end_pos)
        
        # A* Algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: abs(end_node[0]-start_node[0]) + abs(end_node[1]-start_node[1])}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end_node:
                return self.reconstruct_path(came_from, current)
            
            # Neighbors: Up, Down, Left, Right
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_walkable(neighbor):
                    continue
                    
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + abs(end_node[0]-neighbor[0]) + abs(end_node[1]-neighbor[1])
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
                    
        return []

    def reconstruct_path(self, came_from, current):
        total_path = [self.get_world_pos(current)]
        while current in came_from:
            current = came_from[current]
            total_path.append(self.get_world_pos(current))
        return total_path[::-1]


# --- CLASS: DRONE ---
class Drone(pygame.sprite.Sprite):
    def __init__(self, x, y, color, speed_limit=5.0):
        super().__init__()
        self.image = pygame.Surface((20, 20))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(x, y))
        self.max_speed = speed_limit
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])

    def apply_force(self, force):
        self.acc += force

    def update(self):
        self.vel += self.acc
        self.vel *= FRICTION
        
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        
        self.pos += self.vel
        self.rect.center = self.pos
        self.acc = np.array([0.0, 0.0])

        # Boundary checks
        if self.pos[0] < 0: self.pos[0] = 0; self.vel[0] *= -0.5
        if self.pos[0] > WIDTH: self.pos[0] = WIDTH; self.vel[0] *= -0.5
        if self.pos[1] < 0: self.pos[1] = 0; self.vel[1] *= -0.5
        if self.pos[1] > HEIGHT: self.pos[1] = HEIGHT; self.vel[1] *= -0.5

    def check_collision(self, obstacles):
        for obs in obstacles:
            if self.rect.colliderect(obs.rect):
                clip = self.rect.clip(obs.rect)
                overlap_x = clip.width
                overlap_y = clip.height
                
                if overlap_x < overlap_y:
                    self.vel[0] *= -1
                    if self.rect.centerx < obs.rect.centerx: self.pos[0] -= (overlap_x + 2) 
                    else: self.pos[0] += (overlap_x + 2)
                else:
                    self.vel[1] *= -1
                    if self.rect.centery < obs.rect.centery: self.pos[1] -= (overlap_y + 2)
                    else: self.pos[1] += (overlap_y + 2)
                
                self.rect.center = self.pos


# --- CLASS: PARTICLE FILTER ---
class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, vx, vy):
        super().__init__()
        self.image = pygame.Surface((4, 4))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([float(vx), float(vy)])

    def update(self):
        self.pos += self.vel
        self.pos += np.random.normal(0, 1.5, 2)
        self.rect.center = self.pos
        if self.pos[0] < 0 or self.pos[0] > WIDTH or self.pos[1] < 0 or self.pos[1] > HEIGHT:
            self.kill()

class ParticleFilter:
    def __init__(self, num_particles=200):
        self.num = num_particles
        self.particles = pygame.sprite.Group()
        self.is_initialized = False
        self.frame_timer = 0

    def predict(self):
        for p in self.particles:
            p.update()

    def update(self, observation, intruder_vel):
        # 1. THROTTLE: Keep the 10-frame delay (It's good)
        self.frame_timer += 1
        if self.frame_timer % 10 != 0:
            return 

        if observation is not None:
            self.is_initialized = True
            obs_x, obs_y = observation
            vel_x, vel_y = intruder_vel 
            
            # --- OPTIMIZATION: RECYCLE PARTICLES ---
            # Get list of currently alive particles
            existing_particles = self.particles.sprites()
            
            # If we don't have enough, create some new ones
            while len(existing_particles) < 100:
                new_p = Particle(obs_x, obs_y, vel_x, vel_y)
                self.particles.add(new_p)
                existing_particles.append(new_p)
            
            # If we have too many, kill the extras
            while len(existing_particles) > 100:
                p = existing_particles.pop()
                p.kill()

            # RE-USE EXISTING PARTICLES (Teleport them)
            for p in existing_particles:
                # Reset position to observation + noise
                p.pos[0] = obs_x + np.random.normal(0, 5)
                p.pos[1] = obs_y + np.random.normal(0, 5)
                # Reset velocity
                p.vel[0] = vel_x
                p.vel[1] = vel_y
                p.rect.center = p.pos
                
        elif self.is_initialized:
            # TARGET LOST: Do nothing, let them drift naturally
            pass


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.image.fill(GRAY)
        self.rect = self.image.get_rect(topleft=(x, y))

def check_line_of_sight(hunter, intruder, obstacles):
    line_start = hunter.rect.center
    line_end = intruder.rect.center
    for obs in obstacles:
        if obs.rect.clipline(line_start, line_end):
            return False, line_start, line_end
    return True, line_start, line_end


# --- MAIN GAME LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("S6 Project: Drone Interceptor Arena")
    clock = pygame.time.Clock()

    # Create Sprites
    hunter = Drone(100, 300, BLUE)
    intruder = Drone(200, 550, RED)
    
    # Create Walls (Urban Canyon)
    walls = pygame.sprite.Group()
    walls.add(Obstacle(300, 100, 50, 400)) # Central skyscraper
    walls.add(Obstacle(500, 0, 50, 250))   # Top block
    walls.add(Obstacle(500, 350, 50, 250))  

    # Initialize Systems
    pf = ParticleFilter(num_particles=200)
    
    # Initialize A* Pilot (High Resolution Grid)
    pathfinder = Pathfinder(walls, grid_size=20) 
    current_path = [] 
    path_timer = 0
    
    # Variables
    frame_count = 0
    mcts_interval = 10 
    MAX_BLIND_TIME = 10000 
    last_seen_time = pygame.time.get_ticks() 
    current_time = 0
    smooth_target = None
    running = True
    while running:
        screen.fill(BLACK)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- STEP 1: PHYSICS ---
        hunter.update()
        hunter.check_collision(walls)
        intruder.update()
        
# --- STEP 2: RED DRONE AI (THE SURVIVOR) ---
        # We run Red MCTS every frame because it needs to react fast.
        # Since we lowered SIMs to 50, it should be fine.
        
        # 1. Run Red MCTS
        # Note: Red sees Blue perfectly (it's hard to hide a drone)
        red_force = pomcp.run_red_mcts(intruder, hunter, walls)
        
        # 2. Safety Override (A* Lite)
        # Even MCTS makes mistakes. If MCTS says "Fly Right" but there is a wall,
        # we override it with a simple bounce.
        if not pomcp.is_safe_move(intruder.pos, intruder.vel, red_force, walls):
             # Simple Evasion: Try random safe directions
             red_force = np.array([0.0, 0.0]) # Brake
             
        # 3. Apply Force
        intruder.apply_force(red_force)

        # --- STEP 3: SENSORS & BELIEF ---
        can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
        
        observation = None
        if can_see:
            observation = intruder.pos
            last_seen_time = current_time
        
        time_since_seen = current_time - last_seen_time

        pf.predict()
        pf.update(observation, intruder.vel)

        # --- STEP 4: HUNTER AI (THE SPOTTER & PILOT) ---
        frame_count += 1
        
        # A. THE SPOTTER (MCTS Strategy Layer)
        if frame_count % mcts_interval == 0:
            if len(pf.particles) > 0:
                # 1. Ask MCTS for best intercept vector
                strategic_force = pomcp.run_mcts(hunter, pf.particles, []) 
                
                # 2. Raw Target calculation
                raw_target = hunter.pos + (strategic_force * 150)
                
                # --- SMOOTHING FIX ---
                if smooth_target is None:
                    smooth_target = raw_target
                else:
                    # Blend: 80% Old Position + 20% New Position
                    # This creates a "Heavy" target that resists jitter
                    smooth_target = (smooth_target * 0.8) + (raw_target * 0.2)

                # 3. Snap SMOOTH target to Reality
                final_target = pathfinder.get_nearest_walkable(smooth_target)
                
                # 4. Update A* Path (Only if target moved significantly)
                # Check distance against the smoothed target to avoid micro-updates
                current_grid_target = pathfinder.get_world_pos(pathfinder.get_grid_pos(hunter.pos))
                
                if not current_path or np.linalg.norm(final_target - current_grid_target) > 40:
                     new_path = pathfinder.find_path(hunter.pos, final_target)
                     if new_path: current_path = new_path

        # B. THE PILOT (A* Control Layer)
        steering = np.array([0.0, 0.0])
        
        if current_path:
            target_node = current_path[0]
            if np.linalg.norm(hunter.pos - target_node) < 30:
                current_path.pop(0)
                if current_path: target_node = current_path[0]
            
            desired = target_node - hunter.pos
            if np.linalg.norm(desired) > 0:
                desired_vel = (desired / np.linalg.norm(desired)) * 4.0
                steering = desired_vel - hunter.vel

        # C. SAFETY REFLEX (Wall Repulsion)
        for obs in walls:
            if obs.rect.collidepoint(hunter.pos[0] + hunter.vel[0]*10, hunter.pos[1] + hunter.vel[1]*10):
                steering += np.array([-hunter.vel[1], hunter.vel[0]]) * 3.0
        
        if np.linalg.norm(steering) > 0.5:
             steering = (steering / np.linalg.norm(steering)) * 0.5
        
        hunter.apply_force(steering)


        # --- STEP 5: DRAWING ---
        walls.draw(screen)
        
        # Draw A* Path (Cyan)
        if len(current_path) > 1:
            pygame.draw.lines(screen, (0, 255, 255), False, current_path, 2)

        # Draw MCTS Strategic Intent (Red)
        if len(pf.particles) > 0 and 'final_target' in locals():
            pygame.draw.line(screen, (255, 0, 0), hunter.pos, final_target, 1)
            pygame.draw.circle(screen, (255, 0, 0), (int(final_target[0]), int(final_target[1])), 4)

        # UI Text
        font = pygame.font.SysFont(None, 36)
        mode_text = "TRACKING" if len(pf.particles) > 0 else "PATROL"
        color = (0, 255, 0) if mode_text == "TRACKING" else (255, 0, 0)
        img = font.render(f"Mode: {mode_text} (Blind: {time_since_seen/1000:.1f}s)", True, color)
        screen.blit(img, (10, 10))
        
        pf.particles.draw(screen)
        screen.blit(hunter.image, hunter.rect)
        
        if can_see:
            pygame.draw.line(screen, YELLOW, hunter.rect.center, intruder.rect.center, 2)
            screen.blit(intruder.image, intruder.rect)
        else:
            pygame.draw.line(screen, (50, 50, 50), hunter.rect.center, intruder.rect.center, 1)
            ghost_image = intruder.image.copy()
            ghost_image.set_alpha(50)
            screen.blit(ghost_image, intruder.rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()