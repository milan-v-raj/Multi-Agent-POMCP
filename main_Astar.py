import pygame
import math
import numpy as np
import random
import pomcp
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
FRICTION = 0.95  # Air resistance (1.0 = no friction, 0.9 = thick mud)
import heapq

class Pathfinder:
    def __init__(self, walls, grid_size=40):
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
        # Create a rectangle for the grid cell
        cell_rect = pygame.Rect(c * self.grid_size, r * self.grid_size, 
                                self.grid_size, self.grid_size)
        
        # Check collision with any wall (inflated slightly for safety buffer)
        for obs in self.walls:
            if cell_rect.colliderect(obs.rect.inflate(10, 10)):
                return False
        return True

    def find_path(self, start_pos, end_pos):
        start_node = self.get_grid_pos(start_pos)
        end_node = self.get_grid_pos(end_pos)
        
        # A* Algorithm Standard Implementation
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
                    
        return [] # Path not found

    def reconstruct_path(self, came_from, current):
        total_path = [self.get_world_pos(current)]
        while current in came_from:
            current = came_from[current]
            total_path.append(self.get_world_pos(current))
        return total_path[::-1] # Return reversed path (Start -> End)
class Drone(pygame.sprite.Sprite):
    def __init__(self, x, y, color,speed_limit=5.0):
        super().__init__()
        self.image = pygame.Surface((20, 20))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(x, y))
        self.max_speed=speed_limit
        # Physics Vectors (Float for precision)
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])

    def apply_force(self, force):
        self.acc += force

    def update(self):
        # 1. Update Velocity
        self.vel += self.acc
        self.vel *= FRICTION  # Apply Drag
        
        # 2. Limit Speed
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed: # Use the instance variable, not global
            self.vel = (self.vel / speed) * self.max_speed
        
        # 3. Update Position
        self.pos += self.vel
        self.rect.center = self.pos
        
        # 4. Reset Acceleration for next frame
        self.acc = np.array([0.0, 0.0])

        # 5. Screen Boundary Collision
        if self.pos[0] < 0: self.pos[0] = 0; self.vel[0] *= -0.5
        if self.pos[0] > WIDTH: self.pos[0] = WIDTH; self.vel[0] *= -0.5
        if self.pos[1] < 0: self.pos[1] = 0; self.vel[1] *= -0.5
        if self.pos[1] > HEIGHT: self.pos[1] = HEIGHT; self.vel[1] *= -0.5
    def check_collision(self, obstacles):
        has_collided = False
        for obs in obstacles:
            if self.rect.colliderect(obs.rect):
                has_collided = True
                
                # Calculate how deep we are inside the wall
                clip = self.rect.clip(obs.rect)
                overlap_x = clip.width
                overlap_y = clip.height
                
                # Resolve collision on the axis with the SMALLEST overlap
                if overlap_x < overlap_y:
                    self.vel[0] *= -1 # Bounce
                    # Push out with a small buffer (+2) to prevent sticking
                    if self.rect.centerx < obs.rect.centerx:
                        self.pos[0] -= (overlap_x + 2) 
                    else:
                        self.pos[0] += (overlap_x + 2)
                else:
                    self.vel[1] *= -1 # Bounce
                    # Push out with a small buffer (+2)
                    if self.rect.centery < obs.rect.centery:
                        self.pos[1] -= (overlap_y + 2)
                    else:
                        self.pos[1] += (overlap_y + 2)
                
                self.rect.center = self.pos
        
        return has_collided

    def move_towards(self, target_pos):
        """
        Greedy Logic: Accelerate towards a specific point (x, y).
        """
        # Calculate vector from self to target
        desired = target_pos - self.pos
        dist = np.linalg.norm(desired)
        
        if dist > 0:
            # Normalize and scale to max acceleration
            desired = (desired / dist) * ACCELERATION
            self.apply_force(desired)

    def wander(self, hit_wall_signal=False, enemy_pos=None):
        # 1. EVASION: If enemy is close (Radius 250), RUN!
        if enemy_pos is not None:
            dist = np.linalg.norm(self.pos - enemy_pos)
            if dist < 250: # Increased fear radius so it reacts sooner
                flee_dir = self.pos - enemy_pos
                # Add slight noise so it doesn't run in a perfectly straight line (predictable)
                noise = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                flee_dir += noise
                
                if np.linalg.norm(flee_dir) > 0:
                    flee_dir = (flee_dir / np.linalg.norm(flee_dir)) * ACCELERATION
                    self.apply_force(flee_dir)
                return
        # 1. Initialize waypoint if needed
        if not hasattr(self, 'waypoint'):
            self.waypoint = np.array([random.uniform(50, WIDTH-50), 
                                      random.uniform(50, HEIGHT-50)])
        
        # 2. PATROL: If we haven't seen anyone, move randomly
        # (Existing wander code...)
        if hit_wall_signal or not hasattr(self, 'waypoint'):
            # Pick a point FAR AWAY to ensure it crosses the map
            self.waypoint = np.array([random.uniform(50, WIDTH-50), 
                                      random.uniform(50, HEIGHT-50)])

        # 3. Move towards the waypoint
        self.move_towards(self.waypoint)
        
        # 4. If close to waypoint, pick a new one
        if np.linalg.norm(self.pos - self.waypoint) < 50:
             self.waypoint = np.array([random.uniform(50, WIDTH-50), 
                                       random.uniform(50, HEIGHT-50)])
        
        # If close to waypoint, pick a new one
        if np.linalg.norm(self.pos - self.waypoint) < 50:
             self.waypoint = np.array([random.uniform(50, WIDTH-50), 
                                       random.uniform(50, HEIGHT-50)])
             # Simple check to ensure waypoint isn't inside a wall would go here    
# --- PHASE 3: BELIEF STATE (PARTICLE FILTER) ---

class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, vx, vy):
        super().__init__()
        self.image = pygame.Surface((4, 4))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([float(vx), float(vy)])

    def update(self, obstacles=None):
        # 1. Move
        self.pos += self.vel
        self.pos += np.random.normal(0, 1.5, 2) # Keep the noise for the cloud effect
        self.rect.center = self.pos
        
        # 2. Kill ONLY if out of screen (The "Infinity" limit)
        if self.pos[0] < 0 or self.pos[0] > WIDTH or self.pos[1] < 0 or self.pos[1] > HEIGHT:
            self.kill()
            return

        # REMOVED: Wall collision check. 
        # Particles now fly through walls like X-Rays.

class ParticleFilter:
    def __init__(self, num_particles=200):
        self.num = num_particles
        self.particles = pygame.sprite.Group()
        self.is_initialized = False

    def predict(self):
        # Move all particles forward in time
        for p in self.particles:
            p.update()

    def update(self, observation, intruder_vel):
        """
        observation: (x, y) if seen, else None
        intruder_vel: The velocity vector of the intruder (to help prediction)
        """
        if observation is not None:
            # --- CASE A: WE SEE THE TARGET ---
            # All particles collapse to the real target location
            self.particles.empty()
            self.is_initialized = True
            
            obs_x, obs_y = observation
            vel_x, vel_y = intruder_vel # We assume we can measure velocity with Radar
            
            for _ in range(self.num):
                # Create particles EXACTLY at target with tiny noise
                px = obs_x + np.random.normal(0, 5)
                py = obs_y + np.random.normal(0, 5)
                # They inherit the target's current speed
                self.particles.add(Particle(px, py, vel_x, vel_y))
                
        elif self.is_initialized:
            # --- CASE B: TARGET LOST ---
            # We don't delete particles! We let them drift.
            # (Prediction step handled in predict())
            pass

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.image.fill(GRAY)
        self.rect = self.image.get_rect(topleft=(x, y))

def check_line_of_sight(hunter, intruder, obstacles):
    """
    Ray-Casting: Checks if a line from Hunter to Intruder hits any wall.
    """
    line_start = hunter.rect.center
    line_end = intruder.rect.center
    
    # Check if the line intersects with any obstacle rect
    for obs in obstacles:
        if obs.rect.clipline(line_start, line_end):
            return False, line_start, line_end # Vision Blocked
            
    return True, line_start, line_end # Vision Clear
class RailroadPatrol:
    def __init__(self):
        # Strict path: Top-Left -> Top-Gap -> Top-Right -> Bottom-Right -> Bottom-Gap -> Bottom-Left
        self.waypoints = [
            np.array([100, 100]), np.array([400, 50]),  np.array([700, 100]),
            np.array([700, 500]), np.array([400, 550]), np.array([100, 500])
        ]
        self.current_idx = 0
        self.radius = 50
        self.stuck_timer = 0 # Counter to detect getting stuck

    def get_search_force(self, hunter_pos, hunter_vel, obstacles):
        target = self.waypoints[self.current_idx]
        desired = target - hunter_pos
        dist = np.linalg.norm(desired)

        # 1. CHECK ARRIVAL
        if dist < self.radius:
            self.current_idx = (self.current_idx + 1) % len(self.waypoints)
            self.stuck_timer = 0 # Reset timer
            
        # 2. STUCK BREAKER (The Fix)
        # If we spend > 300 frames (5 seconds) trying to reach one point, SKIP IT.
        self.stuck_timer += 1
        if self.stuck_timer > 300:
            self.current_idx = (self.current_idx + 1) % len(self.waypoints)
            self.stuck_timer = 0
            
        # 3. CALCULATE FORCE
        # Strong pull towards target
        steering = np.array([0.0, 0.0])
        if np.linalg.norm(desired) > 0:
            desired_vel = (desired / np.linalg.norm(desired)) * 4.0
            steering = desired_vel - hunter_vel
            
        # 4. WALL REJECTION (Strong Push)
        for obs in obstacles:
            # Vector from wall center to drone
            diff = hunter_pos - np.array(obs.rect.center)
            dist_wall = np.linalg.norm(diff)
            
            # If inside danger zone (100px), PUSH HARD
            if dist_wall < 100:
                push = (diff / np.linalg.norm(diff)) * 5.0 # Max force
                steering += push

        # Normalize
        if np.linalg.norm(steering) > 0.5:
            steering = (steering / np.linalg.norm(steering)) * 0.5
            
        return steering

    
# --- MAIN GAME LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("S6 Project: Drone Interceptor Arena")
    clock = pygame.time.Clock()

    # --- SPAWN CONFIGURATION ---
    # 1. Hunter (Blue)
    hunter = Drone(250, 400, BLUE, speed_limit=5)
    hunter.vel = np.array([0.0, -1.0]) 
    
    # 2. Intruder (Red)
    intruder = Drone(450, 200, RED, speed_limit=9)
    intruder.vel = np.array([-2.0, 1.0])
    
    # Create Walls
    walls = pygame.sprite.Group()
    walls.add(Obstacle(300, 100, 50, 400)) # Central skyscraper
    walls.add(Obstacle(500, 0, 50, 250))   # Top block
    walls.add(Obstacle(500, 350, 50, 250)) # Bottom block

    # Initialize Systems
    pf = ParticleFilter(num_particles=200)
    
    # Initialize A* Pathfinder
    pathfinder = Pathfinder(walls, grid_size=40)
    current_path = [] 
    path_timer = 0
    
    # Timer variables
    MAX_BLIND_TIME = 10000 
    last_seen_time = pygame.time.get_ticks() 
    current_time = 0

    running = True
    while running:
        screen.fill(BLACK)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- STEP 1: PHYSICS & COLLISIONS ---
        hunter.update()
        hunter.check_collision(walls)
        intruder.update()
        
        # --- STEP 2: CONTROLS (RED DRONE) ---
        keys = pygame.key.get_pressed()
        input_vec = np.array([0.0, 0.0])
        if keys[pygame.K_w]: input_vec[1] = -1
        if keys[pygame.K_s]: input_vec[1] = 1
        if keys[pygame.K_a]: input_vec[0] = -1
        if keys[pygame.K_d]: input_vec[0] = 1
        
        if np.linalg.norm(input_vec) > 0:
            input_vec = (input_vec / np.linalg.norm(input_vec)) * ACCELERATION
            intruder.apply_force(input_vec)

        # --- STEP 3: SENSORS & BELIEF ---
        can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
        
        observation = None
        if can_see:
            observation = intruder.pos
            last_seen_time = current_time
        
        # Calculate time blind
        time_since_seen = current_time - last_seen_time

        # Update Particles
        # Note: We removed wall collision from particles so they fly through walls (Infinity)
        for p in pf.particles:
            p.update() 
            
        pf.update(observation, intruder.vel)

        # --- STEP 4: HUNTER AI (A* LOGIC) ---
        
        # 1. DETERMINE GOAL
        goal = None
        
        if time_since_seen < MAX_BLIND_TIME and len(pf.particles) > 0:
            # MODE: TRACKING
            mean_x = np.mean([p.pos[0] for p in pf.particles])
            mean_y = np.mean([p.pos[1] for p in pf.particles])
            goal = np.array([mean_x, mean_y])
        else:
            # MODE: PATROL
            if not current_path:
                 goal = np.array([random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)])

        # 2. RE-CALCULATE PATH
        path_timer += 1
        if goal is not None and path_timer > 30:
            path_timer = 0
            new_path = pathfinder.find_path(hunter.pos, goal)
            if new_path:
                current_path = new_path

        # 3. FOLLOW PATH
        steering = np.array([0.0, 0.0])
        
        if current_path:
            target_node = current_path[0]
            if np.linalg.norm(hunter.pos - target_node) < 40:
                current_path.pop(0)
                if current_path:
                    target_node = current_path[0]

            desired = target_node - hunter.pos
            if np.linalg.norm(desired) > 0:
                desired_vel = (desired / np.linalg.norm(desired)) * 4.0
                steering = desired_vel - hunter.vel
                
        # 4. SAFETY BUFFER
        for obs in walls:
            if obs.rect.collidepoint(hunter.pos[0] + hunter.vel[0]*10, hunter.pos[1] + hunter.vel[1]*10):
                steering += np.array([-hunter.vel[1], hunter.vel[0]]) * 2.0

        if np.linalg.norm(steering) > 0.5:
             steering = (steering / np.linalg.norm(steering)) * 0.5
        
        hunter.apply_force(steering)


        # --- STEP 5: DRAWING ---
        walls.draw(screen)
        
        # Draw A* Path
        if len(current_path) > 1:
            pygame.draw.lines(screen, (0, 255, 255), False, current_path, 2)

        # Draw State Text
        font = pygame.font.SysFont(None, 36)
        if time_since_seen < MAX_BLIND_TIME:
            state_text = f"Mode: TRACKING (Blind: {time_since_seen/1000:.1f}s)"
            color = (0, 255, 0)
        else:
            state_text = "Mode: PATROL"
            color = (255, 0, 0)
        img = font.render(state_text, True, color)
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