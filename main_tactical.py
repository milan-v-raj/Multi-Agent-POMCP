import pygame
import math
import numpy as np
import random
import pomcp_tact

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
FRICTION = 0.95  

import heapq

# --- CLASS: PATHFINDER ---
class Pathfinder:
    def __init__(self, walls, grid_size=20): 
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
        c, r = grid_pos
        if c < 0 or c >= self.cols or r < 0 or r >= self.rows:
            return False
        cell_rect = pygame.Rect(c * self.grid_size, r * self.grid_size, 
                                self.grid_size, self.grid_size)
        for obs in self.walls:
            if cell_rect.colliderect(obs.rect.inflate(5, 5)):
                return False
        return True

    def get_nearest_walkable(self, pos):
        """Snap an invalid point (inside wall) to the nearest valid hallway."""
        start_node = self.get_grid_pos(pos)
        if self.is_walkable(start_node): return pos
            
        queue = [start_node]
        visited = set()
        visited.add(start_node)
        
        while queue and len(visited) < 200:
            current = queue.pop(0)
            if self.is_walkable(current):
                return self.get_world_pos(current)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in visited and 0 <= neighbor[0] < self.cols and 0 <= neighbor[1] < self.rows:
                    visited.add(neighbor)
                    queue.append(neighbor)      
        return pos 

    def find_path(self, start_pos, end_pos):
        start_node = self.get_grid_pos(start_pos)
        end_node = self.get_grid_pos(end_pos)
        
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: abs(end_node[0]-start_node[0]) + abs(end_node[1]-start_node[1])}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end_node:
                return self.reconstruct_path(came_from, current)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_walkable(neighbor): continue
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

        if self.pos[0] < 0: self.pos[0] = 0; self.vel[0] *= -0.5
        if self.pos[0] > WIDTH: self.pos[0] = WIDTH; self.vel[0] *= -0.5
        if self.pos[1] < 0: self.pos[1] = 0; self.vel[1] *= -0.5
        if self.pos[1] > HEIGHT: self.pos[1] = HEIGHT; self.vel[1] *= -0.5

    def check_collision(self, obstacles):
        for obs in obstacles:
            if self.rect.colliderect(obs.rect):
                clip = self.rect.clip(obs.rect)
                if clip.width < clip.height:
                    self.vel[0] *= -1
                    if self.rect.centerx < obs.rect.centerx: self.pos[0] -= (clip.width + 2) 
                    else: self.pos[0] += (clip.width + 2)
                else:
                    self.vel[1] *= -1
                    if self.rect.centery < obs.rect.centery: self.pos[1] -= (clip.height + 2)
                    else: self.pos[1] += (clip.height + 2)
                self.rect.center = self.pos


# --- CLASS: PARTICLE (UNBOUNDED) ---
class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, vx, vy):
        super().__init__()
        # Create a surface with per-pixel alpha (transparency)
        self.radius = 6
        self.image = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        
        # Draw faint green circle (Heatmap style)
        pygame.draw.circle(self.image, (0, 255, 0, 50), (self.radius, self.radius), self.radius)
        
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([float(vx), float(vy)])

    def update(self):
        # Continues moving in the last known direction
        self.pos += self.vel
        
        # High Spread Noise
        self.pos += np.random.normal(0, 1.5, 2) 
        
        self.rect.center = self.pos
        
        # --- THE FIX ---
        # REMOVED the "kill if off screen" check.
        # Particles now float into infinity, keeping the drone in TRACKING mode
        # until the 10-second timer in the main loop kills them.

class ParticleFilter:
    def __init__(self, num_particles=200):
        self.num = num_particles
        self.particles = pygame.sprite.Group()
        self.is_initialized = False
        self.frame_timer = 0

    def predict(self):
        # Moves particles every frame (Drifting logic)
        for p in self.particles:
            p.update()

    def update(self, observation, intruder_vel):
        # THROTTLE: Only resample every 5 frames
        self.frame_timer += 1
        
        if observation is not None:
            if self.frame_timer % 5 == 0:
                self.particles.empty()
                self.is_initialized = True
                obs_x, obs_y = observation
                vel_x, vel_y = intruder_vel 
                
                # Use 100 particles for performance
                for _ in range(100): 
                    # Initial Spread (5.0)
                    px = obs_x + np.random.normal(0, 5)
                    py = obs_y + np.random.normal(0, 5)
                    self.particles.add(Particle(px, py, vel_x, vel_y))
        else:
            # BLIND: Do nothing (Particles drift via predict)
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


# --- HELPER: DRAW GRID ---
def draw_grid(screen):
    screen.fill((10, 20, 40)) 
    GRID_SPACING = 50
    for x in range(0, WIDTH, GRID_SPACING):
        pygame.draw.line(screen, (30, 50, 80), (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, GRID_SPACING):
        pygame.draw.line(screen, (30, 50, 80), (0, y), (WIDTH, y), 1)

# --- HELPER: DRAW VECTOR BOID ---
def draw_boid(screen, drone, color):
    angle = math.atan2(-drone.vel[1], drone.vel[0]) 
    center = drone.pos
    size = 15
    points = [
        (center[0] + size * math.cos(angle), center[1] - size * math.sin(angle)),
        (center[0] + size/2 * math.cos(angle + 2.5), center[1] - size/2 * math.sin(angle + 2.5)),
        (center[0] + size/2 * math.cos(angle - 2.5), center[1] - size/2 * math.sin(angle - 2.5))
    ]
    pygame.draw.polygon(screen, color, points)
    
    if color == BLUE:
        arc_len = 100
        arc_points = [center]
        angle_spread = 0.5 
        arc_points.append((center[0] + arc_len * math.cos(angle - angle_spread), 
                           center[1] - arc_len * math.sin(angle - angle_spread)))
        arc_points.append((center[0] + arc_len * math.cos(angle + angle_spread), 
                           center[1] - arc_len * math.sin(angle + angle_spread)))
        cone_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(cone_surf, (50, 50, 200, 30), arc_points)
        screen.blit(cone_surf, (0,0))

# --- CLASS: TACTICAL GRAPH ---
class TacticalGraph:
    def __init__(self):
        # Define 6 Key Nodes (The "Skeleton" of your map)
        self.nodes = {
            0: np.array([100.0, 100.0]),  # Top Left
            1: np.array([400.0, 50.0]),   # Top Gap
            2: np.array([700.0, 100.0]),  # Top Right
            3: np.array([700.0, 500.0]),  # Bottom Right
            4: np.array([400.0, 550.0]),  # Bottom Gap
            5: np.array([100.0, 500.0])   # Bottom Left
        }
        
        # Define Connections (Adjacency List)
        # Who connects to whom? (0 connects to 1 and 5)
        self.edges = {
            0: [1, 5],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3, 5],
            5: [4, 0]
        }

    def get_closest_node(self, pos):
        """Finds the nearest node index to a given position."""
        best_dist = float('inf')
        best_idx = -1
        for idx, node_pos in self.nodes.items():
            dist = np.linalg.norm(pos - node_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

# --- MAIN GAME LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("S6 Project: Drone Interceptor Arena")
    clock = pygame.time.Clock()

    # --- SPAWN CONFIGURATION ---
    hunter = Drone(100, 300, BLUE, speed_limit=5)
    hunter.vel = np.array([1.0, 1.0]) 
    
    intruder = Drone(700, 300, RED, speed_limit=9)
    intruder.vel = np.array([-1.0, -1.0])
    
    walls = pygame.sprite.Group()
    walls.add(Obstacle(300, 100, 50, 400)) 
    walls.add(Obstacle(500, 0, 50, 250))   
    walls.add(Obstacle(500, 350, 50, 250))  

    # Initialize Systems
    pf = ParticleFilter(num_particles=200)
    pathfinder = Pathfinder(walls, grid_size=20) 
    current_path = [] 
    tactical_graph = TacticalGraph()
    path_timer = 0
    
    frame_count = 0
    mcts_interval = 10 
    MAX_BLIND_TIME = 10000 
    last_seen_time = pygame.time.get_ticks() 
    current_time = 0

    running = True
    while running:
        # 1. Background Grid (Radar Style)
        draw_grid(screen)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- STEP 1: PHYSICS ---
        hunter.update()
        hunter.check_collision(walls)
        intruder.update()
        
        # --- STEP 2: RED DRONE CONTROLS (WASD) ---
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
        
        time_since_seen = current_time - last_seen_time

        # THE 10-SECOND RULE:
        # Only kill particles if timer exceeds limit. 
        # Since we removed boundary checks in Particle class, they survive off-screen.
        if time_since_seen > MAX_BLIND_TIME:
            pf.particles.empty()

        pf.predict() 
        pf.update(observation, intruder.vel)

        # --- STEP 4: HUNTER AI (THE SPOTTER & PILOT) ---
        frame_count += 1
        
        # A. THE SPOTTER (Graph MCTS Strategy)
        if frame_count % mcts_interval == 0:
            if len(pf.particles) > 0:
                # 1. Ask MCTS: "Which Node is next?"
                # It returns an Integer (e.g., 2)
                target_node_id = pomcp_tact.run_mcts(hunter.pos, pf.particles)
                
                # 2. Get Coordinate of that Node
                # We access the hardcoded nodes from the imported module or class
                final_target = tactical_graph.nodes[target_node_id]
                
                # 3. Destination Locking Check (Prevent jitter)
                should_update = False
                if not current_path:
                    should_update = True
                else:
                    # If we picked a NEW node different from where we are going
                    current_goal_node_idx = tactical_graph.get_closest_node(current_path[-1])
                    if current_goal_node_idx != target_node_id:
                        should_update = True
                
                # 4. A* Pilot takes us to the Node
                if should_update:
                     new_path = pathfinder.find_path(hunter.pos, final_target)
                     if new_path: current_path = new_path
            
            else:
                # PATROL MODE (Triggered when particle cloud is empty)
                if not current_path:
                     patrol_target = np.array([random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)])
                     valid_patrol = pathfinder.get_nearest_walkable(patrol_target)
                     new_path = pathfinder.find_path(hunter.pos, valid_patrol)
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

        # --- STEP 5: VISUALIZATION ---
        
        # 2. Draw Walls
        for obs in walls:
            pygame.draw.rect(screen, (40, 50, 60), obs.rect)
            pygame.draw.rect(screen, (100, 120, 140), obs.rect, 2)
        
        # 3. Draw Paths
        if len(current_path) > 1:
            pygame.draw.lines(screen, (0, 200, 255), False, current_path, 2)
            for node in current_path:
                pygame.draw.circle(screen, (0, 200, 255), (int(node[0]), int(node[1])), 2)

        # MCTS Intent
        if len(pf.particles) > 0 and 'final_target' in locals():
            pygame.draw.line(screen, (255, 50, 50), hunter.pos, final_target, 1)
            tx, ty = int(final_target[0]), int(final_target[1])
            pygame.draw.circle(screen, (255, 50, 50), (tx, ty), 5, 1)

        # 4. Draw Particles
        pf.particles.draw(screen)

        # 5. Draw Drones
        draw_boid(screen, hunter, (100, 200, 255)) 
        
        # Ghost Drone
        if not can_see:
             ghost_pos = intruder.pos 
             ghost_surf = pygame.Surface((20,20), pygame.SRCALPHA)
             pygame.draw.circle(ghost_surf, (255, 50, 50, 50), (10,10), 8)
             screen.blit(ghost_surf, (intruder.rect.x, intruder.rect.y))
        
        # Real Intruder
        if can_see:
             draw_boid(screen, intruder, (255, 80, 80)) 
             pygame.draw.line(screen, YELLOW, hunter.pos, intruder.pos, 1)

        # 6. HUD
        pygame.draw.rect(screen, (10, 20, 40), (5, 5, 300, 40)) 
        pygame.draw.rect(screen, (50, 80, 100), (5, 5, 300, 40), 1) 
        
        font = pygame.font.SysFont("monospace", 18, bold=True)
        
        if len(pf.particles) > 0:
             status = "TRACKING TARGET"
             color = (0, 255, 100)
        else:
             status = "SEARCHING SECTOR"
             color = (255, 150, 0)
             
        if time_since_seen < MAX_BLIND_TIME:
            ratio = time_since_seen / MAX_BLIND_TIME
            bar_width = 100
            pygame.draw.rect(screen, (50, 0, 0), (190, 15, bar_width, 10))
            pygame.draw.rect(screen, (255, 50, 50), (190, 15, bar_width * (1-ratio), 10))
             
        text_surf = font.render(f"SYS_STATUS: {status}", True, color)
        screen.blit(text_surf, (15, 15))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()