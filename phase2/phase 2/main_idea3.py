import pygame
import math
import numpy as np
import random
import pomcp_real

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # Intruder
BLUE = (50, 50, 200)    # Hunter 1
CYAN = (50, 200, 255)   # Hunter 2 (Lighter blue to tell them apart)
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
        self.radius = 6
        self.image = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (0, 255, 0, 50), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([float(vx), float(vy)])

    def update(self):
        self.pos += self.vel
        self.pos += np.random.normal(0, 1.5, 2) 
        self.rect.center = self.pos

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
        self.frame_timer += 1
        if observation is not None:
            if self.frame_timer % 5 == 0:
                self.particles.empty()
                self.is_initialized = True
                obs_x, obs_y = observation
                vel_x, vel_y = intruder_vel 
                for _ in range(100): 
                    px = obs_x + np.random.normal(0, 5)
                    py = obs_y + np.random.normal(0, 5)
                    self.particles.add(Particle(px, py, vel_x, vel_y))

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
    
def partition_belief(particles):
    """Ultra-fast 2D K-Means to split the belief cloud in half."""
    if len(particles) < 2:
        return particles.sprites(), []
        
    positions = np.array([p.pos for p in particles])
    
    # Randomly pick two distinct particles as initial centers
    idx = np.random.choice(len(positions), 2, replace=False)
    c1, c2 = positions[idx[0]], positions[idx[1]]
    
    # Run 3 quick iterations to stabilize the centers
    for _ in range(3):
        dist_to_c1 = np.linalg.norm(positions - c1, axis=1)
        dist_to_c2 = np.linalg.norm(positions - c2, axis=1)
        
        # Group 1 gets particles closer to c1, Group 2 gets particles closer to c2
        mask1 = dist_to_c1 < dist_to_c2
        
        if np.any(mask1): c1 = np.mean(positions[mask1], axis=0)
        if np.any(~mask1): c2 = np.mean(positions[~mask1], axis=0)

    # Convert back to sprite lists
    sprites = particles.sprites()
    group1 = [sprites[i] for i in range(len(sprites)) if mask1[i]]
    group2 = [sprites[i] for i in range(len(sprites)) if not mask1[i]]
    
    return group1, group2


# --- MAIN GAME LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("S6 Project: Multi-Agent Drone Swarm")
    clock = pygame.time.Clock()

    # --- DUAL HUNTER SPAWN CONFIGURATION ---
    hunter1 = Drone(100, 150, BLUE, speed_limit=5)
    hunter1.vel = np.array([1.0, 1.0]) 
    
    hunter2 = Drone(100, 450, CYAN, speed_limit=5)
    hunter2.vel = np.array([1.0, -1.0]) 
    
    intruder = Drone(700, 300, RED, speed_limit=9)
    intruder.vel = np.array([-1.0, -1.0])
    
    walls = pygame.sprite.Group()
    walls.add(Obstacle(300, 100, 50, 400)) 
    walls.add(Obstacle(500, 0, 50, 250))   
    walls.add(Obstacle(500, 350, 50, 250))  

    # Initialize Systems
    pf = ParticleFilter(num_particles=200)
    pathfinder = Pathfinder(walls, grid_size=20) 
    
    # Tracking Variables for Both Hunters
    current_path1 = [] 
    current_path2 = []
    final_target1 = None
    final_target2 = None
    
    frame_count = 0
    mcts_interval = 30 
    MAX_BLIND_TIME = 10000 
    last_seen_time = pygame.time.get_ticks() 
    current_time = 0

    running = True
    while running:
        draw_grid(screen)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- STEP 1: PHYSICS ---
        hunter1.update()
        hunter1.check_collision(walls)
        
        hunter2.update()
        hunter2.check_collision(walls)
        
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

        # --- STEP 3: SENSORS & SHARED BELIEF (PHASE 2) ---
        can_see1, _, _ = check_line_of_sight(hunter1, intruder, walls)
        can_see2, _, _ = check_line_of_sight(hunter2, intruder, walls)
        
        # Swarm Logic: If ANY drone sees the target, the whole swarm knows.
        can_see_global = can_see1 or can_see2
        
        observation = None
        if can_see_global:
            observation = intruder.pos
            last_seen_time = current_time
        
        time_since_seen = current_time - last_seen_time

        if time_since_seen > MAX_BLIND_TIME:
            pf.particles.empty()

        pf.predict() 
        pf.update(observation, intruder.vel)

        # --- PHASE 3: SWARM TACTICS ANALYZER ---
        swarm_mode = "SEARCHING"
        particles_h1 = pf.particles.sprites() # Default: Both look at whole cloud
        particles_h2 = pf.particles.sprites()
        
        if len(pf.particles) > 0:
            all_positions = np.array([p.pos for p in pf.particles])
            belief_center = np.mean(all_positions, axis=0)
            
            # Calculate how spread out the cloud is (Uncertainty)
            cloud_spread = np.std(all_positions) 
            
            if cloud_spread > 80: 
                # MODE B: HIGH UNCERTAINTY -> DIVIDE AND CONQUER
                swarm_mode = "PARTITION (SPLIT)"
                group1, group2 = partition_belief(pf.particles)
                
                # Assign groups based on which Hunter is closer to the cluster center
                center1 = np.mean([p.pos for p in group1], axis=0) if group1 else belief_center
                # Calculate center2 just for distance checking
                center2 = np.mean([p.pos for p in group2], axis=0) if group2 else belief_center 
                
                if np.linalg.norm(hunter1.pos - center1) < np.linalg.norm(hunter2.pos - center1):
                    particles_h1, particles_h2 = group1, group2
                else:
                    particles_h1, particles_h2 = group2, group1

            else:
                # MODE A: LOW UNCERTAINTY -> HAMMER AND ANVIL
                swarm_mode = "ASSAULT (PINCER)"
                
                # Who is closer to the target?
                dist1 = np.linalg.norm(hunter1.pos - belief_center)
                dist2 = np.linalg.norm(hunter2.pos - belief_center)
                
                if dist1 < dist2:
                    # Hunter 1 is the Hammer (Chases normal particles)
                    # Hunter 2 is the Anvil (We trick its MCTS by shifting the particles forward)
                    particles_h2 = []
                    for p in pf.particles:
                        fake_p = Particle(p.pos[0] + p.vel[0]*30, p.pos[1] + p.vel[1]*30, 0, 0)
                        particles_h2.append(fake_p)
                else:
                    # Hunter 2 is the Hammer, Hunter 1 is the Anvil
                    particles_h1 = []
                    for p in pf.particles:
                        fake_p = Particle(p.pos[0] + p.vel[0]*30, p.pos[1] + p.vel[1]*30, 0, 0)
                        particles_h1.append(fake_p)

        # --- STEP 4: HUNTER AI (THE SPOTTERS & PILOTS) ---
        frame_count += 1
        
        # ==========================================
        # HUNTER 1: STRATEGY (Runs on frames 0, 30, 60...)
        # ==========================================
        if frame_count % mcts_interval == 0:
            if len(pf.particles) > 0:
                # Figure out where Hunter 2 is going
                ally_tgt = final_target2 if 'final_target2' in locals() and final_target2 is not None else hunter2.pos
                
                # --- FIX: Pass ally_target into MCTS ---
                strategic_force1 = pomcp_real.run_mcts(hunter1, particles_h1, walls, ally_target=ally_tgt)
                
                ignore_mcts1 = False
                speed1 = np.linalg.norm(hunter1.vel)
                if speed1 > 2.0:
                    norm_vel1 = hunter1.vel / speed1
                    mcts_len1 = np.linalg.norm(strategic_force1)
                    if mcts_len1 > 0:
                        norm_force1 = strategic_force1 / mcts_len1
                        alignment1 = np.dot(norm_vel1, norm_force1)
                        if alignment1 < -0.5: ignore_mcts1 = True

                if not ignore_mcts1:
                    raw_target1 = hunter1.pos + (strategic_force1 * 150)
                    final_target1 = pathfinder.get_nearest_walkable(raw_target1)
                    
                    should_update1 = False
                    if not current_path1:
                        should_update1 = True
                    else:
                        current_goal1 = current_path1[-1] 
                        if np.linalg.norm(final_target1 - current_goal1) > 60: 
                            should_update1 = True
                    
                    if should_update1:
                         new_path1 = pathfinder.find_path(hunter1.pos, final_target1)
                         if new_path1: current_path1 = new_path1
            else:
                # Patrol 1
                if not current_path1:
                     patrol1 = np.array([random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)])
                     current_path1 = pathfinder.find_path(hunter1.pos, pathfinder.get_nearest_walkable(patrol1))

        # ==========================================
        # HUNTER 2: STRATEGY (Runs on frames 15, 45, 75...)
        # ==========================================
        if (frame_count + 15) % mcts_interval == 0:
            if len(pf.particles) > 0:
                # Figure out where Hunter 1 is going
                ally_tgt = final_target1 if 'final_target1' in locals() and final_target1 is not None else hunter1.pos
                
                # --- FIX: Pass ally_target into MCTS ---
                strategic_force2 = pomcp_real.run_mcts(hunter2, particles_h2, walls, ally_target=ally_tgt)
                
                ignore_mcts2 = False
                speed2 = np.linalg.norm(hunter2.vel)
                if speed2 > 2.0:
                    norm_vel2 = hunter2.vel / speed2
                    mcts_len2 = np.linalg.norm(strategic_force2)
                    if mcts_len2 > 0:
                        norm_force2 = strategic_force2 / mcts_len2
                        alignment2 = np.dot(norm_vel2, norm_force2)
                        if alignment2 < -0.5: ignore_mcts2 = True

                if not ignore_mcts2:
                    raw_target2 = hunter2.pos + (strategic_force2 * 150)
                    final_target2 = pathfinder.get_nearest_walkable(raw_target2)
                    
                    should_update2 = False
                    if not current_path2:
                        should_update2 = True
                    else:
                        current_goal2 = current_path2[-1] 
                        if np.linalg.norm(final_target2 - current_goal2) > 60: 
                            should_update2 = True
                    
                    if should_update2:
                         new_path2 = pathfinder.find_path(hunter2.pos, final_target2)
                         if new_path2: current_path2 = new_path2
            else:
                # Patrol 2
                if not current_path2:
                     patrol2 = np.array([random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)])
                     current_path2 = pathfinder.find_path(hunter2.pos, pathfinder.get_nearest_walkable(patrol2))

        # ==========================================
        # THE PILOTS (A* Control Layer)
        # ==========================================
        steering1 = np.array([0.0, 0.0])
        if current_path1:
            if np.linalg.norm(hunter1.pos - current_path1[0]) < 30: current_path1.pop(0)
            if current_path1:
                desired1 = current_path1[0] - hunter1.pos
                if np.linalg.norm(desired1) > 0:
                    steering1 = ((desired1 / np.linalg.norm(desired1)) * 4.0) - hunter1.vel

        steering2 = np.array([0.0, 0.0])
        if current_path2:
            if np.linalg.norm(hunter2.pos - current_path2[0]) < 30: current_path2.pop(0)
            if current_path2:
                desired2 = current_path2[0] - hunter2.pos
                if np.linalg.norm(desired2) > 0:
                    steering2 = ((desired2 / np.linalg.norm(desired2)) * 4.0) - hunter2.vel

        # --- SAFETY REFLEX (Wall Repulsion) ---
        for obs in walls:
            # Hunter 1
            if obs.rect.collidepoint(hunter1.pos[0] + hunter1.vel[0]*10, hunter1.pos[1] + hunter1.vel[1]*10):
                steering1 += np.array([-hunter1.vel[1], hunter1.vel[0]]) * 3.0
            # Hunter 2
            if obs.rect.collidepoint(hunter2.pos[0] + hunter2.vel[0]*10, hunter2.pos[1] + hunter2.vel[1]*10):
                steering2 += np.array([-hunter2.vel[1], hunter2.vel[0]]) * 3.0
                
        # --- FIX: DRONE-TO-DRONE SEPARATION (Moved before normalization) ---
        dist_between = np.linalg.norm(hunter1.pos - hunter2.pos)
        if dist_between < 40: # If closer than 40 pixels
            push_away = hunter1.pos - hunter2.pos
            push_force = (push_away / dist_between) * 1.5
            steering1 += push_force
            steering2 -= push_force
        
        if np.linalg.norm(steering1) > 0.5: steering1 = (steering1 / np.linalg.norm(steering1)) * 0.5
        if np.linalg.norm(steering2) > 0.5: steering2 = (steering2 / np.linalg.norm(steering2)) * 0.5
        
        hunter1.apply_force(steering1)
        hunter2.apply_force(steering2)
        

        # --- STEP 5: VISUALIZATION ---
        # Draw Walls
        for obs in walls:
            pygame.draw.rect(screen, (40, 50, 60), obs.rect)
            pygame.draw.rect(screen, (100, 120, 140), obs.rect, 2)
        
        # Draw Paths
        if len(current_path1) > 1:
            pygame.draw.lines(screen, BLUE, False, current_path1, 2)
        if len(current_path2) > 1:
            pygame.draw.lines(screen, CYAN, False, current_path2, 2)

        # Draw MCTS Intents
        if len(pf.particles) > 0:
            if final_target1 is not None:
                pygame.draw.circle(screen, BLUE, (int(final_target1[0]), int(final_target1[1])), 5, 1)
            if final_target2 is not None:
                pygame.draw.circle(screen, CYAN, (int(final_target2[0]), int(final_target2[1])), 5, 1)

        # Draw Particles
        pf.particles.draw(screen)

        # Draw Drones
        draw_boid(screen, hunter1, BLUE) 
        draw_boid(screen, hunter2, CYAN) 
        
        # Draw Intruder
        if not can_see_global:
             ghost_surf = pygame.Surface((20,20), pygame.SRCALPHA)
             pygame.draw.circle(ghost_surf, (255, 50, 50, 50), (10,10), 8)
             screen.blit(ghost_surf, (intruder.rect.x, intruder.rect.y))
        else:
             draw_boid(screen, intruder, RED) 
             if can_see1: pygame.draw.line(screen, YELLOW, hunter1.pos, intruder.pos, 1)
             if can_see2: pygame.draw.line(screen, YELLOW, hunter2.pos, intruder.pos, 1)

        # --- FIX: UPDATED HUD to show swarm_mode ---
        pygame.draw.rect(screen, (10, 20, 40), (5, 5, 300, 40)) 
        pygame.draw.rect(screen, (50, 80, 100), (5, 5, 300, 40), 1) 
        
        font = pygame.font.SysFont("monospace", 18, bold=True)
        
        if len(pf.particles) > 0:
             # Shows whether they are in Assault or Partition mode
             status = f"MODE: {swarm_mode}"
             color = (0, 255, 100) if swarm_mode == "ASSAULT (PINCER)" else (200, 100, 255)
        else:
             status = "MODE: SEARCHING"
             color = (255, 150, 0)
             
        if time_since_seen < MAX_BLIND_TIME:
            ratio = time_since_seen / MAX_BLIND_TIME
            bar_width = 100
            pygame.draw.rect(screen, (50, 0, 0), (190, 15, bar_width, 10))
            pygame.draw.rect(screen, (255, 50, 50), (190, 15, bar_width * (1-ratio), 10))
             
        text_surf = font.render(status, True, color)
        screen.blit(text_surf, (15, 15))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()