import pygame
import math
import numpy as np
import random
import pomcp_blunt
import time

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


# --- MAIN BENCHMARK LOOP ---
def run_benchmark(total_episodes=100, visualize=True, log_filename="benchmark_log.txt"):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("S6 Project: Benchmark Data Collection")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)

    # ==========================================
    # NEW COMPLEX MAP: Symmetric Occlusion Maze
    # ==========================================
    walls = pygame.sprite.Group()
    # Left Wing Obstacles
    walls.add(Obstacle(150, 150, 50, 300))  
    walls.add(Obstacle(200, 275, 100, 50))  
    # Central Choke Points
    walls.add(Obstacle(400, 0, 50, 200))    
    walls.add(Obstacle(400, 400, 50, 200))  
    # Right Wing Occlusions
    walls.add(Obstacle(600, 250, 50, 200))  
    walls.add(Obstacle(500, 100, 150, 50))
    # A central pillar to split the particle filter
    walls.add(Obstacle(400, 275, 50, 50))   
    
    pathfinder = Pathfinder(walls, grid_size=20) 

    # --- IEEE METRICS TRACKERS ---
    global_success_count = 0
    global_frames_list = []
    global_latency_list = []

    print("="*50)
    print(f"Starting Benchmark: {total_episodes} Episodes")
    print(f"Map Complexity: HIGH (Symmetric Occlusions)")
    print("="*50)

    for episode in range(1, total_episodes + 1):
        # --- RESET MAP FOR NEW EPISODE ---
        hunter1 = Drone(50, 50, BLUE, speed_limit=5)
        hunter1.vel = np.array([1.0, 1.0]) 
        
        hunter2 = Drone(50, 550, CYAN, speed_limit=5)
        hunter2.vel = np.array([1.0, -1.0]) 
        
        # Spawn intruder randomly on the right side of the map
        valid_spawn = pathfinder.get_nearest_walkable(np.array([random.uniform(500, 750), random.uniform(50, 550)]))
        intruder = Drone(valid_spawn[0], valid_spawn[1], RED, speed_limit=7)
        intruder.vel = np.array([-1.0, -1.0])
        
        pf = ParticleFilter(num_particles=200)
        
        current_path1 = [] 
        current_path2 = []
        final_target1 = None
        final_target2 = None
        
        frame_count = 0
        mcts_interval = 30 
        MAX_BLIND_TIME = 10000 
        MAX_EPISODE_FRAMES = 1500 # Timeout after ~25 seconds of simulation
        last_seen_time = pygame.time.get_ticks() if visualize else 0
        
        episode_running = True
        outcome = "TIMEOUT"
        episode_latencies = []

        while episode_running:
            current_time = pygame.time.get_ticks() if visualize else frame_count * 16

            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            # --- STEP 1: PHYSICS ---
            hunter1.update()
            hunter1.check_collision(walls)
            hunter2.update()
            hunter2.check_collision(walls)
            intruder.update()
            intruder.check_collision(walls)
            
            # --- STEP 2: SMART AUTOMATED EVADER (8-Way Raycast) ---
            best_score = -float('inf')
            best_dir = np.array([0.0, 0.0])
            
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                dir_vec = np.array([math.cos(rad), math.sin(rad)])
                test_pos = intruder.pos + (dir_vec * 30)
                wall_penalty = 0
                
                if not (20 < test_pos[0] < WIDTH-20 and 20 < test_pos[1] < HEIGHT-20): 
                    wall_penalty = 1000 
                else:
                    test_rect = pygame.Rect(test_pos[0]-10, test_pos[1]-10, 20, 20)
                    for obs in walls:
                        if test_rect.colliderect(obs.rect.inflate(40, 40)): 
                            wall_penalty = 1000
                            break
                            
                nearest_hunter_dist = min(np.linalg.norm(test_pos - hunter1.pos), np.linalg.norm(test_pos - hunter2.pos))
                score = nearest_hunter_dist - wall_penalty
                if score > best_score:
                    best_score = score
                    best_dir = dir_vec
                    
            if best_score > -500: 
                intruder.apply_force(best_dir * ACCELERATION)

            # --- STEP 3: SENSORS & SHARED BELIEF ---
            can_see1, _, _ = check_line_of_sight(hunter1, intruder, walls)
            can_see2, _, _ = check_line_of_sight(hunter2, intruder, walls)
            can_see_global = can_see1 or can_see2
            
            observation = intruder.pos if can_see_global else None
            if can_see_global: last_seen_time = current_time
            if current_time - last_seen_time > MAX_BLIND_TIME: pf.particles.empty()

            pf.predict() 
            pf.update(observation, intruder.vel)

            # --- STEP 4: HUNTER AI (MCTS WITH LATENCY TRACKING) ---
            frame_count += 1
            
            if frame_count % mcts_interval == 0:
                if len(pf.particles) > 0:
                    ally_tgt = final_target2 if 'final_target2' in locals() and final_target2 is not None else hunter2.pos
                    
                    # --- MCTS TIMER START ---
                    start_time = time.perf_counter()
                    # IMPORTANT: Make sure this points to the right POMCP file!
                    strategic_force1 = pomcp_blunt.run_mcts(hunter1, pf.particles.sprites(), walls, ally_target=ally_tgt)
                    end_time = time.perf_counter()
                    episode_latencies.append((end_time - start_time) * 1000)
                    # --- MCTS TIMER END ---
                    
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
                    if not current_path1:
                         patrol1 = np.array([random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)])
                         current_path1 = pathfinder.find_path(hunter1.pos, pathfinder.get_nearest_walkable(patrol1))

            if (frame_count + 15) % mcts_interval == 0:
                if len(pf.particles) > 0:
                    ally_tgt = final_target1 if 'final_target1' in locals() and final_target1 is not None else hunter1.pos
                    
                    # --- MCTS TIMER START ---
                    start_time = time.perf_counter()
                    strategic_force2 = pomcp_blunt.run_mcts(hunter2, pf.particles.sprites(), walls, ally_target=ally_tgt)
                    end_time = time.perf_counter()
                    episode_latencies.append((end_time - start_time) * 1000)
                    # --- MCTS TIMER END ---
                    
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
                    if not current_path2:
                         patrol2 = np.array([random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)])
                         current_path2 = pathfinder.find_path(hunter2.pos, pathfinder.get_nearest_walkable(patrol2))

            # --- STEP 5: PILOTS & SAFETY REFLEX ---
            steering1, steering2 = np.array([0.0, 0.0]), np.array([0.0, 0.0])
            
            if current_path1:
                if np.linalg.norm(hunter1.pos - current_path1[0]) < 30: current_path1.pop(0)
                if current_path1:
                    desired1 = current_path1[0] - hunter1.pos
                    if np.linalg.norm(desired1) > 0:
                        steering1 = ((desired1 / np.linalg.norm(desired1)) * 4.0) - hunter1.vel

            if current_path2:
                if np.linalg.norm(hunter2.pos - current_path2[0]) < 30: current_path2.pop(0)
                if current_path2:
                    desired2 = current_path2[0] - hunter2.pos
                    if np.linalg.norm(desired2) > 0:
                        steering2 = ((desired2 / np.linalg.norm(desired2)) * 4.0) - hunter2.vel

            for obs in walls:
                if obs.rect.collidepoint(hunter1.pos[0] + hunter1.vel[0]*10, hunter1.pos[1] + hunter1.vel[1]*10):
                    steering1 += np.array([-hunter1.vel[1], hunter1.vel[0]]) * 3.0
                if obs.rect.collidepoint(hunter2.pos[0] + hunter2.vel[0]*10, hunter2.pos[1] + hunter2.vel[1]*10):
                    steering2 += np.array([-hunter2.vel[1], hunter2.vel[0]]) * 3.0
            
            if np.linalg.norm(steering1) > 0.5: steering1 = (steering1 / np.linalg.norm(steering1)) * 0.5
            if np.linalg.norm(steering2) > 0.5: steering2 = (steering2 / np.linalg.norm(steering2)) * 0.5
            
            hunter1.apply_force(steering1)
            hunter2.apply_force(steering2)
            
            # BOIDS SEPARATION
            dist_between = np.linalg.norm(hunter1.pos - hunter2.pos)
            if dist_between < 40: 
                push_force = ((hunter1.pos - hunter2.pos) / dist_between) * 1.5
                steering1 += push_force
                steering2 -= push_force

            # --- STEP 6: WIN/LOSS CHECK ---
            if np.linalg.norm(hunter1.pos - intruder.pos) < 40 or np.linalg.norm(hunter2.pos - intruder.pos) < 40:
                outcome = "CAPTURED (WIN)"
                episode_running = False

            if frame_count >= MAX_EPISODE_FRAMES:
                outcome = "ESCAPED (TIMEOUT)"
                episode_running = False

            # --- STEP 7: VISUALIZATION ---
            if visualize:
                draw_grid(screen)
                for obs in walls:
                    pygame.draw.rect(screen, (40, 50, 60), obs.rect)
                    pygame.draw.rect(screen, (100, 120, 140), obs.rect, 2)
                
                if len(current_path1) > 1: pygame.draw.lines(screen, BLUE, False, current_path1, 2)
                if len(current_path2) > 1: pygame.draw.lines(screen, CYAN, False, current_path2, 2)

                pf.particles.draw(screen)
                draw_boid(screen, hunter1, BLUE) 
                draw_boid(screen, hunter2, CYAN) 
                
                if not can_see_global:
                     ghost_surf = pygame.Surface((20,20), pygame.SRCALPHA)
                     pygame.draw.circle(ghost_surf, (255, 50, 50, 50), (10,10), 8)
                     screen.blit(ghost_surf, (intruder.rect.x, intruder.rect.y))
                else:
                     draw_boid(screen, intruder, RED) 

                status = "TRACKING" if len(pf.particles) > 0 else "SEARCHING"
                color = (0, 255, 100) if status == "TRACKING" else (255, 150, 0)
                
                pygame.draw.rect(screen, (0, 0, 0), (0, 0, WIDTH, 40))
                txt = font.render(f"EP: {episode}/{total_episodes} | FR: {frame_count} | {status}", True, color)
                screen.blit(txt, (10, 10))

                pygame.display.flip()
                clock.tick(FPS)
                
        # --- END OF EPISODE LOGGING ---
        if outcome == "CAPTURED (WIN)":
            global_success_count += 1
            global_frames_list.append(frame_count)
            
        if episode_latencies:
            global_latency_list.extend(episode_latencies)
            
        print(f"Episode {episode:03d}: {outcome} in {frame_count} frames.")

    # ==========================================
    # FINAL IEEE BENCHMARK OUTPUT & FILE LOGGING
    # ==========================================
    output_lines = [
        "\n" + "="*50,
        "=== FINAL IEEE BENCHMARK RESULTS ===",
        "="*50,
        f"Total Episodes Run   : {total_episodes}",
        f"Capture Success Rate : {(global_success_count / total_episodes) * 100:.1f}%"
    ]
    
    if len(global_frames_list) > 0:
        output_lines.append(f"Avg Time-to-Capture  : {np.mean(global_frames_list):.1f} frames")
        output_lines.append(f"Std Dev (Frames)     : ±{np.std(global_frames_list):.1f}")
    if len(global_latency_list) > 0:
        output_lines.append(f"Avg Planning Latency : {np.mean(global_latency_list):.2f} ms")
        output_lines.append(f"Std Dev (Latency)    : ±{np.std(global_latency_list):.2f} ms")
    output_lines.append("="*50 + "\n")

    # 1. Print to console
    final_text = "\n".join(output_lines)
    print(final_text)

    # 2. Append to a persistent text file
    with open(log_filename, "a") as f:
        f.write(final_text)
    print(f"[INFO] Metrics successfully saved to {log_filename}")

    pygame.quit()

if __name__ == "__main__":
    # Set visualize=False to run 100 games instantly in the background!
    run_benchmark(total_episodes=100, visualize=False, log_filename="benchmark_log2.txt")