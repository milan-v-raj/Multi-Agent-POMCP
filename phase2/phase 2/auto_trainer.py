import pygame
import math
import numpy as np
import random
import pomcp_beforeNN

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800, 600
FPS = 120 
WHITE = (255, 255, 255)
RED = (200, 50, 50)     
BLUE = (50, 50, 200)    
CYAN = (50, 200, 255)   
GRAY = (100, 100, 100)  
YELLOW = (255, 255, 0)  

MAX_SPEED = 5
ACCELERATION = 0.2
FRICTION = 0.95  

# --- IMPORT CLASSES ---
import heapq

class Pathfinder:
    def __init__(self, walls, grid_size=20): 
        self.walls = walls
        self.grid_size = grid_size
        self.cols = WIDTH // grid_size
        self.rows = HEIGHT // grid_size

    def get_grid_pos(self, pos): return (int(pos[0] // self.grid_size), int(pos[1] // self.grid_size))
    def get_world_pos(self, grid_pos): return np.array([grid_pos[0] * self.grid_size + self.grid_size/2, grid_pos[1] * self.grid_size + self.grid_size/2])
    def is_walkable(self, grid_pos):
        c, r = grid_pos
        if c < 0 or c >= self.cols or r < 0 or r >= self.rows: return False
        cell_rect = pygame.Rect(c * self.grid_size, r * self.grid_size, self.grid_size, self.grid_size)
        for obs in self.walls:
            if cell_rect.colliderect(obs.rect.inflate(5, 5)): return False
        return True

    def get_nearest_walkable(self, pos):
        start_node = self.get_grid_pos(pos)
        if self.is_walkable(start_node): return pos
        queue = [start_node]
        visited = set([start_node])
        while queue and len(visited) < 200:
            current = queue.pop(0)
            if self.is_walkable(current): return self.get_world_pos(current)
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
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end_node:
                path = [self.get_world_pos(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(self.get_world_pos(current))
                return path[::-1]
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_walkable(neighbor): continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + abs(end_node[0]-neighbor[0]) + abs(end_node[1]-neighbor[1])
                    heapq.heappush(open_set, (f, neighbor))
        return []

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

    def apply_force(self, force): self.acc += force
    def update(self):
        self.vel += self.acc
        self.vel *= FRICTION
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed: self.vel = (self.vel / speed) * self.max_speed
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
                    self.pos[0] += (clip.width + 2) if self.rect.centerx > obs.rect.centerx else -(clip.width + 2)
                else:
                    self.vel[1] *= -1
                    self.pos[1] += (clip.height + 2) if self.rect.centery > obs.rect.centery else -(clip.height + 2)
                self.rect.center = self.pos

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
        self.particles = pygame.sprite.Group()
        self.frame_timer = 0

    def predict(self):
        for p in self.particles: p.update()

    def update(self, observation, intruder_vel):
        self.frame_timer += 1
        if observation is not None and self.frame_timer % 5 == 0:
            self.particles.empty()
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
        if obs.rect.clipline(line_start, line_end): return False
    return True

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
    if len(particles) < 2: return particles.sprites(), []
    positions = np.array([p.pos for p in particles])
    idx = np.random.choice(len(positions), 2, replace=False)
    c1, c2 = positions[idx[0]], positions[idx[1]]
    for _ in range(3):
        dist_to_c1 = np.linalg.norm(positions - c1, axis=1)
        dist_to_c2 = np.linalg.norm(positions - c2, axis=1)
        mask1 = dist_to_c1 < dist_to_c2
        if np.any(mask1): c1 = np.mean(positions[mask1], axis=0)
        if np.any(~mask1): c2 = np.mean(positions[~mask1], axis=0)
    sprites = particles.sprites()
    return [sprites[i] for i in range(len(sprites)) if mask1[i]], [sprites[i] for i in range(len(sprites)) if not mask1[i]]


# --- AUTOMATED ENVIRONMENT RUNNER ---
def run_automated_games(total_episodes=50, visualize_first_n=20):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DRL Data Collector: Advanced Harvester")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)

    walls = pygame.sprite.Group()
    walls.add(Obstacle(300, 100, 50, 400)) 
    walls.add(Obstacle(500, 0, 50, 250))   
    walls.add(Obstacle(500, 350, 50, 250))  
    pathfinder = Pathfinder(walls, grid_size=20)

    # --- THE RL DATASET TENSORS ---
    X_data = [] # Will hold the 22-number stacked state arrays
    Y_data = [] # Will hold the discounted rewards

    MAX_SPREAD = 500.0 # Approximate max standard deviation for our screen size
    GAMMA = 0.99       # The discount factor

    for episode in range(1, total_episodes + 1):
        visualize = episode <= visualize_first_n
        mode_text = "VISUAL MODE" if visualize else "HEADLESS ACCELERATED"
        print(f"--- Starting Episode {episode}/{total_episodes} [{mode_text}] ---")
        
        hunter1 = Drone(100, 150, BLUE, speed_limit=5)
        hunter2 = Drone(100, 450, CYAN, speed_limit=5)
        
        valid_spawn = pathfinder.get_nearest_walkable(np.array([random.uniform(50, 750), random.uniform(50, 550)]))
        intruder = Drone(valid_spawn[0], valid_spawn[1], RED, speed_limit=7)
        
        pf = ParticleFilter(num_particles=200)
        current_path1, current_path2 = [], []
        final_target1, final_target2 = None, None
        
        frame_count = 0
        mcts_interval = 30 
        MAX_BLIND_TIME = 10000 
        MAX_EPISODE_FRAMES = 1500 
        last_seen_time = pygame.time.get_ticks() if visualize else 0 
        
        episode_running = True
        outcome = "TIMEOUT"

        # --- NEW: Episode Tracking ---
        episode_states = []
        prev_state_tensor = None 

        while episode_running:
            current_time = pygame.time.get_ticks() if visualize else frame_count * 16
            
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            # --- STEP 1 & 2: PHYSICS & INTRUDER AI ---
            hunter1.update(); hunter1.check_collision(walls)
            hunter2.update(); hunter2.check_collision(walls)
            intruder.update(); intruder.check_collision(walls)

            best_score = -float('inf')
            best_dir = np.array([0.0, 0.0])
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                dir_vec = np.array([math.cos(rad), math.sin(rad)])
                test_pos = intruder.pos + (dir_vec * 30)
                wall_penalty = 0
                if not (20 < test_pos[0] < WIDTH-20 and 20 < test_pos[1] < HEIGHT-20): wall_penalty = 1000 
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
            if best_score > -500: intruder.apply_force(best_dir * ACCELERATION)

            # --- STEP 3: SENSORS & SWARM MODE ---
            can_see_global = check_line_of_sight(hunter1, intruder, walls) or check_line_of_sight(hunter2, intruder, walls)
            if can_see_global: last_seen_time = current_time
            if current_time - last_seen_time > MAX_BLIND_TIME: pf.particles.empty()
            
            pf.predict()
            pf.update(intruder.pos if can_see_global else None, intruder.vel)

            swarm_mode = "SEARCHING"
            particles_h1, particles_h2 = pf.particles.sprites(), pf.particles.sprites()
            belief_center = np.array([WIDTH/2, HEIGHT/2]) # Fallback
            cloud_spread = 0.0

            if len(pf.particles) > 0:
                all_positions = np.array([p.pos for p in pf.particles])
                belief_center = np.mean(all_positions, axis=0)
                cloud_spread = np.std(all_positions) 
                
                if cloud_spread > 80: 
                    swarm_mode = "PARTITION"
                    g1, g2 = partition_belief(pf.particles)
                    c1 = np.mean([p.pos for p in g1], axis=0) if g1 else belief_center
                    if np.linalg.norm(hunter1.pos - c1) < np.linalg.norm(hunter2.pos - c1):
                        particles_h1, particles_h2 = g1, g2
                    else:
                        particles_h1, particles_h2 = g2, g1
                else:
                    swarm_mode = "ASSAULT"
                    if np.linalg.norm(hunter1.pos - belief_center) < np.linalg.norm(hunter2.pos - belief_center):
                        particles_h2 = [Particle(p.pos[0] + p.vel[0]*30, p.pos[1] + p.vel[1]*30, 0, 0) for p in pf.particles]
                    else:
                        particles_h1 = [Particle(p.pos[0] + p.vel[0]*30, p.pos[1] + p.vel[1]*30, 0, 0) for p in pf.particles]

            # --- NEW: ADVANCED DATA HARVESTING (NORMALIZED & STACKED) ---
            if frame_count % mcts_interval == 0:
                # 1. Normalize the state
                current_state_tensor = [
                    hunter1.pos[0] / WIDTH, hunter1.pos[1] / HEIGHT,
                    hunter1.vel[0] / MAX_SPEED, hunter1.vel[1] / MAX_SPEED,
                    hunter2.pos[0] / WIDTH, hunter2.pos[1] / HEIGHT,
                    hunter2.vel[0] / MAX_SPEED, hunter2.vel[1] / MAX_SPEED,
                    belief_center[0] / WIDTH, belief_center[1] / HEIGHT,
                    min(cloud_spread / MAX_SPREAD, 1.0) # Cap at 1.0 just in case
                ]

                # 2. Frame Stacking
                if prev_state_tensor is None:
                    prev_state_tensor = current_state_tensor # Duplicate for the very first frame
                
                # Combine current and previous to create a 22D vector capturing motion
                stacked_state = current_state_tensor + prev_state_tensor
                episode_states.append(stacked_state)
                
                prev_state_tensor = current_state_tensor # Save for next loop

            # --- STEP 4: MCTS PLANNERS ---
            frame_count += 1
            if frame_count % mcts_interval == 0:
                if len(pf.particles) > 0:
                    ally_tgt = final_target2 if final_target2 is not None else hunter2.pos
                    sf1 = pomcp_beforeNN.run_mcts(hunter1, particles_h1, [], ally_target=ally_tgt)
                    if np.dot(hunter1.vel / (np.linalg.norm(hunter1.vel)+0.01), sf1 / (np.linalg.norm(sf1)+0.01)) >= -0.5:
                        final_target1 = pathfinder.get_nearest_walkable(hunter1.pos + (sf1 * 150))
                        if not current_path1 or np.linalg.norm(final_target1 - current_path1[-1]) > 60:
                            current_path1 = pathfinder.find_path(hunter1.pos, final_target1)
                else:
                    if not current_path1: current_path1 = pathfinder.find_path(hunter1.pos, pathfinder.get_nearest_walkable(np.array([random.uniform(50, WIDTH//2-50), random.uniform(50, HEIGHT-50)])))

            if (frame_count + 15) % mcts_interval == 0:
                if len(pf.particles) > 0:
                    ally_tgt = final_target1 if final_target1 is not None else hunter1.pos
                    sf2 = pomcp_beforeNN.run_mcts(hunter2, particles_h2, [], ally_target=ally_tgt)
                    if np.dot(hunter2.vel / (np.linalg.norm(hunter2.vel)+0.01), sf2 / (np.linalg.norm(sf2)+0.01)) >= -0.5:
                        final_target2 = pathfinder.get_nearest_walkable(hunter2.pos + (sf2 * 150))
                        if not current_path2 or np.linalg.norm(final_target2 - current_path2[-1]) > 60:
                            current_path2 = pathfinder.find_path(hunter2.pos, final_target2)
                else:
                    if not current_path2: current_path2 = pathfinder.find_path(hunter2.pos, pathfinder.get_nearest_walkable(np.array([random.uniform(WIDTH//2+50, WIDTH-50), random.uniform(50, HEIGHT-50)])))

            # --- STEP 5: PILOT & SAFETY ---
            s1, s2 = np.array([0.0, 0.0]), np.array([0.0, 0.0])
            if current_path1:
                if np.linalg.norm(hunter1.pos - current_path1[0]) < 30: current_path1.pop(0)
                if current_path1: s1 = ((current_path1[0] - hunter1.pos) / np.linalg.norm(current_path1[0] - hunter1.pos) * 4.0) - hunter1.vel
            if current_path2:
                if np.linalg.norm(hunter2.pos - current_path2[0]) < 30: current_path2.pop(0)
                if current_path2: s2 = ((current_path2[0] - hunter2.pos) / np.linalg.norm(current_path2[0] - hunter2.pos) * 4.0) - hunter2.vel

            for obs in walls:
                if obs.rect.collidepoint(hunter1.pos[0] + hunter1.vel[0]*10, hunter1.pos[1] + hunter1.vel[1]*10): s1 += np.array([-hunter1.vel[1], hunter1.vel[0]]) * 3.0
                if obs.rect.collidepoint(hunter2.pos[0] + hunter2.vel[0]*10, hunter2.pos[1] + hunter2.vel[1]*10): s2 += np.array([-hunter2.vel[1], hunter2.vel[0]]) * 3.0
            
            dist_h = np.linalg.norm(hunter1.pos - hunter2.pos)
            if dist_h < 40:
                push = ((hunter1.pos - hunter2.pos) / dist_h) * 1.5
                s1 += push; s2 -= push
                
            if np.linalg.norm(s1) > 0.5: s1 = (s1 / np.linalg.norm(s1)) * 0.5
            if np.linalg.norm(s2) > 0.5: s2 = (s2 / np.linalg.norm(s2)) * 0.5
            hunter1.apply_force(s1); hunter2.apply_force(s2)

            # --- STEP 6: CHECK WIN/LOSS CONDITIONS ---
            if np.linalg.norm(hunter1.pos - intruder.pos) < 30 or np.linalg.norm(hunter2.pos - intruder.pos) < 30:
                outcome = "CAPTURED (WIN)"
                episode_running = False

            if frame_count >= MAX_EPISODE_FRAMES:
                outcome = "ESCAPED (LOSS)"
                episode_running = False

            # --- STEP 7: VISUALIZATION ---
            if visualize:
                screen.fill((10, 20, 40)) 
                for obs in walls:
                    pygame.draw.rect(screen, (40, 50, 60), obs.rect)
                    pygame.draw.rect(screen, (100, 120, 140), obs.rect, 2)
                
                pf.particles.draw(screen)
                draw_boid(screen, hunter1, BLUE) 
                draw_boid(screen, hunter2, CYAN) 
                draw_boid(screen, intruder, RED) 

                pygame.draw.rect(screen, (0, 0, 0), (0, 0, WIDTH, 40))
                txt = font.render(f"EPISODE: {episode} | TIME: {frame_count}/{MAX_EPISODE_FRAMES} | TACTIC: {swarm_mode}", True, WHITE)
                screen.blit(txt, (10, 10))

                pygame.display.flip()
                clock.tick(FPS)
            
        # --- NEW: DISCOUNTED REWARD ASSIGNMENT ---
        print(f"Result: {outcome} in {frame_count} frames. Recorded {len(episode_states)} states.")
        
        base_reward = 1.0 if outcome == "CAPTURED (WIN)" else -1.0
        T = len(episode_states)

        for t, state in enumerate(episode_states):
            # t=0 is the first frame, t=T-1 is the terminal frame. 
            # We want the terminal frame to have exactly base_reward (+1 or -1)
            discounted_reward = base_reward * (GAMMA ** (T - t - 1))
            X_data.append(state)
            Y_data.append(discounted_reward)

    # --- SAVE TO HARD DRIVE ---
    print("\n--- DATA HARVESTING COMPLETE ---")
    
    X_array = np.array(X_data, dtype=np.float32)
    Y_array = np.array(Y_data, dtype=np.float32)
    
    # Check dataset balance
    win_ratio = sum(1 for y in Y_array if y > 0) / len(Y_array) if len(Y_array) > 0 else 0
    print(f"Dataset Win Ratio: {win_ratio:.2f} (Ideal is between 0.30 and 0.70)")
    
    np.save("swarm_states_X.npy", X_array)
    np.save("swarm_labels_Y.npy", Y_array)
    
    print(f"Successfully saved {len(X_array)} stacked states to disk!")
    pygame.quit()

if __name__ == "__main__":
    # Runs 50 games total, but ONLY draws the screen for the first 20.
    run_automated_games(total_episodes=2000, visualize_first_n=0)