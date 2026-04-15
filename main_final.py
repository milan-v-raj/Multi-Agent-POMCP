import pygame
import math
import numpy as np
import random
import pomcp_final
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
class HybridPatrol:
    def __init__(self):
        # The Figure-8 Waypoints (Good Coverage)
        self.waypoints = [
            np.array([100, 100]), np.array([400, 50]), np.array([700, 100]),
            np.array([700, 300]), np.array([700, 500]), np.array([400, 550]),
            np.array([100, 500]), np.array([100, 300]), np.array([400, 300])
        ]
        self.current_idx = 0
        self.threshold = 50 

    def get_search_force(self, hunter_pos, hunter_vel, obstacles):
        target = self.waypoints[self.current_idx]
        
        # 1. SEEK FORCE (Drive towards the waypoint)
        desired = target - hunter_pos
        dist = np.linalg.norm(desired)
        
        # Switch waypoint if close
        if dist < self.threshold:
            self.current_idx = (self.current_idx + 1) % len(self.waypoints)
            target = self.waypoints[self.current_idx]
            desired = target - hunter_pos
            
        seek_force = np.array([0.0, 0.0])
        if np.linalg.norm(desired) > 0:
            # Normalize and scale
            seek_force = (desired / np.linalg.norm(desired)) * 0.5 

        # 2. AVOID FORCE (Steer away from walls)
        avoid_force = np.array([0.0, 0.0])
        whisker = hunter_pos + (hunter_vel * 20) # Look 20 frames ahead
        
        for obs in obstacles:
            # Check simple proximity to wall centers
            diff = hunter_pos - np.array(obs.rect.center)
            dist_to_wall = np.linalg.norm(diff)
            
            # If we are too close (Radius 80)
            if dist_to_wall < 80:
                push = (diff / np.linalg.norm(diff)) * ((80 - dist_to_wall) / 80.0)
                avoid_force += push * 2.0 # Strong repulsion

        # 3. COMBINE (Seek + Avoid)
        # Avoidance has higher priority (2.0 vs 0.5) so it overrides seeking near walls
        total_force = seek_force + avoid_force
        
        return total_force

    
# --- MAIN GAME LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("S6 Project: Drone Interceptor Arena")
    clock = pygame.time.Clock()

    # --- SPAWN CONFIGURATION (Face-Off) ---
    
    # 1. Hunter (Blue)
    # Bottom-Left of the center building, looking UP
    hunter = Drone(250, 400, BLUE, speed_limit=5)
    hunter.vel = np.array([0.0, -1.0]) # Start moving UP towards the gap
    
    # 2. Intruder (Red)
    # Top-Right of the center building, moving DOWN-LEFT into the open
    intruder = Drone(450, 200, RED, speed_limit=9)
    intruder.vel = np.array([-2.0, 1.0]) # Moving diagonally towards the Hunter
    
    # Create Walls (Urban Canyon)
    walls = pygame.sprite.Group()
    walls.add(Obstacle(300, 100, 50, 400)) # Central skyscraper
    walls.add(Obstacle(500, 0, 50, 250))   # Top block
    walls.add(Obstacle(500, 350, 50, 250)) # Bottom block

    # Initialize Particle Filter
    pf = ParticleFilter(num_particles=200)
    patrol_brain = HybridPatrol() # <--- ADD NEW
    # NEW: Watchdog Timer
    last_seen_time = pygame.time.get_ticks() # Records current time in milliseconds
    last_known_vel = np.array([0.0, 0.0]) # Store enemy speed
    MAX_BLIND_TIME = 10000 # 3 Seconds (If blind for 3s, give up)
    # NEW: Timer variables
    frame_count = 0
    mcts_interval = 10  # Think every 10 frames (approx 0.16 seconds)
    current_action = np.array([0.0, 0.0]) # Store the decision
    running = True
    while running:
        screen.fill(BLACK)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- STEP 1: PHYSICS & COLLISIONS (MUST BE FIRST) ---
        # We must calculate collisions NOW so the AI knows if it hit a wall
        hunter.update()
        hunter.check_collision(walls)
        
        intruder.update()
        #intruder_hit_wall = intruder.check_collision(walls) # <--- Variable defined here

        # --- STEP 2: AI LOGIC (MUST BE SECOND) ---
        
        # A. INTRUDER (Red)
        # Pass Hunter's position to Intruder so it can run away
        #intruder.wander(intruder_hit_wall, enemy_pos=hunter.pos)
        # --- STEP 2: CONTROLS & AI ---
        
        # A. INTRUDER (RED) - MANUAL CONTROL (WASD)
        keys = pygame.key.get_pressed()
        input_vec = np.array([0.0, 0.0])
        
        # WASD for Red Drone
        if keys[pygame.K_w]: input_vec[1] = -1
        if keys[pygame.K_s]: input_vec[1] = 1
        if keys[pygame.K_a]: input_vec[0] = -1
        if keys[pygame.K_d]: input_vec[0] = 1
        
        # Apply force if a key is pressed
        if np.linalg.norm(input_vec) > 0:
            input_vec = (input_vec / np.linalg.norm(input_vec)) * ACCELERATION
            intruder.apply_force(input_vec)
        else:
            # If no key pressed, apply friction (already handled in update)
            pass

        # B. SENSORS
        can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
        
        if can_see:
            observation = intruder.pos
            last_seen_time = current_time
            last_known_vel = intruder.vel # <--- Save this!
        else:
            observation = None
            
        # C. BELIEF UPDATE
        pf.predict()
        pf.update(observation, intruder.vel)
        # D. HUNTER (BLUE) - FINAL LOGIC
        frame_count += 1
        current_time = pygame.time.get_ticks()
        
        # Calculate time blind
        if can_see:
            last_seen_time = current_time
            time_since_seen = 0
        else:
            time_since_seen = current_time - last_seen_time
        
        # D. HUNTER (BLUE) - FINAL LOGIC WITH EXTRAPOLATION
        if time_since_seen < MAX_BLIND_TIME:
            # --- TRACKING MODE ---
            
            if len(pf.particles) > 0:
                # CASE A: We have particles. Use MCTS.
                if frame_count % mcts_interval == 0:
                    current_action = pomcp_final.run_mcts(hunter, pf.particles, walls)
                hunter.apply_force(current_action)
                # --- VISUALIZE INTENTION ---
                # Draw a red line showing the "Force" the AI is trying to apply
                # If line points INTO a wall, the AI is dumb. If it points AWAY, it's smart.
                end_point = hunter.pos + (current_action * 200) # Scale up for visibility
                pygame.draw.line(screen, (255, 0, 0), hunter.rect.center, end_point, 3)
            else:
                # CASE B: Blind & No Particles (They all hit the wall).
                # EXTRAPOLATE: Fly in the direction of the last known velocity.
                # This makes it "chase" the predicted path even without particles.
                
                if np.linalg.norm(last_known_vel) > 0:
                    # Normalize the enemy's last velocity to get direction
                    pred_dir = last_known_vel / np.linalg.norm(last_known_vel)
                    # Apply force in that direction
                    hunter.apply_force(pred_dir * 0.5)
                    
        else:
            # --- SEARCH MODE ---
            # 1. Kill particles? NO. You said "run to infinity".
            # So we DON'T empty the list. We just ignore them for MCTS if the timer is up.
            # (Optional: You can clear them if you want a fresh start, but keeping them is fine)

            # 2. Run Smart Patrol
            # It steers ITSELF. No stuck walls.
            # In Search Mode Loop
            search_force = patrol_brain.get_search_force(hunter.pos, hunter.vel, walls)
            hunter.apply_force(search_force)
                
            # Visual Debug: White Circle (Searching)
            pygame.draw.circle(screen, (255, 255, 255), (int(hunter.pos[0]), int(hunter.pos[1])), 8)

        # B. SENSORS
        can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
        
        if can_see:
            observation = intruder.pos
        else:
            observation = None
            
        # C. BELIEF UPDATE
        pf.predict()
        # C. BELIEF UPDATE
        # Manually update particles so we can pass 'walls'
        for p in pf.particles:
            p.update(walls) # <--- Pass walls here!
            
        pf.update(observation, intruder.vel)

        # D. HUNTER (Blue) - MCTS BRAIN
        best_force = pomcp_final.run_mcts(hunter, pf.particles, walls)
        hunter.apply_force(best_force)

        # --- STEP 3: DRAW ---
        walls.draw(screen)
        # --- DEBUG UI ---
        font = pygame.font.SysFont(None, 36)
        
        # 1. Show State
        if time_since_seen < MAX_BLIND_TIME:
            state_text = f"Mode: TRACKING (Time Blind: {time_since_seen/1000:.1f}s)"
            color = (0, 255, 0)
        else:
            state_text = "Mode: SEARCHING (PATROL)"
            color = (255, 0, 0)
            
        img = font.render(state_text, True, color)
        screen.blit(img, (10, 10))
        # Draw Belief State (Green Cloud)
        pf.particles.draw(screen)
        
        screen.blit(hunter.image, hunter.rect)
        
        if can_see:
            # Draw real line and real intruder
            pygame.draw.line(screen, YELLOW, hunter.rect.center, intruder.rect.center, 2)
            screen.blit(intruder.image, intruder.rect)
        else:
            # Draw blocked line and "Ghost" intruder (for debugging)
            pygame.draw.line(screen, (50, 50, 50), hunter.rect.center, intruder.rect.center, 1)
            ghost_image = intruder.image.copy()
            ghost_image.set_alpha(50)
            screen.blit(ghost_image, intruder.rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()