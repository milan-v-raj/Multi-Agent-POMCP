import pygame
import math
import numpy as np
import random
import pomcp_old
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
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface((20, 20))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(x, y))
        
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
        if speed > MAX_SPEED:
            self.vel = (self.vel / speed) * MAX_SPEED
            
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

    def wander(self, hit_wall_signal=False):
        # 1. Initialize waypoint if needed
        if not hasattr(self, 'waypoint'):
            self.waypoint = np.array([random.uniform(50, WIDTH-50), 
                                      random.uniform(50, HEIGHT-50)])
        
        # 2. IF WE HIT A WALL: Pick a new waypoint immediately!
        # This stops it from trying to drill through the wall.
        if hit_wall_signal:
            # Pick a random point, but maybe try to pick one AWAY from the current wall?
            # For now, random is fine.
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
        self.image = pygame.Surface((4, 4))  # Tiny dots
        self.image.fill((0, 255, 0))         # Green color
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([float(vx), float(vy)])

    def update(self):
        # 1. Move based on last known velocity (Inertia)
        self.pos += self.vel
        
        # 2. Add "Noise" (Uncertainty spreads over time)
        # The longer we don't see the target, the more the cloud spreads
        self.pos += np.random.normal(0, 1.0, 2) 
        
        self.rect.center = self.pos

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

# --- MAIN GAME LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("S6 Project: Drone Interceptor Arena")
    clock = pygame.time.Clock()

    # Create Sprites
    hunter = Drone(100, 300, BLUE)
    intruder = Drone(700, 300, RED)
    
    # Create Walls (Urban Canyon)
    walls = pygame.sprite.Group()
    walls.add(Obstacle(300, 100, 50, 400)) # Central skyscraper
    walls.add(Obstacle(500, 0, 50, 250))   # Top block
    walls.add(Obstacle(500, 350, 50, 250)) # Bottom block

    # Initialize Particle Filter
    pf = ParticleFilter(num_particles=200)

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
        intruder_hit_wall = intruder.check_collision(walls) # <--- Variable defined here

        # --- STEP 2: AI LOGIC (MUST BE SECOND) ---
        
        # A. INTRUDER (Red)
        # Now we can safely pass the variable because it was defined in Step 1
        intruder.wander(intruder_hit_wall) 
        
        # B. SENSORS
        can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
        
        if can_see:
            observation = intruder.pos
        else:
            observation = None
            
        # C. BELIEF UPDATE
        pf.predict()
        pf.update(observation, intruder.vel)

        # D. HUNTER (Blue) - MCTS BRAIN
        best_force = pomcp_old.run_mcts(hunter, pf.particles, walls)
        hunter.apply_force(best_force)

        # --- STEP 3: DRAW ---
        walls.draw(screen)
        
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