import pygame
import math
import numpy as np
import random
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
        # Create a temporary rect to see where we WOULD be
        # (We use the sprite's rect which is integer-based)
        for obs in obstacles:
            if self.rect.colliderect(obs.rect):
                # Simple Bounce Logic:
                # If we hit a wall, reverse velocity and push back slightly
                # This is a basic "elastic" collision
                
                # Check Horizontal Collision
                if abs(self.rect.centerx - obs.rect.centerx) > abs(self.rect.centery - obs.rect.centery):
                    self.vel[0] *= -1 
                    # Push out of wall to prevent sticking
                    if self.rect.centerx < obs.rect.centerx:
                        self.pos[0] -= 5
                    else:
                        self.pos[0] += 5
                
                # Check Vertical Collision
                else:
                    self.vel[1] *= -1
                    if self.rect.centery < obs.rect.centery:
                        self.pos[1] -= 5
                    else:
                        self.pos[1] += 5
    # ... inside Drone class ...

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

    def wander(self):
        """
        Intruder Logic: Pick a random spot, fly there, then pick another.
        """
        # Initialize a target attribute if it doesn't exist
        if not hasattr(self, 'waypoint'):
            self.waypoint = np.array([random.uniform(50, WIDTH-50), 
                                      random.uniform(50, HEIGHT-50)])
        
        # Move towards the waypoint
        self.move_towards(self.waypoint)
        
        # If close to waypoint, pick a new one
        if np.linalg.norm(self.pos - self.waypoint) < 50:
             self.waypoint = np.array([random.uniform(50, WIDTH-50), 
                                       random.uniform(50, HEIGHT-50)])
             # Simple check to ensure waypoint isn't inside a wall would go here                        
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

    running = True
    while running:
        screen.fill(BLACK)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 1. AI LOGIC ---
        
        # A. INTRUDER LOGIC (Random Patrol)
        intruder.wander()

        # B. HUNTER LOGIC (The "Greedy" Baseline)
        # Check if we can see the target
        can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
        
        if can_see:
            # If visible, chase it directly!
            hunter.move_towards(intruder.pos)
        else:
            # If hidden, just brake (Decision Paralysis)
            # THIS IS WHERE MCTS WILL EVENTUALLY GO
            hunter.vel *= 0.9  # Slow down

        # --- 2. UPDATE PHYSICS ---
        hunter.update()
        hunter.check_collision(walls)  # <--- NEW LINE: Check Blue Drone vs Walls
        
        intruder.update()
        intruder.check_collision(walls) # <--- NEW LINE: Check Red Drone vs Walls

        # --- 3. SENSORS (RAY CASTING) ---
        can_see, start_pos, end_pos = check_line_of_sight(hunter, intruder, walls)

        # --- 4. DRAW ---
        walls.draw(screen)
        screen.blit(hunter.image, hunter.rect)
        
        # Only draw Intruder if visible (simulating camera feed)
        # But for debugging, we draw it Red if visible, Faded if hidden
        if can_see:
            pygame.draw.line(screen, YELLOW, start_pos, end_pos, 2) # Sight line
            screen.blit(intruder.image, intruder.rect)
        else:
            pygame.draw.line(screen, (50, 50, 50), start_pos, end_pos, 1) # Blocked line
            # Draw a "Ghost" intruder to show where it actually is (Debug mode)
            ghost_image = intruder.image.copy()
            ghost_image.set_alpha(50) # Transparent
            screen.blit(ghost_image, intruder.rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()