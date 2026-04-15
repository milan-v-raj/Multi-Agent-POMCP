import pygame
import math
import numpy as np

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

        # --- 1. CONTROLS (Manual for now) ---
        keys = pygame.key.get_pressed()
        
        # Hunter Controls (Arrow Keys)
        input_vec = np.array([0.0, 0.0])
        if keys[pygame.K_UP]:    input_vec[1] = -1
        if keys[pygame.K_DOWN]:  input_vec[1] = 1
        if keys[pygame.K_LEFT]:  input_vec[0] = -1
        if keys[pygame.K_RIGHT]: input_vec[0] = 1
        
        if np.linalg.norm(input_vec) > 0:
            input_vec = (input_vec / np.linalg.norm(input_vec)) * ACCELERATION
            hunter.apply_force(input_vec)

        # Intruder Logic (Simple Bounce / AI Placeholder)
        # For now, let's make it move Up/Down automatically to test tracking
        intruder.apply_force(np.array([0.0, 0.05 * math.sin(pygame.time.get_ticks()/500)]))

        # --- 2. UPDATE PHYSICS ---
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