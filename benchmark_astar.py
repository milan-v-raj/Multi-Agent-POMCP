import pygame
import numpy as np
import pandas as pd
import time
import random
# Ensure these imports match your file structure
from test import Drone, Pathfinder, Obstacle, check_line_of_sight, WIDTH, HEIGHT, BLUE, RED
from test import ParticleFilter # We still use particles just to track metrics, not for control

MAX_EPISODES = 10
MAX_TIME = 30
CAPTURE_RADIUS = 30

def run_standard_astar():
    pygame.init()
    screen = pygame.display.set_mode((1, 1), pygame.NOFRAME) # Hidden window
    clock = pygame.time.Clock()
    logs = []

    for episode in range(1, MAX_EPISODES + 1):
        print(f"--- ASTAR BASELINE EPISODE {episode} ---")
        
        # Init Environment
        hunter = Drone(100, 300, BLUE, speed_limit=5)
        intruder = Drone(700, 300, RED, speed_limit=6)
        intruder.vel = np.array([-2.0, -1.0])
        
        walls = pygame.sprite.Group()
        walls.add(Obstacle(300, 100, 50, 400))
        walls.add(Obstacle(500, 0, 50, 250))
        
        pathfinder = Pathfinder(walls, grid_size=20)
        current_path = []
        last_known_pos = None
        
        # We track particles ONLY to calculate RMSE for the graph, 
        # but the Hunter is NOT allowed to use them.
        pf = ParticleFilter(num_particles=200) 
        
        start_time = time.time()
        running = True
        
        while running:
            dt = clock.tick(60) / 1000.0
            duration = time.time() - start_time
            
            # Physics
            hunter.update()
            hunter.check_collision(walls)
            intruder.update()
            
            # Dumb Intruder
            if intruder.pos[0] < 50 or intruder.pos[0] > WIDTH-50: intruder.vel[0] *= -1
            if intruder.pos[1] < 50 or intruder.pos[1] > HEIGHT-50: intruder.vel[1] *= -1
            
            # Sensors
            can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
            
            # Metrics Tracking (Hidden from Hunter)
            pf.predict()
            pf.update(intruder.pos if can_see else None, intruder.vel)

            # ==========================================
            #      STANDARD A* LOGIC (NO MCTS)
            # ==========================================
            target_pos = None
            
            if can_see:
                # 1. If we see it, go straight to it
                target_pos = intruder.pos
                last_known_pos = intruder.pos
            elif last_known_pos is not None:
                # 2. If lost, go to last known location
                target_pos = last_known_pos
                # If we reached the last known spot and it's not there, clear it
                if np.linalg.norm(hunter.pos - last_known_pos) < 50:
                    last_known_pos = None
            else:
                # 3. Patrol Randomly
                if not current_path:
                    rand_target = np.array([random.uniform(50, 750), random.uniform(50, 550)])
                    target_pos = pathfinder.get_nearest_walkable(rand_target)

            # Pathfinding Execution
            if target_pos is not None:
                # Only re-calculate path every 30 frames to save CPU, or if empty
                if frame_count % 30 == 0 or not current_path:
                    safe_target = pathfinder.get_nearest_walkable(target_pos)
                    current_path = pathfinder.find_path(hunter.pos, safe_target)
            
            # Follow Path
            steering = np.array([0.0, 0.0])
            if current_path:
                target_node = current_path[0]
                if np.linalg.norm(hunter.pos - target_node) < 20:
                    current_path.pop(0)
                else:
                    desired = target_node - hunter.pos
                    desired = (desired / np.linalg.norm(desired)) * 5.0
                    steering = desired - hunter.vel
            
            # Wall Avoidance (Reflex)
            for obs in walls:
                if obs.rect.collidepoint(hunter.pos + hunter.vel * 5):
                    steering += np.array([-hunter.vel[1], hunter.vel[0]]) * 5.0

            hunter.apply_force(steering)
            # ==========================================

            # Logging
            dist = np.linalg.norm(hunter.pos - intruder.pos)
            
            logs.append({
                "Episode": episode,
                "Time": round(duration, 2),
                "Result": "Fail", # Default
                "Method": "Standard_AStar"
            })

            if dist < CAPTURE_RADIUS:
                logs[-1]["Result"] = "Success"
                print(f"Capture! {duration:.2f}s")
                running = False
            elif duration > MAX_TIME:
                logs[-1]["Result"] = "Timeout"
                print("Timeout")
                running = False

    # Save separate file
    df = pd.DataFrame(logs)
    df.to_csv("logs_astar.csv", index=False)
    print("Saved logs_astar.csv")
    pygame.quit()

if __name__ == "__main__":
    frame_count=0
    run_standard_astar()