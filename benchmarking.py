import pygame
import numpy as np
import pandas as pd
import time
import random
# Ensure these imports match your file structure. 
# If your main file is named 'main.py', keep 'from main import ...'
from test import Drone, ParticleFilter, Pathfinder, Obstacle, check_line_of_sight, WIDTH, HEIGHT, BLUE, RED, GRAY 
import pomcp_standard as pomcp_standard

# --- CONFIGURATION ---
MAX_EPISODES = 50
MAX_TIME_PER_EPISODE = 30 # seconds
CAPTURE_RADIUS = 30

def run_benchmarking():
    pygame.init()
    # screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Uncomment to watch it (slower)
    screen = pygame.display.set_mode((1, 1), pygame.NOFRAME) # Hidden window (faster)
    clock = pygame.time.Clock()

    # Data Storage
    logs = [] 

    for episode in range(1, MAX_EPISODES + 1):
        print(f"--- STARTING EPISODE {episode}/{MAX_EPISODES} ---")
        
        # 1. Reset Environment
        hunter = Drone(100, 300, BLUE, speed_limit=5)
        intruder = Drone(700, 300, RED, speed_limit=6) 
        intruder.vel = np.array([-2.0, -1.0])
        
        walls = pygame.sprite.Group()
        # A "Bucket" or "U" shape
        walls.add(Obstacle(400, 200, 20, 300)) # Left wall of U
        walls.add(Obstacle(600, 200, 20, 300)) # Right wall of U
        walls.add(Obstacle(400, 500, 220, 20)) # Bottom floor of U
        
        pf = ParticleFilter(num_particles=200)
        
        # --- MISSING COMPONENT 1: PATHFINDER ---
        pathfinder = Pathfinder(walls, grid_size=20)
        
        # --- MISSING COMPONENT 2: STATE VARIABLES ---
        current_path = []
        mcts_cooldown = 0
        last_belief_mean = None
        last_mcts_target = None
        
        start_time = time.time()
        running = True
        frame_count = 0
        
        while running:
            dt = clock.tick(60) / 1000.0 
            current_duration = time.time() - start_time
            frame_count += 1
            
            # --- PHYSICS ---
            hunter.update()
            hunter.check_collision(walls)
            intruder.update()
            
            # Simple AI for Intruder (Bounce around to be a moving target)
            if intruder.pos[0] < 50 or intruder.pos[0] > WIDTH-50: intruder.vel[0] *= -1
            if intruder.pos[1] < 50 or intruder.pos[1] > HEIGHT-50: intruder.vel[1] *= -1
            
            # --- SENSORS ---
            can_see, _, _ = check_line_of_sight(hunter, intruder, walls)
            observation = intruder.pos if can_see else None
            pf.predict()
            pf.update(observation, intruder.vel)

            # =========================================================
            #        YOUR EXACT HUNTER LOGIC (RESTORED)
            # =========================================================
            
            # A. STRATEGY LAYER
            if len(pf.particles) > 0:
                # Reduce cooldown
                if mcts_cooldown > 0:
                    mcts_cooldown -= 1
                
                # Calculate current belief mean
                current_belief_mean = np.mean([p.pos for p in pf.particles], axis=0)
                
                # Check for sudden enemy movement (Emergency Unlock)
                emergency_unlock = False
                if last_belief_mean is not None:
                    belief_shift = np.linalg.norm(current_belief_mean - last_belief_mean)
                    if belief_shift > 100: 
                        emergency_unlock = True
                        mcts_cooldown = 0 

                # --- RUN MCTS ---
                if mcts_cooldown == 0 or emergency_unlock:
                    # Unpack both values (Force + Scores)
                    strategic_force, mcts_scores = pomcp_standard.run_mcts(hunter, pf.particles, [])
                    
                    # 2. Calculate Target
                    raw_target = hunter.pos + (strategic_force * 200)
                    final_target = pathfinder.get_nearest_walkable(raw_target)
                    
                    # 3. Update Path
                    new_path = pathfinder.find_path(hunter.pos, final_target)
                    if new_path: 
                        current_path = new_path
                        last_mcts_target = final_target
                        last_belief_mean = current_belief_mean 
                        mcts_cooldown = 20 # Lock strategy for 0.5s
            
            else:
                # PATROL MODE
                if not current_path:
                    patrol_target = np.array([random.uniform(50, WIDTH-50), random.uniform(50, HEIGHT-50)])
                    valid_patrol = pathfinder.get_nearest_walkable(patrol_target)
                    new_path = pathfinder.find_path(hunter.pos, valid_patrol)
                    if new_path: current_path = new_path

            # B. PILOT LAYER (A*)
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

            # C. SAFETY REFLEX
            for obs in walls:
                if obs.rect.collidepoint(hunter.pos[0] + hunter.vel[0]*10, hunter.pos[1] + hunter.vel[1]*10):
                    steering += np.array([-hunter.vel[1], hunter.vel[0]]) * 3.0
            
            if np.linalg.norm(steering) > 0.5:
                steering = (steering / np.linalg.norm(steering)) * 0.5
            
            hunter.apply_force(steering)
            
            # =========================================================
            #                 END OF HUNTER LOGIC
            # =========================================================

            # --- METRICS & LOGGING ---
            
            # 1. RMSE (Belief Error)
            belief_error = 0.0
            if len(pf.particles) > 0:
                mean_pos = np.mean([p.pos for p in pf.particles], axis=0)
                belief_error = np.linalg.norm(mean_pos - intruder.pos)
            
            # 2. Distance to Target
            dist_to_target = np.linalg.norm(hunter.pos - intruder.pos)
            
            logs.append({
                "Episode": episode,
                "Time": round(current_duration, 2),
                "RMSE_Error": round(belief_error, 2),
                "Distance": round(dist_to_target, 2),
                "Visible": 1 if can_see else 0
            })
            
            # --- TERMINATION ---
            if dist_to_target < CAPTURE_RADIUS:
                print(f"Capture! Time: {current_duration:.2f}s")
                logs[-1]["Result"] = "Success"
                running = False
                
            if current_duration > MAX_TIME_PER_EPISODE:
                print("Timeout!")
                logs[-1]["Result"] = "Timeout"
                running = False

    # Save to CSV
    df = pd.DataFrame(logs)
    df.to_csv("mission_logs_dumb.csv", index=False)
    print("Data saved to mission_logs_dumb.csv")
    pygame.quit()

if __name__ == "__main__":
    run_benchmarking()