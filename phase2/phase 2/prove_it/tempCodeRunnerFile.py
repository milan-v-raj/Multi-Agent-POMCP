# main.py
import pygame
import numpy as np
from config import *
from environment import Arena, Pathfinder, Obstacle
from agents import Drone, ParticleFilter
from ai_heuristic import get_heuristic_action
from ai_deep import get_deep_action

def draw_hud(screen, mode, gate_active):
    pygame.draw.rect(screen, UI_PANEL_COLOR, (10, 10, 250, 90), border_radius=5)
    pygame.draw.rect(screen, WALL_OUTLINE, (10, 10, 250, 90), 2, border_radius=5)
    
    font = pygame.font.SysFont("monospace", 16, bold=True)
    mode_color = (100, 255, 100) if mode == "DEEP-POMCP" else (255, 100, 100)
    text_mode = font.render(f"MODE: {mode}", True, mode_color)
    
    gate_status = "LOCKED" if gate_active else "OPEN"
    gate_color = (255, 50, 50) if gate_active else (50, 255, 50)
    text_gate = font.render(f"GATE: {gate_status}", True, gate_color)
    
    font_small = pygame.font.SysFont("monospace", 12)
    text_inst = font_small.render("[M] Toggle AI  |  [R] Reset", True, UI_TEXT_COLOR)

    screen.blit(text_mode, (20, 20))
    screen.blit(text_gate, (20, 45))
    screen.blit(text_inst, (20, 75))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("A/B Test: Heuristic vs Deep-POMCP")
    clock = pygame.time.Clock()

    # 1. BUILD STRATEGIC ARENA
    arena = Arena()
    arena.walls.empty() 
    
    arena.walls.add(Obstacle(0, 0, WIDTH, 10))
    arena.walls.add(Obstacle(0, HEIGHT-10, WIDTH, 10))
    arena.walls.add(Obstacle(0, 0, 10, HEIGHT))
    arena.walls.add(Obstacle(WIDTH-10, 0, 10, HEIGHT))
    
    arena.walls.add(Obstacle(WIDTH - 200, 10, 10, HEIGHT//2 - 60))
    arena.walls.add(Obstacle(WIDTH - 200, HEIGHT//2 + 60, 10, HEIGHT//2))
    
    switch_rect = pygame.Rect(50, HEIGHT//2 - 30, 60, 60)
    gate_rect = pygame.Rect(WIDTH - 200, HEIGHT//2 - 60, 10, 120)
    
    # 2. SPAWN ACTORS
    def reset_actors():
        h1 = Drone(WIDTH//2, HEIGHT//2 - 50, HUNTER_COLOR_1)
        h2 = Drone(WIDTH//2, HEIGHT//2 + 50, HUNTER_COLOR_2)
        tgt = Drone(WIDTH - 80, HEIGHT//2, TARGET_COLOR, speed_limit=0.0) 
        pf = ParticleFilter()
        pf.update(tgt.pos, tgt.vel)
        return h1, h2, tgt, pf

    hunter1, hunter2, target, pf = reset_actors()
    
    mode = "HEURISTIC" 
    running = True
    frame_count = 0
    mcts_interval = 30
    
    # Initialize the continuous thruster vectors
    current_action1 = np.array([0.0, 0.0])
    current_action2 = np.array([0.0, 0.0])
    
    while running:
        screen.fill(BG_COLOR)
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    mode = "DEEP-POMCP" if mode == "HEURISTIC" else "HEURISTIC"
                    hunter1, hunter2, target, pf = reset_actors()
                    current_action1 = np.array([0.0, 0.0])
                    current_action2 = np.array([0.0, 0.0])
                if event.key == pygame.K_r:
                    hunter1, hunter2, target, pf = reset_actors()
                    current_action1 = np.array([0.0, 0.0])
                    current_action2 = np.array([0.0, 0.0])

        # 3. GATE LOGIC
        gate_active = not (switch_rect.colliderect(hunter1.rect) or switch_rect.colliderect(hunter2.rect))
        
        dynamic_walls = arena.walls.copy()
        if gate_active:
            dynamic_walls.add(Obstacle(gate_rect.x, gate_rect.y, gate_rect.width, gate_rect.height))
            
        current_arena = Arena()
        current_arena.walls = dynamic_walls

        # 4. AI DECISIONS (The Neural Pilot)
        if frame_count % mcts_interval == 0:
            if mode == "HEURISTIC":
                current_action1 = get_heuristic_action(hunter1, target.pos, current_arena)
            else:
                _, spread = pf.get_belief_center_and_spread()
                current_action1 = get_deep_action(hunter1, hunter2.pos, target.pos, spread, current_arena)

        if (frame_count + 15) % mcts_interval == 0:
            if mode == "HEURISTIC":
                current_action2 = get_heuristic_action(hunter2, target.pos, current_arena)
            else:
                _, spread = pf.get_belief_center_and_spread()
                current_action2 = get_deep_action(hunter2, hunter1.pos, target.pos, spread, current_arena)

        # 5. PHYSICS & STEERING (Directly apply AI intention to thrusters!)
        hunter1.apply_force(current_action1 * 0.8)
        hunter2.apply_force(current_action2 * 0.8)
        
        hunter1.update()
        hunter2.update()
        
        hunter1.check_collision(current_arena)
        hunter2.check_collision(current_arena)

        # 6. RENDER
        switch_color = (50, 255, 50) if not gate_active else (20, 100, 20)
        pygame.draw.rect(screen, switch_color, switch_rect)
        pygame.draw.rect(screen, WHITE, switch_rect, 2)
        
        if gate_active:
            pygame.draw.rect(screen, (255, 50, 50, 150), gate_rect)
            
        current_arena.draw(screen)
        
        # Draw a short line showing where the AI is trying to fly
        pygame.draw.line(screen, HUNTER_COLOR_1, hunter1.pos, hunter1.pos + (current_action1 * 30), 2)
        pygame.draw.line(screen, HUNTER_COLOR_2, hunter2.pos, hunter2.pos + (current_action2 * 30), 2)
        
        screen.blit(hunter1.image, hunter1.rect)
        screen.blit(hunter2.image, hunter2.rect)
        screen.blit(target.image, target.rect)
        
        # Check Win
        if np.linalg.norm(hunter1.pos - target.pos) < 20 or np.linalg.norm(hunter2.pos - target.pos) < 20:
            font = pygame.font.SysFont("monospace", 40, bold=True)
            win_text = font.render("TARGET SECURED", True, (50, 255, 50))
            screen.blit(win_text, (WIDTH//2 - 150, HEIGHT//2 - 20))
            pygame.display.flip()
            pygame.time.wait(2000)
            hunter1, hunter2, target, pf = reset_actors()
            current_action1 = np.array([0.0, 0.0])
            current_action2 = np.array([0.0, 0.0])

        draw_hud(screen, mode, gate_active)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()