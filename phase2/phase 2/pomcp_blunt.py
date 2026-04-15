import math
import random
import numpy as np
import pygame  

# --- PARAMETERS ---
EXPLORATION_CONSTANT = 1.414    
DISCOUNT_FACTOR = 0.95        
MAX_DEPTH = 40                
NUM_SIMULATIONS = 150        

FORCE_MAG = 0.5 
ACTIONS = [
    np.array([0.0, -FORCE_MAG]), # UP 
    np.array([0.0, FORCE_MAG]),  # DOWN 
    np.array([-FORCE_MAG, 0.0]), # LEFT 
    np.array([FORCE_MAG, 0.0]),  # RIGHT 
    np.array([0.0, 0.0])         # WAIT 
]
WIDTH, HEIGHT = 800, 600 

# --- HELPER: STRICT SAFETY CHECK ---
def is_safe_move(pos, vel, action_acc, obstacles):
    next_vel = (vel + action_acc) * 0.95
    speed = np.linalg.norm(next_vel)
    if speed > 5:
        next_vel = (next_vel / speed) * 5
        
    next_pos = pos + next_vel
    
    # 1. Check Screen Bounds
    if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT):
        return False

    # 2. Check Obstacles (Strict)
    if len(obstacles) > 0:
        safety_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
        for obs in obstacles:
            if safety_rect.colliderect(obs.rect):
                return False 
            
    return True

class MCTSNode:
    def __init__(self, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None 

    def is_fully_expanded(self, state):
        if self.untried_actions is None:
            self.untried_actions = []
            for i in range(len(ACTIONS)):
                if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], state.obstacles):
                    self.untried_actions.append(i)
            
            if not self.untried_actions:
                self.untried_actions = [4] # Default to WAIT if trapped

        return len(self.untried_actions) == 0

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        best_score = -float('inf')
        best_node = None
        for action_idx, child in self.children.items():
            if child.visits == 0: return child
            exploitation = child.value / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self):
        action_idx = self.untried_actions.pop()
        new_node = MCTSNode(parent=self, action_from_parent=action_idx)
        self.children[action_idx] = new_node
        return new_node, action_idx


class LightweightState:
    def __init__(self, h_pos, h_vel, i_pos, i_vel, obstacles):
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.i_pos = np.copy(i_pos)
        self.i_vel = np.copy(i_vel)
        self.obstacles = obstacles

    def step(self, action_idx):
        # 1. Update Hunter
        acc = ACTIONS[action_idx]
        self.h_vel += acc
        self.h_vel *= 0.95 
        speed = np.linalg.norm(self.h_vel)
        if speed > 5: self.h_vel = (self.h_vel / speed) * 5
            
        next_pos = self.h_pos + self.h_vel

        # Check strict collisions
        hit_wall = False
        if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT):
            hit_wall = True
        elif len(self.obstacles) > 0:
            sim_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
            for obs in self.obstacles:
                if sim_rect.colliderect(obs.rect):
                    hit_wall = True
                    break
        
        if hit_wall:
            # === THE BLUNT POMCP WALL PENALTY ===
            # Instantly terminate the simulation and return a massive negative reward.
            return -100.0, True 

        self.h_pos = next_pos

        # 2. Update Intruder (Simple linear projection)
        self.i_pos += self.i_vel

        # 3. Calculate Reward
        dist = np.linalg.norm(self.h_pos - self.i_pos)
        if dist < 30:
            return 500.0, True 
        
        reward = -(dist / 100.0) 
        return reward, False


def run_mcts(hunter, particles, obstacles, ally_target=None):
    root = MCTSNode()
    
    # Initial Guess
    target_pos_guess = np.array([400.0, 300.0])
    target_vel_guess = np.array([0.0, 0.0])

    for _ in range(NUM_SIMULATIONS):
        node = root
        
        # A. SAMPLING
        if len(particles) > 0:
            p = random.choice(particles)
            target_pos_guess = p.pos
            target_vel_guess = p.vel

        # Create strict state (Obstacles are passed in, so it respects walls)
        state = LightweightState(hunter.pos, hunter.vel, target_pos_guess, target_vel_guess, obstacles)
        
        # B. SELECTION
        depth = 0
        while node.is_fully_expanded(state) and depth < MAX_DEPTH:
            node = node.best_child()
            if node.action_from_parent is not None:
                _, is_done = state.step(node.action_from_parent)
                if is_done: break
            depth += 1

        # C. EXPANSION
        if not node.is_fully_expanded(state) and depth < MAX_DEPTH:
            new_node, action_idx = node.expand()
            reward, is_done = state.step(action_idx)
            node = new_node
        else:
            reward = 0
            is_done = False

        # D. ROLLOUT (The Blunt, Blind Uniform Random Rollout)
        rollout_depth = 0
        cumulative_reward = reward
        current_discount = 1.0
        
        while not is_done and rollout_depth < (MAX_DEPTH - depth):
            # === THE BLUNT ROLLOUT ===
            # Pure random chance. No heuristic alignment geometry.
            action_idx = random.randint(0, 4) 
            
            r, is_done = state.step(action_idx)
            cumulative_reward += r * current_discount
            current_discount *= DISCOUNT_FACTOR
            rollout_depth += 1

        # E. BACKPROPAGATION
        while node is not None:
            node.visits += 1
            node.value += cumulative_reward
            node = node.parent

    if not root.children:
        return np.array([0.0, 0.0])
        
    best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    return ACTIONS[best_action_idx]