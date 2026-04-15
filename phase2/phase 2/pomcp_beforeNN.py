import math
import random
import numpy as np
import pygame  

# --- PARAMETERS ---
EXPLORATION_CONSTANT = 1.414    
DISCOUNT_FACTOR = 0.95        
MAX_DEPTH = 30             
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

# --- HELPER: SAFETY CHECK ---
def is_safe_move(pos, vel, action_acc, obstacles):
    """
    Checks if a move is safe.
    If 'obstacles' is empty (Ghost Mode), this only checks screen boundaries.
    """
    next_vel = (vel + action_acc) * 0.95
    speed = np.linalg.norm(next_vel)
    if speed > 5:
        next_vel = (next_vel / speed) * 5
        
    next_pos = pos + next_vel
    
    # 1. Check Screen Bounds (Always enforced)
    if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT):
        return False

    # 2. Check Obstacles
    if len(obstacles) > 0:
        safety_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
        for obs in obstacles:
            if safety_rect.colliderect(obs.rect):
                return False 
            
    return True

# --- PROXIMITY PENALTY ---
def wall_proximity_penalty(pos, obstacles):
    if len(obstacles) == 0: return 0.0
    
    min_dist = float('inf')
    for obs in obstacles:
        dx = max(obs.rect.left - pos[0], 0, pos[0] - obs.rect.right)
        dy = max(obs.rect.top - pos[1], 0, pos[1] - obs.rect.bottom)
        dist = math.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist

    if min_dist < 40:
        return -(40 - min_dist) * 0.1 
    return 0.0


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
                # Filter out moves that hit walls (or screen bounds)
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
    # --- ADDED ally_target to __init__ ---
    def __init__(self, h_pos, h_vel, i_pos, i_vel, obstacles, ally_target=None):
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.i_pos = np.copy(i_pos)
        self.i_vel = np.copy(i_vel)
        self.obstacles = obstacles
        self.ally_target = ally_target # Where is my teammate going?

    def step(self, action_idx):
        acc = ACTIONS[action_idx]
        self.h_vel += acc
        self.h_vel *= 0.95 
        speed = np.linalg.norm(self.h_vel)
        if speed > 5: self.h_vel = (self.h_vel / speed) * 5
            
        next_pos = self.h_pos + self.h_vel

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
            self.h_pos = next_pos 
            return -10.0, False 

        self.h_pos = next_pos
        self.i_pos += self.i_vel

        dist = np.linalg.norm(self.h_pos - self.i_pos)
        if dist < 30:
            return 500.0, True 
        
        reward = -(dist / 100.0) - 0.2
        reward += wall_proximity_penalty(self.h_pos, self.obstacles)

        # --- PHASE 3: COOPERATIVE STRATEGY (SPATIAL REPULSION) ---
        # If we get too close to our ally's target, penalize the reward!
        # This forces the MCTS to explore flanking maneuvers.
        if self.ally_target is not None:
            dist_to_ally = np.linalg.norm(self.h_pos - self.ally_target)
            if dist_to_ally < 100: # 100px repulsion zone
                reward -= (100 - dist_to_ally) * 0.05 

        return reward, False


# --- ADDED ally_target to run_mcts signature ---
def run_mcts(hunter, particles, obstacles, ally_target=None):
    sim_obstacles = [] 
    root = MCTSNode()
    
    target_pos_guess = np.array([400.0, 300.0])
    target_vel_guess = np.array([0.0, 0.0])

    for _ in range(NUM_SIMULATIONS):
        node = root
        
        if len(particles) > 0:
            p = random.choice(particles) #  it was random.choice(particles.sprites()) earlier
            target_pos_guess = p.pos
            target_vel_guess = p.vel

        # Pass ally_target into the state
        state = LightweightState(hunter.pos, hunter.vel, target_pos_guess, target_vel_guess, sim_obstacles, ally_target)
        
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

        # D. ROLLOUT (Smart Ranked + Ghost Safe)
        rollout_depth = 0
        cumulative_reward = reward
        current_discount = 1.0
        
        while not is_done and rollout_depth < (MAX_DEPTH - depth):
            action_idx = 4 # Default to WAIT
            
            # Strategy: 80% Smart Geometry, 20% Random
            if random.random() < 0.8:
                diff = state.i_pos - state.h_pos
                
                # Rank actions by alignment (Dot Product)
                candidates = []
                for i in range(4): 
                    ax, ay = ACTIONS[i]
                    score = ax * diff[0] + ay * diff[1]
                    candidates.append((score, i))
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Pick best move that is valid (screen bounds only)
                found_move = False
                for score, idx in candidates:
                    if is_safe_move(state.h_pos, state.h_vel, ACTIONS[idx], sim_obstacles):
                        action_idx = idx
                        found_move = True
                        break
                
                if not found_move:
                     possible = [i for i in range(4) if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], sim_obstacles)]
                     if possible: action_idx = random.choice(possible)

            else:
                possible = [i for i in range(4) if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], sim_obstacles)]
                if possible: action_idx = random.choice(possible)

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