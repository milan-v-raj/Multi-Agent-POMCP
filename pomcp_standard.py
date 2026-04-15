import math
import random
import numpy as np
import pygame  

# DOGFIGHT POMCP - STANDARD VERSION (Wall Collisions Enabled)
# PARAMETERS 
EXPLORATION_CONSTANT = 1.414    
DISCOUNT_FACTOR = 0.95        
MAX_DEPTH = 50               
NUM_SIMULATIONS = 125         

FORCE_MAG = 0.5 
ACTIONS = [
    np.array([0.0, -FORCE_MAG]), # UP 
    np.array([0.0, FORCE_MAG]),  # DOWN 
    np.array([-FORCE_MAG, 0.0]), # LEFT 
    np.array([FORCE_MAG, 0.0]),  # RIGHT 
    np.array([0.0, 0.0])         # WAIT 
]
WIDTH, HEIGHT = 800, 600  # Map boundaries

#  HELPER: SAFETY CHECK (Action Masking) 
def is_safe_move(pos, vel, action_acc, obstacles):
    """
    Predicts if applying 'action_acc' will cause a crash in the next frame.
    Returns False if unsafe.
    """
    
    # Predict next velocity 
    next_vel = (vel + action_acc) * 0.95
    
    # (Speed Limit 5.0)
    speed = np.linalg.norm(next_vel)
    if speed > 5:
        next_vel = (next_vel / speed) * 5
        
    next_pos = pos + next_vel
    
    # 1. Check Screen Bounds
    if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT):
        return False

    # 2. Check Obstacles (Safety Buffer 20px)
    safety_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
    
    for obs in obstacles:
        if safety_rect.colliderect(obs.rect):
            return False # Hit wall
            
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
                # In Standard MCTS, we usually allow the agent to TRY unsafe moves
                # so it learns they are bad (via negative reward).
                # But to make it fair, we can keep the safety check or remove it.
                # Removing it makes it a "True" standard MCTS (it must learn safety).
                # Keeping it makes it a "Safe" standard MCTS.
                # Let's keep it consistent with your Hybrid:
                if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], state.obstacles):
                    self.untried_actions.append(i)
            
            if not self.untried_actions:
                self.untried_actions = [4] # Fallback to Wait

        return len(self.untried_actions) == 0

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        best_score = -float('inf')
        best_node = None
        
        for action_idx, child in self.children.items():
            if child.visits == 0: return child
            
            # UCT Formula
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

    def step(self, action_idx, prev_action_idx=None):
        # 1. Update Hunter Velocity
        acc = ACTIONS[action_idx]
        self.h_vel += acc
        self.h_vel *= 0.95 
        
        speed = np.linalg.norm(self.h_vel)
        if speed > 5: self.h_vel = (self.h_vel / speed) * 5
        
        # --- STANDARD MODE PHYSICS (With Collisions) ---
        # 2. Update Position
        self.h_pos += self.h_vel
        
        # 3. Check Collision IMMEDIATELY
        hit_wall = False
        
        # Screen Bounds Check
        if not (0 < self.h_pos[0] < WIDTH and 0 < self.h_pos[1] < HEIGHT):
            hit_wall = True
        
        # Wall Collision Check
        if not hit_wall:
            # Create a rect at the NEW position
            rect = pygame.Rect(self.h_pos[0]-10, self.h_pos[1]-10, 20, 20)
            for wall in self.obstacles:
                if rect.colliderect(wall.rect):
                    hit_wall = True
                    break
        
        if hit_wall:
            # HIT WALL -> STOP & PENALIZE
            self.h_pos -= self.h_vel # Bounce back / Undo move
            self.h_vel *= -0.5 # Lose energy
            return -100.0, True # HUGE PENALTY (-100) and Terminal State (True)

        # 4. Update Intruder
        self.i_pos += self.i_vel

        # 5. Calculate Reward
        dist = np.linalg.norm(self.h_pos - self.i_pos)
        
        # Base Reward (Distance)
        reward = -(dist / 100.0) - 0.1
        
        # Smoothness Penalty
        if prev_action_idx is not None:
            curr_vec = ACTIONS[action_idx]
            prev_vec = ACTIONS[prev_action_idx]
            jerk = np.linalg.norm(curr_vec - prev_vec)
            reward -= jerk * 0.5 

        return reward, False


def run_mcts(hunter, particles, obstacles):
    root = MCTSNode()
    
    # Initial Guess
    target_pos_guess = np.array([400.0, 300.0])
    target_vel_guess = np.array([0.0, 0.0])

    for _ in range(NUM_SIMULATIONS):
        node = root
        
        # A. SAMPLING
        if len(particles) > 0:
            p = random.choice(particles.sprites())
            target_pos_guess = p.pos
            target_vel_guess = p.vel

        state = LightweightState(hunter.pos, hunter.vel, target_pos_guess, target_vel_guess, obstacles)
        
        # B. SELECTION
        depth = 0
        while node.is_fully_expanded(state) and depth < MAX_DEPTH:
            prev_action = node.action_from_parent
            node = node.best_child()
            
            if node.action_from_parent is not None:
                _, is_done = state.step(node.action_from_parent, prev_action)
                if is_done: break # Stop if we hit a wall in selection
            depth += 1

        # C. EXPANSION
        reward = 0
        is_done = False
        if not node.is_fully_expanded(state) and depth < MAX_DEPTH:
            new_node, action_idx = node.expand()
            reward, is_done = state.step(action_idx)
            node = new_node

        # D. ROLLOUT
        rollout_depth = 0
        cumulative_reward = reward
        current_discount = 1.0
        
        # Continue rollout only if we haven't crashed (is_done)
        while not is_done and rollout_depth < (MAX_DEPTH - depth):
            action_idx = 4 # Default Wait
            
            # Simple Heuristic for Rollout
            if random.random() < 0.5:
                possible = [i for i in range(4) if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], obstacles)]
                if possible: action_idx = random.choice(possible)
            else:
                possible = [i for i in range(4) if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], obstacles)]
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

    # SELECT BEST ACTION
    if not root.children:
        return np.array([0.0, 0.0]), {}
        
    best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    scores = {}
    if root.children:
        for action_idx, child in root.children.items():
            if child.visits > 0:
                scores[action_idx] = child.value / child.visits
    
    return ACTIONS[best_action_idx], scores