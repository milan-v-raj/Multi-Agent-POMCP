import math
import random
import numpy as np
import pygame  # Needed for Rect collision checks

# --- TUNED PARAMETERS ---
EXPLORATION_CONSTANT = 2.0    
DISCOUNT_FACTOR = 0.95        
MAX_DEPTH = 15               
NUM_SIMULATIONS = 75         

FORCE_MAG = 0.5 
ACTIONS = [
    np.array([0.0, -FORCE_MAG]), # UP (0)
    np.array([0.0, FORCE_MAG]),  # DOWN (1)
    np.array([-FORCE_MAG, 0.0]), # LEFT (2)
    np.array([FORCE_MAG, 0.0]),  # RIGHT (3)
    np.array([0.0, 0.0])         # WAIT (4)
]
WIDTH, HEIGHT = 800, 600  # Map boundaries

# --- HELPER: SAFETY CHECK (Action Masking) ---
def is_safe_move(pos, vel, action_acc, obstacles):
    """
    Predicts if applying 'action_acc' will cause a crash in the next frame.
    Returns False if unsafe.
    """
    
    # Predict next velocity with friction
    next_vel = (vel + action_acc) * 0.95
    
    # Cap speed (Speed Limit 5.0)
    speed = np.linalg.norm(next_vel)
    if speed > 5:
        next_vel = (next_vel / speed) * 5
        
    next_pos = pos + next_vel
    drone_size = 20
    safety_rect = pygame.Rect(next_pos[0] - drone_size/2, 
                              next_pos[1] - drone_size/2, 
                              drone_size, drone_size)
    # 1. Check Screen Bounds
    if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT):
        return False

    # 2. Check Obstacles (Safety Buffer 20px)
    safety_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
    
    for obs in obstacles:
        if safety_rect.colliderect(obs.rect):
            return False # Hit wall
            
    return True

# --- HELPER: PROXIMITY PENALTY (Soft Constraint) ---
def wall_proximity_penalty(pos, obstacles):
    """
    Returns a negative penalty if the drone is too close to a wall.
    Encourages staying in the center of corridors.
    """
    min_dist = float('inf')
    
    # Simple Distance to nearest wall edge
    for obs in obstacles:
        # Distance to rectangle logic
        dx = max(obs.rect.left - pos[0], 0, pos[0] - obs.rect.right)
        dy = max(obs.rect.top - pos[1], 0, pos[1] - obs.rect.bottom)
        dist = math.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist

    # If within 40 pixels, apply penalty
    if min_dist < 40:
        return -(40 - min_dist) * 0.1 # Small penalty scaling
    return 0.0


class MCTSNode:
    def __init__(self, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None # Will be filled on first access

    def is_fully_expanded(self, state):
        # FIX 1: ACTION MASKING
        # Only define 'untried_actions' once. Filter out dangerous moves.
        if self.untried_actions is None:
            self.untried_actions = []
            for i in range(len(ACTIONS)):
                if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], state.obstacles):
                    self.untried_actions.append(i)
            
            # If TRAPPED (no safe moves), allow WAIT (4) just to survive
            if not self.untried_actions:
                self.untried_actions = [4]

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
        # Pick an action we haven't tried yet
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
        self.h_vel *= 0.95 # Drag
        
        speed = np.linalg.norm(self.h_vel)
        if speed > 5: self.h_vel = (self.h_vel / speed) * 5
            
        next_pos = self.h_pos + self.h_vel

        # FIX 3: TERMINAL CRASH PENALTY (Inside Simulation)
        # If rollout hits wall, immediate death.
        hit_wall = False
        if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT):
            hit_wall = True
        else:
            sim_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
            for obs in self.obstacles:
                if sim_rect.colliderect(obs.rect):
                    hit_wall = True
                    break
        
        if hit_wall:
            return -1000.0, True # Terminal Penalty

        self.h_pos = next_pos

        # 2. Update Intruder
        self.i_pos += self.i_vel

        # 3. Calculate Reward
        dist = np.linalg.norm(self.h_pos - self.i_pos)
        
        # CATCH REWARD
        if dist < 30:
            return 100.0, True 
        
        # STANDARD REWARD: Distance + Time + Proximity
        # Base Penalty
        reward = -(dist / 100.0) - 0.2
        
        # Add Wall Proximity Penalty (Soft Constraint)
        reward += wall_proximity_penalty(self.h_pos, self.obstacles)

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

        # D. ROLLOUT (Biased)
        rollout_depth = 0
        cumulative_reward = reward
        current_discount = 1.0
        
        while not is_done and rollout_depth < (MAX_DEPTH - depth):
            # 50% Greedy, 50% Safe Random
            if random.random() < 0.5:
                diff = state.i_pos - state.h_pos
                if abs(diff[0]) > abs(diff[1]):
                    best_action = 3 if diff[0] > 0 else 2
                else:
                    best_action = 1 if diff[1] > 0 else 0
                
                # Check if greedy move kills us
                if is_safe_move(state.h_pos, state.h_vel, ACTIONS[best_action], obstacles):
                    action_idx = best_action
                else:
                    action_idx = 4 # Wait if greedy is dangerous
            else:
                # Pick random SAFE move
                possible = []
                for i in range(len(ACTIONS)):
                    if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], obstacles):
                        possible.append(i)
                action_idx = random.choice(possible) if possible else 4
            
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
        return np.array([0.0, 0.0])
        
    best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    return ACTIONS[best_action_idx]