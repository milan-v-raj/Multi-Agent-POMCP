import math
import random
import numpy as np

# --- TUNED PARAMETERS ---
EXPLORATION_CONSTANT = 2.0    # Increased to encourage trying new paths
DISCOUNT_FACTOR = 0.95        
MAX_DEPTH = 15                # Look further into the future
NUM_SIMULATIONS = 150         # MORE SIMULATIONS = SMARTER (was 50)

# Increased Force (was 0.2) to make it more decisive
FORCE_MAG = 0.5 
ACTIONS = [
    np.array([0.0, -FORCE_MAG]), # UP
    np.array([0.0, FORCE_MAG]),  # DOWN
    np.array([-FORCE_MAG, 0.0]), # LEFT
    np.array([FORCE_MAG, 0.0]),  # RIGHT
    np.array([0.0, 0.0])         # WAIT
]

class MCTSNode:
    def __init__(self, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(ACTIONS)

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
        
        # Speed Limit (Matches main.py)
        speed = np.linalg.norm(self.h_vel)
        if speed > 5: self.h_vel = (self.h_vel / speed) * 5
            
        self.h_pos += self.h_vel

        # 2. Update Intruder (Simple constant velocity prediction)
        self.i_pos += self.i_vel

        # 3. Calculate Reward
        dist = np.linalg.norm(self.h_pos - self.i_pos)
        
        # CATCH REWARD
        if dist < 30:
            return 100.0, True 
        
        # --- THE FIX: TIME PENALTY ---
        # - (dist / 100.0): Existing distance penalty
        # - 0.2: NEW Time penalty. 
        # This forces the drone to CHOOSE ACTION over WAITING.
        return -(dist / 100.0) - 0.2, False

def run_mcts(hunter, particles, obstacles):
    root = MCTSNode()
    
    # Safety check: If particles are empty, assume center of map
    target_pos_guess = np.array([400.0, 300.0])
    target_vel_guess = np.array([0.0, 0.0])

    for _ in range(NUM_SIMULATIONS):
        node = root
        
        # A. SAMPLING
        if len(particles) > 0:
            # Pick a random "Simulated Reality"
            p = random.choice(particles.sprites())
            target_pos_guess = p.pos
            target_vel_guess = p.vel

        state = LightweightState(hunter.pos, hunter.vel, target_pos_guess, target_vel_guess, obstacles)
        
        # B. SELECTION
        depth = 0
        while node.is_fully_expanded() and depth < MAX_DEPTH:
            node = node.best_child()
            if node.action_from_parent is not None:
                _, is_done = state.step(node.action_from_parent)
                if is_done: break
            depth += 1

        # C. EXPANSION
        if not node.is_fully_expanded() and depth < MAX_DEPTH:
            untried = [i for i in range(len(ACTIONS)) if i not in node.children]
            action_idx = random.choice(untried)
            new_node = MCTSNode(parent=node, action_from_parent=action_idx)
            node.children[action_idx] = new_node
            node = new_node
            reward, is_done = state.step(action_idx)
        else:
            reward = 0
            is_done = False

        # D. ROLLOUT (THE FIX: HEURISTIC ROLLOUT)
        # Instead of completely random, we bias the simulation towards the target
        # This makes the AI "realize" that getting closer is good
        rollout_depth = 0
        cumulative_reward = reward
        current_discount = 1.0
        
        while not is_done and rollout_depth < (MAX_DEPTH - depth):
            # Simple Logic: 50% chance to move towards target, 50% random
            # This "guides" the random search significantly
            if random.random() < 0.5:
                # Calculate direction to target
                diff = state.i_pos - state.h_pos
                if abs(diff[0]) > abs(diff[1]):
                    # Move Horizontally
                    action_idx = 3 if diff[0] > 0 else 2
                else:
                    # Move Vertically
                    action_idx = 1 if diff[1] > 0 else 0
            else:
                action_idx = random.randint(0, len(ACTIONS)-1)
            
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
        
    # Pick child with highest visit count
    best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    return ACTIONS[best_action_idx]