import math
import random
import numpy as np
import pygame  
import torch
import torch.nn as nn

# --- 1. THE BRAIN (PyTorch Setup) ---
class SwarmValueNet(nn.Module):
    def __init__(self):
        super(SwarmValueNet, self).__init__()
        # CRITICAL FIX: This matches the Deep 256-neuron architecture you just trained!
        self.network = nn.Sequential(
            nn.Linear(22, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.network(x)

value_net = SwarmValueNet()
try:
    value_net.load_state_dict(torch.load("swarm_value_net.pth", map_location=torch.device('cpu'), weights_only=True))
    value_net.eval()
    print("🧠 Deep-POMCP Neural Network Loaded Successfully!")
except FileNotFoundError:
    print("⚠️ WARNING: swarm_value_net.pth not found. MCTS will fail.")

# --- 2. PARAMETERS ---
EXPLORATION_CONSTANT = 0.5  # Tuned for bounded Neural Outputs
MAX_DEPTH = 30              # Deep enough to see around corners, short enough to be fast
NUM_SIMULATIONS = 60        # Dropped from 150 to keep framerates buttery smooth

FORCE_MAG = 0.5 
# WE DELETED "WAIT". The drone MUST move.
ACTIONS = [
    np.array([0.0, -FORCE_MAG]), # UP 
    np.array([0.0, FORCE_MAG]),  # DOWN 
    np.array([-FORCE_MAG, 0.0]), # LEFT 
    np.array([FORCE_MAG, 0.0])   # RIGHT 
]
WIDTH, HEIGHT, MAX_SPEED, MAX_SPREAD = 800, 600, 5.0, 500.0

# --- 3. HELPER: SAFETY CHECK ---
def is_safe_move(pos, vel, action_acc, obstacles):
    next_vel = (vel + action_acc) * 0.95
    speed = np.linalg.norm(next_vel)
    if speed > MAX_SPEED: next_vel = (next_vel / speed) * MAX_SPEED
    next_pos = pos + next_vel
    
    if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT): return False
    if len(obstacles) > 0:
        safety_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
        for obs in obstacles:
            if safety_rect.colliderect(obs.rect): return False 
    return True

# --- 4. THE MCTS TREE ---
class MCTSNode:
    def __init__(self, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None 
        self.is_terminal = False

    def is_fully_expanded(self, state):
        if self.untried_actions is None:
            self.untried_actions = []
            for i in range(len(ACTIONS)):
                if is_safe_move(state.h_pos, state.h_vel, ACTIONS[i], state.obstacles):
                    self.untried_actions.append(i)
            # If completely trapped, default to UP just to do something
            if not self.untried_actions: self.untried_actions = [0] 

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


class PureNeuralState:
    def __init__(self, h_pos, h_vel, i_pos, i_vel, obstacles):
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.i_pos = np.copy(i_pos)
        self.i_vel = np.copy(i_vel)
        self.obstacles = obstacles

    def step(self, action_idx):
        # 1. Move Hunter
        self.h_vel += ACTIONS[action_idx]
        self.h_vel *= 0.95 
        speed = np.linalg.norm(self.h_vel)
        if speed > MAX_SPEED: self.h_vel = (self.h_vel / speed) * MAX_SPEED
        next_pos = self.h_pos + self.h_vel

        # 2. Check Walls (SOFT WALL FIX)
        hit_wall = False
        if not (0 < next_pos[0] < WIDTH and 0 < next_pos[1] < HEIGHT): hit_wall = True
        if len(self.obstacles) > 0:
            sim_rect = pygame.Rect(next_pos[0]-10, next_pos[1]-10, 20, 20)
            for obs in self.obstacles:
                if sim_rect.colliderect(obs.rect): 
                    hit_wall = True
                    break
        
        if hit_wall:
            # Stop momentum, but DO NOT terminate! 
            # Apply a tiny penalty so it prefers open space, but survives to slide around corners.
            self.h_vel *= 0.0 
            return -0.1, False 

        self.h_pos = next_pos

        # 3. Move Intruder & Check Win
        self.i_pos += self.i_vel
        if np.linalg.norm(self.h_pos - self.i_pos) < 30: return 1.0, True 

        return 0.0, False

# --- 5. THE PURE NEURAL MCTS LOOP ---
def run_mcts(hunter, particles, obstacles, ally_target=None):
    sim_obstacles = obstacles # FIX: Actually pass the real walls to the simulation!
    root = MCTSNode()
    
    target_pos_guess = np.array([400.0, 300.0])
    target_vel_guess = np.array([0.0, 0.0])
    ally_pos = ally_target if ally_target is not None else np.array([400.0, 300.0])

    for _ in range(NUM_SIMULATIONS):
        node = root
        
        # Sample a belief
        if len(particles) > 0:
            p = random.choice(particles) if isinstance(particles, list) else random.choice(particles.sprites())
            target_pos_guess, target_vel_guess = p.pos, p.vel

        state = PureNeuralState(hunter.pos, hunter.vel, target_pos_guess, target_vel_guess, sim_obstacles)
        depth = 0
        
        
        # --- THE FIX: Penalty Tracker ---
        is_done = False
        r = 0.0
        cumulative_penalty = 0.0 
        
        # A. SELECTION
        while node.is_fully_expanded(state) and not node.is_terminal and depth < MAX_DEPTH:
            node = node.best_child(c_param=EXPLORATION_CONSTANT)
            r, is_done = state.step(node.action_from_parent)
            if r < 0: cumulative_penalty += r # Accumulate wall hits!
            if is_done: break
            depth += 1

        # B. EXPANSION
        if not is_done and not node.is_fully_expanded(state) and depth < MAX_DEPTH:
            new_node, action_idx = node.expand()
            r, is_done = state.step(action_idx)
            if r < 0: cumulative_penalty += r # Accumulate wall hits!
            node = new_node

        # C. HYBRID EVALUATION (Properly Penalized)
        if is_done:
            final_value = r + cumulative_penalty 
            node.is_terminal = True
        else:
            # 1. NORMALIZED TACTICAL HEURISTIC
            dist = np.linalg.norm(state.h_pos - state.i_pos)
            tactical_reward = -(dist / 1000.0) 
            
            # --- MISSING SPREAD CALCULATION RESTORED ---
            c_spread = 0.0
            if len(particles) > 0:
                all_pos = np.array([p.pos for p in particles] if isinstance(particles, list) else [p.pos for p in particles.sprites()])
                c_spread = np.std(all_pos)
            # -------------------------------------------

            # 2. STRATEGIC NEURAL NET
            current_state_11d = [
                state.h_pos[0] / WIDTH, state.h_pos[1] / HEIGHT,
                state.h_vel[0] / MAX_SPEED, state.h_vel[1] / MAX_SPEED,
                ally_pos[0] / WIDTH, ally_pos[1] / HEIGHT,
                0.0, 0.0, 
                state.i_pos[0] / WIDTH, state.i_pos[1] / HEIGHT, 
                min(c_spread / MAX_SPREAD, 1.0) 
            ]
            
            prev_h_pos = state.h_pos - (state.h_vel * 30.0)
            prev_i_pos = state.i_pos - (state.i_vel * 30.0)
            
            prev_state_11d = [
                prev_h_pos[0] / WIDTH, prev_h_pos[1] / HEIGHT,
                state.h_vel[0] / MAX_SPEED, state.h_vel[1] / MAX_SPEED,
                ally_pos[0] / WIDTH, ally_pos[1] / HEIGHT,
                0.0, 0.0, 
                prev_i_pos[0] / WIDTH, prev_i_pos[1] / HEIGHT, 
                min(c_spread / MAX_SPREAD, 1.0) 
            ]
            
            tensor = torch.FloatTensor(current_state_11d + prev_state_11d).unsqueeze(0)
            
            with torch.no_grad():
                neural_strategy_value = value_net(tensor).item() 
                
            # 3. THE BLEND
            final_value = (tactical_reward * 0.85) + (neural_strategy_value * 0.15) + cumulative_penalty

        # D. BACKPROPAGATION
        while node is not None:
            node.visits += 1
            node.value += final_value
            node = node.parent

    # # Return Best Action
    # if not root.children: return np.array([0.0, 0.0])
    # best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    # return ACTIONS[best_action_idx]

    # --- DIAGNOSTIC PRINT ---
    if not root.children: return np.array([0.0, 0.0])
    
    # Optional: Uncomment this to watch the brain think in your terminal
    print("--- MCTS ROOT EVALUATION ---")
    for action_idx, child in root.children.items():
        avg_val = child.value / child.visits if child.visits > 0 else 0
        print(f"Action {action_idx} | Visits: {child.visits} | Avg Score: {avg_val:.3f}")
        
    best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    
    # Scale the 0.5 magnitude back up to a 1.0 unit vector so A* can clear walls
    chosen_action = ACTIONS[best_action_idx]
    if np.linalg.norm(chosen_action) > 0:
        chosen_action = chosen_action / np.linalg.norm(chosen_action)
        
    return chosen_action