import math
import random
import numpy as np

# --- PARAMETERS ---
EXPLORATION_CONSTANT = 2.0  # Higher because graph has fewer options
DISCOUNT_FACTOR = 0.90      # We care about the near future
MAX_DEPTH = 30              # Depth 10 in graph = ~30 seconds of flight! (Huge horizon)
NUM_SIMULATIONS = 150       # Fast simulation allows more runs

# We need the graph structure inside POMCP now
GRAPH_NODES = {
    0: np.array([100.0, 100.0]), 1: np.array([400.0, 50.0]),
    2: np.array([700.0, 100.0]), 3: np.array([700.0, 500.0]),
    4: np.array([400.0, 550.0]), 5: np.array([100.0, 500.0])
}
GRAPH_EDGES = {
    0: [1, 5], 1: [0, 2], 2: [1, 3],
    3: [2, 4], 4: [3, 5], 5: [4, 0]
}

class GraphState:
    def __init__(self, hunter_node, i_pos, i_vel):
        self.h_node = hunter_node # Index (0-5)
        self.i_pos = np.copy(i_pos)
        self.i_vel = np.copy(i_vel)

    def step(self, next_node_idx):
        # 1. Calculate Travel Time
        # How long does it take to fly between these two nodes?
        curr_pos = GRAPH_NODES[self.h_node]
        next_pos = GRAPH_NODES[next_node_idx]
        
        dist = np.linalg.norm(next_pos - curr_pos)
        speed = 5.0 # Average Hunter Speed
        travel_time = dist / speed # Frames needed (approx)
        
        # 2. Advance Hunter
        self.h_node = next_node_idx
        
        # 3. Advance Intruder (Project forward in time)
        # "While I am flying to Node B, the enemy keeps moving for T seconds"
        self.i_pos += self.i_vel * travel_time
        
        # 4. Calculate Reward
        # Are we close to the enemy at the end of this leg?
        hunter_pos = GRAPH_NODES[self.h_node]
        dist_to_enemy = np.linalg.norm(hunter_pos - self.i_pos)
        
        # Capture logic
        if dist_to_enemy < 100: # Broad capture radius for graph logic
            return 1000.0, True
            
        # Heuristic: We want to minimize distance
        return -dist_to_enemy, False

class MCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action # This is a Node Index now (0-5)
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_moves = None # Valid neighbors

    def get_untried_moves(self, current_node_idx):
        if self.untried_moves is None:
            # You can only move to connected neighbors
            self.untried_moves = list(GRAPH_EDGES[current_node_idx])
        return self.untried_moves

    def is_fully_expanded(self, current_node_idx):
        moves = self.get_untried_moves(current_node_idx)
        return len(moves) == 0

    def best_child(self):
        best_score = -float('inf')
        best_node = None
        for child in self.children.values():
            if child.visits == 0: return child
            score = (child.value / child.visits) + EXPLORATION_CONSTANT * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self, current_node_idx):
        moves = self.get_untried_moves(current_node_idx)
        next_node = moves.pop() # Pick one neighbor
        child_node = MCTSNode(parent=self, action=next_node)
        self.children[next_node] = child_node
        return child_node, next_node

def run_mcts(hunter_pos, particles):
    # 1. Determine where the Hunter IS on the graph
    # Find closest node to snap to start
    start_node = -1
    min_dist = float('inf')
    for idx, pos in GRAPH_NODES.items():
        d = np.linalg.norm(hunter_pos - pos)
        if d < min_dist:
            min_dist = d
            start_node = idx
            
    root = MCTSNode()
    
    # Run Sims
    for _ in range(NUM_SIMULATIONS):
        node = root
        
        # Sample Enemy
        i_pos, i_vel = np.array([400.0, 300.0]), np.array([0.0,0.0])
        if len(particles) > 0:
            p = random.choice(particles.sprites())
            i_pos, i_vel = p.pos, p.vel
            
        state = GraphState(start_node, i_pos, i_vel)
        
        # Selection
        depth = 0
        while node.is_fully_expanded(state.h_node) and depth < MAX_DEPTH:
            node = node.best_child()
            if node.action is not None:
                _, done = state.step(node.action)
                if done: break
            depth += 1
            
        # Expansion
        if not node.is_fully_expanded(state.h_node) and depth < MAX_DEPTH:
            node, action = node.expand(state.h_node)
            reward, done = state.step(action)
        else:
            reward = 0
            
        # Rollout (Random Walk on Graph)
        while not done and depth < MAX_DEPTH:
            neighbors = GRAPH_EDGES[state.h_node]
            random_next = random.choice(neighbors)
            r, done = state.step(random_next)
            reward += r * (DISCOUNT_FACTOR ** depth)
            depth += 1
            
        # Backprop
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
            
    # Return best next Node ID
    if not root.children: return start_node
    best_child = max(root.children.values(), key=lambda c: c.visits)
    return best_child.action