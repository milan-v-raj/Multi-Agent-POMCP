# ai_heuristic.py
import numpy as np
import math
import random
from config import *

# Unit vectors for movement
ACTIONS = [
    np.array([0.0, -1.0]), # UP
    np.array([0.0, 1.0]),  # DOWN
    np.array([-1.0, 0.0]), # LEFT
    np.array([1.0, 0.0])   # RIGHT
]

class MCTSNode:
    def __init__(self, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(range(len(ACTIONS)))
        self.is_terminal = False

    def best_child(self, c_param=1.414):
        best_score = -float('inf')
        best_node = None
        for child in self.children.values():
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


class HeuristicSimState:
    def __init__(self, h_pos, h_vel, target_pos, arena):
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.target_pos = np.copy(target_pos)
        self.arena = arena

    def step(self, action_idx):
        # 1. Apply Physics
        force = ACTIONS[action_idx] * 0.5
        self.h_vel += force
        self.h_vel *= FRICTION
        speed = np.linalg.norm(self.h_vel)
        if speed > MAX_SPEED_HUNTER:
            self.h_vel = (self.h_vel / speed) * MAX_SPEED_HUNTER
        
        next_pos = self.h_pos + self.h_vel

        # 2. Check Walls (The gate acts as a wall here)
        if not self.arena.is_walkable(next_pos, padding=5):
            self.h_vel *= 0.0 # Stop momentum
            return -5.0, False # Penalty for hitting a wall

        self.h_pos = next_pos

        # 3. Check Win Condition
        dist_to_target = np.linalg.norm(self.h_pos - self.target_pos)
        if dist_to_target < 20:
            return 100.0, True # Massive reward for catching

        # 4. The "Greedy" Heuristic
        # We penalize the AI based on its distance to the target. 
        # This mathematically forces it to take the straightest line possible.
        return -(dist_to_target / 100.0), False


def get_heuristic_action(hunter, target_guess, arena, num_simulations=40, max_depth=15):
    """
    Runs a standard greedy MCTS. It simulates future moves and picks the path
    that minimizes the Euclidean distance to the target.
    """
    root = MCTSNode()

    for _ in range(num_simulations):
        node = root
        state = HeuristicSimState(hunter.pos, hunter.vel, target_guess, arena)
        depth = 0
        is_done = False
        r = 0.0
        cumulative_penalty = 0.0

        # A. Selection
        while not node.untried_actions and not node.is_terminal and depth < max_depth:
            node = node.best_child()
            r, is_done = state.step(node.action_from_parent)
            if r < 0: cumulative_penalty += r
            if is_done: break
            depth += 1

        # B. Expansion
        if not is_done and node.untried_actions and depth < max_depth:
            new_node, action_idx = node.expand()
            r, is_done = state.step(action_idx)
            if r < 0: cumulative_penalty += r
            node = new_node

        # C. Evaluation (The flaw of the baseline)
        if is_done:
            final_value = r + cumulative_penalty
            node.is_terminal = True
        else:
            # Evaluate the final state solely by its distance to the target
            dist = np.linalg.norm(state.h_pos - state.target_pos)
            final_value = -(dist / 50.0) + cumulative_penalty

        # D. Backpropagation
        while node is not None:
            node.visits += 1
            node.value += final_value
            node = node.parent

    # Return the best action as a steering vector
    if not root.children: return np.array([0.0, 0.0])
    best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    
    chosen_action = ACTIONS[best_action_idx]
    if np.linalg.norm(chosen_action) > 0:
        chosen_action = chosen_action / np.linalg.norm(chosen_action)
        
    return chosen_action