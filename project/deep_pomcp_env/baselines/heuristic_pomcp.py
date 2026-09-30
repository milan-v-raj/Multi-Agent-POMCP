import math
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pygame
from .base_policy import BasePursuerPolicy
from ..scenarios import Obstacle
from ..pathfinder import Pathfinder

class MCTSNodeHeuristic:
    def __init__(self, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self, state, action_list):
        if self.untried_actions is None:
            self.untried_actions = list(range(len(action_list)))
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.414):
        best_score = -float('inf')
        best_node = None
        for action_idx, child in self.children.items():
            if child.visits == 0:
                return child
            exploitation = child.value / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self):
        action_idx = self.untried_actions.pop()
        new_node = MCTSNodeHeuristic(parent=self, action_from_parent=action_idx)
        self.children[action_idx] = new_node
        return new_node, action_idx


class GhostState:
    def __init__(self, h_pos: np.ndarray, h_vel: np.ndarray, i_pos: np.ndarray, i_vel: np.ndarray,
                 width: int, height: int, ally_target: Optional[np.ndarray] = None):
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.i_pos = np.copy(i_pos)
        self.i_vel = np.copy(i_vel)
        self.width = width
        self.height = height
        self.ally_target = ally_target

    def step(self, action_vec: np.ndarray):
        self.h_vel += action_vec
        self.h_vel *= 0.95
        speed = np.linalg.norm(self.h_vel)
        if speed > 5.0:
            self.h_vel = (self.h_vel / speed) * 5.0
        next_pos = self.h_pos + self.h_vel

        hit_wall = False
        if not (0 < next_pos[0] < self.width and 0 < next_pos[1] < self.height):
            hit_wall = True

        if hit_wall:
            self.h_pos = next_pos
            return -10.0, False

        self.h_pos = next_pos
        self.i_pos += self.i_vel

        dist = np.linalg.norm(self.h_pos - self.i_pos)
        if dist < 30.0:
            return 500.0, True

        reward = -(dist / 100.0) - 0.2

        if self.ally_target is not None:
            dist_to_ally = np.linalg.norm(self.h_pos - self.ally_target)
            if dist_to_ally < 100.0:
                reward -= (100.0 - dist_to_ally) * 0.05

        return reward, False


class HeuristicPOMCPPolicy(BasePursuerPolicy):
    def __init__(self, num_simulations: int = 150, max_depth: int = 40, name: str = 'Heuristic POMCP'):
        super().__init__(name)
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_param = 1.414
        self.discount_factor = 0.95
        self.action_vectors = [
            np.array([0.0, -0.5], dtype=np.float32),
            np.array([0.0, 0.5], dtype=np.float32),
            np.array([-0.5, 0.0], dtype=np.float32),
            np.array([0.5, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32)
        ]
        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}
        self.final_targets: Dict[int, np.ndarray] = {}

    def reset(self):
        self.current_paths.clear()
        self.final_targets.clear()

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int, obstacles: List[Obstacle], width: int, height: int) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos = info['hunter_positions'][agent_id]
        h_vel = info['hunter_velocities'][agent_id]
        b_mean = info['belief_mean']
        step = info.get('current_step', 0)
        particles = info.get('particles', None)

        if agent_id not in self.current_paths:
            self.current_paths[agent_id] = []

        is_active = info.get('is_belief_active', False) and (b_mean is not None)

        plan_offset = 0 if agent_id == 0 else 15
        if (step + plan_offset) % 30 == 0:
            if is_active:
                other_id = 1 if agent_id == 0 else 0
                ally_tgt = self.final_targets.get(other_id, None)

                strategic_dir = self._run_mcts(h_pos, h_vel, b_mean, width, height, ally_tgt, particles=particles)

                ignore_mcts = False
                speed = np.linalg.norm(h_vel)
                mcts_len = np.linalg.norm(strategic_dir)
                if speed > 2.0 and mcts_len > 0:
                    alignment = np.dot(h_vel / speed, strategic_dir / mcts_len)
                    if alignment < -0.5:
                        ignore_mcts = True

                if not ignore_mcts and mcts_len > 0:
                    raw_target = h_pos + (strategic_dir / mcts_len) * 150.0
                    final_target = self.pathfinder.get_nearest_walkable(raw_target)
                    self.final_targets[agent_id] = final_target

                    should_update = False
                    if not self.current_paths[agent_id]:
                        should_update = True
                    else:
                        current_goal = self.current_paths[agent_id][-1]
                        if np.linalg.norm(final_target - current_goal) > 60.0:
                            should_update = True

                    if should_update:
                        new_path = self.pathfinder.find_path(h_pos, final_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path
            else:
                # Target unobserved: patrol respective map sectors
                if agent_id == 0:
                    patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60), np.random.uniform(60, height * 0.45)])
                else:
                    patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60), np.random.uniform(height * 0.55, height - 60)])
                walkable_target = self.pathfinder.get_nearest_walkable(patrol_pt)
                new_path = self.pathfinder.find_path(h_pos, walkable_target)
                if new_path:
                    self.current_paths[agent_id] = new_path

        if not self.current_paths[agent_id]:
            if is_active:
                walkable_mean = self.pathfinder.get_nearest_walkable(b_mean)
                new_path = self.pathfinder.find_path(h_pos, walkable_mean)
                if new_path:
                    self.current_paths[agent_id] = new_path
            else:
                patrol_pt = np.array([width * 0.5, height * 0.5])
                walkable_pt = self.pathfinder.get_nearest_walkable(patrol_pt)
                new_path = self.pathfinder.find_path(h_pos, walkable_pt)
                if new_path:
                    self.current_paths[agent_id] = new_path

        steer_force = np.array([0.0, 0.0], dtype=np.float32)
        if self.current_paths[agent_id]:
            if np.linalg.norm(h_pos - self.current_paths[agent_id][0]) < 25.0:
                self.current_paths[agent_id].pop(0)

            if self.current_paths[agent_id]:
                desired_dir = self.current_paths[agent_id][0] - h_pos
                norm_dir = np.linalg.norm(desired_dir)
                if norm_dir > 0:
                    desired_vel = (desired_dir / norm_dir) * 4.0
                    steer_force = desired_vel - h_vel

        for obs_item in obstacles:
            lookahead = h_pos + h_vel * 8.0
            if obs_item.collides_point(lookahead[0], lookahead[1], buffer=6.0):
                steer_force += np.array([-h_vel[1], h_vel[0]], dtype=np.float32) * 2.0

        for j in range(len(info['hunter_positions'])):
            if j != agent_id:
                other_pos = info['hunter_positions'][j]
                dist = np.linalg.norm(h_pos - other_pos)
                if 0 < dist < 35.0:
                    steer_force += ((h_pos - other_pos) / dist) * 1.5

        if np.linalg.norm(steer_force) > 0.01:
            return max(range(4), key=lambda a: np.dot(self.action_vectors[a], steer_force))
        return 4

    def _run_mcts(self, h_pos: np.ndarray, h_vel: np.ndarray, b_mean: np.ndarray,
                  width: int, height: int, ally_target: Optional[np.ndarray],
                  particles: Optional[np.ndarray] = None) -> np.ndarray:
        root = MCTSNodeHeuristic()
        
        vel_est = np.array([0.0, 0.0], dtype=np.float32)
        if particles is not None and len(particles) > 0:
            vel_est = np.mean(particles[:, 2:4], axis=0) * 7.0

        for _ in range(self.num_simulations):
            node = root
            i_pos_guess = b_mean + np.random.normal(0, 8.0, size=2)
            i_vel_guess = vel_est + np.random.normal(0, 0.5, size=2).astype(np.float32)

            state = GhostState(h_pos, h_vel, i_pos_guess, i_vel_guess, width, height, ally_target)

            depth = 0
            while node.is_fully_expanded(state, self.action_vectors) and depth < self.max_depth:
                node = node.best_child(self.c_param)
                if node.action_from_parent is not None:
                    r, is_done = state.step(self.action_vectors[node.action_from_parent])
                    if is_done:
                        break
                depth += 1

            if not node.is_fully_expanded(state, self.action_vectors) and depth < self.max_depth:
                new_node, action_idx = node.expand()
                reward, is_done = state.step(self.action_vectors[action_idx])
                node = new_node
            else:
                reward = 0.0
                is_done = False

            rollout_depth = 0
            cumulative_reward = reward
            current_discount = 1.0

            while not is_done and rollout_depth < (self.max_depth - depth):
                if random.random() < 0.80:
                    diff = state.i_pos - state.h_pos
                    scores = [np.dot(self.action_vectors[a], diff) for a in range(4)]
                    action_idx = int(np.argmax(scores))
                else:
                    action_idx = random.randint(0, 3)

                r, is_done = state.step(self.action_vectors[action_idx])
                cumulative_reward += r * current_discount
                current_discount *= self.discount_factor
                rollout_depth += 1

            while node is not None:
                node.visits += 1
                node.value += cumulative_reward
                node = node.parent

        if not root.children:
            return np.array([0.0, 0.0])

        best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
        return self.action_vectors[best_action_idx]

    def _run_mcts_with_distribution(self, h_pos: np.ndarray, h_vel: np.ndarray, b_mean: np.ndarray,
                                   width: int, height: int, ally_target: Optional[np.ndarray],
                                   particles: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        root = MCTSNodeHeuristic()
        
        vel_est = np.array([0.0, 0.0], dtype=np.float32)
        if particles is not None and len(particles) > 0:
            vel_est = np.mean(particles[:, 2:4], axis=0) * 7.0

        for _ in range(self.num_simulations):
            node = root
            i_pos_guess = b_mean + np.random.normal(0, 8.0, size=2)
            i_vel_guess = vel_est + np.random.normal(0, 0.5, size=2).astype(np.float32)

            state = GhostState(h_pos, h_vel, i_pos_guess, i_vel_guess, width, height, ally_target)

            depth = 0
            while node.is_fully_expanded(state, self.action_vectors) and depth < self.max_depth:
                node = node.best_child(self.c_param)
                if node.action_from_parent is not None:
                    r, is_done = state.step(self.action_vectors[node.action_from_parent])
                    if is_done:
                        break
                depth += 1

            if not node.is_fully_expanded(state, self.action_vectors) and depth < self.max_depth:
                new_node, action_idx = node.expand()
                reward, is_done = state.step(self.action_vectors[action_idx])
                node = new_node
            else:
                reward = 0.0
                is_done = False

            rollout_depth = 0
            cumulative_reward = reward
            current_discount = 1.0

            while not is_done and rollout_depth < (self.max_depth - depth):
                if random.random() < 0.80:
                    diff = state.i_pos - state.h_pos
                    scores = [np.dot(self.action_vectors[a], diff) for a in range(4)]
                    action_idx = int(np.argmax(scores))
                else:
                    action_idx = random.randint(0, 3)

                r, is_done = state.step(self.action_vectors[action_idx])
                cumulative_reward += r * current_discount
                current_discount *= self.discount_factor
                rollout_depth += 1

            while node is not None:
                node.visits += 1
                node.value += cumulative_reward
                node = node.parent

        visits = np.zeros(len(self.action_vectors), dtype=np.float32)
        for act_idx, child in root.children.items():
            visits[act_idx] = float(child.visits)

        total_visits = float(np.sum(visits))
        if total_visits > 0:
            probs = visits / total_visits
        else:
            probs = np.full(len(self.action_vectors), 1.0 / len(self.action_vectors), dtype=np.float32)

        if not root.children:
            return np.array([0.0, 0.0]), probs

        best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
        return self.action_vectors[best_action_idx], probs
