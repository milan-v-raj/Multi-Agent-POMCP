import math
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import pygame
from .base_policy import BasePursuerPolicy
from ..scenarios import Obstacle
from ..pathfinder import Pathfinder
from ..nets.policy_value_net import DeepPOMCPNet

class MCTSNodeDeep:
    def __init__(self, parent=None, action_from_parent: Optional[int] = None, prior: float = 0.2):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = prior
        self.children: Dict[int, MCTSNodeDeep] = {}
        self.visits = 0
        self.total_value = 0.0
        self.is_terminal = False

    @property
    def q_value(self) -> float:
        return (self.total_value / self.visits) if self.visits > 0 else 0.0

    def select_puct_child(self, c_puct: float = 1.5, default_q: float = 0.0) -> Tuple[int, 'MCTSNodeDeep']:
        total_visits_sqrt = math.sqrt(max(1, self.visits))
        visited_q = [child.q_value for child in self.children.values() if child.visits > 0]
        q_min = min(visited_q) if visited_q else default_q
        q_max = max(visited_q) if visited_q else default_q
        q_range = max(q_max - q_min, 1e-4)

        best_score = -float('inf')
        best_action = 0
        best_child = None

        for action_idx, child in self.children.items():
            if child.visits > 0:
                norm_q = (child.q_value - q_min) / q_range
            else:
                norm_q = (default_q - q_min) / q_range

            u_score = c_puct * child.prior * (total_visits_sqrt / (1.0 + child.visits))
            score = norm_q + u_score
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        return best_action, best_child


class DeepSimulationState:
    def __init__(self, h_pos: np.ndarray, h_vel: np.ndarray, i_pos: np.ndarray, i_vel: np.ndarray,
                 width: int, height: int, ally_pos: np.ndarray, ally_target: Optional[np.ndarray] = None):
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.i_pos = np.copy(i_pos)
        self.i_vel = np.copy(i_vel)
        self.width = width
        self.height = height
        self.ally_pos = np.copy(ally_pos)
        self.ally_target = ally_target

    def step(self, action_vec: np.ndarray) -> Tuple[float, bool]:
        self.h_vel += action_vec
        self.h_vel *= 0.95
        speed = np.linalg.norm(self.h_vel)
        if speed > 5.0:
            self.h_vel = (self.h_vel / speed) * 5.0
        self.h_pos += self.h_vel
        self.i_pos += self.i_vel

        hit_wall = not (20 < self.h_pos[0] < self.width - 20 and 20 < self.h_pos[1] < self.height - 20)

        dist = np.linalg.norm(self.h_pos - self.i_pos)
        if dist < 30.0:
            return 1.0, True

        reward = -(dist / 400.0)
        if hit_wall:
            reward -= 0.5

        if self.ally_target is not None:
            dist_to_ally = np.linalg.norm(self.h_pos - self.ally_target)
            if dist_to_ally < 100.0:
                reward -= (100.0 - dist_to_ally) * 0.005

        return reward, False

    def step_multi(self, action_vec: np.ndarray, frames: int = 5) -> Tuple[float, bool]:
        cum_reward = 0.0
        for _ in range(frames):
            r, is_done = self.step(action_vec)
            cum_reward += r
            if is_done:
                return 10.0, True
        return cum_reward, False


import os

class DeepPOMCPPolicy(BasePursuerPolicy):
    def __init__(self, net: Optional[DeepPOMCPNet] = None,
                 weights_path: Optional[str] = "deep_pomcp_weights.pth",
                 num_simulations: int = 40,
                 max_depth: int = 6,
                 c_puct: float = 1.5,
                 name: str = 'Deep-POMCP (Ours)'):
        super().__init__(name)
        self.net = net if net is not None else DeepPOMCPNet()
        
        # Load weights if available
        if weights_path and os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location="cpu")
                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                self.net.load_state_dict(state_dict)
                print(f"[{name}] Successfully loaded neural weights from '{weights_path}'")
            except Exception as e:
                print(f"[{name}] Warning: Failed to load weights from '{weights_path}': {e}")

        self.net.eval()
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
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

        plan_offset = 0 if agent_id == 0 else 15
        if (step + plan_offset) % 30 == 0:
            other_id = 1 if agent_id == 0 else 0
            ally_pos = info['hunter_positions'][other_id]
            ally_tgt = self.final_targets.get(other_id, None)

            strategic_dir = self._run_deep_mcts(obs, h_pos, h_vel, ally_pos, b_mean, width, height, ally_tgt, particles=particles)

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

        if not self.current_paths[agent_id]:
            walkable_mean = self.pathfinder.get_nearest_walkable(b_mean)
            new_path = self.pathfinder.find_path(h_pos, walkable_mean)
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

    def _run_deep_mcts(self, obs: np.ndarray, h_pos: np.ndarray, h_vel: np.ndarray,
                       ally_pos: np.ndarray, b_mean: np.ndarray,
                       width: int, height: int, ally_target: Optional[np.ndarray] = None,
                       particles: Optional[np.ndarray] = None) -> np.ndarray:
        kinematics_tensor = torch.FloatTensor(obs[0:8]).unsqueeze(0)
        local_grid_tensor = torch.FloatTensor(obs[15:136]).unsqueeze(0)
        
        if particles is not None and len(particles) == 200:
            particles_tensor = torch.FloatTensor(particles).unsqueeze(0)
        else:
            dummy_particles = np.zeros((200, 4), dtype=np.float32)
            dummy_particles[:, 0:2] = (b_mean + np.random.normal(0, 8.0, size=(200, 2))) / np.array([width, height])
            particles_tensor = torch.FloatTensor(dummy_particles).unsqueeze(0)

        priors, root_value = self.net.predict_priors_and_value(particles_tensor, kinematics_tensor, local_grid_tensor)

        root = MCTSNodeDeep(prior=1.0)
        for a_idx in range(len(self.action_vectors)):
            root.children[a_idx] = MCTSNodeDeep(parent=root, action_from_parent=a_idx, prior=float(priors[a_idx].item()))

        for _ in range(self.num_simulations):
            node = root
            i_pos_guess = b_mean + np.random.normal(0, 8.0, size=2)
            i_vel_guess = np.array([0.0, 0.0], dtype=np.float32)

            state = DeepSimulationState(h_pos, h_vel, i_pos_guess, i_vel_guess, width, height, ally_pos, ally_target)
            initial_dist = np.linalg.norm(h_pos - i_pos_guess)

            depth = 0
            search_path = [node]
            cum_reward = 0.0

            while len(node.children) > 0 and depth < self.max_depth:
                action_idx, child_node = node.select_puct_child(self.c_puct, default_q=root_value)
                r, is_terminal = state.step_multi(self.action_vectors[action_idx], frames=5)
                cum_reward += r * (self.discount_factor ** depth)
                node = child_node
                search_path.append(node)
                depth += 1
                if is_terminal:
                    node.is_terminal = True
                    break

            if not node.is_terminal and depth < self.max_depth and len(node.children) == 0:
                for a_idx in range(len(self.action_vectors)):
                    node.children[a_idx] = MCTSNodeDeep(parent=node, action_from_parent=a_idx, prior=float(priors[a_idx].item()))

            if node.is_terminal:
                leaf_value = 10.0
            else:
                final_dist = np.linalg.norm(state.h_pos - state.i_pos)
                progress = (initial_dist - final_dist) / 10.0
                leaf_value = cum_reward + progress + root_value * 2.0

            for visited_node in reversed(search_path):
                visited_node.visits += 1
                visited_node.total_value += leaf_value
                leaf_value *= self.discount_factor

        # Select the directional action with the highest visit count
        directional_actions = [a for a in range(4) if a in root.children]
        if directional_actions:
            best_act = max(directional_actions, key=lambda a: root.children[a].visits)
        else:
            best_act = max(root.children, key=lambda a: root.children[a].visits)

        return self.action_vectors[best_act]
