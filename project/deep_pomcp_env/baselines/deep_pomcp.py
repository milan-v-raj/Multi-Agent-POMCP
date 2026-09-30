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

    def select_puct_child(self, c_puct: float = 1.5, default_q: float = 0.0) -> Tuple[int, "MCTSNodeDeep"]:
        total_visits_sqrt = math.sqrt(max(1, self.visits))
        visited_q = [child.q_value for child in self.children.values() if child.visits > 0]
        q_min = min(visited_q) if visited_q else default_q
        q_max = max(visited_q) if visited_q else default_q
        q_range = max(q_max - q_min, 1e-4)

        best_score = -float("inf")
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


# ---------------------------------------------------------------------------
# A1: Adversarial evader simulation state
# The simulated evader now responds to both hunters using a lightweight
# 12-ray flee sweep, at half the real evader's force (0.25 vs 0.50).
# Reference: "Know your Enemy" (arXiv 2305.13206) — opponent models in MCTS
# improve performance in tactical pursuit games.
# ---------------------------------------------------------------------------
class DeepSimulationState:
    def __init__(self, h_pos: np.ndarray, h_vel: np.ndarray,
                 i_pos: np.ndarray, i_vel: np.ndarray,
                 width: int, height: int,
                 ally_pos: np.ndarray,
                 ally_target: Optional[np.ndarray] = None,
                 pincer_dist: float = 200.0):
        self.h_pos      = np.copy(h_pos)
        self.h_vel      = np.copy(h_vel)
        self.i_pos      = np.copy(i_pos)
        self.i_vel      = np.copy(i_vel)
        self.width      = width
        self.height     = height
        self.ally_pos   = np.copy(ally_pos) if ally_pos is not None else None
        self.ally_target = np.copy(ally_target) if ally_target is not None else None
        # A2 hyperparameter: distance threshold to switch from spread to pincer phase
        self.pincer_dist = pincer_dist

    def _sim_evader_flee(self, max_force: float = 0.25, n_rays: int = 12) -> np.ndarray:
        """A1: Lightweight adversarial flee force for the simulated evader.
        Uses the committed ally waypoint (ally_target) rather than current ally
        position — that is the macro-level intention the evader should anticipate.
        Half-strength (0.25) avoids a pessimistic tree where every action looks bad.
        """
        # Use committed waypoint as the reference for the ally hunter
        ally_ref = self.ally_target if self.ally_target is not None else self.ally_pos
        best_score = -1e9
        best_dir   = np.array([1.0, 0.0], dtype=np.float32)

        for k in range(n_rays):
            ang = (2.0 * math.pi * k) / n_rays
            d  = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
            tp = self.i_pos + d * 40.0

            # Boundary clearance
            if not (20 < tp[0] < self.width - 20 and 20 < tp[1] < self.height - 20):
                continue

            # Score: maximize distance from active hunter
            score = np.linalg.norm(tp - self.h_pos)
            # Also flee from ally (committed waypoint)
            if ally_ref is not None:
                score += np.linalg.norm(tp - ally_ref) * 0.7

            if score > best_score:
                best_score = score
                best_dir   = d

        return best_dir * max_force

    def step(self, action_vec: np.ndarray) -> Tuple[float, bool]:
        # Hunter physics
        self.h_vel += action_vec
        self.h_vel *= 0.95
        speed = np.linalg.norm(self.h_vel)
        if speed > 5.0:
            self.h_vel = (self.h_vel / speed) * 5.0
        self.h_pos += self.h_vel

        # A1: Adversarial evader response (replaces static i_pos += i_vel)
        flee = self._sim_evader_flee()
        self.i_vel = self.i_vel * 0.95 + flee
        i_spd = np.linalg.norm(self.i_vel)
        if i_spd > 7.0:
            self.i_vel = (self.i_vel / i_spd) * 7.0
        self.i_pos += self.i_vel

        hit_wall = not (20 < self.h_pos[0] < self.width - 20
                        and 20 < self.h_pos[1] < self.height - 20)

        dist = np.linalg.norm(self.h_pos - self.i_pos)
        if dist < 30.0:
            return 500.0, True

        reward = -(dist / 100.0) - 0.2
        if hit_wall:
            reward -= 10.0

        # A2: Encirclement Advantage (EA) metric reward
        # Phase-switch at pincer_dist: in close range, reward 180-degree angular
        # separation; at long range, penalise hunters being too close together.
        # Fast implementation uses dot-product gate (cos_sep < 0 ↔ angle > 90°)
        # to avoid arccos inside the hot MCTS loop.
        ally_ref = self.ally_target if self.ally_target is not None else self.ally_pos
        if ally_ref is not None:
            if dist < self.pincer_dist:
                # Pincer phase: reward angular encirclement
                v1 = self.h_pos - self.i_pos
                v2 = ally_ref  - self.i_pos
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_sep = np.dot(v1, v2) / (n1 * n2)
                    if cos_sep < 0:  # hunters are more than 90 deg apart
                        reward += (-cos_sep) * 2.0  # 0..2, max at 180 deg
            else:
                # Spread phase: penalise hunters crowding each other
                dist_to_ally = np.linalg.norm(self.h_pos - ally_ref)
                if dist_to_ally < 100.0:
                    reward -= (100.0 - dist_to_ally) * 0.05

        return reward, False

    def step_multi(self, action_vec: np.ndarray, frames: int = 5) -> Tuple[float, bool]:
        cum_reward = 0.0
        for _ in range(frames):
            r, is_done = self.step(action_vec)
            cum_reward += r
            if is_done:
                return r, True
        return cum_reward, False


import os

class DeepPOMCPPolicy(BasePursuerPolicy):
    def __init__(self, net: Optional[DeepPOMCPNet] = None,
                 weights_path: Optional[str] = "deep_pomcp_weights.pth",
                 num_simulations: int = 60,
                 max_depth: int = 6,
                 c_puct: float = 1.5,
                 # A2 hyperparameter: pincer phase threshold (px)
                 pincer_dist: float = 200.0,
                 # A3 hyperparameters: blind-period adaptive budget
                 blind_threshold: int = 80,
                 blind_sim_multiplier: float = 2.0,
                 name: str = "Deep-POMCP (Ours)"):
        super().__init__(name)
        self.net = net if net is not None else DeepPOMCPNet()

        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.net.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.net.load_state_dict(checkpoint)
                print(f"[{name}] Successfully loaded neural weights from '{weights_path}'")
            except Exception as e:
                print(f"[{name}] Warning: Failed to load weights: {e}")

        self.net.eval()
        self.num_simulations    = num_simulations
        self.max_depth          = max_depth
        self.c_puct             = c_puct
        self.pincer_dist        = pincer_dist
        self.blind_threshold    = blind_threshold
        self.blind_sim_multiplier = blind_sim_multiplier
        self.discount_factor    = 0.95

        self.action_vectors = [
            np.array([0.0, -0.5], dtype=np.float32),
            np.array([0.0,  0.5], dtype=np.float32),
            np.array([-0.5, 0.0], dtype=np.float32),
            np.array([0.5,  0.0], dtype=np.float32),
            np.array([0.0,  0.0], dtype=np.float32)
        ]
        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}
        self.final_targets: Dict[int, np.ndarray] = {}
        # A3: per-agent blind-step counters (reset on LOS recovery)
        self._blind_steps: Dict[int, int] = {}

    def reset(self):
        self.current_paths.clear()
        self.final_targets.clear()
        self._blind_steps.clear()

    def get_action(self, obs: np.ndarray, info: Dict[str, Any],
                   agent_id: int, obstacles: List[Obstacle],
                   width: int, height: int) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos     = info["hunter_positions"][agent_id]
        h_vel     = info["hunter_velocities"][agent_id]
        b_mean    = info["belief_mean"]
        step      = info.get("current_step", 0)
        particles = info.get("particles", None)
        is_active = info.get("is_belief_active", False) and (b_mean is not None)

        if agent_id not in self.current_paths:
            self.current_paths[agent_id] = []

        # A3: update blind-step counter for this agent
        if is_active:
            self._blind_steps[agent_id] = 0          # LOS recovered, reset
        else:
            self._blind_steps[agent_id] = self._blind_steps.get(agent_id, 0) + 1

        plan_offset = 0 if agent_id == 0 else 7
        switch_pos = info.get('switch_pos', None)
        gate_open = info.get('gate_open', True)

        # Environmental Affordance & Cooperative Role Specialization (Switch-Door)
        is_holding_switch = False
        if switch_pos is not None:
            other_id = 1 if agent_id == 0 else 0
            other_pos = info['hunter_positions'][other_id]
            d_self = np.linalg.norm(h_pos - switch_pos)
            d_other = np.linalg.norm(other_pos - switch_pos)
            
            # Operator Role: Move to switch and maintain hold until ally breaches
            if d_self <= d_other:
                ally_breached = (other_pos[0] > 600.0)
                if not ally_breached:
                    is_holding_switch = True
                    target_pt = switch_pos
                    walkable_target = self.pathfinder.get_nearest_walkable(target_pt)
                    self.final_targets[agent_id] = walkable_target
                    if (step + plan_offset) % 10 == 0 or len(self.current_paths[agent_id]) == 0:
                        new_path = self.pathfinder.find_path(h_pos, walkable_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path
            else:
                # Breacher Role: Stage at gate when locked, breach when open
                if not gate_open:
                    is_holding_switch = True
                    gate_obs = info.get('gate_obstacle', None)
                    if gate_obs is not None:
                        target_pt = np.array([gate_obs.x - 50.0, gate_obs.y + gate_obs.height / 2.0], dtype=np.float32)
                    else:
                        target_pt = np.array([520.0, height / 2.0], dtype=np.float32)
                    walkable_target = self.pathfinder.get_nearest_walkable(target_pt)
                    self.final_targets[agent_id] = walkable_target
                    if (step + plan_offset) % 10 == 0 or len(self.current_paths[agent_id]) == 0:
                        new_path = self.pathfinder.find_path(h_pos, walkable_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path

        if (step + plan_offset) % 15 == 0 and not is_holding_switch:
            if is_active:
                other_id  = 1 if agent_id == 0 else 0
                ally_pos  = info["hunter_positions"][other_id]
                ally_tgt  = self.final_targets.get(other_id, None)

                # A3: adaptive simulation budget
                blind_steps = self._blind_steps.get(agent_id, 0)
                n_sims = int(self.num_simulations * self.blind_sim_multiplier) \
                         if blind_steps > self.blind_threshold \
                         else self.num_simulations

                strategic_dir = self._run_deep_mcts(
                    obs, h_pos, h_vel, ally_pos, b_mean,
                    width, height, ally_tgt,
                    particles=particles, n_sims=n_sims
                )

                ignore_mcts = False
                speed    = np.linalg.norm(h_vel)
                mcts_len = np.linalg.norm(strategic_dir)
                if speed > 2.0 and mcts_len > 0:
                    alignment = np.dot(h_vel / speed, strategic_dir / mcts_len)
                    if alignment < -0.5:
                        ignore_mcts = True

                if not ignore_mcts and mcts_len > 0:
                    dist_to_belief = np.linalg.norm(h_pos - b_mean)
                    proj_dist = min(150.0, max(50.0, dist_to_belief * 1.15))
                    raw_target   = h_pos + (strategic_dir / mcts_len) * proj_dist
                    final_target = self.pathfinder.get_nearest_walkable(raw_target)
                    self.final_targets[agent_id] = final_target

                    should_update = False
                    if not self.current_paths[agent_id]:
                        should_update = True
                    else:
                        current_goal = self.current_paths[agent_id][-1]
                        if np.linalg.norm(final_target - current_goal) > 40.0:
                            should_update = True

                    if should_update:
                        new_path = self.pathfinder.find_path(h_pos, final_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path
            else:
                if agent_id == 0:
                    patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60),
                                          np.random.uniform(60, height * 0.45)])
                else:
                    patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60),
                                          np.random.uniform(height * 0.55, height - 60)])
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
                    desired_vel = (desired_dir / norm_dir) * 4.8
                    steer_force = desired_vel - h_vel

        for obs_item in obstacles:
            lookahead = h_pos + h_vel * 8.0
            if obs_item.collides_point(lookahead[0], lookahead[1], buffer=6.0):
                steer_force += np.array([-h_vel[1], h_vel[0]], dtype=np.float32) * 2.0

        for j in range(len(info["hunter_positions"])):
            if j != agent_id:
                other_pos = info["hunter_positions"][j]
                dist = np.linalg.norm(h_pos - other_pos)
                if 0 < dist < 35.0:
                    steer_force += ((h_pos - other_pos) / dist) * 1.5

        if np.linalg.norm(steer_force) > 0.01:
            return max(range(4), key=lambda a: np.dot(self.action_vectors[a], steer_force))
        return 4

    def _run_deep_mcts(self, obs: np.ndarray, h_pos: np.ndarray, h_vel: np.ndarray,
                       ally_pos: np.ndarray, b_mean: np.ndarray,
                       width: int, height: int,
                       ally_target: Optional[np.ndarray] = None,
                       particles: Optional[np.ndarray] = None,
                       n_sims: Optional[int] = None) -> np.ndarray:
        if n_sims is None:
            n_sims = self.num_simulations

        kinematics_tensor  = torch.FloatTensor(obs[0:8]).unsqueeze(0)
        local_grid_tensor  = torch.FloatTensor(obs[15:136]).unsqueeze(0)

        if particles is not None and len(particles) == 200:
            particles_tensor = torch.FloatTensor(particles).unsqueeze(0)
        else:
            dummy_particles = np.zeros((200, 4), dtype=np.float32)
            dummy_particles[:, 0:2] = (
                b_mean + np.random.normal(0, 8.0, size=(200, 2))
            ) / np.array([width, height])
            particles_tensor = torch.FloatTensor(dummy_particles).unsqueeze(0)

        priors, root_value = self.net.predict_priors_and_value(
            particles_tensor, kinematics_tensor, local_grid_tensor
        )
        root_priors = priors.cpu().numpy()

        root = MCTSNodeDeep(prior=1.0)
        for a_idx in range(len(self.action_vectors)):
            root.children[a_idx] = MCTSNodeDeep(
                parent=root, action_from_parent=a_idx,
                prior=float(root_priors[a_idx])
            )

        vel_est = np.array([0.0, 0.0], dtype=np.float32)
        if particles is not None and len(particles) > 0:
            vel_est = np.mean(particles[:, 2:4], axis=0) * 7.0

        for _ in range(n_sims):
            node = root
            i_pos_guess = b_mean + np.random.normal(0, 8.0, size=2)
            i_vel_guess = vel_est + np.random.normal(0, 0.5, size=2).astype(np.float32)

            state = DeepSimulationState(
                h_pos, h_vel, i_pos_guess, i_vel_guess,
                width, height, ally_pos, ally_target,
                pincer_dist=self.pincer_dist
            )
            initial_dist = np.linalg.norm(h_pos - i_pos_guess)

            depth       = 0
            search_path = [node]
            cum_reward  = 0.0

            while len(node.children) > 0 and depth < self.max_depth:
                action_idx, child_node = node.select_puct_child(
                    self.c_puct, default_q=root_value
                )
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
                    node.children[a_idx] = MCTSNodeDeep(
                        parent=node, action_from_parent=a_idx,
                        prior=float(priors[a_idx].item())
                    )

            if node.is_terminal:
                leaf_value = cum_reward
            else:
                final_dist = np.linalg.norm(state.h_pos - state.i_pos)
                progress   = (initial_dist - final_dist) / 10.0
                leaf_value = cum_reward + progress + root_value * 5.0

            for visited_node in reversed(search_path):
                visited_node.visits      += 1
                visited_node.total_value += leaf_value
                leaf_value *= self.discount_factor

        directional_actions = [a for a in range(4) if a in root.children]
        if directional_actions:
            best_act = max(directional_actions,
                           key=lambda a: root.children[a].visits)
        else:
            best_act = max(root.children, key=lambda a: root.children[a].visits)

        return self.action_vectors[best_act]
