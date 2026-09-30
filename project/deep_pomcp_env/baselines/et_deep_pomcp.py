import os
"""
Information-Theoretic Event-Triggered Deep-POMCP (ET-MCTS).
Replaces rigid periodic clock-based replanning (t % 15 == 0) with an asynchronous
Information-Theoretic Event Engine that triggers MCTS lookahead on:
  1. Belief Entropy / Spread Flux Events (|ΔH(B_t)| > τ)
  2. Observation State Transition Events (LOS locked <-> lost)
  3. Geometric Path Invalidation / Blocked Waypoints
  4. Bounded Safety Horizon Watchdogs (Max idle limit)
"""

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
from .deep_pomcp import MCTSNodeDeep, DeepSimulationState

class EventTriggeredDeepPOMCPPolicy(BasePursuerPolicy):
    def __init__(self,
                 net: Optional[DeepPOMCPNet] = None,
                 weights_path: Optional[str] = "deep_pomcp_weights.pth",
                 num_simulations: int = 60,
                 max_depth: int = 6,
                 c_puct: float = 1.5,
                 pincer_dist: float = 200.0,
                 # Event Engine Hyperparameters
                 entropy_threshold: float = 22.0,   # ΔSpread in px triggering replan
                 min_cooldown: int = 5,             # Minimum frames between MCTS dispatches
                 max_watchdog: int = 35,            # Max frames before forced sync replan
                 name: str = "ET-Deep-POMCP (Ours)"):
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
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.pincer_dist = pincer_dist
        self.discount_factor = 0.95

        # Event-Triggering Hyperparameters
        self.entropy_threshold = entropy_threshold
        self.min_cooldown = min_cooldown
        self.max_watchdog = max_watchdog

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

        # Telemetry & Event Tracking
        self._last_plan_step: Dict[int, int] = {}
        self._last_belief_spread: Dict[int, float] = {}
        self._last_los: Dict[int, bool] = {}
        self._last_trigger_reason: Dict[int, str] = {}
        self.total_mcts_calls = 0
        self.event_trigger_counts: Dict[str, int] = {
            "ENTROPY_FLUX": 0,
            "LOS_TRANSITION": 0,
            "PATH_INVALID": 0,
            "WATCHDOG_SYNC": 0,
            "INITIAL_PLAN": 0
        }

    def reset(self):
        self.current_paths.clear()
        self.final_targets.clear()
        self._last_plan_step.clear()
        self._last_belief_spread.clear()
        self._last_los.clear()
        self._last_trigger_reason.clear()
        self.total_mcts_calls = 0
        for k in self.event_trigger_counts:
            self.event_trigger_counts[k] = 0

    def _evaluate_event_trigger(self, agent_id: int, b_spread: float, can_see: bool, obstacles: List[Obstacle], step: int) -> Tuple[bool, str]:
        last_step = self._last_plan_step.get(agent_id, -999)
        steps_since = step - last_step

        # Cooldown guard: Inhibit rapid back-to-back MCTS calls
        if steps_since < self.min_cooldown:
            return False, "COOLDOWN_HOLD"

        # 1. Initial Planning / No Path
        if not self.current_paths.get(agent_id):
            return True, "INITIAL_PLAN"

        # 2. Observation State Transition Event
        prev_los = self._last_los.get(agent_id, can_see)
        if can_see != prev_los:
            return True, "LOS_TRANSITION"

        # 3. Information Surprise / Belief Entropy Flux Event
        prev_spread = self._last_belief_spread.get(agent_id, b_spread)
        delta_spread = abs(b_spread - prev_spread)
        if delta_spread > self.entropy_threshold:
            return True, "ENTROPY_FLUX"

        # 4. Path Invalidation / Waypoint Blocked Event
        path = self.current_paths.get(agent_id, [])
        if len(path) <= 1:
            return True, "PATH_INVALID"
        for pt in path[:4]:
            for obs in obstacles:
                if obs.collides_point(pt[0], pt[1], buffer=6.0):
                    return True, "PATH_INVALID"

        # 5. Bounded Safety Horizon Watchdog
        if steps_since >= self.max_watchdog:
            return True, "WATCHDOG_SYNC"

        return False, "IDLE_TRACKING"

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int, obstacles: List[Obstacle], width: int, height: int) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos     = info["hunter_positions"][agent_id]
        h_vel     = info["hunter_velocities"][agent_id]
        b_mean    = info["belief_mean"]
        b_spread  = info.get("belief_spread", 0.0)
        can_see   = info.get("can_see_global", False)
        step      = info.get("current_step", 0)
        particles = info.get("particles", None)
        is_active = info.get("is_belief_active", False) and (b_mean is not None)

        if agent_id not in self.current_paths:
            self.current_paths[agent_id] = []

        # Cooperative Switch-Door Affordance Check
        switch_pos = info.get("switch_pos", None)
        gate_open = info.get("gate_open", True)
        is_holding_switch = False

        if switch_pos is not None:
            other_id = 1 if agent_id == 0 else 0
            other_pos = info["hunter_positions"][other_id]
            d_self = np.linalg.norm(h_pos - switch_pos)
            d_other = np.linalg.norm(other_pos - switch_pos)

            if d_self <= d_other:
                ally_breached = (other_pos[0] > 600.0)
                if not ally_breached:
                    is_holding_switch = True
                    target_pt = switch_pos
                    walkable_target = self.pathfinder.get_nearest_walkable(target_pt)
                    self.final_targets[agent_id] = walkable_target
                    if (step % 10 == 0) or len(self.current_paths[agent_id]) == 0:
                        new_path = self.pathfinder.find_path(h_pos, walkable_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path
            else:
                if not gate_open:
                    is_holding_switch = True
                    gate_obs = info.get("gate_obstacle", None)
                    if gate_obs is not None:
                        target_pt = np.array([gate_obs.x - 50.0, gate_obs.y + gate_obs.height / 2.0], dtype=np.float32)
                    else:
                        target_pt = np.array([520.0, height / 2.0], dtype=np.float32)
                    walkable_target = self.pathfinder.get_nearest_walkable(target_pt)
                    self.final_targets[agent_id] = walkable_target
                    if (step % 10 == 0) or len(self.current_paths[agent_id]) == 0:
                        new_path = self.pathfinder.find_path(h_pos, walkable_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path

        # Event-Triggered MCTS Dispatch
        if not is_holding_switch:
            should_replan, trigger_reason = self._evaluate_event_trigger(agent_id, b_spread, can_see, obstacles, step)
            self._last_trigger_reason[agent_id] = trigger_reason

            if should_replan:
                self._last_plan_step[agent_id] = step
                self.total_mcts_calls += 1
                if trigger_reason in self.event_trigger_counts:
                    self.event_trigger_counts[trigger_reason] += 1

                if is_active:
                    other_id = 1 if agent_id == 0 else 0
                    ally_pos = info["hunter_positions"][other_id]
                    ally_tgt = self.final_targets.get(other_id, None)

                    strategic_dir = self._run_deep_mcts(
                        obs, h_pos, h_vel, ally_pos, b_mean,
                        width, height, ally_tgt,
                        particles=particles, n_sims=self.num_simulations
                    )

                    mcts_len = np.linalg.norm(strategic_dir)
                    if mcts_len > 0:
                        dist_to_belief = np.linalg.norm(h_pos - b_mean)
                        proj_dist = min(150.0, max(50.0, dist_to_belief * 1.15))
                        raw_target = h_pos + (strategic_dir / mcts_len) * proj_dist
                        final_target = self.pathfinder.get_nearest_walkable(raw_target)
                        self.final_targets[agent_id] = final_target

                        new_path = self.pathfinder.find_path(h_pos, final_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path
                else:
                    # Patrol Search Mode
                    if agent_id == 0:
                        patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60), np.random.uniform(60, height * 0.45)])
                    else:
                        patrol_pt = np.array([np.random.uniform(width * 0.4, width - 60), np.random.uniform(height * 0.55, height - 60)])
                    walkable_target = self.pathfinder.get_nearest_walkable(patrol_pt)
                    new_path = self.pathfinder.find_path(h_pos, walkable_target)
                    if new_path:
                        self.current_paths[agent_id] = new_path

        # Update telemetry history
        self._last_belief_spread[agent_id] = b_spread
        self._last_los[agent_id] = can_see

        if not self.current_paths[agent_id]:
            if is_active:
                walkable_mean = self.pathfinder.get_nearest_walkable(b_mean)
                new_path = self.pathfinder.find_path(h_pos, walkable_mean)
                if new_path:
                    self.current_paths[agent_id] = new_path

        # Low-Level A* Path Tracking & Continuous Steering
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

        # Obstacle avoidance buffer
        for obs_item in obstacles:
            lookahead = h_pos + h_vel * 8.0
            if obs_item.collides_point(lookahead[0], lookahead[1], buffer=6.0):
                steer_force += np.array([-h_vel[1], h_vel[0]], dtype=np.float32) * 2.0

        # Ally separation buffer
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

        kinematics_tensor = torch.FloatTensor(obs[0:8]).unsqueeze(0)
        local_grid_tensor = torch.FloatTensor(obs[15:136]).unsqueeze(0)

        if particles is not None and len(particles) == 200:
            particles_tensor = torch.FloatTensor(particles).unsqueeze(0)
        else:
            dummy_particles = np.zeros((200, 4), dtype=np.float32)
            dummy_particles[:, 0:2] = (b_mean + np.random.normal(0, 8.0, size=(200, 2))) / np.array([width, height])
            particles_tensor = torch.FloatTensor(dummy_particles).unsqueeze(0)

        priors, root_value = self.net.predict_priors_and_value(
            particles_tensor, kinematics_tensor, local_grid_tensor
        )
        root_priors = priors.cpu().numpy()

        root = MCTSNodeDeep(prior=1.0)
        for a_idx in range(len(self.action_vectors)):
            root.children[a_idx] = MCTSNodeDeep(parent=root, action_from_parent=a_idx, prior=float(root_priors[a_idx]))

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
                leaf_value = cum_reward
            else:
                final_dist = np.linalg.norm(state.h_pos - state.i_pos)
                progress = (initial_dist - final_dist) / 10.0
                leaf_value = cum_reward + progress + root_value * 5.0

            for visited_node in reversed(search_path):
                visited_node.visits += 1
                visited_node.total_value += leaf_value
                leaf_value *= self.discount_factor

        directional_actions = [a for a in range(4) if a in root.children]
        if directional_actions:
            best_act = max(directional_actions, key=lambda a: root.children[a].visits)
        else:
            best_act = max(root.children, key=lambda a: root.children[a].visits)

        return self.action_vectors[best_act]
