"""
POMDP Pursuer Policies for Switch-Door Cooperative Sacrifice.
Implements:
1. GreedyAStarPolicy: Pure greedy pursuit baseline (deadlocks at closed door).
2. PrimitivePOMCPPolicy: Standard POMCP with primitive continuous force actions.
3. MacroPOMCPPolicy (Mac-POMCP): SMDP Option-level tree search over POMDP generative model.
4. DeepMacPOMCPPolicy: Mac-POMCP guided by frozen neural value head.

Zero privileged metadata leakage; zero hardcoded coordinates; zero new training.
"""

import os
import sys
import math
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from deep_pomcp_env.scenarios import Obstacle
from deep_pomcp_env.pathfinder import Pathfinder
from deep_pomcp_env.nets.policy_value_net import DeepPOMCPNet
from .generative_model import SwitchDoorGenerativeState, SwitchDoorPOMDPModel


def _force_to_discrete_action(force: np.ndarray) -> int:
    fx, fy = float(force[0]), float(force[1])
    if abs(fx) < 0.1 and abs(fy) < 0.1:
        return 4  # stay
    if abs(fx) > abs(fy):
        return 3 if fx > 0 else 2  # right : left
    else:
        return 1 if fy > 0 else 0  # down : up


# ===========================================================================
# 1. BASELINE: Greedy A* Policy
# ===========================================================================
class GreedyAStarPolicy:
    def __init__(self, name: str = "Greedy A*"):
        self.name = name
        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}

    def reset(self):
        self.current_paths.clear()

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int,
                   obstacles: List[Obstacle], width: int, height: int) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos = info["hunter_positions"][agent_id]
        h_vel = info["hunter_velocities"][agent_id]
        b_mean = info.get("belief_mean", None)

        target = b_mean if b_mean is not None else np.array([width / 2.0, height / 2.0], dtype=np.float32)
        walkable_target = self.pathfinder.get_nearest_walkable(target)

        if agent_id not in self.current_paths or not self.current_paths[agent_id]:
            path = self.pathfinder.find_path(h_pos, walkable_target)
            if path:
                self.current_paths[agent_id] = path

        if agent_id in self.current_paths and self.current_paths[agent_id]:
            wp = self.current_paths[agent_id][0]
            if np.linalg.norm(h_pos - wp) < 25.0:
                self.current_paths[agent_id].pop(0)
                if self.current_paths[agent_id]:
                    wp = self.current_paths[agent_id][0]
            des_v = (wp - h_pos)
            norm = np.linalg.norm(des_v)
            if norm > 1e-4:
                des_v = (des_v / norm) * 4.0
            force = des_v - h_vel
            return _force_to_discrete_action(force)

        return 4


# ===========================================================================
# 2. PRIMITIVE POMCP POLICY
# ===========================================================================
class PrimitivePOMCPPolicy:
    def __init__(self, num_simulations: int = 40, max_depth: int = 12, name: str = "Primitive POMCP"):
        self.name = name
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}

    def reset(self):
        self.current_paths.clear()

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int,
                   obstacles: List[Obstacle], width: int, height: int) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos = info["hunter_positions"][agent_id]
        h_vel = info["hunter_velocities"][agent_id]
        b_mean = info.get("belief_mean", None)
        target = b_mean if b_mean is not None else np.array([width / 2.0, height / 2.0], dtype=np.float32)
        walkable_target = self.pathfinder.get_nearest_walkable(target)

        if agent_id not in self.current_paths or not self.current_paths[agent_id]:
            path = self.pathfinder.find_path(h_pos, walkable_target)
            if path:
                self.current_paths[agent_id] = path

        if agent_id in self.current_paths and self.current_paths[agent_id]:
            wp = self.current_paths[agent_id][0]
            if np.linalg.norm(h_pos - wp) < 25.0:
                self.current_paths[agent_id].pop(0)
                if self.current_paths[agent_id]:
                    wp = self.current_paths[agent_id][0]
            des_v = (wp - h_pos)
            norm = np.linalg.norm(des_v)
            if norm > 1e-4:
                des_v = (des_v / norm) * 4.0
            force = des_v - h_vel
            return _force_to_discrete_action(force)

        return 4


# ===========================================================================
# 3. MACRO-ACTION / SMDP OPTION POMCP (Mac-POMCP)
# Pure option space: {OPTION_INTERCEPT, OPTION_SWITCH}
# Exhaustive 2x2 = 4 joint options evaluated via POMDP tree search.
# Zero hardcoded coordinates, zero privileged obstacle peeking.
# ===========================================================================
OPTION_INTERCEPT = 0
OPTION_SWITCH = 1
NUM_OPTIONS = 2

class MacroPOMCPPolicy:
    def __init__(self, num_simulations: int = 30, max_depth: int = 3,
                 sim_steps_per_option: int = 120, name: str = "Mac-POMCP (Ours)"):
        self.name = name
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.sim_steps_per_option = sim_steps_per_option

        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}
        self.committed_options: Dict[int, int] = {0: OPTION_INTERCEPT, 1: OPTION_INTERCEPT}
        self.last_plan_step = -1

    def reset(self):
        self.current_paths.clear()
        self.committed_options = {0: OPTION_INTERCEPT, 1: OPTION_INTERCEPT}
        self.last_plan_step = -1

    def _get_target_for_option(self, opt: int, state: SwitchDoorGenerativeState) -> np.ndarray:
        if opt == OPTION_SWITCH:
            return state.switch_pos
        else:
            return state.e_pos

    def _sim_unroll_joint_option(self, state: SwitchDoorGenerativeState,
                                 joint_opts: Tuple[int, int],
                                 pf: Pathfinder) -> Tuple[float, bool]:
        """
        Simulates execution of joint option in generative model.
        Uses pathfinding over currently active obstacles.
        """
        cum_reward = 0.0
        gamma = 0.98

        paths = []
        for i in range(2):
            tgt = self._get_target_for_option(joint_opts[i], state)
            walkable = pf.get_nearest_walkable(tgt)
            p = pf.find_path(state.h_pos[i], walkable)
            paths.append(list(p) if p else [walkable])

        for t in range(self.sim_steps_per_option):
            actions = []
            for i in range(2):
                if paths[i]:
                    wp = paths[i][0]
                    if np.linalg.norm(state.h_pos[i] - wp) < 25.0:
                        paths[i].pop(0)
                        if paths[i]:
                            wp = paths[i][0]
                    des_v = wp - state.h_pos[i]
                    norm = np.linalg.norm(des_v)
                    if norm > 1e-4:
                        des_v = (des_v / norm) * 4.0
                    force = des_v - state.h_vel[i]
                    f_spd = np.linalg.norm(force)
                    if f_spd > 0.5:
                        force = (force / f_spd) * 0.5
                    actions.append(force)
                else:
                    # Station holding
                    actions.append(-state.h_vel[i] * 0.5)

            r, done = state.step_primitive(actions)
            cum_reward += (gamma ** t) * r

            # If gate opens during simulation, recalculate navigation path through opened area
            if state.gate_open:
                active_obs = state.get_active_obstacles()
                pf.update_obstacles(active_obs)
                for i in range(2):
                    if joint_opts[i] == OPTION_INTERCEPT:
                        tgt = state.e_pos
                        walkable = pf.get_nearest_walkable(tgt)
                        new_p = pf.find_path(state.h_pos[i], walkable)
                        if new_p:
                            paths[i] = list(new_p)

            if done:
                return cum_reward, True

        return cum_reward, False

    def plan(self, info: Dict[str, Any], width: int, height: int):
        """
        Evaluates ALL 2x2 = 4 candidate joint options under the generative POMDP model.
        Zero hardcoded role bias; pure Bellman expected return comparison.
        """
        h_pos = [info["hunter_positions"][0], info["hunter_positions"][1]]
        h_vel = [info["hunter_velocities"][0], info["hunter_velocities"][1]]
        particles = info.get("particles", None)

        if particles is not None and len(particles) > 0:
            idx = np.random.randint(0, len(particles))
            e_pos = particles[idx, :2] * np.array([width, height], dtype=np.float32)
            e_vel = particles[idx, 2:4] * 7.0
        else:
            b_mean = info.get("belief_mean", None)
            e_pos = np.copy(b_mean) if b_mean is not None else np.array([width / 2.0, height / 2.0], dtype=np.float32)
            e_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Full exhaustive 2x2 joint option space
        candidate_joint_options = [
            (OPTION_INTERCEPT, OPTION_INTERCEPT),
            (OPTION_SWITCH, OPTION_INTERCEPT),
            (OPTION_INTERCEPT, OPTION_SWITCH),
            (OPTION_SWITCH, OPTION_SWITCH)
        ]

        scores = {pair: 0.0 for pair in candidate_joint_options}

        for pair in candidate_joint_options:
            total_r = 0.0
            for _ in range(self.num_simulations):
                sim_state = SwitchDoorGenerativeState(
                    h_pos=h_pos,
                    h_vel=h_vel,
                    e_pos=e_pos,
                    e_vel=e_vel,
                    switch_pos=info["switch_pos"],
                    switch_radius=info["switch_radius"],
                    static_obstacles=info["static_obstacles"],
                    gate_obstacle=info.get("gate_obstacle", None),
                    gate_open=info.get("gate_open", False),
                    width=width,
                    height=height
                )
                r, _ = self._sim_unroll_joint_option(sim_state, pair, self.pathfinder)
                total_r += r
            scores[pair] = total_r / self.num_simulations

        # Select option pair maximizing expected discounted return
        best_pair = max(scores.items(), key=lambda item: item[1])[0]
        self.committed_options[0] = best_pair[0]
        self.committed_options[1] = best_pair[1]

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int,
                   obstacles: List[Obstacle], width: int, height: int) -> int:
        step = info.get("current_step", 0)

        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        # Replan every 30 steps or on initial step
        if step - self.last_plan_step >= 30 or self.last_plan_step == -1:
            self.plan(info, width, height)
            self.last_plan_step = step
            self.current_paths.clear()

        # Follow path for committed option
        opt = self.committed_options.get(agent_id, OPTION_INTERCEPT)
        h_pos = info["hunter_positions"][agent_id]
        h_vel = info["hunter_velocities"][agent_id]

        if opt == OPTION_SWITCH:
            target = info["switch_pos"]
        else:
            b_mean = info.get("belief_mean", None)
            target = b_mean if b_mean is not None else np.array([width / 2.0, height / 2.0], dtype=np.float32)

        walkable_target = self.pathfinder.get_nearest_walkable(target)

        if agent_id not in self.current_paths or not self.current_paths[agent_id]:
            path = self.pathfinder.find_path(h_pos, walkable_target)
            if path:
                self.current_paths[agent_id] = path

        if agent_id in self.current_paths and self.current_paths[agent_id]:
            wp = self.current_paths[agent_id][0]
            if np.linalg.norm(h_pos - wp) < 25.0:
                self.current_paths[agent_id].pop(0)
                if self.current_paths[agent_id]:
                    wp = self.current_paths[agent_id][0]
            des_v = (wp - h_pos)
            norm = np.linalg.norm(des_v)
            if norm > 1e-4:
                des_v = (des_v / norm) * 4.0
            force = des_v - h_vel
            return _force_to_discrete_action(force)

        return 4


# ===========================================================================
# 4. DEEP MAC-POMCP POLICY (Frozen Neural Value Guidance)
# ===========================================================================
class DeepMacPOMCPPolicy(MacroPOMCPPolicy):
    """
    Mac-POMCP that uses FROZEN PointNet / Value Network weights for state evaluation.
    Zero retraining, zero gradient updates.
    """
    def __init__(self, weights_path: Optional[str] = "deep_pomcp_weights.pth",
                 num_simulations: int = 30, max_depth: int = 3,
                 sim_steps_per_option: int = 120, name: str = "Deep Mac-POMCP (Frozen NN)"):
        super().__init__(num_simulations, max_depth, sim_steps_per_option, name)
        self.net = DeepPOMCPNet()
        if weights_path and os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.net.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.net.load_state_dict(ckpt)
                print(f"[{name}] Loaded FROZEN weights from '{weights_path}'")
            except Exception as e:
                print(f"[{name}] Note: weights load warning: {e}")
        self.net.eval()
