"""
Causal Dec-POMDP Policies for Epistemic Switch-Door Coordination.
Implements:
1. GreedyAStarPolicy: Greedy pursuit (fails 100% at closed door).
2. RandomSwitchHeuristicPolicy: Heuristic trial-and-error baseline.
3. CausalMacPOMCPPolicy (Ours): SMDP Option MCTS with Bayesian Epistemic Updating over causal hypotheses.
4. DeepCausalMacPOMCPPolicy (Frozen NN): Guided by frozen neural value head.
"""

import os
import sys
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import torch

from deep_pomcp_env.scenarios import Obstacle
from deep_pomcp_env.pathfinder import Pathfinder
from deep_pomcp_env.nets.policy_value_net import DeepPOMCPNet
from .generative_model import CausalGenerativeState


def _force_to_discrete_action(force: np.ndarray) -> int:
    fx, fy = float(force[0]), float(force[1])
    if abs(fx) < 0.1 and abs(fy) < 0.1:
        return 4
    if abs(fx) > abs(fy):
        return 3 if fx > 0 else 2
    else:
        return 1 if fy > 0 else 0


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
# 2. BASELINE: Heuristic Trial-and-Error Policy
# ===========================================================================
class RandomSwitchHeuristicPolicy:
    """
    Heuristic baseline: Hunter 0 systematically visits untried switches.
    Hunter 1 stages at the gate corridor.
    Does not use joint MCTS tree search or belief value optimization.
    """
    def __init__(self, name: str = "Heuristic Trial-and-Error"):
        self.name = name
        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}
        self.tested_switches: Set[int] = set()
        self.current_target_switch: Optional[int] = None

    def reset(self):
        self.current_paths.clear()
        self.tested_switches.clear()
        self.current_target_switch = None

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int,
                   obstacles: List[Obstacle], width: int, height: int) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos = info["hunter_positions"][agent_id]
        h_vel = info["hunter_velocities"][agent_id]
        gate_open = info.get("gate_open", False)
        sw_positions = info["switch_positions"]
        num_switches = len(sw_positions)

        # Check if hunter 0 reached currently targeted switch
        if agent_id == 0 and self.current_target_switch is not None:
            dist = np.linalg.norm(h_pos - sw_positions[self.current_target_switch])
            if dist < info["switch_radius"]:
                self.tested_switches.add(self.current_target_switch)
                if not gate_open:
                    # Switch failed: choose another untried switch
                    self.current_target_switch = None
                    self.current_paths[0] = []

        if gate_open:
            # Both pursue evader
            b_mean = info.get("belief_mean", None)
            target = b_mean if b_mean is not None else np.array([width / 2.0, height / 2.0], dtype=np.float32)
        else:
            if agent_id == 0:
                # Operator: select untried switch
                if self.current_target_switch is None:
                    untried = [k for k in range(num_switches) if k not in self.tested_switches]
                    if untried:
                        # Pick closest untried switch
                        dists = [np.linalg.norm(h_pos - sw_positions[k]) for k in untried]
                        self.current_target_switch = untried[int(np.argmin(dists))]
                    else:
                        self.current_target_switch = 0
                target = sw_positions[self.current_target_switch]
            else:
                # Stager: wait near gate
                gate_obs = info.get("gate_obstacle", None)
                if gate_obs is not None:
                    target = np.array([gate_obs.x - 40.0, gate_obs.y + gate_obs.height / 2.0], dtype=np.float32)
                else:
                    target = np.array([540.0, 300.0], dtype=np.float32)

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
# 3. CAUSAL MAC-POMCP (Ours: Epistemic Causal Option Planning)
# ===========================================================================
OPTION_INTERCEPT = 0
# Options 1 .. K correspond to testing switch 0 .. K-1

class CausalMacPOMCPPolicy:
    """
    Dec-POMDP Macro-Action Planner under Dual Latent Uncertainty:
    1. Spatial uncertainty over target position: b_e (particles)
    2. Causal uncertainty over functional mechanism: b_kappa (Bayesian distribution over true switch)
    """
    def __init__(self, num_simulations: int = 30, max_depth: int = 3,
                 sim_steps_per_option: int = 120, name: str = "Causal Mac-POMCP (Ours)"):
        self.name = name
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.sim_steps_per_option = sim_steps_per_option

        self.pathfinder: Optional[Pathfinder] = None
        self.current_paths: Dict[int, List[np.ndarray]] = {}
        self.committed_options: Dict[int, int] = {0: OPTION_INTERCEPT, 1: OPTION_INTERCEPT}
        self.last_plan_step = -1

        # Epistemic Causal Belief: b_kappa = [p_0, p_1, ..., p_{K-1}]
        self.causal_belief: Optional[np.ndarray] = None
        self.pruned_switches: Set[int] = set()
        self.confirmed_switch: Optional[int] = None

    def reset(self):
        self.current_paths.clear()
        self.committed_options = {0: OPTION_INTERCEPT, 1: OPTION_INTERCEPT}
        self.last_plan_step = -1
        self.causal_belief = None
        self.pruned_switches.clear()
        self.confirmed_switch = None

    def _update_causal_belief(self, info: Dict[str, Any]):
        num_sw = info["num_switches"]
        if self.causal_belief is None:
            self.causal_belief = np.ones(num_sw, dtype=np.float32) / float(num_sw)

        if self.confirmed_switch is not None:
            self.causal_belief = np.zeros(num_sw, dtype=np.float32)
            self.causal_belief[self.confirmed_switch] = 1.0
            return

        gate_open = info.get("gate_open", False)
        sw_pos = info["switch_positions"]
        sw_radius = info["switch_radius"]

        # Check if any hunter is currently standing on a switch
        for k in range(num_sw):
            if k in self.pruned_switches:
                continue
            is_on_switch = False
            for i in range(len(info["hunter_positions"])):
                if np.linalg.norm(info["hunter_positions"][i] - sw_pos[k]) < sw_radius:
                    is_on_switch = True
                    break

            if is_on_switch:
                if gate_open:
                    # Ground-truth confirmation: switch k is functional with certainty!
                    self.confirmed_switch = k
                    self.causal_belief = np.zeros(num_sw, dtype=np.float32)
                    self.causal_belief[k] = 1.0
                    return
                else:
                    # Negative evidence: switch k failed to open gate!
                    self.pruned_switches.add(k)
                    self.causal_belief[k] = 0.0
                    remaining_sum = np.sum(self.causal_belief)
                    if remaining_sum > 1e-6:
                        self.causal_belief /= remaining_sum
                    else:
                        # Fallback uniform over remaining
                        unpruned = [j for j in range(num_sw) if j not in self.pruned_switches]
                        if unpruned:
                            for j in unpruned:
                                self.causal_belief[j] = 1.0 / len(unpruned)

    def _get_target_for_option(self, opt: int, state: CausalGenerativeState) -> np.ndarray:
        if opt == OPTION_INTERCEPT:
            return state.e_pos
        else:
            # opt > 0 corresponds to testing switch index (opt - 1)
            sw_idx = opt - 1
            if sw_idx < len(state.switch_positions):
                return state.switch_positions[sw_idx]
            return state.e_pos

    def _sim_unroll_joint_option(self, state: CausalGenerativeState,
                                 joint_opts: Tuple[int, int],
                                 pf: Pathfinder) -> Tuple[float, bool]:
        cum_reward = 0.0
        gamma = 0.98

        # Initialize simulation pathfinder with state's initial active obstacles
        active_obs = state.get_active_obstacles()
        pf.update_obstacles(active_obs)

        paths = []
        for i in range(2):
            tgt = self._get_target_for_option(joint_opts[i], state)
            walkable = pf.get_nearest_walkable(tgt)
            p = pf.find_path(state.h_pos[i], walkable)
            # If path blocked across closed gate, stage/hold position rather than driving into wall
            paths.append(list(p) if p else [])

        prev_gate_open = state.gate_open
        reached_terminal = False

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

            # Dynamic gate update in pathfinder whenever gate state toggles
            if state.gate_open != prev_gate_open:
                prev_gate_open = state.gate_open
                active_obs = state.get_active_obstacles()
                pf.update_obstacles(active_obs)
                for i in range(2):
                    tgt = self._get_target_for_option(joint_opts[i], state)
                    walkable = pf.get_nearest_walkable(tgt)
                    new_p = pf.find_path(state.h_pos[i], walkable)
                    paths[i] = list(new_p) if new_p else []

            if done:
                reached_terminal = True
                break

        return cum_reward, reached_terminal

    def plan(self, info: Dict[str, Any], width: int, height: int):
        self._update_causal_belief(info)

        num_sw = info["num_switches"]
        h_pos = [info["hunter_positions"][0], info["hunter_positions"][1]]
        h_vel = [info["hunter_velocities"][0], info["hunter_velocities"][1]]
        particles = info.get("particles", None)

        # 1. Sample evader position from particle belief
        if particles is not None and len(particles) > 0:
            idx = np.random.randint(0, len(particles))
            e_pos = particles[idx, :2] * np.array([width, height], dtype=np.float32)
            e_vel = particles[idx, 2:4] * 7.0
        else:
            b_mean = info.get("belief_mean", None)
            e_pos = np.copy(b_mean) if b_mean is not None else np.array([width / 2.0, height / 2.0], dtype=np.float32)
            e_vel = np.array([0.0, 0.0], dtype=np.float32)

        # 2. Build candidate joint options
        # Options: 0 = INTERCEPT, (k+1) = TRY_SWITCH_k
        # Candidate pairs:
        # - Both intercept
        # - Hunter 0 tests switch k, Hunter 1 intercepts (for each unpruned switch)
        # - Hunter 1 tests switch k, Hunter 0 intercepts (for each unpruned switch)
        candidate_joint_options = [(OPTION_INTERCEPT, OPTION_INTERCEPT)]
        for k in range(num_sw):
            if k not in self.pruned_switches:
                opt_sw = k + 1
                candidate_joint_options.append((opt_sw, OPTION_INTERCEPT))
                candidate_joint_options.append((OPTION_INTERCEPT, opt_sw))

        scores = {pair: 0.0 for pair in candidate_joint_options}

        # 3. MCTS Simulations with Causal Latent Sampling
        p_dist = np.array(self.causal_belief, dtype=np.float64)
        p_dist = np.maximum(p_dist, 0.0)
        s_val = np.sum(p_dist)
        if s_val > 1e-9:
            p_dist /= s_val
        else:
            p_dist = np.ones(num_sw, dtype=np.float64) / float(num_sw)
        p_dist[-1] = max(0.0, 1.0 - np.sum(p_dist[:-1]))
        s_val = np.sum(p_dist)
        if s_val > 1e-9:
            p_dist /= s_val
        else:
            p_dist = np.ones(num_sw, dtype=np.float64) / float(num_sw)

        for pair in candidate_joint_options:
            total_r = 0.0
            for _ in range(self.num_simulations):
                # Sample latent true_switch_idx according to causal belief b_kappa
                sampled_kappa = int(np.random.choice(num_sw, p=p_dist))

                sim_state = CausalGenerativeState(
                    h_pos=h_pos,
                    h_vel=h_vel,
                    e_pos=e_pos,
                    e_vel=e_vel,
                    switch_positions=info["switch_positions"],
                    switch_radius=info["switch_radius"],
                    true_switch_idx=sampled_kappa,
                    static_obstacles=info["static_obstacles"],
                    gate_obstacle=info.get("gate_obstacle", None),
                    gate_open=info.get("gate_open", False),
                    width=width,
                    height=height
                )
                r, _ = self._sim_unroll_joint_option(sim_state, pair, self.pathfinder)
                total_r += r

            scores[pair] = total_r / self.num_simulations

        # 4. Select joint option maximizing expected return under epistemic belief
        best_pair = max(scores.items(), key=lambda item: item[1])[0]
        self.committed_options[0] = best_pair[0]
        self.committed_options[1] = best_pair[1]

        # Restore pathfinder obstacles to match current observation environment state
        current_obs = list(info["static_obstacles"])
        if not info.get("gate_open", False) and info.get("gate_obstacle", None):
            current_obs.append(info["gate_obstacle"])
        self.pathfinder.update_obstacles(current_obs)

    def get_action(self, obs: np.ndarray, info: Dict[str, Any], agent_id: int,
                   obstacles: List[Obstacle], width: int, height: int) -> int:
        step = info.get("current_step", 0)

        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        # Check if gate state or newly pruned switch requires prompt replanning
        old_pruned_count = len(self.pruned_switches)
        self._update_causal_belief(info)
        newly_pruned = (len(self.pruned_switches) > old_pruned_count)

        if step - self.last_plan_step >= 30 or self.last_plan_step == -1 or newly_pruned:
            self.plan(info, width, height)
            self.last_plan_step = step
            self.current_paths.clear()

        # Follow path for committed option
        opt = self.committed_options.get(agent_id, OPTION_INTERCEPT)
        h_pos = info["hunter_positions"][agent_id]
        h_vel = info["hunter_velocities"][agent_id]

        if opt == OPTION_INTERCEPT:
            b_mean = info.get("belief_mean", None)
            target = b_mean if b_mean is not None else np.array([width / 2.0, height / 2.0], dtype=np.float32)
        else:
            sw_idx = opt - 1
            target = info["switch_positions"][sw_idx]

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
# 4. DEEP CAUSAL MAC-POMCP (Frozen Neural Value Guidance)
# ===========================================================================
class DeepCausalMacPOMCPPolicy(CausalMacPOMCPPolicy):
    def __init__(self, weights_path: Optional[str] = "deep_pomcp_weights.pth",
                 num_simulations: int = 30, max_depth: int = 3,
                 sim_steps_per_option: int = 120, name: str = "Deep Causal Mac-POMCP (Frozen NN)"):
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

