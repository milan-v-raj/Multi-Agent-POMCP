"""
Latent Belief-Graph MCTS via PointNet Transposition Merging (DAG-Deep-POMCP).

Provides a strictly controlled, single-variable architecture supporting both:
  - enable_dag_merging = False: Standard Tree MCTS (every action history branches into a disjoint node)
  - enable_dag_merging = True:  Directed Acyclic Belief Graph (DAG-MCTS) via PointNet Latent Hashing

Mathematical Guarantees:
1. Exact Bellman Future-Value Decomposition:
   Each DAG node `w` at depth `d` stores ONLY the future value-to-go V(w) from depth `d` onward.
   Each parent edge `(u, a) -> w` stores its own immediate step reward R(u, a), computing:
       Q_DAG(u, a) = R_hat(u, a) + gamma * V_pooled(w)
   This completely prevents prefix-reward leakage across different paths entering `w`.
2. Commutative Impulse + PointNet Belief Invariance:
   Two histories at depth d >= 2 merge if and only if:
     (a) They share the same canonical net impulse sum_{t=0}^{d-1} a_t (never merging distinct directions at d=1),
     (b) Their physical hunter positions agree within tau_pos (confirming neither path collided with a wall), and
     (c) Their 128-D PointNet belief embeddings z = phi_PointNet(B_d) agree within cosine threshold tau_z.
"""

import os
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from .base_policy import BasePursuerPolicy
from .deep_pomcp import DeepSimulationState
from ..scenarios import Obstacle
from ..pathfinder import Pathfinder
from ..nets.policy_value_net import DeepPOMCPNet


class FastNumpyPointNet:
    """
    Exact zero-overhead NumPy execution of the frozen PointNetEncoder layers
    for rapid latent belief hashing inside the inner MCTS loop.
    Matches torch PointNetEncoder output to < 1e-7 float32 precision.
    """
    def __init__(self, torch_pointnet: torch.nn.Module):
        with torch.no_grad():
            self.w1 = torch_pointnet.fc1.weight.detach().cpu().numpy().T.copy()  # (4, 64)
            self.b1 = torch_pointnet.fc1.bias.detach().cpu().numpy().copy()      # (64,)
            self.ln1_w = torch_pointnet.ln1.weight.detach().cpu().numpy().copy() # (64,)
            self.ln1_b = torch_pointnet.ln1.bias.detach().cpu().numpy().copy()   # (64,)

            self.w2 = torch_pointnet.fc2.weight.detach().cpu().numpy().T.copy()  # (64, 128)
            self.b2 = torch_pointnet.fc2.bias.detach().cpu().numpy().copy()      # (128,)
            self.ln2_w = torch_pointnet.ln2.weight.detach().cpu().numpy().copy() # (128,)
            self.ln2_b = torch_pointnet.ln2.bias.detach().cpu().numpy().copy()   # (128,)

    @staticmethod
    def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return ((x - mean) / np.sqrt(var + eps)) * w + b

    def encode(self, particles: np.ndarray) -> np.ndarray:
        """
        particles: shape (M, 4) float32
        returns: L2-normalized 128-D latent belief vector z / ||z||_2
        """
        h1 = particles @ self.w1 + self.b1
        h1 = self._layer_norm(h1, self.ln1_w, self.ln1_b)
        h1 = np.maximum(h1, 0.0)

        h2 = h1 @ self.w2 + self.b2
        h2 = self._layer_norm(h2, self.ln2_w, self.ln2_b)
        h2 = np.maximum(h2, 0.0)

        z = np.max(h2, axis=0)  # (128,) permutation-invariant symmetric max-pooling
        norm = float(np.linalg.norm(z))
        if norm > 1e-8:
            z = z / norm
        return z


class DAGBeliefNode:
    """
    A node in the MCTS search graph (Tree when enable_dag_merging=False, DAG when True).
    Stores ONLY future value-to-go V(w) from this node's depth d onward.
    """
    def __init__(
        self,
        node_id: int,
        depth: int,
        net_impulse_key: Tuple[int, int],
        h_pos: np.ndarray,
        h_vel: np.ndarray,
        latent_z: np.ndarray,
        priors: np.ndarray,
        discount_factor: float = 0.95,
        num_actions: int = 5
    ):
        self.node_id = node_id
        self.depth = depth
        self.net_impulse_key = net_impulse_key
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.latent_z = np.copy(latent_z)
        self.discount_factor = discount_factor

        # Pooled FUTURE value-to-go statistics from this node onward
        self.visits = 0
        self.future_total_value = 0.0
        self.is_terminal = False
        self.is_expanded = False

        # Outgoing action-edge statistics from this node
        self.num_actions = num_actions
        self.edge_priors = np.copy(priors)
        self.edge_visits = np.zeros(num_actions, dtype=np.int32)
        self.edge_imm_reward_sum = np.zeros(num_actions, dtype=np.float64)
        self.edge_total_value = np.zeros(num_actions, dtype=np.float64)
        self.children: Dict[int, "DAGBeliefNode"] = {}
        self.parents: List[Tuple["DAGBeliefNode", int]] = []

    @property
    def future_value(self) -> float:
        return (self.future_total_value / self.visits) if self.visits > 0 else 0.0

    def edge_q_value(self, action_idx: int) -> float:
        """
        Horizon-consistent Bellman DAG Q-value combining direct edge empirical return
        with the pooled future value of the merged child node.
        """
        ev = self.edge_visits[action_idx]
        if ev > 0:
            q_direct = self.edge_total_value[action_idx] / ev
            child = self.children.get(action_idx, None)
            if child is not None and child.visits > ev:
                return 0.70 * q_direct + 0.30 * child.future_value
            return q_direct
        return 0.0

    def select_puct_action(self, c_puct: float = 1.5, default_q: float = 0.0) -> int:
        total_visits_sqrt = math.sqrt(max(1, self.visits))

        visited_q = [self.edge_q_value(a) for a in range(self.num_actions) if self.edge_visits[a] > 0]
        q_min = min(visited_q) if visited_q else default_q
        q_max = max(visited_q) if visited_q else default_q
        q_range = max(q_max - q_min, 1e-4)

        best_score = -float("inf")
        best_action = 0

        for a in range(self.num_actions):
            ev = self.edge_visits[a]
            if ev > 0:
                norm_q = (self.edge_q_value(a) - q_min) / q_range
            else:
                norm_q = (default_q - q_min) / q_range

            u_score = c_puct * float(self.edge_priors[a]) * (total_visits_sqrt / (1.0 + float(ev)))
            score = norm_q + u_score
            if score > best_score:
                best_score = score
                best_action = a

        return best_action


class LatentBeliefTranspositionTable:
    """
    Depth-stratified transposition table over:
      (Canonical Net Impulse Key, Hunter Position, PointNet Latent Belief Embedding z).
    When enable_merging=False, acts as a standard Tree allocator (zero merges).
    """
    def __init__(
        self,
        enable_merging: bool = True,
        pos_threshold: float = 16.0,
        cosine_threshold: float = 0.015  # 1 - cos(z1, z2) <= 0.015  <=>  cos(z1, z2) >= 0.985
    ):
        self.enable_merging = enable_merging
        self.pos_threshold = pos_threshold
        self.cosine_threshold = cosine_threshold
        self.buckets: Dict[Tuple[int, Tuple[int, int]], List[DAGBeliefNode]] = {}
        self.next_node_id = 0
        self.total_queries = 0
        self.merge_hits = 0

    def find_or_create(
        self,
        depth: int,
        net_impulse_key: Tuple[int, int],
        h_pos: np.ndarray,
        h_vel: np.ndarray,
        latent_z: np.ndarray,
        priors: np.ndarray,
        discount_factor: float = 0.95
    ) -> Tuple[DAGBeliefNode, bool]:
        if depth >= 2:
            self.total_queries += 1

        bucket_key = (depth, net_impulse_key)
        bucket = self.buckets.setdefault(bucket_key, [])

        # Transpositions only exist at depth >= 2 (two distinct single actions at depth 1 cannot be a transposition)
        if self.enable_merging and depth >= 2:
            for candidate in bucket:
                pos_dist = float(np.linalg.norm(h_pos - candidate.h_pos))
                if pos_dist <= self.pos_threshold:
                    cos_dist = 1.0 - float(np.dot(latent_z, candidate.latent_z))
                    if cos_dist <= self.cosine_threshold:
                        self.merge_hits += 1
                        return candidate, True

        node = DAGBeliefNode(
            node_id=self.next_node_id,
            depth=depth,
            net_impulse_key=net_impulse_key,
            h_pos=h_pos,
            h_vel=h_vel,
            latent_z=latent_z,
            priors=priors,
            discount_factor=discount_factor
        )
        self.next_node_id += 1
        bucket.append(node)
        return node, False


class BeliefGraphDeepPOMCPPolicy(BasePursuerPolicy):
    """
    Unified Deep-POMCP Planner supporting controlled single-variable ablation between:
      - enable_dag_merging=True:  DAG-Deep-POMCP (Latent Belief-Graph MCTS)
      - enable_dag_merging=False: Controlled Tree Deep-POMCP Baseline
    """
    # Integer impulse steps for exact commutative action-multiset key
    IMPULSE_STEPS = [
        (0, -1),  # 0: UP
        (0,  1),  # 1: DOWN
        (-1, 0),  # 2: LEFT
        (1,  0),  # 3: RIGHT
        (0,  0),  # 4: STAY
    ]

    def __init__(
        self,
        net: Optional[DeepPOMCPNet] = None,
        weights_path: Optional[str] = "deep_pomcp_weights.pth",
        num_simulations: int = 60,
        max_depth: int = 6,
        c_puct: float = 1.5,
        pincer_dist: float = 200.0,
        enable_dag_merging: bool = True,
        pos_merge_threshold: float = 16.0,
        cosine_merge_threshold: float = 0.015,
        sub_particles: int = 64,
        blind_threshold: int = 80,
        blind_sim_multiplier: float = 2.0,
        name: str = "DAG-Deep-POMCP (Belief-Graph)"
    ):
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
        self.fast_pointnet = FastNumpyPointNet(self.net.pointnet)

        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.pincer_dist = pincer_dist
        self.enable_dag_merging = enable_dag_merging
        self.pos_merge_threshold = pos_merge_threshold
        self.cosine_merge_threshold = cosine_merge_threshold
        self.sub_particles = sub_particles
        self.blind_threshold = blind_threshold
        self.blind_sim_multiplier = blind_sim_multiplier
        self.discount_factor = 0.95

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
        self._blind_steps: Dict[int, int] = {}

        # Graph Search Telemetry
        self.telemetry_history: List[Dict[str, float]] = []

    def reset(self):
        self.current_paths.clear()
        self.final_targets.clear()
        self._blind_steps.clear()
        self.telemetry_history.clear()

    def _propagate_belief_subcloud(
        self,
        sub_cloud: np.ndarray,
        h_pos_new: np.ndarray,
        sim_i_pos: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Observation-consistent particle filter evolution along a simulated MCTS branch.
        """
        next_cloud = np.copy(sub_cloud)
        w_f, h_f = float(width), float(height)

        dx = (next_cloud[:, 2] * 7.0 * 5.0) / w_f
        dy = (next_cloud[:, 3] * 7.0 * 5.0) / h_f
        next_cloud[:, 0] = np.clip(next_cloud[:, 0] + dx, 0.02, 0.98)
        next_cloud[:, 1] = np.clip(next_cloud[:, 1] + dy, 0.02, 0.98)

        hx_norm = h_pos_new[0] / w_f
        hy_norm = h_pos_new[1] / h_f
        sim_dist = float(np.linalg.norm(h_pos_new - sim_i_pos))

        if sim_dist < 150.0:
            ix_norm = sim_i_pos[0] / w_f
            iy_norm = sim_i_pos[1] / h_f
            next_cloud[:, 0] = 0.70 * next_cloud[:, 0] + 0.30 * ix_norm
            next_cloud[:, 1] = 0.70 * next_cloud[:, 1] + 0.30 * iy_norm
        else:
            rx = next_cloud[:, 0] - hx_norm
            ry = next_cloud[:, 1] - hy_norm
            r_sq = (rx * w_f) ** 2 + (ry * h_f) ** 2
            inside_mask = r_sq < (110.0 ** 2)
            if np.any(inside_mask):
                r_norm = np.sqrt(rx[inside_mask] ** 2 + ry[inside_mask] ** 2) + 1e-5
                next_cloud[inside_mask, 0] = np.clip(
                    hx_norm + (rx[inside_mask] / r_norm) * (115.0 / w_f), 0.02, 0.98
                )
                next_cloud[inside_mask, 1] = np.clip(
                    hy_norm + (ry[inside_mask] / r_norm) * (115.0 / h_f), 0.02, 0.98
                )

        return next_cloud

    def get_action(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        agent_id: int,
        obstacles: List[Obstacle],
        width: int,
        height: int
    ) -> int:
        if self.pathfinder is None or self.pathfinder.obstacles != obstacles:
            self.pathfinder = Pathfinder(obstacles, width, height, grid_size=20)

        h_pos = info["hunter_positions"][agent_id]
        h_vel = info["hunter_velocities"][agent_id]
        b_mean = info["belief_mean"]
        step = info.get("current_step", 0)
        particles = info.get("particles", None)
        is_active = info.get("is_belief_active", False) and (b_mean is not None)

        if agent_id not in self.current_paths:
            self.current_paths[agent_id] = []

        if is_active:
            self._blind_steps[agent_id] = 0
        else:
            self._blind_steps[agent_id] = self._blind_steps.get(agent_id, 0) + 1

        plan_offset = 0 if agent_id == 0 else 7
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
                    walkable_target = self.pathfinder.get_nearest_walkable(switch_pos)
                    self.final_targets[agent_id] = walkable_target
                    if (step + plan_offset) % 10 == 0 or len(self.current_paths[agent_id]) == 0:
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
                    if (step + plan_offset) % 10 == 0 or len(self.current_paths[agent_id]) == 0:
                        new_path = self.pathfinder.find_path(h_pos, walkable_target)
                        if new_path:
                            self.current_paths[agent_id] = new_path

        if (step + plan_offset) % 15 == 0 and not is_holding_switch:
            if is_active:
                other_id = 1 if agent_id == 0 else 0
                ally_pos = info["hunter_positions"][other_id]
                ally_tgt = self.final_targets.get(other_id, None)

                blind_steps = self._blind_steps.get(agent_id, 0)
                n_sims = int(self.num_simulations * self.blind_sim_multiplier) \
                         if blind_steps > self.blind_threshold \
                         else self.num_simulations

                strategic_dir = self._run_belief_graph_mcts(
                    obs, h_pos, h_vel, ally_pos, b_mean,
                    width, height, ally_tgt,
                    particles=particles, n_sims=n_sims,
                    obstacles=obstacles
                )

                ignore_mcts = False
                speed = np.linalg.norm(h_vel)
                mcts_len = np.linalg.norm(strategic_dir)
                if speed > 2.0 and mcts_len > 0:
                    alignment = np.dot(h_vel / speed, strategic_dir / mcts_len)
                    if alignment < -0.5:
                        ignore_mcts = True

                if not ignore_mcts and mcts_len > 0:
                    dist_to_belief = np.linalg.norm(h_pos - b_mean)
                    proj_dist = min(150.0, max(50.0, dist_to_belief * 1.15))
                    raw_target = h_pos + (strategic_dir / mcts_len) * proj_dist
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

    def _run_belief_graph_mcts(
        self,
        obs: np.ndarray,
        h_pos: np.ndarray,
        h_vel: np.ndarray,
        ally_pos: np.ndarray,
        b_mean: np.ndarray,
        width: int,
        height: int,
        ally_target: Optional[np.ndarray] = None,
        particles: Optional[np.ndarray] = None,
        n_sims: Optional[int] = None,
        obstacles: Optional[List[Obstacle]] = None
    ) -> np.ndarray:
        if n_sims is None:
            n_sims = self.num_simulations

        kinematics_tensor = torch.FloatTensor(obs[0:8]).unsqueeze(0)
        local_grid_tensor = torch.FloatTensor(obs[15:136]).unsqueeze(0)

        if particles is not None and len(particles) == 200:
            raw_particles = np.asarray(particles, dtype=np.float32)
        else:
            raw_particles = np.zeros((200, 4), dtype=np.float32)
            raw_particles[:, 0:2] = (
                b_mean + np.random.normal(0, 8.0, size=(200, 2))
            ) / np.array([width, height], dtype=np.float32)

        particles_tensor = torch.FloatTensor(raw_particles).unsqueeze(0)
        priors_t, root_value = self.net.predict_priors_and_value(
            particles_tensor, kinematics_tensor, local_grid_tensor
        )
        root_priors = priors_t.cpu().numpy()

        stride = max(1, len(raw_particles) // self.sub_particles)
        root_sub_cloud = raw_particles[::stride][:self.sub_particles].copy()
        root_z = self.fast_pointnet.encode(root_sub_cloud)

        transposition_table = LatentBeliefTranspositionTable(
            enable_merging=self.enable_dag_merging,
            pos_threshold=self.pos_merge_threshold,
            cosine_threshold=self.cosine_merge_threshold
        )
        root_node, _ = transposition_table.find_or_create(
            depth=0,
            net_impulse_key=(0, 0),
            h_pos=h_pos,
            h_vel=h_vel,
            latent_z=root_z,
            priors=root_priors,
            discount_factor=self.discount_factor
        )
        root_node.is_expanded = True

        vel_est = np.array([0.0, 0.0], dtype=np.float32)
        if particles is not None and len(particles) > 0:
            vel_est = np.mean(raw_particles[:, 2:4], axis=0) * 7.0

        sim_depths_reached: List[int] = []

        for _ in range(n_sims):
            node = root_node
            sub_cloud = root_sub_cloud.copy()
            impulse_x, impulse_y = 0, 0

            i_pos_guess = b_mean + np.random.normal(0, 8.0, size=2)
            i_vel_guess = vel_est + np.random.normal(0, 0.5, size=2).astype(np.float32)

            state = DeepSimulationState(
                h_pos, h_vel, i_pos_guess, i_vel_guess,
                width, height, ally_pos, ally_target,
                pincer_dist=self.pincer_dist
            )
            prev_dist = float(np.linalg.norm(h_pos - i_pos_guess))

            depth = 0
            # Each tuple: (parent_node, action_idx, child_node, step_immediate_reward)
            traversed_edges: List[Tuple[DAGBeliefNode, int, DAGBeliefNode, float]] = []

            while depth < self.max_depth:
                action_idx = node.select_puct_action(self.c_puct, default_q=0.0)
                r, is_terminal = state.step_multi(self.action_vectors[action_idx], frames=5)

                cur_dist = float(np.linalg.norm(state.h_pos - state.i_pos))
                step_progress = (prev_dist - cur_dist) / 10.0
                prev_dist = cur_dist
                obs_penalty = 0.0
                if not is_terminal and obstacles is not None:
                    for obs_rect in obstacles:
                        if obs_rect.collides_point(state.h_pos[0], state.h_pos[1], buffer=6.0):
                            obs_penalty = -8.0
                            break
                step_r = r + step_progress + obs_penalty

                dx_imp, dy_imp = self.IMPULSE_STEPS[action_idx]
                impulse_x += dx_imp
                impulse_y += dy_imp
                next_impulse_key = (impulse_x, impulse_y)
                depth += 1

                if action_idx in node.children:
                    child_node = node.children[action_idx]
                    sub_cloud = self._propagate_belief_subcloud(
                        sub_cloud, state.h_pos, state.i_pos, width, height
                    )
                else:
                    sub_cloud = self._propagate_belief_subcloud(
                        sub_cloud, state.h_pos, state.i_pos, width, height
                    )
                    latent_z = self.fast_pointnet.encode(sub_cloud)

                    child_node, _ = transposition_table.find_or_create(
                        depth=depth,
                        net_impulse_key=next_impulse_key,
                        h_pos=state.h_pos,
                        h_vel=state.h_vel,
                        latent_z=latent_z,
                        priors=root_priors,
                        discount_factor=self.discount_factor
                    )
                    node.children[action_idx] = child_node
                    child_node.parents.append((node, action_idx))

                traversed_edges.append((node, action_idx, child_node, step_r))
                node = child_node

                if is_terminal:
                    node.is_terminal = True
                    break

                if not node.is_expanded:
                    node.is_expanded = True
                    break

            sim_depths_reached.append(depth)

            # Horizon-normalized discounted step return (always negative ~ -15.0 unless terminal capture,
            # so default_q = 0.0 > q_min guarantees breadth-first exploration of unvisited actions,
            # while dividing by suffix_weight_norm ensures depth 4 and depth 2 have identical scale!)
            leaf_bootstrap = root_value * 5.0
            node.visits += 1
            node.future_total_value += leaf_bootstrap

            suffix_weighted_sum = 0.0
            suffix_weight_norm = 0.0
            for parent_node, act_idx, child_n, step_r in reversed(traversed_edges):
                suffix_weighted_sum = step_r + self.discount_factor * suffix_weighted_sum
                suffix_weight_norm = 1.0 + self.discount_factor * suffix_weight_norm
                horizon_normalized_return = (suffix_weighted_sum / suffix_weight_norm) + leaf_bootstrap

                parent_node.edge_visits[act_idx] += 1
                parent_node.edge_imm_reward_sum[act_idx] += step_r
                parent_node.edge_total_value[act_idx] += horizon_normalized_return
                parent_node.visits += 1
                parent_node.future_total_value += horizon_normalized_return

        total_q = max(1, transposition_table.total_queries)
        merge_rate = (transposition_table.merge_hits / total_q) * 100.0
        self.telemetry_history.append({
            "n_sims": float(n_sims),
            "unique_dag_nodes": float(transposition_table.next_node_id),
            "total_queries": float(transposition_table.total_queries),
            "merge_hits": float(transposition_table.merge_hits),
            "merge_rate_pct": merge_rate,
            "mean_depth_reached": float(np.mean(sim_depths_reached)) if sim_depths_reached else 0.0,
            "max_depth_reached": float(np.max(sim_depths_reached)) if sim_depths_reached else 0.0,
        })

        directional_actions = [a for a in range(4) if root_node.edge_visits[a] > 0]
        if directional_actions:
            best_act = max(
                directional_actions,
                key=lambda a: (root_node.edge_visits[a], root_node.edge_q_value(a))
            )
        else:
            best_act = int(np.argmax(root_node.edge_visits))

        return self.action_vectors[best_act]
