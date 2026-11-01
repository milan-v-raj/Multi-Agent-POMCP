"""
Latent Belief-Graph MCTS via PointNet Transposition Merging (DAG-Deep-POMCP).

Transforms the exponential action-observation history tree of standard POMCP/AlphaZero
into a Directed Acyclic Belief Graph (DAG) by merging belief-equivalent nodes across
different action histories using the permutation-invariant PointNet latent embedding
z = phi_PointNet(B_t) in R^128 and normalized hunter kinematics.
"""

import os
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .base_policy import BasePursuerPolicy
from .deep_pomcp import DeepSimulationState
from ..scenarios import Obstacle
from ..pathfinder import Pathfinder
from ..nets.policy_value_net import DeepPOMCPNet


class FastNumpyPointNet:
    """
    Exact zero-overhead NumPy execution of the frozen PointNetEncoder layers
    for rapid O(1) latent belief hashing inside the inner MCTS loop.
    Matches torch PointNetEncoder output to < 1e-6 float32 precision.
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
        returns: normalized 128-D latent belief vector z / ||z||_2
        """
        h1 = particles @ self.w1 + self.b1
        h1 = self._layer_norm(h1, self.ln1_w, self.ln1_b)
        h1 = np.maximum(h1, 0.0)

        h2 = h1 @ self.w2 + self.b2
        h2 = self._layer_norm(h2, self.ln2_w, self.ln2_b)
        h2 = np.maximum(h2, 0.0)

        z = np.max(h2, axis=0)  # (128,) symmetric max-pooling
        norm = float(np.linalg.norm(z))
        if norm > 1e-8:
            z = z / norm
        return z


class DAGBeliefNode:
    """
    A node in the Directed Acyclic Belief Graph (DAG).
    Represents an equivalence class of (Hunter Kinematics, PointNet Belief Embedding z_t)
    at search depth d. Multiple parent (node, action) edges can merge into this single node.
    """
    def __init__(
        self,
        node_id: int,
        depth: int,
        h_pos: np.ndarray,
        h_vel: np.ndarray,
        latent_z: np.ndarray,
        priors: np.ndarray,
        num_actions: int = 5
    ):
        self.node_id = node_id
        self.depth = depth
        self.h_pos = np.copy(h_pos)
        self.h_vel = np.copy(h_vel)
        self.latent_z = np.copy(latent_z)  # L2-normalized 128-D PointNet embedding

        # Pooled node statistics across ALL incoming parent histories
        self.visits = 0
        self.total_value = 0.0
        self.is_terminal = False
        self.is_expanded = False

        # Action-edge statistics from this belief node
        self.num_actions = num_actions
        self.edge_priors = np.copy(priors)
        self.edge_visits = np.zeros(num_actions, dtype=np.int32)
        self.edge_total_value = np.zeros(num_actions, dtype=np.float64)
        self.children: Dict[int, "DAGBeliefNode"] = {}

        # Incoming parent references (parent_node, action_idx) for DAG propagation
        self.parents: List[Tuple["DAGBeliefNode", int]] = []

    @property
    def node_q_value(self) -> float:
        return (self.total_value / self.visits) if self.visits > 0 else 0.0

    def edge_q_value(self, action_idx: int) -> float:
        ev = self.edge_visits[action_idx]
        child = self.children.get(action_idx, None)
        if ev > 0:
            q_edge = self.edge_total_value[action_idx] / ev
            if child is not None and child.visits > ev:
                return 0.70 * q_edge + 0.30 * child.node_q_value
            return q_edge
        return 0.0

    def select_puct_action(self, c_puct: float = 1.5, default_q: float = 0.0) -> int:
        total_visits_sqrt = math.sqrt(max(1, self.visits))

        visited_q = [self.edge_q_value(a) for a in range(self.num_actions) if self.edge_visits[a] > 0]
        if len(visited_q) >= 2:
            q_min = min(visited_q)
            q_max = max(visited_q)
            q_range = max(q_max - q_min, 1.0)
        elif len(visited_q) == 1:
            q_min = visited_q[0] - 1.0
            q_max = visited_q[0] + 1.0
            q_range = 2.0
        else:
            q_min = 0.0
            q_max = 1.0
            q_range = 1.0

        best_score = -float("inf")
        best_action = 0

        for a in range(self.num_actions):
            ev = self.edge_visits[a]
            if ev > 0:
                norm_q = (self.edge_q_value(a) - q_min) / q_range
            else:
                # First-Play Urgency (FPU): ensures all unvisited root/internal actions
                # are explored cleanly without 1e-4 division explosion
                norm_q = 0.65

            u_score = c_puct * float(self.edge_priors[a]) * (total_visits_sqrt / (1.0 + float(ev)))
            score = norm_q + u_score
            if score > best_score:
                best_score = score
                best_action = a

        return best_action


class LatentBeliefTranspositionTable:
    """
    Depth-stratified locality-sensitive transposition table over
    (Hunter Kinematics, PointNet Latent Embedding z).
    """
    def __init__(
        self,
        pos_threshold: float = 24.0,
        vel_threshold: float = 1.6,
        cosine_threshold: float = 0.02
    ):
        self.pos_threshold = pos_threshold
        self.vel_threshold = vel_threshold
        self.cosine_threshold = cosine_threshold
        self.buckets: Dict[int, List[DAGBeliefNode]] = {}
        self.next_node_id = 0
        self.total_queries = 0
        self.merge_hits = 0

    def find_or_create(
        self,
        depth: int,
        h_pos: np.ndarray,
        h_vel: np.ndarray,
        latent_z: np.ndarray,
        priors: np.ndarray
    ) -> Tuple[DAGBeliefNode, bool]:
        if depth > 0:
            self.total_queries += 1

        bucket = self.buckets.setdefault(depth, [])
        for candidate in bucket:
            pos_dist = float(np.linalg.norm(h_pos - candidate.h_pos))
            if pos_dist <= self.pos_threshold:
                vel_dist = float(np.linalg.norm(h_vel - candidate.h_vel))
                if vel_dist <= self.vel_threshold:
                    cos_dist = 1.0 - float(np.dot(latent_z, candidate.latent_z))
                    if cos_dist <= self.cosine_threshold:
                        self.merge_hits += 1
                        return candidate, True

        node = DAGBeliefNode(
            node_id=self.next_node_id,
            depth=depth,
            h_pos=h_pos,
            h_vel=h_vel,
            latent_z=latent_z,
            priors=priors
        )
        self.next_node_id += 1
        bucket.append(node)
        return node, False


class BeliefGraphDeepPOMCPPolicy(BasePursuerPolicy):
    """
    DAG-Deep-POMCP: Latent Belief-Graph MCTS via PointNet Transposition Merging.
    """
    def __init__(
        self,
        net: Optional[DeepPOMCPNet] = None,
        weights_path: Optional[str] = "deep_pomcp_weights.pth",
        num_simulations: int = 60,
        max_depth: int = 6,
        c_puct: float = 1.5,
        pincer_dist: float = 200.0,
        pos_merge_threshold: float = 24.0,
        vel_merge_threshold: float = 1.6,
        cosine_merge_threshold: float = 0.02,
        sub_particles: int = 32,
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
        self.pos_merge_threshold = pos_merge_threshold
        self.vel_merge_threshold = vel_merge_threshold
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
        width: int,
        height: int
    ) -> np.ndarray:
        next_cloud = np.copy(sub_cloud)
        dx = (next_cloud[:, 2] * 7.0 * 5.0) / float(width)
        dy = (next_cloud[:, 3] * 7.0 * 5.0) / float(height)

        hx_norm = h_pos_new[0] / float(width)
        hy_norm = h_pos_new[1] / float(height)
        rx = next_cloud[:, 0] - hx_norm
        ry = next_cloud[:, 1] - hy_norm
        r_dist = np.sqrt(rx * rx + ry * ry) + 1e-4

        next_cloud[:, 0] = np.clip(next_cloud[:, 0] + dx + (rx / r_dist) * 0.008, 0.02, 0.98)
        next_cloud[:, 1] = np.clip(next_cloud[:, 1] + dy + (ry / r_dist) * 0.008, 0.02, 0.98)
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
            pos_threshold=self.pos_merge_threshold,
            vel_threshold=self.vel_merge_threshold,
            cosine_threshold=self.cosine_merge_threshold
        )
        root_node, _ = transposition_table.find_or_create(
            depth=0,
            h_pos=h_pos,
            h_vel=h_vel,
            latent_z=root_z,
            priors=root_priors
        )
        root_node.is_expanded = True

        vel_est = np.array([0.0, 0.0], dtype=np.float32)
        if particles is not None and len(particles) > 0:
            vel_est = np.mean(raw_particles[:, 2:4], axis=0) * 7.0

        sim_depths_reached: List[int] = []

        for _ in range(n_sims):
            node = root_node
            sub_cloud = root_sub_cloud.copy()

            i_pos_guess = b_mean + np.random.normal(0, 8.0, size=2)
            i_vel_guess = vel_est + np.random.normal(0, 0.5, size=2).astype(np.float32)

            state = DeepSimulationState(
                h_pos, h_vel, i_pos_guess, i_vel_guess,
                width, height, ally_pos, ally_target,
                pincer_dist=self.pincer_dist
            )
            initial_dist = float(np.linalg.norm(h_pos - i_pos_guess))
            prev_dist = initial_dist

            depth = 0
            traversed_edges: List[Tuple[DAGBeliefNode, int, DAGBeliefNode]] = []
            cum_reward = 0.0

            while depth < self.max_depth:
                action_idx = node.select_puct_action(self.c_puct, default_q=root_value)
                r, is_terminal = state.step_multi(self.action_vectors[action_idx], frames=5)

                if not is_terminal and obstacles is not None:
                    for obs_rect in obstacles:
                        if obs_rect.collides_point(state.h_pos[0], state.h_pos[1], buffer=6.0):
                            r -= 12.0
                            break

                cum_reward += r
                depth += 1

                if action_idx in node.children:
                    child_node = node.children[action_idx]
                    sub_cloud = self._propagate_belief_subcloud(sub_cloud, state.h_pos, width, height)
                else:
                    sub_cloud = self._propagate_belief_subcloud(sub_cloud, state.h_pos, width, height)
                    latent_z = self.fast_pointnet.encode(sub_cloud)

                    child_node, _ = transposition_table.find_or_create(
                        depth=depth,
                        h_pos=state.h_pos,
                        h_vel=state.h_vel,
                        latent_z=latent_z,
                        priors=root_priors
                    )
                    node.children[action_idx] = child_node
                    child_node.parents.append((node, action_idx))

                traversed_edges.append((node, action_idx, child_node))
                node = child_node

                if is_terminal:
                    node.is_terminal = True
                    break

                if not node.is_expanded:
                    node.is_expanded = True
                    break

            sim_depths_reached.append(depth)

            if node.is_terminal:
                leaf_value = 500.0 + (self.max_depth - depth) * 20.0
            else:
                final_dist = float(np.linalg.norm(state.h_pos - state.i_pos))
                mean_step_r = cum_reward / float(max(1, depth))
                total_progress = (initial_dist - final_dist) / 8.0
                leaf_value = mean_step_r * 2.0 + total_progress + root_value * 5.0

            # Backpropagate along the traversed DAG path and pool statistics at shared nodes
            node.visits += 1
            node.total_value += leaf_value
            cur_val = leaf_value
            for parent_node, act_idx, child_n in reversed(traversed_edges):
                parent_node.edge_visits[act_idx] += 1
                parent_node.edge_total_value[act_idx] += cur_val
                parent_node.visits += 1
                parent_node.total_value += cur_val

                # BROPOP Cross-Parent Propagation across merged DAG edges
                if len(child_n.parents) > 1:
                    for sib_parent, sib_act in child_n.parents:
                        if sib_parent is not parent_node and sib_parent.edge_visits[sib_act] > 0:
                            sib_parent.edge_total_value[sib_act] = (
                                0.85 * sib_parent.edge_total_value[sib_act]
                                + 0.15 * cur_val * sib_parent.edge_visits[sib_act]
                            )

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
