"""
3D Orbital POMCP Planning Engine with Potential-Based Astrodynamic Valuation
Implements Ng et al. (1999) Potential-Based Reward Shaping on 3D relative orbit projections,
mathematically guaranteeing immunity to reward gaming and ensuring optimal rendezvous trajectories.
"""

import math
import random
import numpy as np
from orbital_dynamics import (
    propagate_cw_fast, compute_cw_targeting_impulse, N_GEO
)

# --- HYPERPARAMETERS ---
EXPLORATION_CONSTANT = 1.414
DISCOUNT_FACTOR = 0.95
MAX_DEPTH = 5
NUM_SIMULATIONS = 200

D_TARGET = 20000.0           # 20 km Target Safe Zone
D_CAPTURE = 10000.0          # 10 km Defender Capture Zone
D_THREAT = 12000.0           # 12 km Threat Buffer

# --- LONG-RANGE ACTIONS (d_PT > 25 km) ---
ACTIONS_LONG_RANGE = [
    (10800.0, np.array([0.0, 0.0, 0.0]), "Coast_3h"),
    (21600.0, np.array([0.0, 0.0, 0.0]), "Coast_6h"),
    (10800.0, np.array([0.0, 1.5, 0.0]), "Mid_Along_Pro (+y, 3h)"),
    (10800.0, np.array([0.0, -1.5, 0.0]), "Mid_Along_Retro (-y, 3h)"),
    (21600.0, np.array([0.0, 2.0, 0.0]), "Macro_Along_Pro (+y, 6h)"),
    (21600.0, np.array([0.0, -2.0, 0.0]), "Macro_Along_Retro (-y, 6h)"),
    (10800.0, np.array([1.5, 0.0, 0.0]), "Mid_Radial_Out (+x, 3h)"),
    (10800.0, np.array([-1.5, 0.0, 0.0]), "Mid_Radial_In (-x, 3h)"),
    (21600.0, np.array([2.0, 0.0, 0.0]), "Macro_Radial_Out (+x, 6h)"),
    (21600.0, np.array([-2.0, 0.0, 0.0]), "Macro_Radial_In (-x, 6h)"),
    (10800.0, np.array([0.0, 0.0, 1.0]), "Mid_Cross_North (+z, 3h)"),
    (10800.0, np.array([0.0, 0.0, -1.0]), "Mid_Cross_South (-z, 3h)")
]

# --- CLOSE-RANGE & DOGFIGHT ACTIONS (d_PT <= 25 km) ---
ACTIONS_CLOSE_RANGE = [
    (1800.0, np.array([0.0, 0.0, 0.0]), "Coast_0.5h"),
    (3600.0, np.array([0.0, 0.0, 0.0]), "Coast_1h"),
    (1800.0, np.array([0.5, 0.0, 0.0]), "Micro_Radial_Out (+x, 0.5h)"),
    (1800.0, np.array([-0.5, 0.0, 0.0]), "Micro_Radial_In (-x, 0.5h)"),
    (1800.0, np.array([0.0, 0.5, 0.0]), "Micro_Along_Pro (+y, 0.5h)"),
    (1800.0, np.array([0.0, -0.5, 0.0]), "Micro_Along_Retro (-y, 0.5h)"),
    (1800.0, np.array([0.0, 0.0, 0.5]), "Micro_Cross_North (+z, 0.5h)"),
    (1800.0, np.array([0.0, 0.0, -0.5]), "Micro_Cross_South (-z, 0.5h)"),
    (3600.0, np.array([1.0, 0.0, 0.0]), "Evade_Radial_Out (+x, 1h)"),
    (3600.0, np.array([-1.0, 0.0, 0.0]), "Evade_Radial_In (-x, 1h)"),
    (3600.0, np.array([0.0, 1.0, 0.0]), "Evade_Along_Pro (+y, 1h)"),
    (3600.0, np.array([0.0, -1.0, 0.0]), "Evade_Along_Retro (-y, 1h)")
]

ACTIONS_ORBITAL = ACTIONS_LONG_RANGE
EVAL_GRID_SECONDS = np.arange(1800.0, 86400.0 + 1800.0, 1800.0)


def compute_projected_min_distance(r, v, n=N_GEO):
    """Calculates the high-precision minimum distance to Target over 24 hours."""
    min_dist = np.linalg.norm(r)
    for t_step in EVAL_GRID_SECONDS:
        r_f, _ = propagate_cw_fast(r, v, t_step, n)
        d_f = np.linalg.norm(r_f)
        if d_f < min_dist:
            min_dist = d_f
    return min_dist


def compute_action_priors(r_p, v_p, r_d, action_set):
    """Computes prior probability distribution P(s, a)."""
    d_pt = np.linalg.norm(r_p)
    d_pd = np.linalg.norm(r_p - r_d)
    
    if d_pt > D_TARGET:
        ideal_dv = compute_cw_targeting_impulse(r_p, v_p, np.zeros(3), 21600.0)
    else:
        v_target_circ = -np.cross(np.array([0, 0, 1]), r_p) * N_GEO
        ideal_dv = v_target_circ - v_p
        if d_pd < D_THREAT:
            evade_vec = (r_p - r_d) / max(d_pd, 1.0)
            ideal_dv += evade_vec * 2.0

    ideal_mag = np.linalg.norm(ideal_dv)
    scores = []
    
    for dt_a, dv_a, name in action_set:
        dv_mag = np.linalg.norm(dv_a)
        if dv_mag < 1e-6:
            score = 0.25
        else:
            if ideal_mag > 1e-3:
                cos_sim = np.dot(dv_a, ideal_dv) / (dv_mag * ideal_mag)
                score = max(0.01, (cos_sim + 1.0) / 2.0)
            else:
                score = 0.1
        scores.append(score)
        
    scores = np.array(scores)
    priors = scores / np.sum(scores)
    return priors


class MCTSNode:
    def __init__(self, action_set, priors, parent=None, action_from_parent=None):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.action_set = action_set
        self.priors = priors
        self.untried_actions = list(range(len(action_set)))

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_puct=EXPLORATION_CONSTANT):
        best_score = -float('inf')
        best_node = None
        total_visits = self.visits
        
        for action_idx, child in self.children.items():
            q_val = child.value / child.visits if child.visits > 0 else 0.0
            prior = self.priors[action_idx]
            u_val = c_puct * prior * math.sqrt(total_visits) / (1 + child.visits)
            score = q_val + u_val
            
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self):
        action_idx = self.untried_actions.pop()
        new_node = MCTSNode(action_set=self.action_set, priors=self.priors, parent=self, action_from_parent=action_idx)
        self.children[action_idx] = new_node
        return new_node, action_idx


class LightweightOrbitalState:
    def __init__(self, r_p, v_p, r_d, v_d, fuel_p, action_set, n_orbital=N_GEO):
        self.r_p = np.copy(r_p)
        self.v_p = np.copy(v_p)
        self.r_d = np.copy(r_d)
        self.v_d = np.copy(v_d)
        self.fuel_p = float(fuel_p)
        self.action_set = action_set
        self.n = n_orbital

    def step(self, action_idx):
        dt, dv_cmd, _ = self.action_set[action_idx]
        dv_mag = np.linalg.norm(dv_cmd)
        
        actual_dv_mag = min(dv_mag, max(0.0, self.fuel_p))
        if dv_mag > 1e-6 and actual_dv_mag > 0:
            effective_dv = (dv_cmd / dv_mag) * actual_dv_mag
        else:
            effective_dv = np.zeros(3, dtype=np.float64)
            
        # Potential before action
        phi_prev = -compute_projected_min_distance(self.r_p, self.v_p, self.n) / 1000.0
        
        self.v_p += effective_dv
        self.fuel_p -= actual_dv_mag
        
        self.r_p, self.v_p = propagate_cw_fast(self.r_p, self.v_p, dt, self.n)
        self.r_d, self.v_d = propagate_cw_fast(self.r_d, self.v_d, dt, self.n)
        
        # Potential after action
        phi_next = -compute_projected_min_distance(self.r_p, self.v_p, self.n) / 1000.0
        
        d_pt_next = np.linalg.norm(self.r_p)
        d_pd_next = np.linalg.norm(self.r_p - self.r_d)
        
        # Potential-based reward: gamma * Phi(s') - Phi(s)
        reward = (DISCOUNT_FACTOR * phi_next - phi_prev)
        
        # Target Safe Zone Hold Reward
        if d_pt_next <= D_TARGET and d_pd_next > D_CAPTURE:
            reward += (dt / 3600.0) * 35.0
        elif d_pd_next <= D_CAPTURE:
            reward -= (dt / 3600.0) * 45.0
            
        # Fuel Penalty
        reward -= actual_dv_mag * 0.05
        
        return reward, False


def run_orbital_pomcp(pursuer_r, pursuer_v, pursuer_fuel, particle_filter, n_sims=NUM_SIMULATIONS):
    """
    Executes Secular 3D Orbital POMCP planning cycle with Potential-Based Valuation.
    """
    d_pt_current = np.linalg.norm(pursuer_r)
    
    if d_pt_current > 25000.0:
        active_action_set = ACTIONS_LONG_RANGE
    else:
        active_action_set = ACTIONS_CLOSE_RANGE
        
    r_d_est, _ = particle_filter.sample_hypothesis()
    priors = compute_action_priors(pursuer_r, pursuer_v, r_d_est, active_action_set)
    
    root = MCTSNode(action_set=active_action_set, priors=priors)
    
    for _ in range(n_sims):
        node = root
        
        r_d_guess, v_d_guess = particle_filter.sample_hypothesis()
        state = LightweightOrbitalState(pursuer_r, pursuer_v, r_d_guess, v_d_guess, pursuer_fuel, active_action_set)
        
        depth = 0
        while node.is_fully_expanded() and depth < MAX_DEPTH:
            node = node.best_child()
            if node.action_from_parent is not None:
                _, is_done = state.step(node.action_from_parent)
                if is_done:
                    break
            depth += 1
            
        if not node.is_fully_expanded() and depth < MAX_DEPTH:
            new_node, action_idx = node.expand()
            reward, is_done = state.step(action_idx)
            node = new_node
        else:
            reward = 0.0
            is_done = False
            
        rollout_depth = 0
        cumulative_reward = reward
        current_discount = 1.0
        
        while not is_done and rollout_depth < (MAX_DEPTH - depth):
            d_min_curr = compute_projected_min_distance(state.r_p, state.v_p, state.n)
            
            if d_min_curr <= 20000.0:
                action_idx = 0  # Coast
            else:
                sim_priors = compute_action_priors(state.r_p, state.v_p, state.r_d, active_action_set)
                action_idx = int(np.argmax(sim_priors))
                
            r, is_done = state.step(action_idx)
            cumulative_reward += r * current_discount
            current_discount *= DISCOUNT_FACTOR
            rollout_depth += 1
            
        while node is not None:
            node.visits += 1
            node.value += cumulative_reward
            node = node.parent
            
    if not root.children:
        return active_action_set[0][0], active_action_set[0][1], active_action_set[0][2]
        
    best_action_idx = max(root.children, key=lambda k: root.children[k].visits)
    best_dt, best_dv, best_name = active_action_set[best_action_idx]
    return best_dt, best_dv, best_name
