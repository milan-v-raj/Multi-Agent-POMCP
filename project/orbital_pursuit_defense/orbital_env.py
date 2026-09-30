"""
3D Orbital Pursuit-Defense Environment (Full 24-Hour Horizon & Dual-Metric Evaluation)
Adheres strictly to the physical parameters, multi-tiered reward formulations (Eq. 22-31),
and continuous cumulative duration metrics (t_s^P and t_s^D) established in the research paper:
'Impulsive maneuver spacecraft pursuit-defense game based on multi-agent reinforcement learning'
(Fan Shuhui, Zhang Xiang, Liao Wenhe, 2025).
"""

import math
import numpy as np
from orbital_dynamics import (
    MU_EARTH, A_GEO, N_GEO, propagate_cw_fast, compute_j2_perturbation_lvlh
)


class Spacecraft:
    def __init__(self, name, r_init, v_init, total_fuel, max_single_dv, dt_min, dt_max, t_sam):
        self.name = name
        self.r = np.array(r_init, dtype=np.float64)      # [x, y, z] in meters
        self.v = np.array(v_init, dtype=np.float64)      # [vx, vy, vz] in m/s
        self.total_fuel = float(total_fuel)              # Total delta_v budget (m/s)
        self.fuel_remaining = float(total_fuel)          # Remaining delta_v (m/s)
        self.max_single_dv = float(max_single_dv)        # Max delta_v per impulse (m/s)
        self.dt_min = float(dt_min)                      # Min time between impulses (seconds)
        self.dt_max = float(dt_max)                      # Max time between impulses (seconds)
        self.t_sam = float(t_sam)                        # Perception cycle (seconds)
        
        # Action & Trajectory history
        self.impulse_history = []                        # Tuples of (timestamp, delta_v_vector, actual_dv)
        self.trajectory_history = [(0.0, self.r.copy(), self.v.copy())]
        self.last_decision_r = self.r.copy()

    def apply_impulse(self, delta_v_vector, current_time):
        """Applies an impulsive velocity increment delta_v respecting the fuel budget."""
        self.last_decision_r = self.r.copy()
        dv_mag = np.linalg.norm(delta_v_vector)
        if dv_mag > self.max_single_dv:
            delta_v_vector = (delta_v_vector / dv_mag) * self.max_single_dv
            dv_mag = self.max_single_dv
            
        actual_dv = min(dv_mag, max(0.0, self.fuel_remaining))
        if dv_mag > 1e-6 and actual_dv > 0:
            effective_dv_vector = (delta_v_vector / dv_mag) * actual_dv
        else:
            effective_dv_vector = np.zeros(3, dtype=np.float64)
            
        self.v += effective_dv_vector
        self.fuel_remaining -= actual_dv
        self.impulse_history.append((current_time, effective_dv_vector.copy(), actual_dv))
        return effective_dv_vector


class OrbitalPursuitDefenseEnv:
    def __init__(self, t_max_hours=24.0, enable_perturbations=True):
        self.n = N_GEO
        self.enable_perturbations = enable_perturbations
        self.t_max = t_max_hours * 3600.0  # 24 hours = 86,400s
        
        # Distance thresholds (Table 2 of paper)
        self.D_T = 20000.0          # Target safe distance: 20 km (meters)
        self.D_P = 10000.0          # Pursuer safe distance / capture radius: 10 km (meters)
        
        # Paper reward constants (Table 3 of paper)
        self.k_scale = 1.0 / 600.0  # Scaling factor k = 1/600
        self.r_s_p = 2.0            # r_s^P = 2
        self.r_s_d = 2.0            # r_s^D = 2
        self.r_cap_p = 1.0          # r_cap^P = 1
        self.r_u_p = 1.0            # r_u^P = 1
        self.r_u_d = 1.0            # r_u^D = 1
        self.T_pro = 3600.0         # Prediction horizon T_pro = 1 hour
        
        self.current_time = 0.0
        self.target_pos = np.zeros(3, dtype=np.float64)
        
        # Cumulative duration trackers (Section 2.3 & Tables 6 & 7)
        self.t_s_p = 0.0            # Cumulative seconds with (d_PT <= 20km and d_PD > 10km)
        self.t_s_d = 0.0            # Cumulative seconds with (d_PD <= 10km)
        
        self.pursuer = None
        self.defender = None
        self.reset()

    def reset(self, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.current_time = 0.0
        self.t_s_p = 0.0
        self.t_s_d = 0.0
        
        # --- INITIAL ORBITAL ELEMENTS (Table 1 of paper) ---
        # Defender: delta_theta in [8.65, 8.69] U [8.71, 8.75] deg relative to Target (~20-40km along-track)
        side_d = 1.0 if np.random.rand() > 0.5 else -1.0
        d_theta_d_deg = side_d * np.random.uniform(0.01, 0.05)
        r_y_d = math.radians(d_theta_d_deg) * A_GEO
        r_x_d = np.random.uniform(-2000.0, 2000.0)
        r_z_d = np.random.uniform(-1000.0, 1000.0)
        r_d_init = np.array([r_x_d, r_y_d, r_z_d])
        v_d_init = np.array([0.0, -1.5 * self.n * r_x_d, 0.0])
        
        # Pursuer: a in [42144, 42184] km (+- 20km radial), delta_theta in [8.2, 8.5] U [8.9, 9.2] deg (~150-350km)
        side_p = 1.0 if np.random.rand() > 0.5 else -1.0
        d_theta_p_deg = side_p * np.random.uniform(0.2, 0.5)
        r_y_p = math.radians(d_theta_p_deg) * A_GEO
        r_x_p = np.random.uniform(-20000.0, 20000.0)
        r_z_p = np.random.uniform(-5000.0, 5000.0)
        r_p_init = np.array([r_x_p, r_y_p, r_z_p])
        v_p_init = np.array([0.0, -1.5 * self.n * r_x_p, 0.0])
        
        self.pursuer = Spacecraft(
            name="Pursuer",
            r_init=r_p_init,
            v_init=v_p_init,
            total_fuel=20.0,          # U_P = 20 m/s
            max_single_dv=2.0,        # dv_max = 2.0 m/s
            dt_min=3600.0,            # dt_min = 1.0 h
            dt_max=21600.0,           # dt_max = 6.0 h
            t_sam=600.0               # Perception cycle: 600s
        )
        
        self.defender = Spacecraft(
            name="Defender",
            r_init=r_d_init,
            v_init=v_d_init,
            total_fuel=15.0,          # U_D = 15 m/s
            max_single_dv=1.0,        # dv_max = 1.0 m/s
            dt_min=1800.0,            # dt_min = 0.5 h
            dt_max=21600.0,           # dt_max = 6.0 h
            t_sam=300.0               # Perception cycle: 300s
        )
        
        return self._get_observation()

    def step_ballistic(self, dt):
        """
        Propagates spacecraft over dt seconds and accumulates success durations t_s^P and t_s^D.
        """
        if dt <= 0.0:
            return 0.0, 0.0
            
        r_p_next, v_p_next = propagate_cw_fast(self.pursuer.r, self.pursuer.v, dt, self.n)
        r_d_next, v_d_next = propagate_cw_fast(self.defender.r, self.defender.v, dt, self.n)
        
        if self.enable_perturbations:
            a_j2_p = compute_j2_perturbation_lvlh(self.pursuer.r)
            a_j2_d = compute_j2_perturbation_lvlh(self.defender.r)
            v_p_next += a_j2_p * dt
            v_d_next += a_j2_d * dt
            
        self.pursuer.r, self.pursuer.v = r_p_next, v_p_next
        self.defender.r, self.defender.v = r_d_next, v_d_next
        self.current_time += dt
        
        # Evaluate instantaneous condition
        d_pt = np.linalg.norm(self.pursuer.r)
        d_pd = np.linalg.norm(self.pursuer.r - self.defender.r)
        
        step_ts_p = dt if (d_pt <= self.D_T and d_pd > self.D_P) else 0.0
        step_ts_d = dt if (d_pd <= self.D_P) else 0.0
        
        self.t_s_p += step_ts_p
        self.t_s_d += step_ts_d
        
        self.pursuer.trajectory_history.append((self.current_time, self.pursuer.r.copy(), self.pursuer.v.copy()))
        self.defender.trajectory_history.append((self.current_time, self.defender.r.copy(), self.defender.v.copy()))
        
        return step_ts_p, step_ts_d

    def calculate_paper_reward(self, step_ts_p, step_ts_d, d_pt_prev, d_pd_prev):
        """
        Calculates exact multi-tiered reward according to Equations 22-31 of the paper.
        """
        d_pt = np.linalg.norm(self.pursuer.r)
        d_pd = np.linalg.norm(self.pursuer.r - self.defender.r)
        
        # 1. Passive flight duration reward (Eq. 22 & 23)
        u_p_ratio = max(0.0, self.pursuer.fuel_remaining) / self.pursuer.total_fuel
        u_d_ratio = max(0.0, self.defender.fuel_remaining) / self.defender.total_fuel
        
        R_d_P = self.k_scale * (u_p_ratio * step_ts_p - step_ts_d)
        R_d_D = self.k_scale * (u_d_ratio * step_ts_d)
        
        # 2. Predicted flight guidance reward (Eq. 24 & 25)
        # Forward propagate state by T_pro to forecast trajectory trend
        r_p_pred, _ = propagate_cw_fast(self.pursuer.r, self.pursuer.v, self.T_pro, self.n)
        r_d_pred, _ = propagate_cw_fast(self.defender.r, self.defender.v, self.T_pro, self.n)
        
        d_pt_pred = np.linalg.norm(r_p_pred)
        d_pd_pred = np.linalg.norm(r_p_pred - r_d_pred)
        
        R_g_P = (d_pt_prev - d_pt_pred) / max(d_pt_prev, 1000.0)
        R_g_D = (d_pd_prev - d_pd_pred) / max(d_pd_prev, 1000.0)
        
        # 3. Result reward (Eq. 26 & 27)
        R_r_P = 0.0
        if d_pt_pred <= self.D_T and d_pd_pred > self.D_P:
            R_r_P = self.r_s_p
        elif d_pd_pred <= self.D_P:
            R_r_P = -self.r_cap_p
            
        R_r_D = self.r_s_d if d_pd_pred <= self.D_P else 0.0
        
        # 4. Fuel penalty (Eq. 28 & 29)
        R_u_P = -self.r_u_p if self.pursuer.fuel_remaining < 0 else 0.0
        R_u_D = -self.r_u_d if self.defender.fuel_remaining < 0 else 0.0
        
        # Total Joint Reward (Eq. 30 & 31)
        reward_P = R_d_P + R_g_P + R_r_P + R_u_P
        reward_D = R_d_D + R_g_D + R_r_D + R_u_D
        
        return reward_P, reward_D

    def _get_observation(self):
        """Returns noisy observations for Pursuer and Defender."""
        noise_p = np.random.normal(0.0, 50.0, size=3)
        noise_d = np.random.normal(0.0, 50.0, size=3)
        
        obs = {
            "time": self.current_time,
            "pursuer": {
                "r": self.pursuer.r.copy(),
                "v": self.pursuer.v.copy(),
                "obs_defender_r": self.defender.r.copy() + noise_p,
                "fuel_rem": self.pursuer.fuel_remaining
            },
            "defender": {
                "r": self.defender.r.copy(),
                "v": self.defender.v.copy(),
                "obs_pursuer_r": self.pursuer.r.copy() + noise_d,
                "fuel_rem": self.defender.fuel_remaining
            }
        }
        return obs

    def check_game_status(self):
        """
        Evaluates game termination: The game continues until full horizon t_max (24 hours).
        """
        d_pt = np.linalg.norm(self.pursuer.r - self.target_pos)
        d_pd = np.linalg.norm(self.pursuer.r - self.defender.r)
        
        done = self.current_time >= self.t_max
        status = "IN_PROGRESS"
        
        if done:
            if self.t_s_p > 0 and self.t_s_d == 0:
                status = "PURSUER_DOMINANT_WIN"
            elif self.t_s_p > self.t_s_d:
                status = "PURSUER_ADVANTAGE"
            elif self.t_s_d > self.t_s_p:
                status = "DEFENDER_ADVANTAGE"
            elif self.t_s_d > 0 and self.t_s_p == 0:
                status = "DEFENDER_INTERCEPT_WIN"
            else:
                status = "DRAW_TIMEOUT"
                
        return done, status, d_pt, d_pd, self.t_s_p, self.t_s_d
