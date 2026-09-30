"""
3D Orbital Dynamics Module
Implements Clohessy-Wiltshire (CW) relative orbital equations of motion in the LVLH frame,
analytical state transition matrices, J2 perturbations, and analytical orbital rendezvous guidance.
"""

import math
import numpy as np

# --- PHYSICAL CONSTANTS ---
MU_EARTH = 3.986004418e14  # Earth gravitational parameter (m^3 / s^2)
R_EARTH = 6378137.0         # Earth equatorial radius (m)
J2 = 1.08263e-3            # Earth J2 zonal harmonic coefficient

# Default GEO target semi-major axis (Table 1: 42164.0 km)
A_GEO = 42164000.0         # meters
N_GEO = math.sqrt(MU_EARTH / (A_GEO ** 3))  # ~7.292115e-5 rad/s

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True)
def get_cw_submatrices(dt, n=N_GEO):
    """
    Computes the 3x3 sub-matrices for the 6x6 Clohessy-Wiltshire state transition matrix.
    """
    phi = n * dt
    cos_p = math.cos(phi)
    sin_p = math.sin(phi)
    
    # Phi_rr (Position from Position)
    phi_rr = np.zeros((3, 3), dtype=np.float64)
    phi_rr[0, 0] = 4.0 - 3.0 * cos_p
    phi_rr[1, 0] = 6.0 * (sin_p - phi)
    phi_rr[1, 1] = 1.0
    phi_rr[2, 2] = cos_p

    # Phi_rv (Position from Velocity)
    phi_rv = np.zeros((3, 3), dtype=np.float64)
    phi_rv[0, 0] = sin_p / n
    phi_rv[0, 1] = 2.0 * (1.0 - cos_p) / n
    phi_rv[1, 0] = 2.0 * (cos_p - 1.0) / n
    phi_rv[1, 1] = (4.0 * sin_p - 3.0 * phi) / n
    phi_rv[2, 2] = sin_p / n

    # Phi_vr (Velocity from Position)
    phi_vr = np.zeros((3, 3), dtype=np.float64)
    phi_vr[0, 0] = 3.0 * n * sin_p
    phi_vr[1, 0] = 6.0 * n * (cos_p - 1.0)
    phi_vr[2, 2] = -n * sin_p

    # Phi_vv (Velocity from Velocity)
    phi_vv = np.zeros((3, 3), dtype=np.float64)
    phi_vv[0, 0] = cos_p
    phi_vv[0, 1] = 2.0 * sin_p
    phi_vv[1, 0] = -2.0 * sin_p
    phi_vv[1, 1] = 4.0 * cos_p - 3.0
    phi_vv[2, 2] = cos_p

    return phi_rr, phi_rv, phi_vr, phi_vv


@njit(fastmath=True)
def propagate_cw_fast(r, v, dt, n=N_GEO):
    """
    Propagates 3D position r and velocity v over time interval dt using CW equations.
    """
    phi_rr, phi_rv, phi_vr, phi_vv = get_cw_submatrices(dt, n)
    r_next = np.dot(phi_rr, r) + np.dot(phi_rv, v)
    v_next = np.dot(phi_vr, r) + np.dot(phi_vv, v)
    return r_next, v_next


@njit(fastmath=True)
def compute_cw_targeting_impulse(r_curr, v_curr, r_target, dt, n=N_GEO):
    """
    Exact Analytical Orbital Rendezvous and Phasing Guidance:
    - ry > 0: Burn Prograde (dv_y > 0) to drift backwards
    - ry < 0: Burn Retrograde (dv_y < 0) to drift forwards
    - Near Target: Match circular orbit speed and circularize
    """
    dr = r_curr - r_target
    dist_total = math.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
    
    if dist_total <= 25000.0:
        # Station-keeping circularization at target
        dv_x = -v_curr[0] - 0.5 * n * dr[0]
        dv_y = -v_curr[1] - 1.5 * n * dr[0] - 0.2 * n * dr[1]
        dv_z = -v_curr[2] - 0.5 * n * dr[2]
        return np.array([dv_x, dv_y, dv_z], dtype=np.float64)
        
    t_transit = max(10.0 * 3600.0, dt)
    y_sign = 1.0 if dr[1] >= 0.0 else -1.0
    desired_drift_dv_y = y_sign * min(2.0, abs(dr[1]) / (3.0 * t_transit))
    
    dv_y = desired_drift_dv_y - (v_curr[1] + 1.5 * n * dr[0])
    dv_x = -v_curr[0] * 0.8 - (dr[0] * n * 0.4)
    dv_z = -v_curr[2] * 0.8 - (dr[2] * n * 0.4)
    
    return np.array([dv_x, dv_y, dv_z], dtype=np.float64)


def compute_j2_perturbation_lvlh(r_lvlh, r_target_eci=A_GEO):
    """Calculates differential J2 gravitational perturbation in the LVLH frame."""
    r_mag = r_target_eci + r_lvlh[0]
    if r_mag <= R_EARTH:
        return np.zeros(3)
        
    factor = (3.0 / 2.0) * J2 * MU_EARTH * (R_EARTH ** 2) / (r_mag ** 4)
    a_j2 = np.array([
        -factor * (1.0 - 3.0 * (r_lvlh[2] / r_mag) ** 2),
        0.0,
        -factor * 2.0 * (r_lvlh[2] / r_mag)
    ])
    return a_j2
