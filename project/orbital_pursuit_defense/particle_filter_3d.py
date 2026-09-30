"""
6-DoF Particle Filter for 3D Orbital State Estimation
Tracks relative position [x, y, z] and velocity [vx, vy, vz] of an opponent spacecraft
under intermittent, noisy observations using Clohessy-Wiltshire (CW) kinematics propagation
and Systematic Resampling (SIR).
"""

import math
import numpy as np
from orbital_dynamics import propagate_cw_fast, N_GEO


class Particle3D:
    def __init__(self, r, v, weight=1.0):
        self.r = np.array(r, dtype=np.float64)  # [x, y, z] meters
        self.v = np.array(v, dtype=np.float64)  # [vx, vy, vz] m/s
        self.weight = float(weight)

    def copy(self):
        return Particle3D(self.r.copy(), self.v.copy(), self.weight)


class ParticleFilter3D:
    def __init__(self, num_particles=300, n_orbital=N_GEO, pos_noise_std=50.0, vel_noise_std=0.05):
        self.num_particles = num_particles
        self.n = n_orbital
        self.pos_noise_std = pos_noise_std  # Process noise std for position (meters)
        self.vel_noise_std = vel_noise_std  # Process noise std for velocity (m/s)
        self.particles = []
        self.is_initialized = False

    def initialize_around_state(self, mean_r, mean_v, r_spread=500.0, v_spread=0.5):
        """Initializes particles with a Gaussian distribution around an initial observation."""
        self.particles = []
        for _ in range(self.num_particles):
            r_sample = mean_r + np.random.normal(0.0, r_spread, size=3)
            v_sample = mean_v + np.random.normal(0.0, v_spread, size=3)
            self.particles.append(Particle3D(r_sample, v_sample, 1.0 / self.num_particles))
        self.is_initialized = True

    def predict(self, dt):
        """Propagates all particles forward by dt seconds using CW dynamics and process noise."""
        if not self.is_initialized:
            return

        for p in self.particles:
            r_next, v_next = propagate_cw_fast(p.r, p.v, dt, self.n)
            # Add stochastic perturbation / maneuver uncertainty
            r_noise = np.random.normal(0.0, self.pos_noise_std * math.sqrt(max(1.0, dt / 60.0)), size=3)
            v_noise = np.random.normal(0.0, self.vel_noise_std * math.sqrt(max(1.0, dt / 60.0)), size=3)
            p.r = r_next + r_noise
            p.v = v_next + v_noise

    def update(self, observation, obs_noise_std=100.0):
        """
        Updates particle weights based on a 3D position observation [x, y, z].
        observation: np.ndarray shape (3,) in meters, or None if occluded/no-measurement.
        """
        if observation is None or not self.is_initialized:
            return

        obs = np.array(observation, dtype=np.float64)
        total_weight = 0.0
        var = obs_noise_std ** 2

        for p in self.particles:
            dist_sq = np.sum((p.r - obs) ** 2)
            # Gaussian likelihood
            likelihood = math.exp(-0.5 * dist_sq / var) + 1e-12
            p.weight *= likelihood
            total_weight += p.weight

        # Normalize weights
        if total_weight > 1e-15:
            for p in self.particles:
                p.weight /= total_weight
        else:
            # Degeneracy recovery: reset uniform weights around observation
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

        # Check Effective Sample Size (N_eff) and conditionally perform SIR Resampling
        n_eff = self.compute_effective_sample_size()
        if n_eff < self.num_particles / 2.0:
            self.systematic_resample()

    def compute_effective_sample_size(self):
        sq_sum = sum(p.weight ** 2 for p in self.particles)
        return 1.0 / max(sq_sum, 1e-15)

    def systematic_resample(self):
        """Systematic Independence Resampling (SIR)."""
        weights = np.array([p.weight for p in self.particles])
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Avoid numerical rounding drift

        step = 1.0 / self.num_particles
        u0 = np.random.uniform(0.0, step)
        pointers = u0 + np.arange(self.num_particles) * step

        new_particles = []
        idx = 0
        for ptr in pointers:
            while ptr > cumulative_sum[idx] and idx < self.num_particles - 1:
                idx += 1
            chosen = self.particles[idx]
            new_particles.append(Particle3D(chosen.r.copy(), chosen.v.copy(), 1.0 / self.num_particles))

        self.particles = new_particles

    def get_belief_estimate(self):
        """Returns the weighted mean 3D position, velocity, and spatial covariance spread (sigma)."""
        if not self.is_initialized or not self.particles:
            return np.zeros(3), np.zeros(3), 0.0

        mean_r = np.zeros(3)
        mean_v = np.zeros(3)
        for p in self.particles:
            mean_r += p.weight * p.r
            mean_v += p.weight * p.v

        # Compute spatial standard deviation sigma_cloud
        var_cloud = sum(p.weight * np.sum((p.r - mean_r) ** 2) for p in self.particles)
        sigma_cloud = math.sqrt(max(0.0, var_cloud))

        return mean_r, mean_v, sigma_cloud

    def sample_hypothesis(self):
        """Samples a single state hypothesis (r, v) proportional to particle weights."""
        if not self.is_initialized or not self.particles:
            return np.zeros(3), np.zeros(3)
        weights = [p.weight for p in self.particles]
        chosen = np.random.choice(self.particles, p=weights)
        return chosen.r.copy(), chosen.v.copy()

