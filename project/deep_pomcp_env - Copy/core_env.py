import math
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from .scenarios import Obstacle, ScenarioConfig, ScenarioGenerator
from .evaders import BaseEvader, SmartRaycastEvader

class ParticleFilterInternal:
    """Internal Sequential Monte Carlo particle filter for belief tracking."""
    def __init__(self, num_particles: int = 200, width: int = 800, height: int = 600):
        self.num = num_particles
        self.width = width
        self.height = height
        self.particles_pos = np.zeros((num_particles, 2), dtype=np.float32)
        self.particles_vel = np.zeros((num_particles, 2), dtype=np.float32)
        self.is_initialized = False

    def initialize(self, pos: np.ndarray, vel: np.ndarray, np_rng: np.random.Generator):
        noise_pos = np_rng.normal(0, 5.0, size=(self.num, 2))
        noise_vel = np_rng.normal(0, 0.5, size=(self.num, 2))
        self.particles_pos = pos + noise_pos
        self.particles_vel = vel + noise_vel
        self.is_initialized = True

    def predict(self, obstacles: List[Obstacle], np_rng: np.random.Generator):
        if not self.is_initialized:
            return
        # Drift with stochastic acceleration
        self.particles_pos += self.particles_vel
        self.particles_pos += np_rng.normal(0, 1.5, size=(self.num, 2))
        # Keep inside bounds
        self.particles_pos[:, 0] = np.clip(self.particles_pos[:, 0], 10, self.width - 10)
        self.particles_pos[:, 1] = np.clip(self.particles_pos[:, 1], 10, self.height - 10)

    def update(self, observation: Optional[np.ndarray], evader_vel: np.ndarray, np_rng: np.random.Generator):
        if observation is not None:
            self.particles_pos = observation + np_rng.normal(0, 4.0, size=(self.num, 2))
            self.particles_vel = evader_vel + np_rng.normal(0, 0.3, size=(self.num, 2))
            self.is_initialized = True

    def get_belief_stats(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns (mean_pos, std_pos, cloud_spread_scalar)."""
        if not self.is_initialized:
            return np.array([self.width / 2.0, self.height / 2.0]), np.array([100.0, 100.0]), 100.0
        mean_pos = np.mean(self.particles_pos, axis=0)
        std_pos = np.std(self.particles_pos, axis=0)
        cloud_spread = float(np.mean(std_pos))
        return mean_pos, std_pos, cloud_spread

    def clear(self):
        self.is_initialized = False


class PursuitEvasionEnv(gym.Env):
    """
    Standardized Multi-Agent Pursuit-Evasion Environment (Dec-POMDP) for Deep-POMCP Benchmark.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Action mappings for discrete impulse
    FORCE_MAG = 0.5
    ACTION_VECTORS = [
        np.array([0.0, -FORCE_MAG], dtype=np.float32),  # 0: UP
        np.array([0.0, FORCE_MAG], dtype=np.float32),   # 1: DOWN
        np.array([-FORCE_MAG, 0.0], dtype=np.float32),  # 2: LEFT
        np.array([FORCE_MAG, 0.0], dtype=np.float32),   # 3: RIGHT
        np.array([0.0, 0.0], dtype=np.float32)          # 4: WAIT
    ]

    def __init__(self, config: Optional[ScenarioConfig] = None,
                 evader_policy: Optional[BaseEvader] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.config = config if config is not None else ScenarioConfig()
        self.render_mode = render_mode
        self.evader_policy = evader_policy if evader_policy is not None else SmartRaycastEvader()

        self.width = self.config.width
        self.height = self.config.height
        self.num_hunters = self.config.num_hunters
        self.max_speed_hunter = 5.0
        self.max_speed_evader = 7.0 * self.config.evader_speed_mult
        self.friction = 0.95
        self.max_steps = self.config.max_steps
        self.capture_radius = self.config.capture_radius
        self.d_safe = self.config.d_safe

        self.rng = random.Random()
        self.np_rng = np.random.default_rng()

        # Agents state buffers
        self.hunter_pos = np.zeros((self.num_hunters, 2), dtype=np.float32)
        self.hunter_vel = np.zeros((self.num_hunters, 2), dtype=np.float32)
        self.evader_pos = np.zeros(2, dtype=np.float32)
        self.evader_vel = np.zeros(2, dtype=np.float32)
        self.obstacles: List[Obstacle] = []
        self.occupancy_map = np.zeros((int(self.height / 10.0), int(self.width / 10.0)), dtype=np.float32)

        # Sensing & Belief
        self.pf = ParticleFilterInternal(num_particles=200, width=self.width, height=self.height)
        self.can_see_agents = [False] * self.num_hunters
        self.can_see_global = False
        self.last_seen_step = 0
        self.current_step = 0

        # Define Observation & Action Spaces
        obs_dim = 4 + (4 * (self.num_hunters - 1)) + 7 + 121

        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(len(self.ACTION_VECTORS))
            for i in range(self.num_hunters)
        })

        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for i in range(self.num_hunters)
        })

        # Pygame Rendering elements
        self.screen = None
        self.clock = None
        self.is_pygame_init = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.current_step = 0
        self.last_seen_step = 0

        # Generate scenario
        self.obstacles, h_spawns, e_spawn = ScenarioGenerator.generate(self.config, self.rng)
        self._build_occupancy_map()

        for i in range(self.num_hunters):
            self.hunter_pos[i] = h_spawns[i]
            self.hunter_vel[i] = np.array([0.0, 0.0], dtype=np.float32)

        self.evader_pos = np.copy(e_spawn)
        self.evader_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Initialize Particle Filter
        self.pf.initialize(self.evader_pos, self.evader_vel, self.np_rng)
        self._update_sensing()

        observations = self._get_observations()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, info

    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self.current_step += 1
        wall_hits = [False] * self.num_hunters

        # 1. Apply Hunter Actions & Kinematics
        for i in range(self.num_hunters):
            agent_key = f"agent_{i}"
            action = actions.get(agent_key, 4)
            if isinstance(action, (int, np.integer)):
                force_vec = self.ACTION_VECTORS[action]
            else:
                force_vec = np.array(action, dtype=np.float32)

            self.hunter_vel[i] += force_vec
            self.hunter_vel[i] *= self.friction
            speed = np.linalg.norm(self.hunter_vel[i])
            if speed > self.max_speed_hunter:
                self.hunter_vel[i] = (self.hunter_vel[i] / speed) * self.max_speed_hunter

            next_pos = self.hunter_pos[i] + self.hunter_vel[i]

            # Collision with walls / workspace boundaries
            next_pos, hit = self._resolve_wall_collisions(self.hunter_pos[i], next_pos, radius=10.0)
            if hit:
                wall_hits[i] = True
                self.hunter_vel[i] *= -0.5

            self.hunter_pos[i] = next_pos

        # 2. Update Evader Kinematics
        h_pos_list = [self.hunter_pos[i] for i in range(self.num_hunters)]
        evader_force = self.evader_policy.get_action(
            self.evader_pos, self.evader_vel, h_pos_list, self.obstacles, self.width, self.height
        )
        self.evader_vel += evader_force
        self.evader_vel *= self.friction
        e_speed = np.linalg.norm(self.evader_vel)
        if e_speed > self.max_speed_evader:
            self.evader_vel = (self.evader_vel / e_speed) * self.max_speed_evader

        next_e_pos = self.evader_pos + self.evader_vel
        next_e_pos, _ = self._resolve_wall_collisions(self.evader_pos, next_e_pos, radius=10.0)
        self.evader_pos = next_e_pos

        # 3. Update Sensors & Shared Particle Filter Belief
        self._update_sensing()

        # 4. Check Termination Conditions
        dists_to_evader = [float(np.linalg.norm(self.hunter_pos[i] - self.evader_pos)) for i in range(self.num_hunters)]
        min_dist = min(dists_to_evader)
        captured = min_dist < self.capture_radius
        timeout = self.current_step >= self.max_steps

        terminated = {f"agent_{i}": captured for i in range(self.num_hunters)}
        terminated["__all__"] = captured
        truncated = {f"agent_{i}": timeout for i in range(self.num_hunters)}
        truncated["__all__"] = timeout

        # 5. Compute Dec-POMDP Rewards
        rewards = self._compute_rewards(min_dist, captured, wall_hits)

        # 6. Observations & Info
        observations = self._get_observations()
        info = self._get_info()
        info["captured"] = captured
        info["timeout"] = timeout
        info["min_dist_to_evader"] = min_dist
        info["wall_hits"] = sum(wall_hits)

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, truncated, info

    def _build_occupancy_map(self):
        res = 10.0
        rows = int(self.height / res)
        cols = int(self.width / res)
        self.occupancy_map = np.zeros((rows, cols), dtype=np.float32)
        for obs in self.obstacles:
            r_min = max(0, int(obs.y / res))
            r_max = min(rows, int(math.ceil((obs.y + obs.height) / res)))
            c_min = max(0, int(obs.x / res))
            c_max = min(cols, int(math.ceil((obs.x + obs.width) / res)))
            self.occupancy_map[r_min:r_max, c_min:c_max] = 1.0

    def _get_local_occupancy_grid(self, pos: np.ndarray, grid_size: int = 11, res: float = 10.0) -> np.ndarray:
        rows, cols = self.occupancy_map.shape
        center_r = int(pos[1] / res)
        center_c = int(pos[0] / res)
        half = grid_size // 2

        r_start = center_r - half
        r_end = r_start + grid_size
        c_start = center_c - half
        c_end = c_start + grid_size

        subgrid = np.ones((grid_size, grid_size), dtype=np.float32)

        src_r_min = max(0, r_start)
        src_r_max = min(rows, r_end)
        src_c_min = max(0, c_start)
        src_c_max = min(cols, c_end)

        dst_r_min = src_r_min - r_start
        dst_r_max = dst_r_min + (src_r_max - src_r_min)
        dst_c_min = src_c_min - c_start
        dst_c_max = dst_c_min + (src_c_max - src_c_min)

        if src_r_max > src_r_min and src_c_max > src_c_min:
            subgrid[dst_r_min:dst_r_max, dst_c_min:dst_c_max] = self.occupancy_map[src_r_min:src_r_max, src_c_min:src_c_max]

        return subgrid.flatten()

    def _update_sensing(self):
        self.can_see_global = False
        for i in range(self.num_hunters):
            can_see = self._check_los(self.hunter_pos[i], self.evader_pos)
            self.can_see_agents[i] = can_see
            if can_see:
                self.can_see_global = True

        if self.can_see_global:
            self.last_seen_step = self.current_step
            obs = self.evader_pos + self.np_rng.normal(0, 1.0, size=2)
            self.pf.update(obs, self.evader_vel, self.np_rng)
        else:
            if (self.current_step - self.last_seen_step) > 600:
                self.pf.clear()
            else:
                self.pf.predict(self.obstacles, self.np_rng)

    def _check_los(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        x1, y1 = float(pos1[0]), float(pos1[1])
        x2, y2 = float(pos2[0]), float(pos2[1])
        for obs in self.obstacles:
            if obs.collides_line(x1, y1, x2, y2):
                return False
        return True

    def _resolve_wall_collisions(self, current_pos: np.ndarray, next_pos: np.ndarray, radius: float = 10.0) -> Tuple[np.ndarray, bool]:
        hit = False
        x, y = next_pos[0], next_pos[1]

        if x < radius: x = radius; hit = True
        if x > self.width - radius: x = self.width - radius; hit = True
        if y < radius: y = radius; hit = True
        if y > self.height - radius: y = self.height - radius; hit = True

        for obs in self.obstacles:
            if obs.collides_circle(x, y, radius):
                hit = True
                r = obs.rect
                overlap_l = (x + radius) - r.left
                overlap_r = r.right - (x - radius)
                overlap_t = (y + radius) - r.top
                overlap_b = r.bottom - (y - radius)
                min_overlap = min(overlap_l, overlap_r, overlap_t, overlap_b)

                if min_overlap == overlap_l: x = r.left - radius - 1
                elif min_overlap == overlap_r: x = r.right + radius + 1
                elif min_overlap == overlap_t: y = r.top - radius - 1
                elif min_overlap == overlap_b: y = r.bottom + radius + 1

        return np.array([x, y], dtype=np.float32), hit

    def _compute_rewards(self, min_dist: float, captured: bool, wall_hits: List[bool]) -> Dict[str, float]:
        rewards = {}
        for i in range(self.num_hunters):
            r = 0.0
            if captured:
                r += 500.0

            r -= 0.01 * (min_dist / 10.0)
            r -= 0.2

            if wall_hits[i]:
                r -= 10.0

            min_wall_dist = min(obs.distance_to_point(self.hunter_pos[i][0], self.hunter_pos[i][1]) for obs in self.obstacles)
            if min_wall_dist < self.d_safe:
                r -= 0.1 * (self.d_safe - min_wall_dist)

            for j in range(self.num_hunters):
                if i != j:
                    inter_dist = np.linalg.norm(self.hunter_pos[i] - self.hunter_pos[j])
                    if inter_dist < 30.0:
                        r -= 0.5

            rewards[f"agent_{i}"] = float(r)

        rewards["__all__"] = float(sum(rewards.values()) / self.num_hunters)
        return rewards

    def _get_observations(self) -> Dict[str, np.ndarray]:
        b_mean, b_std, b_spread = self.pf.get_belief_stats()
        blind_time_ratio = min(1.0, (self.current_step - self.last_seen_step) / 600.0)

        obs_dict = {}
        for i in range(self.num_hunters):
            self_feat = [
                self.hunter_pos[i][0] / self.width,
                self.hunter_pos[i][1] / self.height,
                self.hunter_vel[i][0] / self.max_speed_hunter,
                self.hunter_vel[i][1] / self.max_speed_hunter
            ]

            team_feat = []
            for j in range(self.num_hunters):
                if i != j:
                    rel_pos = (self.hunter_pos[j] - self.hunter_pos[i]) / np.array([self.width, self.height])
                    rel_vel = (self.hunter_vel[j] - self.hunter_vel[i]) / self.max_speed_hunter
                    team_feat.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]])

            belief_feat = [
                b_mean[0] / self.width,
                b_mean[1] / self.height,
                b_std[0] / self.width,
                b_std[1] / self.height,
                b_spread / 500.0,
                1.0 if self.can_see_agents[i] else 0.0,
                blind_time_ratio
            ]

            grid_feat = self._get_local_occupancy_grid(self.hunter_pos[i])
            full_vec = np.array(self_feat + team_feat + belief_feat + list(grid_feat), dtype=np.float32)
            obs_dict[f"agent_{i}"] = full_vec

        return obs_dict

    def get_particle_array(self) -> np.ndarray:
        norm_pos = self.pf.particles_pos / np.array([self.width, self.height], dtype=np.float32)
        norm_vel = self.pf.particles_vel / self.max_speed_evader
        return np.hstack([norm_pos, norm_vel]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        b_mean, _, b_spread = self.pf.get_belief_stats()
        return {
            "current_step": self.current_step,
            "evader_ground_truth": np.copy(self.evader_pos),
            "belief_mean": np.copy(b_mean),
            "belief_spread": b_spread,
            "can_see_global": self.can_see_global,
            "hunter_positions": np.copy(self.hunter_pos),
            "hunter_velocities": np.copy(self.hunter_vel)
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if not self.is_pygame_init:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Deep-POMCP: Multi-Agent Pursuit Benchmark")
            else:
                self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.is_pygame_init = True

        self.screen.fill((15, 23, 42))
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, (30, 41, 59), (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, (30, 41, 59), (0, y), (self.width, y), 1)

        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (51, 65, 85), obs.rect)
            pygame.draw.rect(self.screen, (100, 116, 139), obs.rect, 2)
        # Draw Particles
        if self.pf.is_initialized:
            for p_pos in self.pf.particles_pos:
                pygame.draw.circle(self.screen, (34, 197, 94), (int(p_pos[0]), int(p_pos[1])), 2)

        # Draw LOS Vision Rays
        for i in range(self.num_hunters):
            if self.can_see_agents[i]:
                h_pt = (int(self.hunter_pos[i][0]), int(self.hunter_pos[i][1]))
                e_pt = (int(self.evader_pos[0]), int(self.evader_pos[1]))
                pygame.draw.line(self.screen, (234, 179, 8), h_pt, e_pt, 1)

        # Draw Hunters (Vector Boids)
        hunter_colors = [(59, 130, 246), (6, 182, 212), (168, 85, 247)]
        for i in range(self.num_hunters):
            color = hunter_colors[i % len(hunter_colors)]
            self._draw_boid(self.hunter_pos[i], self.hunter_vel[i], color)

        # Draw Evader
        if self.can_see_global:
            self._draw_boid(self.evader_pos, self.evader_vel, (239, 68, 68))
        else:
            # Ghost circle when occluded
            ghost_surf = pygame.Surface((24, 24), pygame.SRCALPHA)
            pygame.draw.circle(ghost_surf, (239, 68, 68, 80), (12, 12), 10)
            self.screen.blit(ghost_surf, (int(self.evader_pos[0] - 12), int(self.evader_pos[1] - 12)))

        # HUD Overlay
        pygame.draw.rect(self.screen, (15, 23, 42), (10, 10, 360, 48), border_radius=4)
        pygame.draw.rect(self.screen, (71, 85, 105), (10, 10, 360, 48), 1, border_radius=4)
        font = pygame.font.SysFont("monospace", 14, bold=True)

        status_text = "SWARM TRACKING" if self.can_see_global else "BELIEF SEARCH"
        status_color = (34, 197, 94) if self.can_see_global else (249, 115, 22)
        hud_line1 = font.render(f"STEP: {self.current_step:04d} | STATUS: {status_text}", True, status_color)
        hud_line2 = font.render(f"PRESET: {self.config.density_preset.upper()} | HUNTERS: {self.num_hunters}", True, (203, 213, 225))
        self.screen.blit(hud_line1, (18, 16))
        self.screen.blit(hud_line2, (18, 34))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))

    def _draw_boid(self, pos: np.ndarray, vel: np.ndarray, color: Tuple[int, int, int]):
        px, py = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])
        speed = math.hypot(vx, vy)
        angle = math.atan2(-vy, vx) if speed > 0.01 else 0.0
        size = 14.0
        p1 = (px + size * math.cos(angle), py - size * math.sin(angle))
        p2 = (px + size * 0.5 * math.cos(angle + 2.5), py - size * 0.5 * math.sin(angle + 2.5))
        p3 = (px + size * 0.5 * math.cos(angle - 2.5), py - size * 0.5 * math.sin(angle - 2.5))
        pygame.draw.polygon(self.screen, color, [p1, p2, p3])

    def close(self):
        if self.is_pygame_init:
            pygame.quit()
            self.is_pygame_init = False
