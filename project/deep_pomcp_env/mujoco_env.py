"""
High-Fidelity Continuous Multi-Body Robotics Simulation Environment for Deep-POMCP
Powered by MuJoCo (Multi-Joint Dynamics with Contact).

Replaces 2D heuristic collision handling with:
  1. Continuous 3D rigid-body contact mechanics & Coulomb friction
  2. Viscous velocity damping & torque-bounded motor actuators
  3. Continuous 3D optical Line-of-Sight (LOS) raycasting via `mj_ray`
  4. Photorealistic 3D off-screen rendering via `mujoco.Renderer`
  5. 100% API parity with `PursuitEvasionEnv` for drop-in policy evaluation
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .scenarios import Obstacle, ScenarioConfig, ScenarioGenerator
from .evaders import BaseEvader, SmartRaycastEvader, DeceptiveFeintingEvader
from .core_env import ParticleFilterInternal


class MujocoPursuitEvasionEnv(gym.Env):
    """
    Standardized Gymnasium environment for Multi-Agent Pursuit-Evasion
    simulated inside the MuJoCo physics engine.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Scaling: 800 px x 600 px mapped to 8.0 m x 6.0 m in MuJoCo space
    # (Center of arena at (0, 0), x in [-4.0, +4.0], y in [-3.0, +3.0])
    SCALE = 0.01  # 1 px = 0.01 meters

    FORCE_MAG = 8.0  # Newtons applied to robot chassis
    ACTION_VECTORS = [
        np.array([0.0,  FORCE_MAG], dtype=np.float32),  # 0: UP (in Pygame y is down, in MuJoCo y is up)
        np.array([0.0, -FORCE_MAG], dtype=np.float32),  # 1: DOWN
        np.array([-FORCE_MAG, 0.0], dtype=np.float32),  # 2: LEFT
        np.array([ FORCE_MAG, 0.0], dtype=np.float32),  # 3: RIGHT
        np.array([0.0, 0.0], dtype=np.float32)          # 4: WAIT
    ]

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        evader_policy: Optional[BaseEvader] = None,
        render_mode: Optional[str] = None,
        physics_substeps: int = 8
    ):
        super().__init__()
        self.config = config if config is not None else ScenarioConfig()
        self.render_mode = render_mode
        self.evader_policy = evader_policy if evader_policy is not None else SmartRaycastEvader()
        self.physics_substeps = physics_substeps
        self.dt_step = self.physics_substeps * 0.005  # 0.040s (25 Hz control loop)

        self.width = self.config.width
        self.height = self.config.height
        self.num_hunters = self.config.num_hunters
        self.max_steps = self.config.max_steps
        self.capture_radius = self.config.capture_radius  # 35 px -> 0.35m

        self.max_speed_hunter = 5.0
        self.max_speed_evader = 7.0 * self.config.evader_speed_mult

        self.rng = random.Random()
        self.np_rng = np.random.default_rng()

        self.obstacles: List[Obstacle] = []
        self.pf = ParticleFilterInternal(num_particles=200, width=self.width, height=self.height)
        self.can_see_agents = [False] * self.num_hunters
        self.can_see_global = False
        self.last_seen_step = 0
        self.current_step = 0

        # MuJoCo Engine Objects
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.renderer: Optional[mujoco.Renderer] = None

        obs_dim = 4 + (4 * (self.num_hunters - 1)) + 7 + 121
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Discrete(len(self.ACTION_VECTORS))
            for i in range(self.num_hunters)
        })
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for i in range(self.num_hunters)
        })

    def _px_to_m(self, pos_px: np.ndarray) -> np.ndarray:
        """Converts [0, 800] x [0, 600] Pygame coords to [-4.0, 4.0] x [-3.0, 3.0] MuJoCo coords."""
        x_m = (pos_px[0] - self.width / 2.0) * self.SCALE
        y_m = -(pos_px[1] - self.height / 2.0) * self.SCALE  # Invert Y so up is positive
        return np.array([x_m, y_m], dtype=np.float64)

    def _m_to_px(self, pos_m: np.ndarray) -> np.ndarray:
        """Converts [-4.0, 4.0] x [-3.0, 3.0] MuJoCo coords to [0, 800] x [0, 600] Pygame coords."""
        px_x = (pos_m[0] / self.SCALE) + self.width / 2.0
        px_y = (-pos_m[1] / self.SCALE) + self.height / 2.0
        return np.array([px_x, px_y], dtype=np.float32)

    def _build_mujoco_xml(
        self,
        obstacles: List[Obstacle],
        h_spawns: List[np.ndarray],
        e_spawn: np.ndarray
    ) -> str:
        """Generates dynamic continuous 3D physics arena XML for MuJoCo."""
        half_w = (self.width / 2.0) * self.SCALE
        half_h = (self.height / 2.0) * self.SCALE
        wall_thick = 0.15
        wall_h = 0.50

        xml_lines = [
            "<mujoco model='deep_pomcp_arena'>",
            "  <option timestep='0.005' gravity='0 0 -9.81' iterations='50'/>",
            "  <visual>",
            "    <global offwidth='960' offheight='720'/>",
            "  </visual>",
            "  <asset>",
            "    <texture name='grid' type='2d' builtin='checker' width='512' height='512' rgb1='0.12 0.16 0.22' rgb2='0.16 0.21 0.28'/>",
            "    <material name='grid_mat' texture='grid' texrepeat='8 6' reflectance='0.15'/>",
            "    <material name='wall_mat' rgba='0.28 0.35 0.45 1.0' reflectance='0.2'/>",
            "    <material name='obs_mat' rgba='0.40 0.48 0.58 1.0' reflectance='0.1'/>",
            "  </asset>",
            "  <worldbody>",
            "    <light diffuse='0.9 0.9 0.9' dir='0 0 -1' pos='0 0 6.0'/>",
            "    <camera name='overhead' pos='0 -0.2 9.2' zaxis='0 0 1' fovy='54'/>",
            "    <camera name='perspective' pos='0 -5.8 5.6' xyaxes='1 0 0 0 0.7 0.7' fovy='58'/>",
            "    <!-- Floor -->",
            f"    <geom name='floor' type='plane' size='{half_w} {half_h} 0.1' material='grid_mat'/>",
            "    <!-- Boundary Walls -->",
            f"    <geom name='wall_n' type='box' pos='0 {half_h + wall_thick} {wall_h/2}' size='{half_w + wall_thick*2} {wall_thick} {wall_h/2}' material='wall_mat'/>",
            f"    <geom name='wall_s' type='box' pos='0 {-half_h - wall_thick} {wall_h/2}' size='{half_w + wall_thick*2} {wall_thick} {wall_h/2}' material='wall_mat'/>",
            f"    <geom name='wall_w' type='box' pos='{-half_w - wall_thick} 0 {wall_h/2}' size='{wall_thick} {half_h} {wall_h/2}' material='wall_mat'/>",
            f"    <geom name='wall_e' type='box' pos='{half_w + wall_thick} 0 {wall_h/2}' size='{wall_thick} {half_h} {wall_h/2}' material='wall_mat'/>",
        ]

        # Add 3D Box Obstacles
        for idx, obs in enumerate(obstacles):
            obs_cx_px = obs.x + obs.width / 2.0
            obs_cy_px = obs.y + obs.height / 2.0
            obs_pos_m = self._px_to_m(np.array([obs_cx_px, obs_cy_px]))
            obs_size_m = np.array([obs.width / 2.0, obs.height / 2.0]) * self.SCALE
            xml_lines.append(
                f"    <geom name='obs_{idx}' type='box' pos='{obs_pos_m[0]:.3f} {obs_pos_m[1]:.3f} {wall_h/2}' "
                f"size='{obs_size_m[0]:.3f} {obs_size_m[1]:.3f} {wall_h/2}' material='obs_mat'/>"
            )

        # Add Pursuer Robots (Hunters)
        hunter_colors = ["0.22 0.52 0.95 1.0", "0.10 0.78 0.85 1.0"]
        for i in range(self.num_hunters):
            h_m = self._px_to_m(h_spawns[i])
            c_rgba = hunter_colors[i % len(hunter_colors)]
            xml_lines.extend([
                f"    <body name='hunter_{i}' pos='{h_m[0]:.3f} {h_m[1]:.3f} 0.12'>",
                f"      <joint name='h{i}_x' type='slide' axis='1 0 0' damping='1.5'/>",
                f"      <joint name='h{i}_y' type='slide' axis='0 1 0' damping='1.5'/>",
                f"      <geom name='h{i}_geom' type='cylinder' size='0.12 0.08' mass='1.0' rgba='{c_rgba}' friction='0.8 0.1 0.1'/>",
                f"      <geom name='h{i}_cap' type='sphere' size='0.08' pos='0 0 0.08' mass='0.05' rgba='0.95 0.95 0.95 1.0'/>",
                "    </body>"
            ])

        # Add Evader Robot
        e_m = self._px_to_m(e_spawn)
        xml_lines.extend([
            f"    <body name='evader' pos='{e_m[0]:.3f} {e_m[1]:.3f} 0.12'>",
            "      <joint name='evader_x' type='slide' axis='1 0 0' damping='1.5'/>",
            "      <joint name='evader_y' type='slide' axis='0 1 0' damping='1.5'/>",
            "      <geom name='evader_geom' type='cylinder' size='0.12 0.08' mass='0.85' rgba='0.92 0.25 0.25 1.0' friction='0.8 0.1 0.1'/>",
            "      <geom name='evader_cap' type='sphere' size='0.08' pos='0 0 0.08' mass='0.05' rgba='0.95 0.95 0.20 1.0'/>",
            "    </body>"
        ])

        xml_lines.append("  </worldbody>")

        # Continuous Motor Actuators
        xml_lines.append("  <actuator>")
        for i in range(self.num_hunters):
            xml_lines.extend([
                f"    <motor joint='h{i}_x' name='motor_h{i}_x' gear='1' ctrlrange='-35 35'/>",
                f"    <motor joint='h{i}_y' name='motor_h{i}_y' gear='1' ctrlrange='-35 35'/>"
            ])
        xml_lines.extend([
            "    <motor joint='evader_x' name='motor_evader_x' gear='1' ctrlrange='-35 35'/>",
            "    <motor joint='evader_y' name='motor_evader_y' gear='1' ctrlrange='-35 35'/>"
        ])
        xml_lines.append("  </actuator>")
        xml_lines.append("</mujoco>")

        return "\n".join(xml_lines)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.current_step = 0
        self.last_seen_step = 0

        # Generate scenario obstacles & spawns
        self.obstacles, h_spawns, e_spawn = ScenarioGenerator.generate(self.config, self.rng)
        self._build_occupancy_map()

        # Build & compile MuJoCo continuous physics model
        xml_str = self._build_mujoco_xml(self.obstacles, h_spawns, e_spawn)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        if hasattr(self.evader_policy, "reset_state"):
            self.evader_policy.reset_state()

        self.pf.clear()
        self._update_sensing()

        if self.render_mode in ("rgb_array", "human"):
            self.renderer = mujoco.Renderer(self.model, 720, 960)

        observations = self._get_observations()
        info = self._get_info()
        return observations, info

    def step(
        self,
        actions: Dict[str, Union[int, np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self.current_step += 1

        # 1. Apply Hunter Continuous Motor Torques
        ctrl_idx = 0
        for i in range(self.num_hunters):
            act = actions.get(f"agent_{i}", 4)
            if isinstance(act, (int, np.integer)):
                # Convert discrete action index (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY)
                # Note: in Pygame Y is downward; in MuJoCo Y is upward.
                if act == 0:    f_x, f_y =  0.0,  self.FORCE_MAG
                elif act == 1:  f_x, f_y =  0.0, -self.FORCE_MAG
                elif act == 2:  f_x, f_y = -self.FORCE_MAG, 0.0
                elif act == 3:  f_x, f_y =  self.FORCE_MAG, 0.0
                else:           f_x, f_y =  0.0,  0.0
            else:
                f_x = float(act[0]) * (self.FORCE_MAG / 0.5)
                f_y = -float(act[1]) * (self.FORCE_MAG / 0.5)

            self.data.ctrl[ctrl_idx]     = f_x
            self.data.ctrl[ctrl_idx + 1] = f_y
            ctrl_idx += 2

        # 2. Compute Evader Continuous Force
        h_pos_px = [self._get_hunter_pos_px(i) for i in range(self.num_hunters)]
        e_pos_px = self._get_evader_pos_px()
        e_vel_px = self._get_evader_vel_px()

        e_force = self.evader_policy.get_action(
            e_pos_px, e_vel_px, h_pos_px, self.obstacles, self.width, self.height
        )
        e_fx_m = float(e_force[0]) * (self.FORCE_MAG / 0.5)
        e_fy_m = -float(e_force[1]) * (self.FORCE_MAG / 0.5)

        self.data.ctrl[ctrl_idx]     = e_fx_m
        self.data.ctrl[ctrl_idx + 1] = e_fy_m

        # 3. Step MuJoCo Continuous Multi-Body Dynamics with Physical Speed Clamping
        v_max_hunter_m_s = (self.max_speed_hunter * self.SCALE) / self.dt_step
        v_max_evader_m_s = (self.max_speed_evader * self.SCALE) / self.dt_step

        for _ in range(self.physics_substeps):
            mujoco.mj_step(self.model, self.data)

            # Enforce non-linear physical velocity bounds on joint dynamics
            for i in range(self.num_hunters):
                vx = self.data.qvel[i * 2]
                vy = self.data.qvel[i * 2 + 1]
                spd = math.hypot(vx, vy)
                if spd > v_max_hunter_m_s:
                    scale = v_max_hunter_m_s / spd
                    self.data.qvel[i * 2] = vx * scale
                    self.data.qvel[i * 2 + 1] = vy * scale

            evx = self.data.qvel[self.num_hunters * 2]
            evy = self.data.qvel[self.num_hunters * 2 + 1]
            espd = math.hypot(evx, evy)
            if espd > v_max_evader_m_s:
                escale = v_max_evader_m_s / espd
                self.data.qvel[self.num_hunters * 2] = evx * escale
                self.data.qvel[self.num_hunters * 2 + 1] = evy * escale

        # 4. Sensing & Continuous 3D Optical LOS Raycasts
        self._update_sensing()

        # 5. Check Termination Conditions (Continuous Contact / Distance Threshold)
        dists_to_evader = [
            float(np.linalg.norm(self._get_hunter_pos_px(i) - e_pos_px))
            for i in range(self.num_hunters)
        ]
        min_dist = min(dists_to_evader)
        captured = min_dist < self.capture_radius
        timeout = self.current_step >= self.max_steps

        terminated = {f"agent_{i}": captured for i in range(self.num_hunters)}
        terminated["__all__"] = captured
        truncated = {f"agent_{i}": timeout for i in range(self.num_hunters)}
        truncated["__all__"] = timeout

        # 6. Rewards, Observations, and Info
        wall_hits = [self._check_hunter_contact_wall(i) for i in range(self.num_hunters)]
        rewards = self._compute_rewards(min_dist, captured, wall_hits)

        observations = self._get_observations()
        info = self._get_info()
        info["captured"] = captured
        info["timeout"] = timeout
        info["min_dist_to_evader"] = min_dist
        info["wall_hits"] = sum(wall_hits)
        info["mujoco_contacts"] = int(self.data.ncon)

        return observations, rewards, terminated, truncated, info

    def _get_hunter_pos_px(self, agent_id: int) -> np.ndarray:
        m_pos = self.data.body(f"hunter_{agent_id}").xpos[:2]
        return self._m_to_px(m_pos)

    def _get_hunter_vel_px(self, agent_id: int) -> np.ndarray:
        vx_m_s = float(self.data.qvel[agent_id * 2])
        vy_m_s = float(self.data.qvel[agent_id * 2 + 1])
        vx_px = (vx_m_s * self.dt_step) / self.SCALE
        vy_px = -(vy_m_s * self.dt_step) / self.SCALE  # Invert Y for Pygame coordinate consistency
        return np.array([vx_px, vy_px], dtype=np.float32)

    def _get_evader_pos_px(self) -> np.ndarray:
        m_pos = self.data.body("evader").xpos[:2]
        return self._m_to_px(m_pos)

    def _get_evader_vel_px(self) -> np.ndarray:
        evx_m_s = float(self.data.qvel[self.num_hunters * 2])
        evy_m_s = float(self.data.qvel[self.num_hunters * 2 + 1])
        evx_px = (evx_m_s * self.dt_step) / self.SCALE
        evy_px = -(evy_m_s * self.dt_step) / self.SCALE
        return np.array([evx_px, evy_px], dtype=np.float32)

    def _check_hunter_contact_wall(self, agent_id: int) -> bool:
        """Queries MuJoCo contact manifold for collision with obstacles or outer boundary."""
        h_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"h{agent_id}_geom")
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == h_geom_id or con.geom2 == h_geom_id:
                other_geom = con.geom2 if con.geom1 == h_geom_id else con.geom1
                other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                if other_name and (other_name.startswith("obs_") or other_name.startswith("wall_")):
                    return True
        return False

    def _update_sensing(self):
        """Executes exact 3D optical Line-of-Sight raycasts inside MuJoCo."""
        self.can_see_global = False
        e_pos_3d = self.data.body("evader").xpos.copy()

        for i in range(self.num_hunters):
            h_pos_3d = self.data.body(f"hunter_{i}").xpos.copy()
            ray_dir = e_pos_3d - h_pos_3d
            ray_len = float(np.linalg.norm(ray_dir))

            can_see = True
            if ray_len > 1e-4:
                ray_unit = ray_dir / ray_len
                geom_id = np.zeros(1, dtype=np.int32)
                h_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hunter_{i}")

                # Cast ray from hunter toward evader excluding own hunter body
                hit_dist = mujoco.mj_ray(
                    self.model, self.data, h_pos_3d, ray_unit,
                    None, True, h_body_id, geom_id
                )

                if hit_dist >= 0:
                    hit_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id[0])
                    # If ray hit an obstacle or wall before reaching the evader (within 0.15m tolerance)
                    if hit_geom_name and (hit_geom_name.startswith("obs_") or hit_geom_name.startswith("wall_")):
                        if hit_dist < (ray_len - 0.12):
                            can_see = False

            self.can_see_agents[i] = can_see
            if can_see:
                self.can_see_global = True

        e_pos_px = self._get_evader_pos_px()
        e_vel_px = self._get_evader_vel_px()

        if self.can_see_global:
            self.last_seen_step = self.current_step
            obs = e_pos_px + self.np_rng.normal(0, 1.0, size=2)
            if not self.pf.is_initialized:
                self.pf.initialize(obs, e_vel_px, self.np_rng)
            else:
                self.pf.update(obs, e_vel_px, self.np_rng)
        else:
            if self.pf.is_initialized:
                if (self.current_step - self.last_seen_step) > 600:
                    self.pf.clear()
                else:
                    self.pf.predict(self.obstacles, self.np_rng)

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

    def _get_local_occupancy_grid(self, pos_px: np.ndarray, grid_size: int = 11, res: float = 10.0) -> np.ndarray:
        rows, cols = self.occupancy_map.shape
        center_r = int(pos_px[1] / res)
        center_c = int(pos_px[0] / res)
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
            rewards[f"agent_{i}"] = float(r)
        rewards["__all__"] = float(sum(rewards.values()) / self.num_hunters)
        return rewards

    def _get_observations(self) -> Dict[str, np.ndarray]:
        b_mean, b_std, b_spread = self.pf.get_belief_stats()
        blind_time_ratio = min(1.0, (self.current_step - self.last_seen_step) / 600.0)

        h_positions = [self._get_hunter_pos_px(i) for i in range(self.num_hunters)]
        h_velocities = [self._get_hunter_vel_px(i) for i in range(self.num_hunters)]

        obs_dict = {}
        for i in range(self.num_hunters):
            self_feat = [
                h_positions[i][0] / self.width,
                h_positions[i][1] / self.height,
                h_velocities[i][0] / self.max_speed_hunter,
                h_velocities[i][1] / self.max_speed_hunter
            ]
            team_feat = []
            for j in range(self.num_hunters):
                if i != j:
                    rel_pos = (h_positions[j] - h_positions[i]) / np.array([self.width, self.height])
                    rel_vel = (h_velocities[j] - h_velocities[i]) / self.max_speed_hunter
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
            grid_feat = self._get_local_occupancy_grid(h_positions[i])
            full_vec = np.array(self_feat + team_feat + belief_feat + list(grid_feat), dtype=np.float32)
            obs_dict[f"agent_{i}"] = full_vec
        return obs_dict

    def get_particle_array(self) -> np.ndarray:
        norm_pos = self.pf.particles_pos / np.array([self.width, self.height], dtype=np.float32)
        norm_vel = self.pf.particles_vel / self.max_speed_evader
        return np.hstack([norm_pos, norm_vel]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        b_mean, _, b_spread = self.pf.get_belief_stats()
        h_positions = np.array([self._get_hunter_pos_px(i) for i in range(self.num_hunters)], dtype=np.float32)
        h_velocities = np.array([self._get_hunter_vel_px(i) for i in range(self.num_hunters)], dtype=np.float32)

        return {
            "current_step": self.current_step,
            "evader_ground_truth": np.copy(self._get_evader_pos_px()),
            "belief_mean": np.copy(b_mean) if self.pf.is_initialized else None,
            "belief_spread": b_spread if self.pf.is_initialized else 0.0,
            "is_belief_active": self.pf.is_initialized,
            "can_see_global": self.can_see_global,
            "hunter_positions": h_positions,
            "hunter_velocities": h_velocities,
            "particles": self.get_particle_array() if self.pf.is_initialized else None
        }

    def render(self, camera_name: str = "perspective") -> np.ndarray:
        """Renders photorealistic 3D camera view of the MuJoCo simulation."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 720, 960)
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
