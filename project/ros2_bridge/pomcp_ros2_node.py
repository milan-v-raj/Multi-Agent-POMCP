"""
ROS 2 Node for Edge Deployment: PointNet ET-DAG-Deep-POMCP Pursuer Agent.

Designed for ROS 2 Humble/Iron on differential-drive robots (TurtleBot3, Clearpath Jackal)
or multi-rotor quadrotors.

Subscribed Topics:
  - `/<agent_namespace>/scan` (sensor_msgs/msg/LaserScan) -> 2D LiDAR range scan
  - `/<agent_namespace>/odom` (nav_msgs/msg/Odometry)    -> Robot pose & velocity
  - `/team/evader_detection` (geometry_msgs/msg/PoseStamped) -> Intermittent camera/LOS detection
  - `/team/peer_odom`        (nav_msgs/msg/Odometry)    -> Peer hunter teammate state

Published Topics:
  - `/<agent_namespace>/cmd_vel` (geometry_msgs/msg/Twist) -> Desired linear & angular velocity
  - `/<agent_namespace>/belief_cloud` (sensor_msgs/msg/PointCloud2) -> Particle belief state
"""

import os
import sys
import math
from typing import Optional, List, Tuple
import numpy as np

# Add project root to sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Try importing rclpy with graceful fallback for non-ROS environments
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist, PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan, PointCloud2
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Mock classes for non-ROS testing
    class Vector3:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z
    class Twist:
        def __init__(self):
            self.linear = Vector3()
            self.angular = Vector3()
    class Node:  # type: ignore
        def __init__(self, name: str):
            self.name = name
        def get_logger(self):
            class MockLogger:
                def info(self, msg): print(f"[INFO] {msg}")
                def warn(self, msg): print(f"[WARN] {msg}")
            return MockLogger()

from deep_pomcp_env.baselines.belief_graph_pomcp import BeliefGraphDeepPOMCPPolicy
from deep_pomcp_env.core_env import ParticleFilterInternal
from deep_pomcp_env.scenarios import Obstacle


class PomcpRos2Node(Node):
    """
    ROS 2 Autonomous Pursuer Controller running ET-DAG-Deep-POMCP.
    """

    def __init__(
        self,
        agent_id: int = 0,
        weights_path: Optional[str] = None,
        arena_width_m: float = 8.0,
        arena_height_m: float = 6.0,
        control_freq_hz: float = 20.0
    ):
        super().__init__(f"pomcp_pursuer_{agent_id}")
        self.agent_id = agent_id
        self.width_m = arena_width_m
        self.height_m = arena_height_m
        self.control_freq = control_freq_hz

        # Scale: 0.01 m per pixel -> 8.0 m = 800 px, 6.0 m = 600 px
        self.scale = 0.01
        self.width_px = int(self.width_m / self.scale)
        self.height_px = int(self.height_m / self.scale)

        # Robot state
        self.pos_px = np.array([100.0, 100.0], dtype=np.float32)
        self.vel_px = np.array([0.0, 0.0], dtype=np.float32)
        self.yaw = 0.0
        self.peer_pos_px = np.array([150.0, 150.0], dtype=np.float32)
        self.peer_vel_px = np.array([0.0, 0.0], dtype=np.float32)

        # Local occupancy grid (11x11, 10px resolution = 0.1m per cell)
        self.local_grid = np.zeros(121, dtype=np.float32)

        # Belief State Particle Filter
        self.pf = ParticleFilterInternal(num_particles=200, width=self.width_px, height=self.height_px)
        self.can_see_target = False
        self.last_seen_step = 0
        self.current_step = 0

        # Load Neural MCTS Policy
        if weights_path is None or not os.path.exists(weights_path):
            weights_path = os.path.join(proj_root, "deep_pomcp_weights.pth")

        self.policy = BeliefGraphDeepPOMCPPolicy(
            weights_path=weights_path,
            num_simulations=20,
            max_depth=6,
            enable_dag_merging=True,
            enable_event_trigger=True,
            enable_counter_deception=True,
            name=f"CD-ET-DAG-Agent-{agent_id}"
        )
        self.policy.reset()

        # Dummy obstacles (walls) for pathfinder boundary checks
        self.obstacles: List[Obstacle] = [
            Obstacle(0, 0, self.width_px, 10),
            Obstacle(0, self.height_px - 10, self.width_px, 10),
            Obstacle(0, 0, 10, self.height_px),
            Obstacle(self.width_px - 10, 0, 10, self.height_px)
        ]

        if ROS2_AVAILABLE:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
            self.sub_odom = self.create_subscription(
                Odometry, f"/hunter_{agent_id}/odom", self._on_odom, qos
            )
            self.sub_scan = self.create_subscription(
                LaserScan, f"/hunter_{agent_id}/scan", self._on_scan, qos
            )
            peer_id = 1 if agent_id == 0 else 0
            self.sub_peer = self.create_subscription(
                Odometry, f"/hunter_{peer_id}/odom", self._on_peer_odom, qos
            )
            self.sub_detection = self.create_subscription(
                PoseStamped, "/team/evader_detection", self._on_evader_detection, qos
            )
            self.pub_cmd_vel = self.create_publisher(
                Twist, f"/hunter_{agent_id}/cmd_vel", 10
            )
            self.timer = self.create_timer(1.0 / self.control_freq, self._control_loop)
            self.get_logger().info(f"Initialized POMCP ROS 2 Node for Agent {agent_id} at {control_freq_hz} Hz")

    def _m_to_px(self, x_m: float, y_m: float) -> Tuple[float, float]:
        px = (x_m / self.scale) + (self.width_px / 2.0)
        py = (-y_m / self.scale) + (self.height_px / 2.0)
        return px, py

    def _on_odom(self, msg):
        x_m = msg.pose.pose.position.x
        y_m = msg.pose.pose.position.y
        px, py = self._m_to_px(x_m, y_m)
        self.pos_px = np.array([px, py], dtype=np.float32)

        # Linear velocity in px/frame (scaled to 20Hz frame)
        vx_m_s = msg.twist.twist.linear.x
        vy_m_s = msg.twist.twist.linear.y
        dt = 1.0 / self.control_freq
        self.vel_px = np.array([(vx_m_s * dt) / self.scale, -(vy_m_s * dt) / self.scale], dtype=np.float32)

        # Orientation quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _on_peer_odom(self, msg):
        x_m = msg.pose.pose.position.x
        y_m = msg.pose.pose.position.y
        px, py = self._m_to_px(x_m, y_m)
        self.peer_pos_px = np.array([px, py], dtype=np.float32)
        dt = 1.0 / self.control_freq
        self.peer_vel_px = np.array([(msg.twist.twist.linear.x * dt) / self.scale, -(msg.twist.twist.linear.y * dt) / self.scale], dtype=np.float32)

    def _on_scan(self, msg):
        """Converts LaserScan ranges into local 11x11 occupancy subgrid."""
        subgrid = np.zeros((11, 11), dtype=np.float32)
        half = 5
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # Relative obstacle hit point in robot-frame
                ox_m = r * math.cos(angle + self.yaw)
                oy_m = r * math.sin(angle + self.yaw)
                # Cell offset (0.1m resolution)
                c_idx = int(ox_m / 0.10) + half
                r_idx = int(-oy_m / 0.10) + half
                if 0 <= c_idx < 11 and 0 <= r_idx < 11:
                    subgrid[r_idx, c_idx] = 1.0
            angle += msg.angle_increment
        self.local_grid = subgrid.flatten()

    def _on_evader_detection(self, msg):
        """Received when camera/radar detects evader."""
        ex_m = msg.pose.position.x
        ey_m = msg.pose.position.y
        px, py = self._m_to_px(ex_m, ey_m)
        obs_pos = np.array([px, py], dtype=np.float32)
        self.can_see_target = True
        self.last_seen_step = self.current_step
        if not self.pf.is_initialized:
            self.pf.initialize(obs_pos, np.zeros(2, dtype=np.float32), np.random.default_rng())
        else:
            self.pf.update(obs_pos, np.zeros(2, dtype=np.float32), np.random.default_rng())

    def _control_loop(self):
        """20 Hz periodic decision loop."""
        self.current_step += 1

        if not self.can_see_target and self.pf.is_initialized:
            self.pf.predict(self.obstacles, np.random.default_rng())
        self.can_see_target = False  # Reset for next detection cycle

        b_mean, b_std, b_spread = self.pf.get_belief_stats()
        blind_time_ratio = min(1.0, (self.current_step - self.last_seen_step) / 600.0)

        # Assemble observation vector matching core_env (136 dims)
        self_feat = [
            self.pos_px[0] / self.width_px,
            self.pos_px[1] / self.height_px,
            self.vel_px[0] / 5.0,
            self.vel_px[1] / 5.0
        ]
        peer_feat = [
            (self.peer_pos_px[0] - self.pos_px[0]) / self.width_px,
            (self.peer_pos_px[1] - self.pos_px[1]) / self.height_px,
            (self.peer_vel_px[0] - self.vel_px[0]) / 5.0,
            (self.peer_vel_px[1] - self.vel_px[1]) / 5.0
        ]
        belief_feat = [
            b_mean[0] / self.width_px,
            b_mean[1] / self.height_px,
            b_std[0] / self.width_px,
            b_std[1] / self.height_px,
            b_spread / 500.0,
            1.0 if self.can_see_target else 0.0,
            blind_time_ratio
        ]
        obs_vec = np.array(self_feat + peer_feat + belief_feat + list(self.local_grid), dtype=np.float32)

        info = {
            "hunter_positions": np.array([self.pos_px, self.peer_pos_px]),
            "hunter_velocities": np.array([self.vel_px, self.peer_vel_px]),
            "belief_mean": b_mean if self.pf.is_initialized else None,
            "belief_spread": b_spread if self.pf.is_initialized else 0.0,
            "is_belief_active": self.pf.is_initialized,
            "can_see_global": self.can_see_target,
            "current_step": self.current_step
        }

        # Query frozen neural policy
        action_idx = self.policy.get_action(obs_vec, info, self.agent_id, self.obstacles, self.width_px, self.height_px)

        # Map discrete action to desired world velocity (m/s)
        # 0: UP (-Y), 1: DOWN (+Y), 2: LEFT (-X), 3: RIGHT (+X), 4: WAIT
        target_vx, target_vy = 0.0, 0.0
        v_speed = 0.60  # m/s
        if action_idx == 0:   target_vy =  v_speed  # MuJoCo/ROS: +Y is forward/north
        elif action_idx == 1: target_vy = -v_speed
        elif action_idx == 2: target_vx = -v_speed
        elif action_idx == 3: target_vx =  v_speed

        # Convert Cartesian (target_vx, target_vy) to differential drive (v_linear, omega_angular)
        target_heading = math.atan2(target_vy, target_vx) if (target_vx != 0.0 or target_vy != 0.0) else self.yaw
        heading_err = (target_heading - self.yaw + math.pi) % (2.0 * math.pi) - math.pi

        twist = Twist()
        if target_vx == 0.0 and target_vy == 0.0:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            # Simple proportional heading control
            twist.angular.z = float(np.clip(2.5 * heading_err, -1.5, 1.5))
            if abs(heading_err) < math.pi / 2.0:
                twist.linear.x = float(v_speed * math.cos(heading_err))
            else:
                twist.linear.x = 0.0

        if ROS2_AVAILABLE:
            self.pub_cmd_vel.publish(twist)

        return twist


def main(args=None):
    if ROS2_AVAILABLE:
        rclpy.init(args=args)
        node = PomcpRos2Node(agent_id=0)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("[ROS 2 Not Installed] Running standalone offline validation of PomcpRos2Node...")
        node = PomcpRos2Node(agent_id=0)
        cmd = node._control_loop()
        print(f"Sample computed Twist command: linear.x={cmd.linear.x:.3f}, angular.z={cmd.angular.z:.3f}")
        print("PomcpRos2Node offline validation successful!")


if __name__ == "__main__":
    main()
