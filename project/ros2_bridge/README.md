# ROS 2 & Gazebo Integration Bridge for Deep-POMCP

This package provides a standardized **ROS 2 Humble / Iron** package (`deep_pomcp_ros2`) that deploys the frozen **PointNet ET-DAG-Deep-POMCP** neural Monte Carlo Tree Search policy directly onto physical mobile robots (e.g., TurtleBot3, Clearpath Jackal) or simulated robots inside **Gazebo / Ignition**.

---

## 1. System Architecture

```
                 +-----------------------------------------+
                 |       Gazebo / Physical Robot Hardware  |
                 +-----------------------------------------+
                       | LaserScan                | Odometry
                       v                          v
                 +-----------------------------------------+
                 |      ROS 2 Topics: /scan & /odom        |
                 +-----------------------------------------+
                                      |
                                      v
   +-----------------------------------------------------------------------+
   |                      PomcpRos2Node (rclpy, 20 Hz)                     |
   |                                                                       |
   |  1. Polar LaserScan Range Buffer -> Local Occupancy Subgrid (11x11)   |
   |  2. Intermittent Camera Detection -> SMC Particle Filter (200 pts)    |
   |  3. PointNet Permutation-Invariant Latent Belief Encoder              |
   |  4. ET-DAG Monte Carlo Tree Search (N=20, Transposition Merging)     |
   |  5. Differential Drive Kinematic Mapping:                             |
   |     Target Vector (v_x, v_y) -> Linear Velocity v, Yaw Rate omega     |
   +-----------------------------------------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |       ROS 2 Topic: /hunter_i/cmd_vel    |
                 +-----------------------------------------+
```

---

## 2. Package Structure

```
project/ros2_bridge/
├── package.xml                   # ROS 2 package manifest (ament_python)
├── setup.py                      # Build script and console entry point
├── __init__.py                   # Package initializer
├── pomcp_ros2_node.py            # Primary ROS 2 node implementation
├── launch/
│   └── pursuit_evasion.launch.py # Multi-robot launch file for 2 pursuers
└── README.md                     # Integration and deployment guide
```

---

## 3. Building and Running in ROS 2 (Ubuntu 22.04 / WSL2)

### Step 1: Create or Symlink into ROS 2 Colcon Workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ln -s "/path/to/Btech Project/project/ros2_bridge" deep_pomcp_ros2
```

### Step 2: Install Dependencies and Build
```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install -i --from-path src --rosdistro humble -y
colcon build --packages-select deep_pomcp_ros2
source install/setup.bash
```

### Step 3: Launch Multi-Agent Pursuers in Gazebo
```bash
# In Terminal 1: Launch Gazebo world with TurtleBot3 / Jackal robots
ros2 launch turtlebot3_gazebo multi_turtlebot3_world.launch.py

# In Terminal 2: Launch PointNet ET-DAG-Deep-POMCP pursuer team
ros2 launch deep_pomcp_ros2 pursuit_evasion.launch.py arena_width:=8.0 arena_height:=6.0
```

---

## 4. Topics Subscribed & Published

| Topic Name | Message Type | Direction | Description |
| :--- | :--- | :--- | :--- |
| `/hunter_i/scan` | `sensor_msgs/msg/LaserScan` | Sub | 2D LiDAR range scan converting to local grid |
| `/hunter_i/odom` | `nav_msgs/msg/Odometry` | Sub | Odometry (pose $x, y$ and linear velocities) |
| `/team/evader_detection` | `geometry_msgs/msg/PoseStamped` | Sub | Optical / Line-of-sight sighting of evader |
| `/hunter_i/cmd_vel` | `geometry_msgs/msg/Twist` | Pub | Continuous velocity commands ($v_x, \omega_z$) |

---

## 5. Offline Validation (No ROS 2 Required)
The node includes automatic graceful fallback for systems without ROS 2 installed:
```powershell
python project/ros2_bridge/pomcp_ros2_node.py
```
This tests the full inference loop, local occupancy extraction, and kinematic Twist conversion without requiring a ROS 2 middleware installation.

