<<<<<<< HEAD
# G1_to_Box
For hackathon
=======
# G1 Robot Box Finding in 4x4 Room

Autonomous navigation simulation where a Unitree G1 humanoid robot searches for and approaches a red target box in a confined room environment using vision and LIDAR sensors.

## Simulator Description

**Physics Engine:** MuJoCo (Multi-Joint dynamics with Contact)
- High-fidelity physics simulation with accurate contact dynamics
- Timestep: 0.002s (500Hz)
- Gravity: 9.81 m/s²

**Robot Model:** Unitree G1 Humanoid (23-DOF variant)
- Full articulated humanoid robot with torso, arms, and legs
- Standing height: ~0.82m
- Custom locomotion API for high-level movement control (stand, walk, rotate)

**Sensors:**
- RGB Camera (640×480 resolution, 90° FOV)
- 3D LIDAR (32×16 rays, 5m range)
- IMU (orientation, angular velocity, linear acceleration)

## Scene Description

**Environment:**
- Room dimensions: 4m × 4m × 2m (L × W × H)
- Floor: White surface with high friction
- Walls: Gray concrete appearance (0.15m thick)

**Target Object:**
- Red box: 0.3m × 0.3m × 0.3m
- Material: High-visibility red color for vision detection
- Position: Randomly placed 0.8-1.5m from robot

**Robot Spawn:**
- Random position within ±1.0m from room center
- Random initial orientation (0-360°)

## Launch Instructions

### Prerequisites
```bash
pip install mujoco opencv-python numpy
```

### Running the Simulation

1. Navigate to the project directory:
```bash
cd unitree_ros-master/robots/g1_description
```

2. Run the simulation:
```bash
python3 g1_controlled_navigation.py
```

### Controls
- **Mouse Left-Click + Drag:** Rotate camera view
- **Mouse Right-Click + Drag:** Pan camera
- **Mouse Scroll:** Zoom in/out
- **Space:** Pause/Resume simulation
- **Ctrl+C:** Exit simulation

### Expected Behavior

1. **Initial Phase:** Robot spawns at random position and orientation
2. **Search Phase:** Robot rotates in place scanning environment with camera
3. **Detection Phase:** When red box is detected (~3 frames / 0.1s):
   - Robot stops rotating
   - Centers the box in camera view
4. **Approach Phase:** Robot walks forward toward box using locomotion API
5. **Arrival:** Robot stops at 0.5m distance from target

### Output

The simulation displays:
- **MuJoCo Viewer:** 3D visualization of robot and environment
- **RGB Camera Window:** First-person view with detection overlay
- **3D LIDAR Window:** Point cloud visualization
- **Console Output:** Real-time sensor data every 50 frames
  - IMU readings (orientation, velocity, acceleration)
  - Vision detection (distance, angle)
  - LIDAR scan data
  - Locomotion API status

### Performance Metrics

Typical execution times:
- Detection time: 0.1-0.5 seconds after box enters FOV
- Centering time: 0.3-0.6 seconds
- Approach time: 1-3 seconds (depends on initial distance)
- Total mission time: 2-5 seconds

## Project Structure

```
unitree_ros-master/
└── robots/
    └── g1_description/
        ├── g1_controlled_navigation.py  # Main simulation script
        ├── scene_with_room.xml          # MuJoCo scene definition (4×4 room)
        ├── g1_23dof.xml                 # G1 robot model (23 DOF)
        ├── g1_29dof.xml                 # G1 robot model (29 DOF)
        └── meshes/                      # Robot visual meshes
```

## Notes

- The simulation includes NumPy/matplotlib compatibility warnings which can be safely ignored
- Frame rate is capped at ~30 FPS for realistic real-time performance
- Detection uses HSV color filtering for red box identification
- LIDAR is used for precise distance measurement and obstacle avoidance
>>>>>>> 55665de (Initial commit: G1 controlled navigation with vision and gait)
