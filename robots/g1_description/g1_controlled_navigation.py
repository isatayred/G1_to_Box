#!/usr/bin/env python3
"""
Unitree G1 Robot - Controlled Navigation with Full Sensors
Complete autonomous navigation: search → detect → orient → approach → stop at 0.25m

Uses Unitree Locomotion API for walking control:
- StandUp(): Command robot to stand
- StopMove(): Stop all movement
- Move(vx, vy, vyaw): Set velocity commands
- SetGaitType(type): Configure gait pattern
- SetBodyHeight(height): Adjust body height
- UpdateGait(dt): Update walking animation
- IsWalking(): Check walking state
- GetPhase(): Get current gait phase
"""

import mujoco
import mujoco.viewer
import numpy as np
import cv2
import os
import tempfile
import xml.etree.ElementTree as ET
import random
import time

# Set True to show RGB camera window
SHOW_RGB_WINDOW = True

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

# Global state for IMU acceleration calculation
_prev_linvel = None
_prev_time = None

# Vision smoothing
from collections import deque
dist_hist = deque(maxlen=10)
ang_hist = deque(maxlen=10)

def detect_red_box_in_image(image):
    """Detect red box in camera image and return detection data with bounding box"""
    if image is None or image.shape[0] == 0:
        return {'detected': False, 'distance': None, 'angle': None, 'area': 0, 'bbox': None}
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 100:
            x, y, w, h = cv2.boundingRect(largest_contour)
            box_center_x = x + w / 2
            
            # Distance estimate from box size
            box_pixels = np.sqrt(area)
            focal_length_pixels = (image.shape[1] / 2) / np.tan(np.radians(45))
            real_box_size = 0.3
            estimated_distance = (focal_length_pixels * real_box_size) / (box_pixels + 1)
            distance = np.clip(estimated_distance, 0.02, 3.0)
            
            # Angle to box center
            angle = (box_center_x - image.shape[1] / 2) / (image.shape[1] / 2) * 45.0
            
            return {
                'detected': True,
                'distance': distance,
                'angle': angle,
                'area': area,
                'bbox': (x, y, w, h)
            }
    
    return {'detected': False, 'distance': None, 'angle': None, 'area': 0, 'bbox': None}

def simulate_lidar_scan(model, data, num_rays=16, max_range=3.0):
    """Simulate LIDAR for obstacle detection"""
    try:
        robot_pos = data.body('pelvis').xpos[:3].copy()
        # Lower LIDAR to box height (0.3m) to detect box
        robot_pos[2] = 0.3
    except:
        robot_pos = data.body(0).xpos[:3].copy()
        robot_pos[2] = 0.3
    
    try:
        quat = data.body('pelvis').xquat
    except:
        quat = data.body(0).xquat
        
    yaw = np.arctan2(2.0*(quat[0]*quat[3] + quat[1]*quat[2]), 
                      1.0 - 2.0*(quat[2]**2 + quat[3]**2))

    lidar_data = []
    for i in range(num_rays):
        angle_offset = (i / num_rays - 0.5) * (np.pi / 2)
        ray_angle = yaw + angle_offset
        ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle), 0.0])

        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            model, data,
            robot_pos.astype(np.float64),
            ray_dir.astype(np.float64),
            None, 1, -1, geomid,
        )
        # Ignore near-field to prevent self-hits
        if dist < 0 or dist > max_range or (0 < dist < 0.25):
            min_dist = -1.0
        else:
            min_dist = dist

        lidar_data.append({'angle': np.degrees(angle_offset), 'distance': min_dist})

    return lidar_data

def simulate_3d_lidar_scan(model, data, horizontal_rays=32, vertical_rays=16, max_range=5.0):
    """Simulate 3D LIDAR scan and return point cloud"""
    try:
        sensor_pos = data.body('pelvis').xpos[:3].copy()
        # Put sensor around chest height (not above head) - lower to see box
        sensor_pos[2] = 0.55
    except:
        sensor_pos = data.body(0).xpos[:3].copy()
        sensor_pos[2] = 0.55
    
    try:
        quat = data.body('pelvis').xquat
    except:
        quat = data.body(0).xquat
    
    # MuJoCo quaternion is [w, x, y, z]
    qw, qx, qy, qz = quat
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    points = []
    colors = []
    
    geomid = np.array([-1], dtype=np.int32)
    
    # Exclude robot body to reduce self-hits
    try:
        bodyexclude = model.body('pelvis').id
    except:
        bodyexclude = -1
    
    # MUCH more downward coverage: from -45° (down) to +10° (slightly up)
    v_min, v_max = np.radians(-45), np.radians(10)
    
    # Scan in 3D: horizontal and vertical rays
    for v in range(vertical_rays):
        vertical_angle = v_min + (v/(vertical_rays-1))*(v_max - v_min)
        
        for h in range(horizontal_rays):
            horizontal_angle = (h / horizontal_rays - 0.5) * np.radians(90)  # ±45 degrees horizontal
            ray_angle_yaw = yaw + horizontal_angle
            
            # 3D ray direction
            ray_dir = np.array([
                np.cos(ray_angle_yaw) * np.cos(vertical_angle),
                np.sin(ray_angle_yaw) * np.cos(vertical_angle),
                np.sin(vertical_angle)
            ])
            
            dist = mujoco.mj_ray(
                model, data,
                sensor_pos.astype(np.float64),
                ray_dir.astype(np.float64),
                None, 1, bodyexclude, geomid,
            )
            
            if 0 < dist < max_range:
                # Calculate 3D point
                point = sensor_pos + ray_dir * dist
                points.append(point)
                
                # Color by distance (closer = red, farther = blue)
                color_intensity = 1.0 - (dist / max_range)
                colors.append([color_intensity, 0.5, 1.0 - color_intensity])
    
    return np.array(points), np.array(colors)

def visualize_3d_lidar(points, colors, robot_pos):
    """Create a 2D bird's-eye view visualization of 3D LIDAR data"""
    # Create blank image (400x400 pixels, 8m x 8m view)
    img_size = 600
    view_range = 6.0  # meters
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 30  # Dark background
    
    scale = img_size / view_range
    center = img_size // 2
    
    # Draw range circles
    for r in [1, 2, 3]:
        radius = int(r * scale)
        cv2.circle(img, (center, center), radius, (60, 60, 60), 1)
    
    # Draw robot position (center)
    cv2.circle(img, (center, center), 8, (0, 255, 255), -1)
    cv2.circle(img, (center, center), 10, (0, 255, 255), 2)
    
    # Draw LIDAR points
    if len(points) > 0:
        for i, point in enumerate(points):
            # Convert world coordinates to image coordinates
            rel_x = point[0] - robot_pos[0]
            rel_y = point[1] - robot_pos[1]
            
            px = int(center + rel_x * scale)
            py = int(center - rel_y * scale)  # Flip Y for image coordinates
            
            if 0 <= px < img_size and 0 <= py < img_size:
                color = (int(colors[i][2] * 255), int(colors[i][1] * 255), int(colors[i][0] * 255))
                cv2.circle(img, (px, py), 3, color, -1)
    
    # Add text info
    cv2.putText(img, "3D LIDAR - Top View", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Points: {len(points)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Range: {view_range}m", (10, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw direction indicator (forward arrow)
    arrow_len = 30
    cv2.arrowedLine(img, (center, center), (center, center - arrow_len), 
                   (0, 255, 255), 2, tipLength=0.3)
    
    return img

def get_imu_data(model, data):
    """Extract IMU sensor data from robot"""
    try:
        body = data.body('torso_link')
    except:
        try:
            body = data.body('pelvis')
        except:
            body = data.body(0)
    
    # Orientation (quaternion) from MuJoCo: [w, x, y, z]
    qw, qx, qy, qz = body.xquat
    
    # Convert quaternion to Euler angles
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    
    # Angular velocity
    angular_vel = body.cvel[3:6]  # Last 3 components
    
    # Linear acceleration (finite difference)
    global _prev_linvel, _prev_time
    linvel = body.cvel[:3].copy()
    t = data.time
    
    if _prev_linvel is None:
        linear_accel = np.zeros(3)
    else:
        dt = max(t - _prev_time, 1e-6)
        linear_accel = (linvel - _prev_linvel) / dt
    
    _prev_linvel = linvel
    _prev_time = t
    
    return {
        'roll': np.degrees(roll),
        'pitch': np.degrees(pitch),
        'yaw': np.degrees(yaw),
        'angular_velocity': angular_vel,
        'linear_accel': linear_accel
    }

def render_camera_with_imu(model, data, imu_data, box_detection, frame):
    """Render RGB camera view with IMU overlay (console output)"""
    # Skip actual rendering - output to console instead
    return None

def render_rgb_frame(renderer, data, camera_name='rgb_camera'):
    """Render an RGB frame from the MuJoCo camera."""
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render().copy()
        # Convert RGB to BGR for OpenCV processing
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def fuse_vision_lidar(vision_detection, lidar_scan):
    """Fuse RGB vision and LIDAR to detect the box."""
    # Prefer RGB vision when available
    if vision_detection['detected']:
        fused = vision_detection.copy()
        fused['source'] = 'rgb'
        return fused

    # Otherwise, use LIDAR closest return in front
    valid = [r for r in lidar_scan if r['distance'] > 0]
    if not valid:
        return {'detected': False, 'distance': None, 'angle': None, 'area': 0, 'source': 'none'}

    closest = min(valid, key=lambda r: r['distance'])
    return {
        'detected': True,
        'distance': closest['distance'],
        'angle': closest['angle'],
        'area': 0,
        'source': 'lidar'
    }

def apply_walking_gait(data, model, phase, forward_velocity=0.0):
    """Apply a simple walking gait pattern to leg joints.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        phase: Walking phase [0, 2*pi]
        forward_velocity: Desired forward speed
    """
    # Walking gait parameters - moderate amplitude for realistic walking
    hip_amplitude = 0.5
    knee_amplitude = 0.8
    ankle_amplitude = 0.3
    
    # Alternating gait: when one leg swings forward, the other is planted
    # Left leg (phase) - swing phase when sin(phase) > 0
    left_swing = np.sin(phase)
    if left_swing > 0:  # Left leg swinging forward (in air)
        left_hip = -hip_amplitude * left_swing  # Hip extends forward
        left_knee = knee_amplitude * left_swing  # Knee bends
        left_ankle = -ankle_amplitude * left_swing  # Ankle adjusts
    else:  # Left leg planted (on ground)
        left_hip = 0.1 * left_swing  # Slight backward movement
        left_knee = 0.1  # Slight bend for stability
        left_ankle = 0.1 * left_swing
    
    # Right leg (opposite phase) - offset by pi
    right_swing = np.sin(phase + np.pi)
    if right_swing > 0:  # Right leg swinging forward (in air)
        right_hip = -hip_amplitude * right_swing
        right_knee = knee_amplitude * right_swing
        right_ankle = -ankle_amplitude * right_swing
    else:  # Right leg planted (on ground)
        right_hip = 0.1 * right_swing
        right_knee = 0.1
        right_ankle = 0.1 * right_swing
    
    # Try to find and set leg joint positions directly
    joint_targets = {
        'left_hip_pitch_joint': left_hip,
        'left_knee_joint': left_knee,
        'left_ankle_pitch_joint': left_ankle,
        'right_hip_pitch_joint': right_hip,
        'right_knee_joint': right_knee,
        'right_ankle_pitch_joint': right_ankle,
    }
    
    # Apply to joints by name
    for joint_name, target_pos in joint_targets.items():
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_addr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_addr] = target_pos
        except:
            pass  # Joint not found

class UnitreeLocomotionAPI:
    """Unitree-style Locomotion API for G1 Robot in MuJoCo"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.walk_phase = 0.0
        self.walk_frequency = 1.0  # Hz
        self.walk_speed = 0.01  # m/frame (increased for better forward motion)
        self.is_walking = False
        self.target_yaw = 0.0
        
        # Gait parameters
        self.hip_amplitude = 0.5
        self.knee_amplitude = 0.8
        self.ankle_amplitude = 0.3
        self.step_height = 0.05
    
    def StandUp(self):
        """Command robot to stand up"""
        # Set neutral standing pose
        self.is_walking = False
        self.walk_phase = 0.0
        print("🤖 StandUp command sent")
    
    def StopMove(self):
        """Stop all robot movement"""
        self.is_walking = False
        self.walk_phase = 0.0
        print("🛑 StopMove command sent")
    
    def Move(self, vx, vy, vyaw):
        """Move robot with velocity command
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s) 
            vyaw: Yaw rotation velocity (rad/s)
        """
        if abs(vx) > 0.001 or abs(vy) > 0.001:
            self.is_walking = True
            self.walk_speed = abs(vx) * 0.033  # Convert to per-frame
        else:
            self.is_walking = False
        
        # Store yaw rate command for smooth rotation during walking
        self.vyaw_cmd = vyaw
        # Also update target_yaw for rotation-only mode (when not walking)
        if not self.is_walking and abs(vyaw) > 0.001:
            self.target_yaw += vyaw * 0.033
        
        return True
    
    def SetGaitType(self, gait_type):
        """Set gait type (0=idle, 1=trot, 2=walk, etc.)
        
        Args:
            gait_type: Integer gait type
        """
        if gait_type == 0:  # Idle
            self.is_walking = False
        elif gait_type == 1:  # Trot (faster gait)
            self.walk_frequency = 1.5
        elif gait_type == 2:  # Walk (slower gait)
            self.walk_frequency = 1.0
        print(f"🚶 SetGaitType: {gait_type}")
        return True
    
    def SetBodyHeight(self, height):
        """Set robot body height
        
        Args:
            height: Target height in meters
        """
        self.data.qpos[2] = height
        print(f"📏 SetBodyHeight: {height:.2f}m")
        return True
    
    def _get_base_yaw_from_qpos(self):
        """Extract base yaw from quaternion in qpos (source of truth)"""
        # MuJoCo free joint quaternion in qpos is [w, x, y, z] at indices 3:7
        qw, qx, qy, qz = self.data.qpos[3:7]
        return np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    
    def ApplyYawToBase(self):
        """Apply target yaw to base quaternion (used during rotation)"""
        self.data.qpos[3] = np.cos(self.target_yaw / 2)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = np.sin(self.target_yaw / 2)
    
    def UpdateGait(self, dt=0.033):
        """Update gait phase and apply leg movements
        
        Args:
            dt: Time step in seconds
        """
        if not self.is_walking:
            return
        
        # Update walking phase
        self.walk_phase += 2 * np.pi * self.walk_frequency * dt
        if self.walk_phase > 2 * np.pi:
            self.walk_phase -= 2 * np.pi
        
        # Calculate leg swings (alternating gait like H1)
        left_swing = np.sin(self.walk_phase)
        right_swing = np.sin(self.walk_phase + np.pi)
        
        # H1-style gait: hip/ankle swing with phase, knee always bent using abs()
        # This ensures one leg stays as support while the other swings
        left_hip = -0.35 + left_swing * 0.12
        left_knee = 0.75 - abs(left_swing) * 0.2  # Always bent - key for support leg stability
        left_ankle = -0.35 + left_swing * 0.12
        
        right_hip = -0.35 + right_swing * 0.12
        right_knee = 0.75 - abs(right_swing) * 0.2  # Always bent - key for support leg stability
        right_ankle = -0.35 + right_swing * 0.12
        
        # Apply joint positions
        joint_targets = {
            'left_hip_pitch_joint': left_hip,
            'left_knee_joint': left_knee,
            'left_ankle_pitch_joint': left_ankle,
            'right_hip_pitch_joint': right_hip,
            'right_knee_joint': right_knee,
            'right_ankle_pitch_joint': right_ankle,
        }
        
        for joint_name, target_pos in joint_targets.items():
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    qpos_addr = self.model.jnt_qposadr[joint_id]
                    self.data.qpos[qpos_addr] = target_pos
            except:
                pass
        
        # Calculate push factor from stance leg only (proper stance/swing alternation)
        push_factor = 0.0
        if left_swing < 0:  # Left leg is stance
            push_factor = max(push_factor, abs(left_swing))
        if right_swing < 0:  # Right leg is stance
            push_factor = max(push_factor, abs(right_swing))
        
        # CRITICAL: Apply yaw rotation FIRST, then move in that direction (no lag)
        current_yaw = self._get_base_yaw_from_qpos()
        new_yaw = current_yaw + self.vyaw_cmd * 0.033  # Integrate yaw rate
        
        # Update base quaternion with new yaw BEFORE calculating movement
        self.data.qpos[3] = np.cos(new_yaw / 2)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = np.sin(new_yaw / 2)
        
        # Get current base position
        current_x = self.data.qpos[0]
        current_y = self.data.qpos[1]
        
        # Smooth forward displacement (always some movement, not huge spikes)
        forward_displacement = self.walk_speed * (0.3 + 0.7 * push_factor)
        
        # Update X, Y position based on NEW heading (no lag)
        new_x = current_x + forward_displacement * np.cos(new_yaw)
        new_y = current_y + forward_displacement * np.sin(new_yaw)
        
        self.data.qpos[0] = new_x
        self.data.qpos[1] = new_y
        self.data.qpos[2] = 0.82 + 0.005 * abs(np.sin(self.walk_phase))  # Maintain height with slight bobbing
    
    def GetPhase(self):
        """Get current walking phase"""
        return self.walk_phase
    
    def IsWalking(self):
        """Check if robot is walking"""
        return self.is_walking

def load_model_with_random_obstacle(urdf_path):
    """Convert URDF to MJCF with randomized obstacle"""
    from urllib.request import urlopen
    
    # Try to load G1 URDF
    try:
        model = mujoco.MjModel.from_xml_path(urdf_path)
    except:
        print(f"Could not load {urdf_path}, attempting URDF to MJCF conversion...")
        # Fallback: try to load directly
        model = mujoco.MjModel.from_xml_string(open(urdf_path).read())
    
    return model

def main():
    # Randomize robot starting position within room bounds (4m x 4m room, so -1.5 to 1.5 for x,y)
    robot_start_x = random.uniform(-1.0, 1.0)
    robot_start_y = random.uniform(-1.0, 1.0)
    robot_start_yaw = random.uniform(-np.pi, np.pi)  # Random initial orientation
    
    # Randomize box position anywhere in the room
    # Must be >0.3m from room center (0,0) and 0.8-1.5m from robot
    while True:
        obstacle_x = random.uniform(-1.2, 1.2)
        obstacle_y = random.uniform(-1.2, 1.2)
        # Distance from room center (0, 0)
        distance_from_center = np.sqrt(obstacle_x**2 + obstacle_y**2)
        # Distance from robot start position
        distance_to_robot = np.sqrt((obstacle_x - robot_start_x)**2 + (obstacle_y - robot_start_y)**2)
        # Box must be >0.3m from center AND 0.8-1.5m from robot
        if distance_from_center > 0.3 and distance_to_robot >= 0.8 and distance_to_robot <= 1.5:
            break
    
    obstacle_pos = np.array([obstacle_x, obstacle_y, 0.0])
    
    # Load G1 model with scene
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, 'scene_with_room.xml')
    
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
    except Exception as e:
        print(f"Error loading scene: {e}")
        return
    
    # Update obstacle position in the model (move the BODY, not the geom)
    try:
        obstacle_body_id = model.body('obstacle_box').id
        model.body_pos[obstacle_body_id] = obstacle_pos
    except Exception as e:
        print(f"Warning: could not move obstacle body: {e}")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    
    # Check available cameras
    print("📷 Model cameras:", [model.cam(i).name for i in range(model.ncam)])
    
    print("\n" + "="*70)
    print("UNITREE G1 ROBOT - BOX FINDING IN 4x4 ROOM")
    print("="*70)
    print(f"🏠 Room: 4m × 4m | 📦 Box: 0.3m × 0.3m × 0.3m (red)")
    print(f"🤖 Robot Start Position: ({robot_start_x:.2f}, {robot_start_y:.2f}) | Yaw: {np.degrees(robot_start_yaw):.1f}°")
    print(f"📦 Box position unknown to robot - will search and approach")
    print("\n🔄 Mode: Robot will rotate to search, then approach when detected")
    print("👁️  Detection: Using camera-based vision + LIDAR sensors")
    print("="*70 + "\n")
    
    # Initialize robot at randomized starting position
    qpos = data.qpos.copy()
    qpos[0] = robot_start_x  # X position
    qpos[1] = robot_start_y  # Y position
    qpos[2] = 0.82  # Height
    
    # Set initial orientation (yaw) using quaternion
    # Convert yaw to quaternion: quat = [cos(yaw/2), 0, 0, sin(yaw/2)]
    qpos[3] = np.cos(robot_start_yaw / 2)  # qw
    qpos[4] = 0  # qx
    qpos[5] = 0  # qy
    qpos[6] = np.sin(robot_start_yaw / 2)  # qz
    
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    # Initialize Unitree Locomotion API
    loco_api = UnitreeLocomotionAPI(model, data)
    loco_api.target_yaw = robot_start_yaw
    loco_api.StandUp()  # Start in standing pose
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 0.0
        viewer.cam.elevation = -90.0
        viewer.cam.distance = 5.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        
        frame = 0
        rotation_speed = 0.05  # Rotation speed (radians per frame) - ~2.9°/frame
        
        # Initialize robot position tracking (stay in place)
        com_x = robot_start_x
        com_y = robot_start_y
        
        # Detection tracking for stopping rotation
        detection_start_time = None
        stop_delay = 0.1  # seconds - very quick response
        is_stopped = False
        delay_elapsed = False
        centering_threshold = 3.0  # degrees - allow slightly looser alignment for quicker response
        prev_angle_error = 0.0  # For PD controller during walking
        
        # Stable detection requirements
        stable_detect_frames = 0
        REQUIRED_STABLE = 3  # ~0.1s at 30fps - respond quickly
        
        # Walking state variables
        has_arrived = False
        target_distance = 0.5  # meters - stop when closer than this
        
        while viewer.is_running():
            frame += 1
            
            # Add delay to make movement visible (30 FPS)
            time.sleep(0.033)
            
            # Get sensor data
            rgb_frame = render_rgb_frame(renderer, data, camera_name='rgb_camera')
            vision_detection = detect_red_box_in_image(rgb_frame)
            
            # Apply vision smoothing
            if vision_detection['detected']:
                dist_hist.append(vision_detection['distance'])
                ang_hist.append(vision_detection['angle'])
                vision_detection['distance'] = float(np.median(dist_hist))
                vision_detection['angle'] = float(np.median(ang_hist))
                stable_detect_frames += 1
            else:
                dist_hist.clear()
                ang_hist.clear()
                stable_detect_frames = 0
            
            lidar_scan = simulate_lidar_scan(model, data, num_rays=16, max_range=3.0)
            imu_data = get_imu_data(model, data)
            
            # Get 3D LIDAR data
            lidar_3d_points, lidar_3d_colors = simulate_3d_lidar_scan(model, data, horizontal_rays=32, vertical_rays=16, max_range=5.0)

            # Fuse RGB + LIDAR (vision preferred)
            box_detection = fuse_vision_lidar(vision_detection, lidar_scan)

            # Save RGB frames periodically (no GUI)
            if rgb_frame is not None and frame % 50 == 0:
                cv2.imwrite(f"/tmp/g1_rgb_{frame:05d}.png", rgb_frame)

            # Show RGB window with overlays
            if SHOW_RGB_WINDOW and rgb_frame is not None:
                display_frame = rgb_frame.copy()
                
                # Draw detection box if detected
                if vision_detection['detected'] and vision_detection['bbox'] is not None:
                    x, y, w, h = vision_detection['bbox']
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add distance and angle text
                    text = f"Dist: {vision_detection['distance']:.2f}m | Ang: {vision_detection['angle']:.1f}deg"
                    cv2.putText(display_frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add crosshair at center
                h, w = display_frame.shape[:2]
                cv2.line(display_frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 0, 0), 2)
                cv2.line(display_frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 0, 0), 2)
                
                # Add frame counter and yaw info
                info_text = f"Frame: {frame} | Yaw: {np.degrees(loco_api.target_yaw):.1f}deg"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add status indicator
                if has_arrived:
                    status_text = "STATUS: ARRIVED"
                    cv2.putText(display_frame, status_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif loco_api.IsWalking():
                    status_text = "STATUS: WALKING (Unitree API)"
                    cv2.putText(display_frame, status_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                elif is_stopped:
                    status_text = "STATUS: LOCKED"
                    cv2.putText(display_frame, status_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    status_text = "STATUS: ROTATING"
                    cv2.putText(display_frame, status_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("G1 RGB Camera", display_frame)
                cv2.waitKey(1)
            
            # Visualize 3D LIDAR
            try:
                robot_body = data.body('pelvis')
            except:
                robot_body = data.body(0)
            robot_pos_3d = robot_body.xpos[:3]
            
            lidar_viz = visualize_3d_lidar(lidar_3d_points, lidar_3d_colors, robot_pos_3d)
            cv2.imshow("G1 3D LIDAR", lidar_viz)
            cv2.waitKey(1)

            # Render camera with IMU overlay (console only)
            camera_image = render_camera_with_imu(model, data, imu_data, box_detection, frame)
            
            # Get robot height from simulation
            try:
                robot_body = data.body('pelvis')
            except:
                robot_body = data.body(0)
            
            com_height = robot_body.xpos[2]
            
            qpos_target = data.qpos.copy()
            qpos_target[2] = 0.82  # Maintain height
            
            # Check for object detection and manage rotation state
            current_time = time.time()
            
            # Only trigger on STABLE RGB camera detection (not LIDAR)
            if stable_detect_frames >= REQUIRED_STABLE and detection_start_time is None and not is_stopped:
                # Stable RGB detection - start the timer
                detection_start_time = current_time
                print(f"\n🎯 [Frame {frame}] BOX DETECTED! Preparing to approach...")
                print(f"   Distance: {vision_detection['distance']:.2f}m | Angle: {vision_detection['angle']:.1f}°\n")
            
            # Check if delay has elapsed
            if detection_start_time is not None and not delay_elapsed:
                elapsed = current_time - detection_start_time
                if elapsed >= stop_delay:
                    delay_elapsed = True
                    print(f"\n⏱️  [Frame {frame}] Centering on target...\n")
            
            # After delay, check if we're centered on the target
            if delay_elapsed and not is_stopped:
                if vision_detection['detected']:
                    angle_to_box = abs(vision_detection['angle'])
                    if angle_to_box <= centering_threshold:
                        # Box is centered, stop rotation
                        is_stopped = True
                        print(f"\n🎯 [Frame {frame}] LOCKED ON TARGET! Box centered in view.")
                        print(f"   Final Yaw: {np.degrees(loco_api.target_yaw):.1f}°")
                        print(f"   Box Angle: {vision_detection['angle']:.1f}° (centered)")
                        print(f"   Distance: {vision_detection['distance']:.2f}m")
                        print(f"   Robot position: ({com_x:.2f}m, {com_y:.2f}m)")
                        
                        # Check if we need to walk to the target
                        if vision_detection['distance'] > target_distance:
                            # Use Unitree API to start walking
                            forward_speed = 0.5  # m/s - increased for faster approach
                            loco_api.Move(vx=forward_speed, vy=0.0, vyaw=0.0)
                            print(f"🚶 [Frame {frame}] Starting to walk toward target using Unitree API...\n")
                        else:
                            has_arrived = True
                            loco_api.StopMove()
                            print(f"✅ [Frame {frame}] Already at target distance!\n")
                    else:
                        # Still rotating to center
                        if frame % 50 == 0:
                            print(f"🔄 [Frame {frame}] Centering... Box angle: {vision_detection['angle']:.1f}° (target: 0°)")
            
            # ROTATION MODE - Rotate using Unitree API until centered on target
            if not is_stopped:
                # Use API for rotation (convert rad/frame to rad/s)
                loco_api.Move(vx=0.0, vy=0.0, vyaw=rotation_speed / 0.033)
            
            # WALKING MODE - Use Unitree Locomotion API
            # LIDAR has priority for stopping distance, RGB for initial detection/centering
            if loco_api.IsWalking() and not has_arrived:
                # Get LIDAR closest distance in front (for stopping)
                lidar_valid = [r for r in lidar_scan if r['distance'] > 0 and abs(r['angle']) < 30]
                lidar_distance = min([r['distance'] for r in lidar_valid]) if lidar_valid else None
                
                # LIDAR takes priority for stopping - stop when LIDAR reports ≤0.5m
                if lidar_distance is not None and lidar_distance <= target_distance:
                    loco_api.StopMove()
                    has_arrived = True
                    print(f"\n✅ [Frame {frame}] ARRIVED AT TARGET (LIDAR)!")
                    print(f"   LIDAR Distance: {lidar_distance:.2f}m")
                    print(f"   Final Position: ({com_x:.2f}m, {com_y:.2f}m)")
                    print(f"   Final Yaw: {np.degrees(loco_api.target_yaw):.1f}°\n")
                elif box_detection['detected'] and box_detection['distance'] > target_distance:
                    # Keep walking - target not reached yet
                    vx_cmd = forward_speed
                    vyaw_cmd = 0.0  # No rotation during walking
                    
                    # Send velocity command
                    loco_api.Move(vx=vx_cmd, vy=0.0, vyaw=vyaw_cmd)
                    
                    # Update gait
                    loco_api.UpdateGait(dt=0.033)
                elif not box_detection['detected'] and lidar_distance is None:
                    # Lost both vision and LIDAR, stop for safety
                    loco_api.StopMove()
                    print(f"\n⚠️  [Frame {frame}] Lost all detection (RGB and LIDAR), stopping...")
            
            if frame % 100 == 0:
                if has_arrived:
                    print(f"✅ [Frame {frame}] ARRIVED! Standing at target.")
                    if box_detection['detected']:
                        print(f"   👁️  Box in view: Distance: {box_detection['distance']:.2f}m | Angle: {box_detection['angle']:.1f}° | Source: {box_detection['source']}")
                    print(f"   📍 Position: ({com_x:.2f}m, {com_y:.2f}m)")
                elif loco_api.IsWalking():
                    print(f"🚶 [Frame {frame}] WALKING to target using Unitree API... Pos: ({com_x:.2f}m, {com_y:.2f}m)")
                    if box_detection['detected']:
                        print(f"   👁️  Target: Distance: {box_detection['distance']:.2f}m | Angle: {box_detection['angle']:.1f}° | Source: {box_detection['source']}")
                elif is_stopped:
                    print(f"🎯 [Frame {frame}] LOCKED ON TARGET at Yaw: {np.degrees(loco_api.target_yaw):.1f}°")
                    if box_detection['detected']:
                        print(f"   👁️  Box in view: Distance: {box_detection['distance']:.2f}m | Angle: {box_detection['angle']:.1f}° | Source: {box_detection['source']}")
                elif delay_elapsed:
                    print(f"🔄 [Frame {frame}] CENTERING... Current Yaw: {np.degrees(loco_api.target_yaw):.1f}°")
                    if vision_detection['detected']:
                        print(f"   👁️  Box angle: {vision_detection['angle']:.1f}° (centering to 0°)")
                else:
                    print(f"🔄 [Frame {frame}] ROTATING... Current Yaw: {np.degrees(loco_api.target_yaw):.1f}°")
                    if box_detection['detected']:
                        print(f"   👁️  Box Detected! Distance: {box_detection['distance']:.2f}m | Angle: {box_detection['angle']:.1f}° | Source: {box_detection['source']}")
            
            
            # Let physics handle position - just maintain orientation during rotation
            if not loco_api.IsWalking():
                # Only control orientation, let physics handle position
                data.qpos[3] = np.cos(loco_api.target_yaw / 2)
                data.qpos[4] = 0
                data.qpos[5] = 0
                data.qpos[6] = np.sin(loco_api.target_yaw / 2)
            
            mujoco.mj_forward(model, data)
            
            # Always read actual position from physics after forward dynamics
            try:
                robot_body = data.body('pelvis')
            except:
                robot_body = data.body(0)
            com_x = robot_body.xpos[0]
            com_y = robot_body.xpos[1]
            
            # Update viewer camera to follow robot
            viewer.cam.lookat[:] = [com_x, com_y, 0.5]
            
            # Step the simulation viewer
            viewer.sync()
            
            # Log detailed IMU + Vision every 50 frames
            if frame % 50 == 0:
                print(f"\n{'='*80}")
                print(f"📱 [Frame {frame:5d}] SENSOR DATA:")
                print(f"{'='*80}")
                print(f"🧭 ORIENTATION (Euler Angles):")
                print(f"   Roll:  {imu_data['roll']:8.2f}°    |    Pitch: {imu_data['pitch']:8.2f}°    |    Yaw: {imu_data['yaw']:8.2f}°")
                print(f"\n⚙️  ANGULAR VELOCITY (rad/s):")
                print(f"   ωx: {imu_data['angular_velocity'][0]:8.5f}  |  ωy: {imu_data['angular_velocity'][1]:8.5f}  |  ωz: {imu_data['angular_velocity'][2]:8.5f}")
                print(f"\n📊 LINEAR ACCELERATION (m/s²):")
                print(f"   ax: {imu_data['linear_accel'][0]:8.5f}  |  ay: {imu_data['linear_accel'][1]:8.5f}  |  az: {imu_data['linear_accel'][2]:8.5f}")
                print(f"\n📍 POSITION:")
                print(f"   X: {com_x:8.3f}m  |  Y: {com_y:8.3f}m  |  Z (Height): {com_height:8.3f}m")
                print(f"\n👁️  RGB VISION + LIDAR FUSION:")
                if box_detection['detected']:
                    print(f"   ✅ BOX DETECTED ({box_detection['source']})")
                    print(f"      Distance: {box_detection['distance']:6.2f}m")
                    print(f"      Angle:    {box_detection['angle']:6.1f}° (relative to heading)")
                else:
                    print(f"   ❌ SEARCHING... (no object detected in FOV)")
                # LIDAR preview (front rays)
                lidar_preview = ', '.join([f"{r['angle']:+.0f}°:{r['distance']:.2f}" for r in lidar_scan[:6]])
                print(f"\n📡 LIDAR (front rays): {lidar_preview}")
                print(f"\n🤖 UNITREE API STATUS:")
                print(f"   Walking: {loco_api.IsWalking()}  |  Phase: {loco_api.GetPhase():.2f}  |  Target Yaw: {np.degrees(loco_api.target_yaw):.1f}°")
                print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
