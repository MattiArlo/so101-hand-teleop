#!/usr/bin/env python3
"""
SO-101 Dual-Hand Real-Time Teleoperation Visualizer
Controls the 6-DOF SO-101 follower arm using TWO hands via laptop webcam:
  - Left Hand  -> Arm Position (3-DOF):
      * Lateral (X)        -> shoulder_pan (Base Rotation Left/Right)
      * Height (Y)         -> shoulder_lift (Arm Elevation Up/Down)
      * Depth/Scale (Z)    -> elbow_flex (Arm Reach Forward/Backward)
  - Right Hand -> End-Effector Orientation & Gripper (3-DOF):
      * Wrist Tilt (Pitch) -> wrist_flex (Gripper Pitch Up/Down)
      * Wrist Roll (Roll)  -> wrist_roll (Gripper Roll CW/CCW)
      * Pinch (Distance)   -> gripper (0% Closed - 100% Open)

Features:
- Complete mechanical decoupling: Moving the arm position does not disturb wrist orientation or pinch.
- Dual guided neutral target boxes on screen for intuitive setup.
- Soft bumpless engagement via software slew-rate limiter.
- Swap hand roles hotkey ([S]) for left-handed or alternate operator preference.
- Independent neutral calibration ([C]) for both hands.
- Push-to-engage clutch ([SPACE]) and toggle continuous tracking ([T]).
- Resizable window (cv2.WINDOW_NORMAL) with left HUD dashboard and right camera stream.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Default SO-101 Follower limits (from matti_follower_arm calibration in degrees / percent)
DEFAULT_LIMITS = {
    "shoulder_pan": (-80.0, 80.0),
    "shoulder_lift": (-95.0, 95.0),
    "elbow_flex": (-90.0, 90.0),
    "wrist_flex": (-75.0, 75.0),
    "wrist_roll": (-170.0, 170.0),
    "gripper": (0.0, 100.0),
}

# Maximum safe movement per frame (at ~30 FPS: 1.4 deg/frame = ~42 deg/sec)
MAX_DEG_PER_FRAME = 1.4
MAX_GRIPPER_PER_FRAME = 4.0

DEFAULT_MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_CALIB_PATH = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / "matti_follower_arm.json"


class DualHandPoseMapper:
    """
    Decoupled Dual-Hand 6-DOF Mapper:
      - Position Hand (default: Left): Controls shoulder_pan, shoulder_lift, elbow_flex
      - Tool Hand (default: Right): Controls wrist_flex, wrist_roll, gripper
    """

    def __init__(self, ema_alpha: float = 0.25):
        self.ema_alpha = ema_alpha

        # Position hand reference (default centered in left half of screen)
        self.pos_neutral_x = 0.27
        self.pos_neutral_y = 0.50
        self.pos_neutral_scale = 0.18
        self.pos_calibrated = False

        # Tool hand reference (default centered in right half of screen)
        self.tool_neutral_roll = 0.0
        self.tool_neutral_pitch = -70.0
        self.tool_calibrated = False

        # Role configuration: False = Left:Pos, Right:Tool; True = Right:Pos, Left:Tool
        self.swap_roles = False

        # Active smoothed joint targets
        self.smoothed_joints: Dict[str, float] = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -30.0,
            "elbow_flex": 60.0,
            "wrist_flex": -30.0,
            "wrist_roll": 0.0,
            "gripper": 100.0,
        }

    def set_pos_neutral(self, wrist_pt: Tuple[float, float], scale: float):
        self.pos_neutral_x = wrist_pt[0]
        self.pos_neutral_y = wrist_pt[1]
        self.pos_neutral_scale = max(scale, 0.05)
        self.pos_calibrated = True

    def set_tool_neutral(self, roll_rad: float, pitch_deg: float):
        self.tool_neutral_roll = roll_rad
        self.tool_neutral_pitch = pitch_deg
        self.tool_calibrated = True

    def compute_position(self, landmarks_norm: List[Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculates 3D arm position (shoulder_pan, shoulder_lift, elbow_flex)
        from the position hand's wrist location and palm scale.
        """
        pts = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks_norm])
        wrist = pts[0]
        middle_mcp = pts[9]

        scale = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))
        if not self.pos_calibrated:
            self.set_pos_neutral((wrist[0], wrist[1]), scale)

        # 1. Shoulder Pan (Horizontal lateral deflection)
        delta_x = wrist[0] - self.pos_neutral_x
        # In mirrored display, moving left yields negative delta_x -> pan left
        # Map +/- 0.16 screen fraction to +/- 55 degrees pan
        target_pan = float(np.clip(delta_x / 0.16 * 55.0, DEFAULT_LIMITS["shoulder_pan"][0], DEFAULT_LIMITS["shoulder_pan"][1]))

        # 2. Shoulder Lift & Elbow Flex (Vertical deflection & palm scale)
        delta_y = wrist[1] - self.pos_neutral_y
        depth_ratio = (scale - self.pos_neutral_scale) / max(self.pos_neutral_scale, 1e-4)

        base_lift = -30.0
        base_elbow = 65.0

        lift_offset = -float(delta_y / 0.16 * 45.0)  # Moving hand up on screen raises lift
        reach_offset = float(depth_ratio * 40.0)     # Pushing hand closer to camera extends elbow

        target_lift = float(np.clip(base_lift + lift_offset - reach_offset * 0.25, DEFAULT_LIMITS["shoulder_lift"][0], DEFAULT_LIMITS["shoulder_lift"][1]))
        target_elbow = float(np.clip(base_elbow - lift_offset * 0.25 + reach_offset, DEFAULT_LIMITS["elbow_flex"][0], DEFAULT_LIMITS["elbow_flex"][1]))

        pos_targets = {
            "shoulder_pan": target_pan,
            "shoulder_lift": target_lift,
            "elbow_flex": target_elbow,
        }

        # Apply EMA smoothing to position joints
        for k in pos_targets:
            self.smoothed_joints[k] = self.ema_alpha * pos_targets[k] + (1.0 - self.ema_alpha) * self.smoothed_joints[k]

        metrics = {
            "wrist_x": wrist[0],
            "wrist_y": wrist[1],
            "scale": scale,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "depth_ratio": depth_ratio,
        }
        return pos_targets, metrics

    def compute_tool(self, landmarks_norm: List[Tuple[float, float, float]], is_right_hand: bool = True) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculates end-effector orientation and gripper (wrist_flex, wrist_roll, gripper)
        from the tool hand's palm rotation, wrist tilt, and pinch ratio.
        """
        pts = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks_norm])
        wrist = pts[0]
        thumb_tip = pts[4]
        index_tip = pts[8]
        middle_mcp = pts[9]
        index_mcp = pts[5]
        pinky_mcp = pts[17]

        scale = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))

        # 1. Gripper Pinch (2D Euclidean distance normalized by palm scale)
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        pinch_ratio = pinch_dist / max(scale, 1e-4)
        min_pinch_ratio = 0.22
        max_pinch_ratio = 0.58
        gripper_pct = np.clip((pinch_ratio - min_pinch_ratio) / (max_pinch_ratio - min_pinch_ratio), 0.0, 1.0) * 100.0

        # 2. Wrist Roll (Palm rotation angle)
        if is_right_hand:
            palm_vec = pinky_mcp[:2] - index_mcp[:2]
        else:
            palm_vec = index_mcp[:2] - pinky_mcp[:2]
        current_roll_rad = math.atan2(palm_vec[1], palm_vec[0])

        # 3. Wrist Flex / Pitch (Vector from wrist to middle MCP)
        wrist_vec = middle_mcp[:2] - wrist[:2]
        pitch_angle_deg = math.degrees(math.atan2(wrist_vec[1], math.sqrt(wrist_vec[0]**2 + 1e-6)))

        if not self.tool_calibrated:
            self.set_tool_neutral(current_roll_rad, pitch_angle_deg)

        roll_diff_rad = current_roll_rad - self.tool_neutral_roll
        roll_diff_rad = (roll_diff_rad + math.pi) % (2 * math.pi) - math.pi
        target_roll = float(np.clip(math.degrees(roll_diff_rad), DEFAULT_LIMITS["wrist_roll"][0], DEFAULT_LIMITS["wrist_roll"][1]))

        base_flex = -30.0
        pitch_offset = (pitch_angle_deg - self.tool_neutral_pitch) * 1.3
        target_flex = float(np.clip(base_flex + pitch_offset, DEFAULT_LIMITS["wrist_flex"][0], DEFAULT_LIMITS["wrist_flex"][1]))

        tool_targets = {
            "wrist_flex": target_flex,
            "wrist_roll": target_roll,
            "gripper": gripper_pct,
        }

        # Apply EMA smoothing to tool joints
        for k in tool_targets:
            self.smoothed_joints[k] = self.ema_alpha * tool_targets[k] + (1.0 - self.ema_alpha) * self.smoothed_joints[k]

        metrics = {
            "wrist_x": wrist[0],
            "wrist_y": wrist[1],
            "scale": scale,
            "pinch_ratio": pinch_ratio,
            "gripper_pct": gripper_pct,
            "roll_deg": math.degrees(roll_diff_rad),
            "pitch_deg": pitch_angle_deg,
        }
        return tool_targets, metrics


def draw_hud(panel: np.ndarray, joints: Dict[str, float], is_clutched: bool,
             left_detected: bool, right_detected: bool,
             left_in_box: bool, right_in_box: bool,
             swap_roles: bool, fps: float, robot_connected: bool) -> None:
    """Renders the dark control dashboard on the LEFT side of the window (380px)."""
    h, w, _ = panel.shape
    panel[:] = (20, 20, 24)

    # Title & FPS
    cv2.putText(panel, "SO-101 DUAL-HAND TELEOP", (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"FPS: {fps:4.1f}", (w - 88, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(panel, (18, 42), (w - 18, 42), (70, 70, 75), 1)

    # Hand Roles & Tracking Status
    y = 66
    left_role_name = "POSITION (3D)" if not swap_roles else "WRIST & GRIP"
    right_role_name = "WRIST & GRIP" if not swap_roles else "POSITION (3D)"

    # Left Hand Badge
    l_color = (0, 255, 120) if left_detected else (0, 140, 255)
    l_status = "TRACKED" if left_detected else "SEARCHING..."
    cv2.circle(panel, (26, y - 5), 5, l_color, -1)
    cv2.putText(panel, f"LEFT: {left_role_name}", (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, l_status, (w - 105, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, l_color, 1, cv2.LINE_AA)

    y += 24
    # Right Hand Badge
    r_color = (0, 255, 120) if right_detected else (0, 140, 255)
    r_status = "TRACKED" if right_detected else "SEARCHING..."
    cv2.circle(panel, (26, y - 5), 5, r_color, -1)
    cv2.putText(panel, f"RIGHT: {right_role_name}", (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, r_status, (w - 105, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, r_color, 1, cv2.LINE_AA)

    # Clutch Status Banner
    y += 28
    if is_clutched:
        cv2.rectangle(panel, (18, y - 16), (w - 18, y + 8), (0, 130, 0), -1)
        cv2.putText(panel, "[ CLUTCH ENGAGED - ACTIVE ]", (35, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(panel, (18, y - 16), (w - 18, y + 8), (45, 45, 80), -1)
        cv2.putText(panel, "[ DISENGAGED - FROZEN ]", (45, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (200, 200, 255), 1, cv2.LINE_AA)

    y += 28
    # Hardware Status
    if robot_connected:
        cv2.putText(panel, "ARM: CONNECTED (/dev/ttyACM1)", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 255, 60), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "ARM: SIMULATION (SAFE)", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 50), 1, cv2.LINE_AA)

    # Operation Guide
    y += 20
    cv2.line(panel, (18, y), (w - 18, y), (70, 70, 75), 1)
    y += 18
    cv2.putText(panel, "OPERATION GUIDE:", (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    y += 17
    step1_ready = left_in_box and right_in_box
    step1_color = (0, 255, 120) if step1_ready else (160, 160, 160)
    cv2.putText(panel, "1. Place hands in Left & Right boxes", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, step1_color, 1, cv2.LINE_AA)
    y += 16
    cv2.putText(panel, "2. Press [C] to calibrate neutral zeros", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    y += 16
    step3_color = (0, 255, 120) if is_clutched else (180, 180, 180)
    cv2.putText(panel, "3. Hold [SPACE] to smoothly drive arm", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, step3_color, 1, cv2.LINE_AA)

    # Joint Gauges - Grouped by Role
    y += 22
    cv2.line(panel, (18, y), (w - 18, y), (70, 70, 75), 1)

    # Section 1: Position Hand
    y += 20
    pos_header = f"--- {'LEFT' if not swap_roles else 'RIGHT'} HAND: ARM POSITION ---"
    cv2.putText(panel, pos_header, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 200, 80), 1, cv2.LINE_AA)
    y += 8

    pos_joints = [
        ("shoulder_pan", "Pan (Base Yaw)", "deg"),
        ("shoulder_lift", "Lift (Shoulder)", "deg"),
        ("elbow_flex", "Reach (Elbow)", "deg"),
    ]

    for key, label, unit in pos_joints:
        y += 20
        _draw_gauge_bar(panel, key, label, unit, joints.get(key, 0.0), y, w, color=(255, 190, 40))

    # Section 2: Tool Hand
    y += 28
    tool_header = f"--- {'RIGHT' if not swap_roles else 'LEFT'} HAND: WRIST & GRIP ---"
    cv2.putText(panel, tool_header, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.40, (80, 230, 120), 1, cv2.LINE_AA)
    y += 8

    tool_joints = [
        ("wrist_flex", "Pitch (Wrist Flex)", "deg"),
        ("wrist_roll", "Roll (Wrist Roll)", "deg"),
        ("gripper", "Gripper Pinch", "%"),
    ]

    for key, label, unit in tool_joints:
        y += 20
        is_grip = (key == "gripper")
        _draw_gauge_bar(panel, key, label, unit, joints.get(key, 0.0), y, w, color=(80, 230, 120), is_grip=is_grip)

    # Keyboard Controls Footer
    cv2.line(panel, (18, h - 130), (w - 18, h - 130), (70, 70, 75), 1)
    controls = [
        "[SPACE] Hold: Engage Clutch",
        "[T] Toggle Continuous Mode",
        "[C] Calibrate Neutral Zeros",
        f"[S] Swap Roles ({'L:Pos, R:Tool' if not swap_roles else 'R:Pos, L:Tool'})",
        "[Q] or [ESC] Exit",
    ]
    cy = h - 110
    for c in controls:
        cv2.putText(panel, c, (18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 170), 1, cv2.LINE_AA)
        cy += 18


def _draw_gauge_bar(panel: np.ndarray, key: str, label: str, unit: str,
                    val: float, y: int, w: int, color: Tuple[int, int, int], is_grip: bool = False) -> None:
    """Helper to draw a single joint gauge bar with limits and center zero marker."""
    lim_min, lim_max = DEFAULT_LIMITS[key]
    ratio = np.clip((val - lim_min) / (lim_max - lim_min + 1e-6), 0.0, 1.0)

    # Label & text value
    cv2.putText(panel, f"{label}:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    val_str = f"{val:5.1f} {unit}"
    cv2.putText(panel, val_str, (w - 95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Bar background
    bar_x = 18
    bar_y = y + 4
    bar_w = w - 36
    bar_h = 6
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 45), -1)

    # Fill
    fill_w = int(bar_w * ratio)
    if is_grip:
        bar_color = (int(255 * (1.0 - ratio)), int(255 * ratio), 40)
    else:
        bar_color = color
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    # Center zero mark
    if lim_min < 0 < lim_max:
        center_x = int(bar_x + bar_w * (-lim_min / (lim_max - lim_min)))
        cv2.line(panel, (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), (180, 180, 180), 1)


def draw_target_box(cam_view: np.ndarray, norm_box: Tuple[float, float, float, float],
                    title: str, subtext: str, in_box: bool, base_color: Tuple[int, int, int]) -> None:
    """Draws a themed corner-bracketed target box with crosshair and title."""
    h, w, _ = cam_view.shape
    x1, y1 = int(norm_box[0] * w), int(norm_box[1] * h)
    x2, y2 = int(norm_box[2] * w), int(norm_box[3] * h)

    color = (0, 255, 120) if in_box else base_color
    corner_len = 24

    # Corners
    cv2.line(cam_view, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)

    cv2.line(cam_view, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)

    cv2.line(cam_view, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)

    cv2.line(cam_view, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)

    # Center crosshair
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(cam_view, (cx - 8, cy), (cx + 8, cy), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(cam_view, (cx, cy - 8), (cx, cy + 8), (100, 100, 100), 1, cv2.LINE_AA)

    # Box Labels
    cv2.putText(cam_view, title, (x1 + 8, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.44, color, 1, cv2.LINE_AA)
    status_label = "READY: Neutral Zone" if in_box else subtext
    cv2.putText(cam_view, status_label, (x1 + 8, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def draw_hand_skeleton(cam_view: np.ndarray, landmarks_norm: List[Tuple[float, float, float]],
                       label: str, role_title: str, is_tool_hand: bool,
                       pinch_pct: float = 100.0, bone_color: Tuple[int, int, int] = (255, 180, 0)) -> None:
    """Draws 21 hand landmarks, connecting bones, and contextual metrics."""
    h, w, _ = cam_view.shape
    pts_px = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks_norm]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    # Draw bones
    for start, end in connections:
        cv2.line(cam_view, pts_px[start], pts_px[end], bone_color, 2, cv2.LINE_AA)

    # Draw joints
    for i, pt in enumerate(pts_px):
        if is_tool_hand and i in [4, 8]:
            cv2.circle(cam_view, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(cam_view, pt, 4, (0, 220, 120), -1, cv2.LINE_AA)

    wrist_pt = pts_px[0]
    tag = f"[{label.upper()}: {role_title}]"
    cv2.putText(cam_view, tag, (wrist_pt[0] - 50, wrist_pt[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)

    # Pinch indicator on tool hand
    if is_tool_hand:
        thumb_pt = pts_px[4]
        index_pt = pts_px[8]
        color_g = int(np.clip(pinch_pct * 2.55, 0, 255))
        color_r = int(np.clip((100.0 - pinch_pct) * 2.55, 0, 255))
        cv2.line(cam_view, thumb_pt, index_pt, (0, color_g, color_r), 3, cv2.LINE_AA)

        mid_x = (thumb_pt[0] + index_pt[0]) // 2
        mid_y = (thumb_pt[1] + index_pt[1]) // 2
        cv2.putText(cam_view, f"Grip: {pinch_pct:3.0f}%", (mid_x + 10, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, color_g, color_r), 1, cv2.LINE_AA)


def run_visualizer(camera_id: int = 0, connect_robot: bool = False,
                   robot_port: str = "/dev/ttyACM1", default_swap: bool = False):
    print("=" * 65)
    print("SO-101 DUAL-HAND REAL-TIME TELEOPERATION")
    print("=" * 65)
    print("Division of Labor:")
    print("  • LEFT HAND  -> 3D Arm Position (Pan, Lift, Reach)")
    print("  • RIGHT HAND -> Tool Orientation & Gripper (Pitch, Roll, Pinch)")
    print("=" * 65)

    robot = None
    current_robot_cmd: Dict[str, float] = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -30.0,
        "elbow_flex": 60.0,
        "wrist_flex": -30.0,
        "wrist_roll": 0.0,
        "gripper": 100.0,
    }

    if connect_robot:
        try:
            from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
            print(f"Connecting to SO-101 Follower arm on {robot_port} (id: matti_follower_arm)...")
            config = SOFollowerRobotConfig(port=robot_port, id="matti_follower_arm", use_degrees=True)
            robot = SOFollower(config)
            robot.connect()
            print("Follower arm connected successfully with torque enabled!")

            # Read initial physical positions so arm never jumps on start
            present_pos = robot.bus.sync_read("Present_Position")
            for k, v in present_pos.items():
                if k in current_robot_cmd:
                    current_robot_cmd[k] = float(v)
            print(f"Loaded physical starting pose: {current_robot_cmd}")
        except Exception as e:
            print(f"Warning: Failed to connect to robot arm ({e}). Continuing in simulation mode.")
            robot = None

    # Load MediaPipe HandLandmarker
    if not DEFAULT_MODEL_PATH.exists():
        print(f"Downloading MediaPipe HandLandmarker model to {DEFAULT_MODEL_PATH}...")
        DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, str(DEFAULT_MODEL_PATH))
        print("Model downloaded successfully!")

    print("Initializing MediaPipe HandLandmarker (num_hands=2)...")
    base_options = python.BaseOptions(model_asset_path=str(DEFAULT_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Open Camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mapper = DualHandPoseMapper(ema_alpha=0.30)
    mapper.swap_roles = default_swap

    # Initialize mapper smoothed joints to current robot pose
    for k, v in current_robot_cmd.items():
        mapper.smoothed_joints[k] = v

    window_name = "SO-101 Dual-Hand Teleoperation Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1340, 720)

    is_clutched = False
    toggle_clutch = False
    last_time = time.perf_counter()
    fps = 30.0

    hud_w = 380
    cam_w = 960
    canvas_h = 720
    canvas_w = hud_w + cam_w

    # Target Box Coordinates in normalized camera space [x1, y1, x2, y2]
    left_box_norm = (0.08, 0.20, 0.46, 0.80)
    right_box_norm = (0.54, 0.20, 0.92, 0.80)

    print("\nVisualizer is running!")
    print("HOW TO OPERATE:")
    print("  1. Place your LEFT hand in the left box, and RIGHT hand in the right box.")
    print("  2. Press [C] to zero/calibrate neutral reference poses for both hands.")
    print("  3. Hold [SPACE] to smoothly drive the arm (or press [T] to toggle continuous mode).")
    print("  4. Press [S] to swap hand roles if desired.")
    print("  5. Drag window borders to resize as desired. Press [Q] to quit.\n")

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("Failed to grab video frame.")
                break

            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Detect on RAW un-flipped frame for true physical handedness
            rgb_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_raw)
            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Mirror frame horizontally for intuitive teleoperation display
            cam_view = cv2.flip(raw_frame, 1)
            cam_view = cv2.resize(cam_view, (cam_w, canvas_h))

            # Match detected hands to physical Left and Right
            left_hand_lms = None
            right_hand_lms = None

            if result.hand_landmarks and result.handedness:
                for idx, handedness in enumerate(result.handedness):
                    detected_label = handedness[0].category_name
                    # In mirrored view, landmark x is flipped
                    lms_mirrored = [(1.0 - lm.x, lm.y, lm.z) for lm in result.hand_landmarks[idx]]
                    if detected_label == "Left" and left_hand_lms is None:
                        left_hand_lms = lms_mirrored
                    elif detected_label == "Right" and right_hand_lms is None:
                        right_hand_lms = lms_mirrored

            left_detected = left_hand_lms is not None
            right_detected = right_hand_lms is not None

            # Determine which hand is Position and which is Tool
            if not mapper.swap_roles:
                pos_lms = left_hand_lms
                tool_lms = right_hand_lms
                pos_hand_label = "Left"
                tool_hand_label = "Right"
            else:
                pos_lms = right_hand_lms
                tool_lms = left_hand_lms
                pos_hand_label = "Right"
                tool_hand_label = "Left"

            # Compute Position Joint Targets
            pos_metrics = None
            if pos_lms is not None:
                _, pos_metrics = mapper.compute_position(pos_lms)

            # Compute Tool Joint Targets
            tool_metrics = None
            if tool_lms is not None:
                is_right = (tool_hand_label == "Right")
                _, tool_metrics = mapper.compute_tool(tool_lms, is_right_hand=is_right)

            # Check if hands are inside their target boxes
            left_in_box = False
            right_in_box = False

            if left_hand_lms is not None:
                lw_x, lw_y = left_hand_lms[0][0], left_hand_lms[0][1]
                left_in_box = (left_box_norm[0] <= lw_x <= left_box_norm[2]) and (left_box_norm[1] <= lw_y <= left_box_norm[3])

            if right_hand_lms is not None:
                rw_x, rw_y = right_hand_lms[0][0], right_hand_lms[0][1]
                right_in_box = (right_box_norm[0] <= rw_x <= right_box_norm[2]) and (right_box_norm[1] <= rw_y <= right_box_norm[3])

            # Draw Target Boxes on Camera View
            left_title = "LEFT: ARM POSITION" if not mapper.swap_roles else "LEFT: WRIST & GRIPPER"
            right_title = "RIGHT: WRIST & GRIPPER" if not mapper.swap_roles else "RIGHT: ARM POSITION"

            draw_target_box(cam_view, left_box_norm, left_title, "Hold Left Hand Here", left_in_box, base_color=(180, 140, 60))
            draw_target_box(cam_view, right_box_norm, right_title, "Hold Right Hand Here", right_in_box, base_color=(60, 160, 180))

            # Draw Hand Skeletons
            if left_hand_lms is not None:
                is_tool = mapper.swap_roles
                grip_val = tool_metrics["gripper_pct"] if is_tool and tool_metrics else 100.0
                draw_hand_skeleton(cam_view, left_hand_lms, "Left", "Tool" if is_tool else "Position",
                                   is_tool_hand=is_tool, pinch_pct=grip_val, bone_color=(255, 180, 0))

            if right_hand_lms is not None:
                is_tool = not mapper.swap_roles
                grip_val = tool_metrics["gripper_pct"] if is_tool and tool_metrics else 100.0
                draw_hand_skeleton(cam_view, right_hand_lms, "Right", "Tool" if is_tool else "Position",
                                   is_tool_hand=is_tool, pinch_pct=grip_val, bone_color=(50, 180, 240))

            # Clutch State: active when engaged and at least one hand is detected
            any_hand_detected = left_detected or right_detected
            active_clutch = (is_clutched or toggle_clutch) and any_hand_detected

            # Smooth Bumpless Dispatch to Robot Arm
            target_joints = mapper.smoothed_joints.copy()

            if active_clutch:
                # Apply slew-rate velocity limiter
                for motor, target in target_joints.items():
                    current = current_robot_cmd[motor]
                    max_step = MAX_GRIPPER_PER_FRAME if motor == "gripper" else MAX_DEG_PER_FRAME
                    step = float(np.clip(target - current, -max_step, max_step))
                    current_robot_cmd[motor] = current + step

                if robot is not None and robot.is_connected:
                    action = {f"{k}.pos": v for k, v in current_robot_cmd.items()}
                    robot.send_action(action)

            # Construct Side-by-Side Canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            hud_panel = canvas[:, :hud_w]
            draw_hud(
                panel=hud_panel,
                joints=current_robot_cmd if active_clutch else target_joints,
                is_clutched=active_clutch,
                left_detected=left_detected,
                right_detected=right_detected,
                left_in_box=left_in_box,
                right_in_box=right_in_box,
                swap_roles=mapper.swap_roles,
                fps=fps,
                robot_connected=(robot is not None and robot.is_connected)
            )
            canvas[:, hud_w:] = cam_view

            cv2.imshow(window_name, canvas)

            # Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 27 = ESC
                print("\nExiting visualizer...")
                break
            elif key == ord(' '):
                is_clutched = True
            elif key in [ord('t'), ord('T')]:
                toggle_clutch = not toggle_clutch
                print(f"Continuous Clutch: {'ENABLED' if toggle_clutch else 'DISABLED'}")
            elif key in [ord('s'), ord('S')]:
                mapper.swap_roles = not mapper.swap_roles
                role_str = "Left=Position, Right=Tool" if not mapper.swap_roles else "Right=Position, Left=Tool"
                print(f"Swapped Hand Roles -> {role_str}")
            elif key in [ord('c'), ord('C')]:
                # Calibrate neutral zeros for whichever hands are currently visible
                calibrated_str = []
                if pos_lms is not None and pos_metrics is not None:
                    wrist_pt = (pos_lms[0][0], pos_lms[0][1])
                    mapper.set_pos_neutral(wrist_pt, pos_metrics["scale"])
                    calibrated_str.append(f"{pos_hand_label} (Position)")

                if tool_lms is not None and tool_metrics is not None:
                    mapper.set_tool_neutral(tool_metrics.get("roll_deg", 0.0) * math.pi / 180.0,
                                            tool_metrics.get("pitch_deg", -70.0))
                    calibrated_str.append(f"{tool_hand_label} (Tool/Gripper)")

                if calibrated_str:
                    print(f"Calibrated neutral references for: {', '.join(calibrated_str)}")
                else:
                    print("Calibration note: No hands detected to calibrate.")
            else:
                is_clutched = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        if robot is not None:
            print("Disconnecting robot arm...")
            robot.disconnect()
        print("Visualizer shutdown cleanly.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 Dual-Hand Real-Time Teleoperation Visualizer")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--swap", action="store_true", help="Swap hand roles (Right=Position, Left=Tool)")
    parser.add_argument("--robot", action="store_true", help="Connect to physical SO-101 follower arm")
    parser.add_argument("--port", type=str, default="/dev/ttyACM1", help="Follower robot serial port (default: /dev/ttyACM1)")
    args = parser.parse_args()

    run_visualizer(camera_id=args.camera, connect_robot=args.robot, robot_port=args.port, default_swap=args.swap)


if __name__ == "__main__":
    main()
