#!/usr/bin/env python3
"""
Phase 1: Real-Time Hand Teleoperation Visualizer for SO-101 Follower Arm
Tracks right-hand gestures via laptop webcam and maps them to SO-101 joint angles.

Key Features & Updates:
- Side-by-side layout: HUD on the LEFT (380px), clean camera feed on the RIGHT (960px)
- Resizable window (cv2.WINDOW_NORMAL) - drag window borders or maximize
- Guided start with a center "Neutral Target Box"
- Bumpless engagement: Software slew-rate limiter prevents arm snapping/jumping
- True physical handedness (detected on raw camera feed before mirroring)
- Push-to-engage clutch (Spacebar) or toggle clutch (T)
- Target hand swap hotkey (H: Right <-> Left)
- Neutral calibration (C)
- Safe live follower arm control (--robot --port /dev/ttyACM1)
"""

import argparse
import json
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

# Maximum safe movement per frame (at ~30 FPS, 1.2 deg/frame = ~36 deg/sec)
MAX_DEG_PER_FRAME = 1.4
MAX_GRIPPER_PER_FRAME = 4.0

DEFAULT_MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_CALIB_PATH = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / "matti_follower_arm.json"


class HandPoseMapper:
    """Extracts intuitive 6-DOF robot joint commands from 21 MediaPipe hand landmarks."""

    def __init__(self, ema_alpha: float = 0.25):
        self.ema_alpha = ema_alpha
        self.calibrated = False

        # Neutral reference pose in normalized camera space [0..1]
        self.neutral_x = 0.5
        self.neutral_y = 0.5
        self.neutral_scale = 0.20
        self.neutral_roll = 0.0

        # Current smoothed joint targets
        self.smoothed_joints: Dict[str, float] = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -30.0,
            "elbow_flex": 60.0,
            "wrist_flex": -30.0,
            "wrist_roll": 0.0,
            "gripper": 100.0,
        }

    def set_neutral(self, wrist_pt: Tuple[float, float], scale: float, roll_rad: float):
        self.neutral_x = wrist_pt[0]
        self.neutral_y = wrist_pt[1]
        self.neutral_scale = max(scale, 0.05)
        self.neutral_roll = roll_rad
        self.calibrated = True

    def compute_joints(self, landmarks_norm: List[Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        landmarks_norm: 21 (x, y, z) coordinates in mirrored display space [0..1].
        """
        pts = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks_norm])

        wrist = pts[0]
        thumb_tip = pts[4]
        index_tip = pts[8]
        middle_mcp = pts[9]
        index_mcp = pts[5]
        pinky_mcp = pts[17]

        # 1. Palm Scale (distance between wrist and middle MCP)
        scale = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))
        if not self.calibrated:
            self.set_neutral((wrist[0], wrist[1]), scale, 0.0)

        # 2. Pinch Distance & Gripper %
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        pinch_ratio = pinch_dist / max(scale, 1e-4)

        # Touching ~ 0.22, fully open ~ 0.60
        min_pinch_ratio = 0.22
        max_pinch_ratio = 0.58
        gripper_pct = np.clip((pinch_ratio - min_pinch_ratio) / (max_pinch_ratio - min_pinch_ratio), 0.0, 1.0) * 100.0

        # 3. Shoulder Pan (Left/Right)
        delta_x = wrist[0] - self.neutral_x
        # Map +/- 0.22 screen fraction to +/- 55 degrees
        target_pan = float(np.clip(delta_x / 0.22 * 55.0, DEFAULT_LIMITS["shoulder_pan"][0], DEFAULT_LIMITS["shoulder_pan"][1]))

        # 4. Shoulder Lift & Elbow Flex (Height & Reach)
        delta_y = wrist[1] - self.neutral_y
        depth_ratio = (scale - self.neutral_scale) / max(self.neutral_scale, 1e-4)

        base_lift = -30.0
        base_elbow = 65.0

        lift_offset = -float(delta_y / 0.22 * 45.0)    # Hand up -> lift increases
        reach_offset = float(depth_ratio * 35.0)       # Hand forward -> elbow extends

        target_lift = float(np.clip(base_lift + lift_offset - reach_offset * 0.25, DEFAULT_LIMITS["shoulder_lift"][0], DEFAULT_LIMITS["shoulder_lift"][1]))
        target_elbow = float(np.clip(base_elbow - lift_offset * 0.25 + reach_offset, DEFAULT_LIMITS["elbow_flex"][0], DEFAULT_LIMITS["elbow_flex"][1]))

        # 5. Wrist Roll
        palm_vec = pinky_mcp[:2] - index_mcp[:2]
        current_roll_rad = math.atan2(palm_vec[1], palm_vec[0])
        roll_diff_rad = current_roll_rad - self.neutral_roll
        roll_diff_rad = (roll_diff_rad + math.pi) % (2 * math.pi) - math.pi
        target_roll = float(np.clip(math.degrees(roll_diff_rad), DEFAULT_LIMITS["wrist_roll"][0], DEFAULT_LIMITS["wrist_roll"][1]))

        # 6. Wrist Flex (Pitch)
        wrist_vec = middle_mcp[:2] - wrist[:2]
        pitch_angle_deg = math.degrees(math.atan2(wrist_vec[1], math.sqrt(wrist_vec[0]**2 + 1e-6)))
        target_flex = float(np.clip(pitch_angle_deg - 70.0, DEFAULT_LIMITS["wrist_flex"][0], DEFAULT_LIMITS["wrist_flex"][1]))

        raw_targets = {
            "shoulder_pan": target_pan,
            "shoulder_lift": target_lift,
            "elbow_flex": target_elbow,
            "wrist_flex": target_flex,
            "wrist_roll": target_roll,
            "gripper": gripper_pct,
        }

        # Apply EMA smoothing
        for k in self.smoothed_joints:
            self.smoothed_joints[k] = self.ema_alpha * raw_targets[k] + (1.0 - self.ema_alpha) * self.smoothed_joints[k]

        metrics = {
            "wrist_x": wrist[0],
            "wrist_y": wrist[1],
            "scale": scale,
            "pinch_ratio": pinch_ratio,
            "gripper_pct": gripper_pct,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "depth_ratio": depth_ratio,
        }

        return self.smoothed_joints.copy(), metrics


def draw_left_hud(panel: np.ndarray, joints: Dict[str, float], is_clutched: bool, 
                  hand_detected: bool, in_neutral_box: bool, fps: float, 
                  robot_connected: bool, target_hand: str = "Right") -> None:
    """Renders the dedicated HUD control panel on the LEFT side of the canvas."""
    h, w, _ = panel.shape
    panel[:] = (20, 20, 24)  # dark background

    # Title Banner
    cv2.putText(panel, "SO-101 HAND TELEOP", (18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"FPS: {fps:4.1f}", (w - 90, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(panel, (18, 44), (w - 18, 44), (70, 70, 75), 1)

    # Status Badges
    y = 70
    if hand_detected:
        cv2.circle(panel, (26, y - 5), 6, (0, 255, 120), -1)
        cv2.putText(panel, f"{target_hand.upper()} HAND: TRACKED", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.circle(panel, (26, y - 5), 6, (0, 140, 255), -1)
        cv2.putText(panel, f"SEARCHING {target_hand.upper()} HAND...", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    y += 28
    if is_clutched:
        cv2.rectangle(panel, (18, y - 16), (w - 18, y + 8), (0, 130, 0), -1)
        cv2.putText(panel, "[ CLUTCH ENGAGED - ACTIVE ]", (35, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(panel, (18, y - 16), (w - 18, y + 8), (40, 40, 90), -1)
        cv2.putText(panel, "[ DISENGAGED - FROZEN ]", (46, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (200, 200, 255), 1, cv2.LINE_AA)

    y += 30
    # Mode badge
    if robot_connected:
        mode_text = "ARM: CONNECTED (/dev/ttyACM1)"
        mode_color = (60, 255, 60)
    else:
        mode_text = "ARM: SIMULATION (SAFE)"
        mode_color = (255, 180, 50)
    cv2.putText(panel, mode_text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, mode_color, 1, cv2.LINE_AA)

    # Operational Step Guide
    y += 22
    cv2.line(panel, (18, y), (w - 18, y), (70, 70, 75), 1)
    y += 20
    cv2.putText(panel, "OPERATION GUIDE:", (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    y += 18
    step1_color = (0, 255, 120) if in_neutral_box else (160, 160, 160)
    cv2.putText(panel, "1. Place hand in center box", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, step1_color, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(panel, "2. Press [C] to zero neutral pose", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    y += 18
    step3_color = (0, 255, 120) if is_clutched else (180, 180, 180)
    cv2.putText(panel, "3. Hold [SPACE] to smoothly drive arm", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, step3_color, 1, cv2.LINE_AA)

    # Joint Gauges
    y += 24
    cv2.line(panel, (18, y), (w - 18, y), (70, 70, 75), 1)
    y += 22

    joint_names = [
        ("shoulder_pan", "Pan (Base)", "deg"),
        ("shoulder_lift", "Lift", "deg"),
        ("elbow_flex", "Elbow", "deg"),
        ("wrist_flex", "Flex (Pitch)", "deg"),
        ("wrist_roll", "Roll", "deg"),
        ("gripper", "Gripper Pinch", "%"),
    ]

    for key, label, unit in joint_names:
        val = joints.get(key, 0.0)
        lim_min, lim_max = DEFAULT_LIMITS[key]
        ratio = np.clip((val - lim_min) / (lim_max - lim_min + 1e-6), 0.0, 1.0)

        # Label & value
        cv2.putText(panel, f"{label}:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        val_str = f"{val:5.1f} {unit}"
        cv2.putText(panel, val_str, (w - 95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Bar background
        bar_x = 18
        bar_y = y + 5
        bar_w = w - 36
        bar_h = 7
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 45), -1)

        # Filled bar
        fill_w = int(bar_w * ratio)
        if key == "gripper":
            bar_color = (int(255 * (1 - ratio)), int(255 * ratio), 40)
        else:
            bar_color = (0, 200, 255)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Center zero line
        if lim_min < 0 < lim_max:
            center_x = int(bar_x + bar_w * (-lim_min / (lim_max - lim_min)))
            cv2.line(panel, (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), (180, 180, 180), 1)

        y += 34

    # Bottom Instructions
    cv2.line(panel, (18, h - 135), (w - 18, h - 135), (70, 70, 75), 1)
    controls = [
        "[SPACE] Hold: Engage Clutch",
        "[T] Toggle Continuous Track",
        f"[H] Swap Hand: {target_hand.upper()}",
        "[C] Calibrate Neutral Zero",
        "[Q] or [ESC] Exit",
    ]
    cy = h - 115
    for c in controls:
        cv2.putText(panel, c, (18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1, cv2.LINE_AA)
        cy += 20


def draw_camera_overlay(cam_view: np.ndarray, landmarks_norm: Optional[List[Tuple[float, float, float]]], 
                        pinch_pct: float, in_neutral_box: bool, hand_label: str = "Right", is_active: bool = True):
    """Draws skeleton, pinch indicator, and the center Neutral Target Box on the camera view."""
    h, w, _ = cam_view.shape

    # Draw Center Neutral Target Box
    box_w, box_h = int(w * 0.40), int(h * 0.50)
    bx1 = (w - box_w) // 2
    by1 = (h - box_h) // 2
    bx2 = bx1 + box_w
    by2 = by1 + box_h

    # Color changes when hand is inside
    if in_neutral_box:
        box_color = (0, 255, 120)
        box_text = "READY: Hand in Neutral Zone"
    else:
        box_color = (80, 140, 200)
        box_text = "Place hand here & press [C]"

    # Draw dashed-like corners
    corner_len = 30
    # Top-left
    cv2.line(cam_view, (bx1, by1), (bx1 + corner_len, by1), box_color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (bx1, by1), (bx1, by1 + corner_len), box_color, 2, cv2.LINE_AA)
    # Top-right
    cv2.line(cam_view, (bx2, by1), (bx2 - corner_len, by1), box_color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (bx2, by1), (bx2, by1 + corner_len), box_color, 2, cv2.LINE_AA)
    # Bottom-left
    cv2.line(cam_view, (bx1, by2), (bx1 + corner_len, by2), box_color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (bx1, by2), (bx1, by2 - corner_len), box_color, 2, cv2.LINE_AA)
    # Bottom-right
    cv2.line(cam_view, (bx2, by2), (bx2 - corner_len, by2), box_color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (bx2, by2), (bx2, by2 - corner_len), box_color, 2, cv2.LINE_AA)

    # Center crosshair
    cx, cy = w // 2, h // 2
    cv2.line(cam_view, (cx - 10, cy), (cx + 10, cy), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(cam_view, (cx, cy - 10), (cx, cy + 10), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(cam_view, box_text, (bx1 + 10, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # Draw skeleton if landmarks present
    if landmarks_norm is not None:
        pts_px = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks_norm]

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        bone_color = (240, 160, 50) if is_active else (80, 80, 80)
        joint_color = (0, 220, 100) if is_active else (110, 110, 110)

        for start, end in connections:
            cv2.line(cam_view, pts_px[start], pts_px[end], bone_color, 2 if is_active else 1, cv2.LINE_AA)

        for i, pt in enumerate(pts_px):
            if is_active and i in [4, 8]:
                cv2.circle(cam_view, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(cam_view, pt, 4 if is_active else 3, joint_color, -1, cv2.LINE_AA)

        wrist_pt = pts_px[0]
        tag = f"[{hand_label.upper()} - ACTIVE]" if is_active else f"[{hand_label.upper()} - IDLE]"
        tag_color = (0, 255, 120) if is_active else (140, 140, 140)
        cv2.putText(cam_view, tag, (wrist_pt[0] - 40, wrist_pt[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_color, 1, cv2.LINE_AA)

        if is_active:
            thumb_pt = pts_px[4]
            index_pt = pts_px[8]
            color_g = int(np.clip(pinch_pct * 2.55, 0, 255))
            color_r = int(np.clip((100.0 - pinch_pct) * 2.55, 0, 255))
            cv2.line(cam_view, thumb_pt, index_pt, (0, color_g, color_r), 3, cv2.LINE_AA)

            mid_x = (thumb_pt[0] + index_pt[0]) // 2
            mid_y = (thumb_pt[1] + index_pt[1]) // 2
            cv2.putText(cam_view, f"Grip: {pinch_pct:3.0f}%", (mid_x + 10, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, color_g, color_r), 1, cv2.LINE_AA)


def run_visualizer(camera_id: int = 0, connect_robot: bool = False, robot_port: str = "/dev/ttyACM1", default_hand: str = "Right"):
    print("=" * 65)
    print("SO-101 REAL-TIME HAND TELEOPERATION VISUALIZER (PHASE 1)")
    print("=" * 65)

    # Robot Arm Initialization & State Tracking
    robot = None
    # current_robot_cmd stores the active commanded joints (for bumpless smoothing)
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

            # Read initial physical positions so we start EXACTLY where the arm is (NO jumping!)
            present_pos = robot.bus.sync_read("Present_Position")
            for k, v in present_pos.items():
                if k in current_robot_cmd:
                    current_robot_cmd[k] = float(v)
            print(f"Arm physical starting pose loaded: {current_robot_cmd}")
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

    print("Initializing MediaPipe HandLandmarker...")
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

    mapper = HandPoseMapper(ema_alpha=0.30)
    # Initialize mapper default pose to current arm pose if connected
    for k, v in current_robot_cmd.items():
        mapper.smoothed_joints[k] = v

    window_name = "SO-101 Hand Teleoperation - Phase 1 Visualizer"
    # Resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1340, 720)

    target_hand = default_hand.capitalize()
    is_clutched = False
    toggle_clutch = False
    last_time = time.perf_counter()
    fps = 30.0

    # Layout dimensions
    hud_w = 380
    cam_w = 960
    canvas_h = 720
    canvas_w = hud_w + cam_w

    print("\nVisualizer is running!")
    print("HOW TO OPERATE:")
    print("  1. Place your right hand in the center target box.")
    print("  2. Press [C] to zero/calibrate your neutral hand position.")
    print("  3. Hold [SPACE] to smoothly drive the arm (or press [T] to toggle continuous mode).")
    print("  4. Press [H] to swap hands (Right <-> Left).")
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

            # Detect on RAW frame for true physical handedness
            rgb_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_raw)
            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Mirror frame horizontally and resize to 960x720
            cam_view = cv2.flip(raw_frame, 1)
            cam_view = cv2.resize(cam_view, (cam_w, canvas_h))

            target_hand_idx = None
            if result.hand_landmarks and result.handedness:
                for idx, handedness in enumerate(result.handedness):
                    detected_label = handedness[0].category_name
                    if detected_label == target_hand:
                        target_hand_idx = idx
                        break

            metrics = None
            hand_detected = target_hand_idx is not None
            active_lm_display = None
            in_neutral_box = False

            # Render non-active hands first
            if result.hand_landmarks and result.handedness:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    if idx != target_hand_idx:
                        label = result.handedness[idx][0].category_name
                        lm_display = [(1.0 - lm.x, lm.y, lm.z) for lm in landmarks]
                        draw_camera_overlay(cam_view, lm_display, 100.0, in_neutral_box=False, hand_label=label, is_active=False)

            # Process active target hand
            if hand_detected:
                landmarks = result.hand_landmarks[target_hand_idx]
                active_lm_display = [(1.0 - lm.x, lm.y, lm.z) for lm in landmarks]
                target_joints, metrics = mapper.compute_joints(active_lm_display)

                # Check if wrist is in center target box (X: 0.30 - 0.70, Y: 0.25 - 0.75)
                wrist_x, wrist_y = metrics["wrist_x"], metrics["wrist_y"]
                in_neutral_box = (0.30 <= wrist_x <= 0.70) and (0.25 <= wrist_y <= 0.75)

                draw_camera_overlay(cam_view, active_lm_display, metrics["gripper_pct"], 
                                    in_neutral_box=in_neutral_box, hand_label=target_hand, is_active=True)
            else:
                target_joints = mapper.smoothed_joints.copy()
                draw_camera_overlay(cam_view, None, 100.0, in_neutral_box=False, hand_label=target_hand, is_active=False)

            # Clutch logic
            active_clutch = (is_clutched or toggle_clutch) and hand_detected

            # Smooth Bumpless Dispatch to Robot
            if active_clutch:
                # Slew-rate limiter: smooth transition, NO sudden jumping
                for motor, target in target_joints.items():
                    current = current_robot_cmd[motor]
                    max_step = MAX_GRIPPER_PER_FRAME if motor == "gripper" else MAX_DEG_PER_FRAME
                    step = float(np.clip(target - current, -max_step, max_step))
                    current_robot_cmd[motor] = current + step

                if robot is not None and robot.is_connected:
                    action = {f"{k}.pos": v for k, v in current_robot_cmd.items()}
                    robot.send_action(action)
            else:
                # When clutch is released, target joints stay synchronized with current pose
                pass

            # Construct Side-by-Side Canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            hud_panel = canvas[:, :hud_w]
            draw_left_hud(
                panel=hud_panel,
                joints=current_robot_cmd if active_clutch else target_joints,
                is_clutched=active_clutch,
                hand_detected=hand_detected,
                in_neutral_box=in_neutral_box,
                fps=fps,
                robot_connected=(robot is not None and robot.is_connected),
                target_hand=target_hand
            )
            canvas[:, hud_w:] = cam_view

            cv2.imshow(window_name, canvas)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 27 = ESC
                print("\nExiting visualizer...")
                break
            elif key == ord(' '):
                is_clutched = True
            elif key in [ord('t'), ord('T')]:
                toggle_clutch = not toggle_clutch
                print(f"Continuous Clutch: {'ENABLED' if toggle_clutch else 'DISABLED'}")
            elif key in [ord('h'), ord('H')]:
                target_hand = "Left" if target_hand == "Right" else "Right"
                mapper.calibrated = False
                print(f"Swapped target hand to: {target_hand}")
            elif key in [ord('c'), ord('C')]:
                if hand_detected and active_lm_display is not None and metrics is not None:
                    wrist_lm = active_lm_display[0]
                    mapper.set_neutral((wrist_lm[0], wrist_lm[1]), metrics["scale"], 0.0)
                    print(f"Calibrated neutral center to (X={wrist_lm[0]:.2f}, Y={wrist_lm[1]:.2f})")
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
    parser = argparse.ArgumentParser(description="SO-101 Hand Teleoperation Visualizer (Phase 1)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--hand", type=str, default="Right", choices=["Right", "Left"], help="Target hand (default: Right)")
    parser.add_argument("--robot", action="store_true", help="Connect to live SO-101 follower arm")
    parser.add_argument("--port", type=str, default="/dev/ttyACM1", help="Follower robot serial port (default: /dev/ttyACM1)")
    args = parser.parse_args()

    run_visualizer(camera_id=args.camera, connect_robot=args.robot, robot_port=args.port, default_hand=args.hand)


if __name__ == "__main__":
    main()
