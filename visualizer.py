#!/usr/bin/env python3
"""
SO-101 Dual-Hand Real-Time Teleoperation Visualizer
Controls the 6-DOF SO-101 follower arm using TWO hands via laptop webcam:

  LEFT HAND → Arm Position (3 axes):
    • Up/Down (hand Y in box)    → arm HEIGHT (coordinated shoulder_lift + elbow_flex)
    • Left/Right (hand X in box) → shoulder_pan (base rotation)
    • Forward/Back (hand depth)  → arm REACH (coordinated shoulder_lift + elbow_flex)

  RIGHT HAND → Tool / Gripper (3 axes):
    • Pinch thumb + index → gripper open/close
    • Rotate wrist        → gripper rotation (wrist_roll)
    • Tilt wrist up/down  → gripper pitch (wrist_flex)

Control Philosophy:
  Height and reach both move shoulder_lift and elbow_flex together in coordinated
  pairs recorded during calibration. This makes the arm feel like it's moving
  in pure up/down and forward/backward directions, rather than individual joints.
  The on-screen target box maps directly to the arm's workspace.
"""

import argparse
import json
import math
import queue
import select
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Default broad hardware limits from STS3215 specifications
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
DEFAULT_LIMITS_FILE = Path(__file__).parent / "arm_limits.json"


# ==============================================================================
# Voice Command Listener (Offline Keyword Spotting for "On" and "Off")
# ==============================================================================

class VoiceCommandListener:
    """
    100% offline, low-latency keyword recognizer using Vosk and sounddevice.
    Listens for 'on' (activates arm clutch) and 'off' (deactivates arm clutch).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_active: bool = False
        self.last_command: Optional[str] = None
        self.last_command_time: float = 0.0
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._audio_queue: queue.Queue = queue.Queue()
        self.available: bool = False

        try:
            import sounddevice as sd
            import vosk
            vosk.SetLogLevel(-1)  # Suppress internal Vosk logging
            self.model = vosk.Model(lang="en-us")
            self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate, '["on", "off", "[unk]"]')
            self.available = True
        except Exception as e:
            print(f"Notice: Voice control disabled ({e}).")
            self.available = False

    def start(self):
        if not self.available:
            return
        import sounddevice as sd

        self._running = True

        def audio_callback(indata, frames, time_info, status):
            if self._running:
                self._audio_queue.put(bytes(indata))

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=audio_callback
            )
            self._stream.start()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            print("✓ Voice Command Control: ACTIVE (Say 'On' to engage arm, 'Off' to disengage)")
        except Exception as e:
            print(f"Warning: Could not open microphone stream for voice control ({e})")
            self.available = False

    def _worker(self):
        while self._running:
            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.rec.AcceptWaveform(data):
                res = json.loads(self.rec.Result())
                text = res.get("text", "").strip().lower()
                words = text.split()
                if "on" in words:
                    self.is_active = True
                    self.last_command = "ON"
                    self.last_command_time = time.time()
                    print("\n[VOICE COMMAND] -> 'ON' detected: ARM ACTIVATED")
                elif "off" in words:
                    self.is_active = False
                    self.last_command = "OFF"
                    self.last_command_time = time.time()
                    print("\n[VOICE COMMAND] -> 'OFF' detected: ARM DEACTIVATED")

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None


# ==============================================================================
# Workspace Calibration & Limits Storage
# ==============================================================================

def load_arm_limits(file_path: Path) -> Optional[Dict]:
    """Loads workspace limits from JSON file if available."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to load limits file {file_path} ({e})")
        return None


def save_arm_limits(limits: Dict, file_path: Path) -> None:
    """Saves workspace limits to JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(limits, f, indent=2)
        print(f"✓ Saved verified workspace limits to: {file_path}")
    except Exception as e:
        print(f"Error saving limits file: {e}")


def wait_for_enter_with_live_readout(robot, step_title: str) -> Dict[str, float]:
    """
    Displays step instructions clearly, then shows a live motor angle readout on a
    SINGLE fixed line below using ANSI escape codes so instructions never scroll away.
    """
    # Print instructions with clear visual separation
    print("\n" + "=" * 60)
    print(step_title)
    print("-" * 60)
    print("  >>> Move the arm by hand, then press [ENTER] to lock <<<")
    print()  # blank line that the readout will overwrite

    last_print = 0.0
    while True:
        # Check non-blocking for ENTER key press
        if select.select([sys.stdin], [], [], 0.04)[0]:
            sys.stdin.readline()
            break

        now = time.time()
        if now - last_print > 0.15:  # slower update = less flicker, still responsive
            last_print = now
            try:
                pos = robot.bus.sync_read("Present_Position")
                pan = float(pos.get("shoulder_pan", 0.0))
                lift = float(pos.get("shoulder_lift", 0.0))
                elbow = float(pos.get("elbow_flex", 0.0))
                # Move cursor up 1 line, clear it, print readout, stay on that line
                sys.stdout.write(f"\033[1A\033[2K  Pan:{pan:6.1f}  Lift:{lift:6.1f}  Elbow:{elbow:6.1f}\n")
                sys.stdout.flush()
            except Exception:
                pass

    # Read final position once locked
    pos = robot.bus.sync_read("Present_Position")
    # Clear the readout line and print the locked result
    sys.stdout.write("\033[1A\033[2K")
    print(f"  ✓ Locked: Lift={float(pos['shoulder_lift']):.1f}°  Pan={float(pos['shoulder_pan']):.1f}°  Elbow={float(pos['elbow_flex']):.1f}°")
    return {k: float(v) for k, v in pos.items()}


def calibrate_arm_limits(robot_port: str = "/dev/ttyACM1", save_path: Path = DEFAULT_LIMITS_FILE) -> Optional[Dict]:
    """
    Interactive guided calibration: disables motor torque, provides live dynamic angle
    readouts, and walks the user through 4 quick steps to teach safe desk boundaries.
    """
    print("\n" + "=" * 65)
    print("SO-101 INTERACTIVE WORKSPACE LIMITS CALIBRATION")
    print("=" * 65)
    print(f"Connecting to SO-101 Follower arm on {robot_port}...")

    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
        config = SOFollowerRobotConfig(port=robot_port, id="matti_follower_arm", use_degrees=True)
        robot = SOFollower(config)
        robot.connect()
        robot.bus.disable_torque()
        print("✓ Connected successfully!")
        print("✓ Motor torque is now DISABLED. You can move the arm freely by hand.\n")
    except Exception as e:
        print(f"Error connecting to robot arm: {e}")
        return None

    try:
        # Step 1: Lowest position touching the desk
        step1 = ("STEP 1: Move the arm so the gripper is at your DESK SURFACE / TABLE\n"
                 "        (This will be your lowest pick position).")
        pos_low = wait_for_enter_with_live_readout(robot, step1)
        raw_down = float(pos_low["shoulder_lift"])

        # Step 2: Highest safe reach in the air
        step2 = ("STEP 2: Move the arm to its HIGHEST safe reach in the air\n"
                 "        (Ensure clearance from shelves, monitors, and cables).")
        pos_high = wait_for_enter_with_live_readout(robot, step2)
        raw_up = float(pos_high["shoulder_lift"])

        # Step 3: Preferred Neutral resting pose
        step3 = ("STEP 3: Move the arm to your preferred NEUTRAL working pose\n"
                 "        (A comfortable mid-height, mid-reach working position).")
        pos_mid = wait_for_enter_with_live_readout(robot, step3)
        mid_lift = float(pos_mid["shoulder_lift"])
        mid_elbow = float(pos_mid["elbow_flex"])
        mid_wrist = float(pos_mid["wrist_flex"])

        # Step 4: Most forward extended reach
        step4 = ("STEP 4: Extend the arm FULLY FORWARD (maximum reach)\n"
                 "        (Straighten elbow, reach as far forward as safe).")
        pos_fwd = wait_for_enter_with_live_readout(robot, step4)
        fwd_lift = float(pos_fwd["shoulder_lift"])
        fwd_elbow = float(pos_fwd["elbow_flex"])

        # Step 5: Most retracted close to base
        step5 = ("STEP 5: Retract the arm CLOSE TO BASE (minimum reach)\n"
                 "        (Bend elbow, pull arm close to the base/body).")
        pos_back = wait_for_enter_with_live_readout(robot, step5)
        back_lift = float(pos_back["shoulder_lift"])
        back_elbow = float(pos_back["elbow_flex"])

        # Step 6: Base pan boundaries
        step6a = "STEP 6a: Rotate the arm base by hand to your LEFT workspace boundary."
        pos_left = wait_for_enter_with_live_readout(robot, step6a)
        pan_left = float(pos_left["shoulder_pan"])

        step6b = "STEP 6b: Rotate the arm base by hand to your RIGHT workspace boundary."
        pos_right = wait_for_enter_with_live_readout(robot, step6b)
        pan_right = float(pos_right["shoulder_pan"])

        # Step 7: Gripper closed & open
        step7a = "STEP 7a: Squeeze the gripper fully CLOSED by hand."
        pos_closed = wait_for_enter_with_live_readout(robot, step7a)
        grip_closed = float(pos_closed["gripper"])

        step7b = "STEP 7b: Open the gripper fully by hand."
        pos_open = wait_for_enter_with_live_readout(robot, step7b)
        grip_open = float(pos_open["gripper"])

        # Sanity Checks & Auto-Corrections
        # On SO-101, smaller/negative angle is UP into the air, larger/positive angle is DOWN toward desk
        up_lift = min(raw_up, raw_down)
        down_lift = max(raw_up, raw_down)

        # Ensure mid_lift is between up and down (in case user let arm collapse under gravity)
        if not (up_lift <= mid_lift <= down_lift):
            print("  * Note: Recorded neutral pose was outside [up, down] bounds. Auto-centering neutral.")
            mid_lift = (up_lift + down_lift) / 2.0

        # Ensure pan range is healthy and not single-sided
        min_pan = min(pan_left, pan_right)
        max_pan = max(pan_left, pan_right)
        if abs(max_pan - min_pan) < 15.0 or (min_pan >= 0 and max_pan >= 0) or (min_pan <= 0 and max_pan <= 0):
            print("  * Note: Pan range was too narrow or one-sided. Auto-setting symmetric safe bounds.")
            bound = max(abs(min_pan), abs(max_pan), 50.0)
            min_pan = -bound
            max_pan = bound

        min_grip = min(grip_closed, grip_open)
        max_grip = max(grip_closed, grip_open)

        limits = {
            "shoulder_lift": {
                "up": round(up_lift, 1),
                "neutral": round(mid_lift, 1),
                "down": round(down_lift, 1),
            },
            "shoulder_pan": {
                "left": round(min_pan, 1),
                "right": round(max_pan, 1),
            },
            "elbow_flex": {
                "neutral": round(mid_elbow, 1),
                "extended": round(fwd_elbow, 1),
                "retracted": round(back_elbow, 1),
            },
            "reach_poses": {
                "forward": {"lift": round(fwd_lift, 1), "elbow": round(fwd_elbow, 1)},
                "backward": {"lift": round(back_lift, 1), "elbow": round(back_elbow, 1)},
                "high": {"lift": round(up_lift, 1), "elbow": round(float(pos_high['elbow_flex']), 1)},
                "low": {"lift": round(down_lift, 1), "elbow": round(float(pos_low['elbow_flex']), 1)},
            },
            "wrist_flex": {
                "neutral": round(mid_wrist, 1),
            },
            "gripper": {
                "closed": round(min_grip, 1),
                "open": round(max_grip, 1),
            },
            "calibrated": True,
            "description": "Custom verified physical workspace limits with reach poses",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        print("\n" + "=" * 65)
        print("CALIBRATION SUMMARY:")
        print(f"  • Elevation (Lift) : [UP: {up_lift:.1f}°] -> [NEUTRAL: {mid_lift:.1f}°] -> [DOWN: {down_lift:.1f}°]")
        print(f"  • Reach (Elbow)    : [RETRACTED: {back_elbow:.1f}°] -> [NEUTRAL: {mid_elbow:.1f}°] -> [EXTENDED: {fwd_elbow:.1f}°]")
        print(f"  • Forward Pose     : Lift={fwd_lift:.1f}°, Elbow={fwd_elbow:.1f}°")
        print(f"  • Backward Pose    : Lift={back_lift:.1f}°, Elbow={back_elbow:.1f}°")
        print(f"  • Base Pan (Yaw)   : [{min_pan:.1f}° (Left) .. {max_pan:.1f}° (Right)]")
        print(f"  • Gripper Range    : [{min_grip:.1f}% (Closed) .. {max_grip:.1f}% (Open)]")
        print("=" * 65)

        save_arm_limits(limits, save_path)

        ans = input("\nWould you like to run a gentle TEST SWEEP now to verify all motions? [Y/n]: ").strip().lower()
        if ans in ["", "y", "yes"]:
            robot.bus.enable_torque()
            run_test_sweep(robot, limits)

        return limits

    finally:
        robot.disconnect()
        print("Robot disconnected. Calibration complete!\n")


def move_arm_smoothly(robot, target_pose: Dict[str, float], current_pose: Dict[str, float],
                      max_deg_step: float = 0.8, fps: int = 50, duration: float = 3.0) -> Dict[str, float]:
    """Smoothly glides the physical arm from current_pose to target_pose using slew-rate control."""
    steps = int(duration * fps)
    for _ in range(steps):
        all_reached = True
        for k in target_pose:
            target = target_pose[k]
            current = current_pose.get(k, target)
            diff = target - current
            if abs(diff) > 0.4:
                all_reached = False
            max_step = 2.5 if k == "gripper" else max_deg_step
            step = float(np.clip(diff, -max_step, max_step))
            current_pose[k] = current + step

        action = {f"{k}.pos": v for k, v in current_pose.items()}
        robot.send_action(action)
        time.sleep(1.0 / fps)
        if all_reached:
            break
    return current_pose


def run_test_sweep(robot, limits: Dict) -> None:
    """Executes a slow, gentle test sweep through verified workspace limits."""
    print("\n" + "=" * 60)
    print("STARTING SAFE WORKSPACE TEST SWEEP")
    print("=" * 60)
    print("The arm will gently cycle through its calibrated limits.")
    print("Press Ctrl+C to abort at any time.\n")

    present_pos = robot.bus.sync_read("Present_Position")
    current_pose = {k: float(v) for k, v in present_pos.items()}

    neutral_lift = limits.get("shoulder_lift", {}).get("neutral", 55.0)
    up_lift = limits.get("shoulder_lift", {}).get("up", 15.0)
    down_lift = limits.get("shoulder_lift", {}).get("down", 95.0)
    left_pan = limits.get("shoulder_pan", {}).get("left", -50.0)
    right_pan = limits.get("shoulder_pan", {}).get("right", 50.0)
    mid_elbow = limits.get("elbow_flex", {}).get("neutral", 65.0)
    mid_wrist = limits.get("wrist_flex", {}).get("neutral", -30.0)

    neutral_pose = {
        "shoulder_pan": 0.0,
        "shoulder_lift": neutral_lift,
        "elbow_flex": mid_elbow,
        "wrist_flex": mid_wrist,
        "wrist_roll": 0.0,
        "gripper": 100.0,
    }

    try:
        print(" -> [Step 1/5] Moving to Neutral Resting Pose...")
        current_pose = move_arm_smoothly(robot, neutral_pose, current_pose, duration=2.5)
        time.sleep(0.4)

        print(f" -> [Step 2/5] Testing Elevation HIGH Reach ({up_lift:.1f}°)...")
        high_pose = neutral_pose.copy()
        high_pose["shoulder_lift"] = up_lift
        current_pose = move_arm_smoothly(robot, high_pose, current_pose, duration=2.5)
        time.sleep(0.4)

        print(f" -> [Step 3/5] Testing Elevation LOW Table Reach ({down_lift:.1f}°)...")
        low_pose = neutral_pose.copy()
        low_pose["shoulder_lift"] = down_lift
        current_pose = move_arm_smoothly(robot, low_pose, current_pose, duration=3.0)
        time.sleep(0.4)

        current_pose = move_arm_smoothly(robot, neutral_pose, current_pose, duration=2.5)
        time.sleep(0.3)

        print(f" -> [Step 4/5] Testing Base Pan: Left ({left_pan:.1f}°) and Right ({right_pan:.1f}°)...")
        pan_l_pose = neutral_pose.copy()
        pan_l_pose["shoulder_pan"] = left_pan
        current_pose = move_arm_smoothly(robot, pan_l_pose, current_pose, duration=2.0)
        time.sleep(0.3)

        pan_r_pose = neutral_pose.copy()
        pan_r_pose["shoulder_pan"] = right_pan
        current_pose = move_arm_smoothly(robot, pan_r_pose, current_pose, duration=2.5)
        time.sleep(0.3)

        current_pose = move_arm_smoothly(robot, neutral_pose, current_pose, duration=2.0)
        time.sleep(0.3)

        print(" -> [Step 5/5] Testing Gripper Cycle (Close -> Open)...")
        grip_close_pose = neutral_pose.copy()
        grip_close_pose["gripper"] = limits.get("gripper", {}).get("closed", 5.0)
        current_pose = move_arm_smoothly(robot, grip_close_pose, current_pose, duration=1.5)
        time.sleep(0.4)

        grip_open_pose = neutral_pose.copy()
        grip_open_pose["gripper"] = limits.get("gripper", {}).get("open", 100.0)
        current_pose = move_arm_smoothly(robot, grip_open_pose, current_pose, duration=1.5)
        time.sleep(0.3)

        print("\n✓ TEST SWEEP COMPLETED SUCCESSFULLY! All motions verified safe.\n")

    except KeyboardInterrupt:
        print("\nTest sweep aborted by user. Holding current position.")


# ==============================================================================
# Dual-Hand Pose Mapper
# ==============================================================================

class DualHandPoseMapper:
    """
    Decoupled Dual-Hand 6-DOF Mapper.

    Position Hand (default: Left) — Absolute Box-Position Mapping:
      - Wrist Y in target box → arm HEIGHT (up/down)
        Uses calibrated paired (shoulder_lift, elbow_flex) poses at high and low
        positions so that height changes feel like pure vertical movement.
      - Wrist X in target box → shoulder_pan (left/right base rotation)
      - Palm scale (hand closer/farther from camera) → arm REACH (forward/backward)
        Uses calibrated paired (shoulder_lift, elbow_flex) poses at forward and
        retracted positions so that reach changes feel like pure depth movement.

    Tool Hand (default: Right):
      - Palm rotation angle → wrist_roll (gripper rotation)
      - Thumb-index pinch distance → gripper (open/close)
      - Wrist-to-MCP pitch → wrist_flex (gripper tilt)
    """

    def __init__(self, ema_alpha: float = 0.25, invert_lift: bool = False,
                 limits: Optional[Dict] = None,
                 pos_box: Tuple[float, float, float, float] = (0.08, 0.18, 0.46, 0.82)):
        self.ema_alpha = ema_alpha
        self.pos_box = pos_box  # (x1, y1, x2, y2) in normalized camera coords

        # Tool hand reference (relative offsets for rotation/pitch)
        self.tool_neutral_roll = 0.0
        self.tool_neutral_pitch = -70.0
        self.tool_calibrated = False

        self.swap_roles = False
        self.invert_lift = invert_lift

        # === Height axis: calibrated paired (lift, elbow) poses ===
        # When the user recorded "arm at highest" and "arm at lowest",
        # both shoulder_lift AND elbow_flex were recorded. We interpolate
        # both together so that moving your hand up/down produces coordinated
        # joint movement that feels like pure vertical motion.
        reach_poses = limits.get("reach_poses", {}) if limits else {}

        high_pose = reach_poses.get("high", {})
        low_pose = reach_poses.get("low", {})
        fwd_pose = reach_poses.get("forward", {})
        back_pose = reach_poses.get("backward", {})

        # Height axis endpoints (paired lift + elbow)
        if limits and "shoulder_lift" in limits:
            self.high_lift = high_pose.get("lift", limits["shoulder_lift"].get("up", 15.0))
            self.high_elbow = high_pose.get("elbow", limits.get("elbow_flex", {}).get("neutral", 65.0))
            self.low_lift = low_pose.get("lift", limits["shoulder_lift"].get("down", 95.0))
            self.low_elbow = low_pose.get("elbow", limits.get("elbow_flex", {}).get("neutral", 65.0))
            self.mid_lift = limits["shoulder_lift"].get("neutral", 55.0)
        else:
            self.high_lift = 15.0
            self.high_elbow = 65.0
            self.low_lift = 95.0
            self.low_elbow = 65.0
            self.mid_lift = 55.0

        # Reach axis endpoints (paired lift + elbow)
        self.fwd_lift = fwd_pose.get("lift", self.mid_lift)
        self.fwd_elbow = fwd_pose.get("elbow", limits.get("elbow_flex", {}).get("extended", 30.0) if limits else 30.0)
        self.back_lift = back_pose.get("lift", self.mid_lift)
        self.back_elbow = back_pose.get("elbow", limits.get("elbow_flex", {}).get("retracted", 85.0) if limits else 85.0)

        # Pan range
        self.pan_left_deg = limits.get("shoulder_pan", {}).get("left", -60.0) if limits else -60.0
        self.pan_right_deg = limits.get("shoulder_pan", {}).get("right", 60.0) if limits else 60.0

        # Neutral elbow for initial pose
        self.base_elbow = limits.get("elbow_flex", {}).get("neutral", 65.0) if limits else 65.0

        # Reference palm scale (set on first detection for depth/reach mapping)
        self.pos_neutral_scale = 0.0
        self.scale_calibrated = False

        # Active smoothed joint targets
        self.smoothed_joints: Dict[str, float] = {
            "shoulder_pan": 0.0,
            "shoulder_lift": self.mid_lift,
            "elbow_flex": self.base_elbow,
            "wrist_flex": -30.0,
            "wrist_roll": 0.0,
            "gripper": 100.0,
        }

    def set_tool_neutral(self, roll_rad: float, pitch_deg: float):
        self.tool_neutral_roll = roll_rad
        self.tool_neutral_pitch = pitch_deg
        self.tool_calibrated = True

    def compute_position(self, landmarks_norm: List[Tuple[float, float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Maps the position hand to arm joints using two independent axes:

        HEIGHT axis (hand Y in box):
          Top of box → (high_lift, high_elbow)   = arm at highest
          Bottom     → (low_lift, low_elbow)     = arm at desk level
          Both joints interpolate together for natural vertical movement.

        REACH axis (palm scale / hand depth):
          Hand forward (larger) → (fwd_lift, fwd_elbow) = arm extended
          Hand back (smaller)   → (back_lift, back_elbow) = arm retracted
          Both joints interpolate together for natural depth movement.

        The final joint angles are a blend:
          shoulder_lift = height_component + reach_offset
          elbow_flex    = height_component + reach_offset
        """
        pts = np.array([[lm[0], lm[1], lm[2]] for lm in landmarks_norm])
        wrist = pts[0]
        middle_mcp = pts[9]

        scale = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))

        # Capture initial palm scale for depth/reach mapping
        if not self.scale_calibrated and scale > 0.02:
            self.pos_neutral_scale = scale
            self.scale_calibrated = True

        bx1, by1, bx2, by2 = self.pos_box

        # === HEIGHT AXIS: hand Y position in box ===
        t_h = float(np.clip((wrist[1] - by1) / (by2 - by1), 0.0, 1.0))
        if self.invert_lift:
            t_h = 1.0 - t_h
        # t_h: 0.0 = top of box (HIGH), 1.0 = bottom of box (LOW)

        # Interpolate paired joints for height
        height_lift = self.high_lift + t_h * (self.low_lift - self.high_lift)
        height_elbow = self.high_elbow + t_h * (self.low_elbow - self.high_elbow)

        # === REACH AXIS: palm scale (forward/backward) ===
        if self.scale_calibrated and self.pos_neutral_scale > 0.02:
            depth_ratio = (scale - self.pos_neutral_scale) / max(self.pos_neutral_scale, 0.01)
            t_r = float(np.clip(0.5 + depth_ratio / 0.5, 0.0, 1.0))
            # t_r: 0.0 = hand far from camera (RETRACTED), 1.0 = hand close to camera (EXTENDED)
        else:
            t_r = 0.5  # neutral reach

        # Compute reach offset relative to the neutral pose
        # At t_r=0.5 (neutral), offset is zero
        # At t_r=1.0 (forward), offset pushes toward forward pose
        # At t_r=0.0 (backward), offset pushes toward backward pose
        mid_reach_lift = (self.fwd_lift + self.back_lift) / 2.0
        mid_reach_elbow = (self.fwd_elbow + self.back_elbow) / 2.0

        if t_r >= 0.5:
            reach_t = (t_r - 0.5) * 2.0  # 0 to 1
            reach_lift_offset = reach_t * (self.fwd_lift - mid_reach_lift)
            reach_elbow_offset = reach_t * (self.fwd_elbow - mid_reach_elbow)
        else:
            reach_t = (0.5 - t_r) * 2.0  # 0 to 1
            reach_lift_offset = reach_t * (self.back_lift - mid_reach_lift)
            reach_elbow_offset = reach_t * (self.back_elbow - mid_reach_elbow)

        # Combine: height baseline + reach offset
        target_lift = height_lift + reach_lift_offset
        target_elbow = height_elbow + reach_elbow_offset

        target_lift = float(np.clip(target_lift, DEFAULT_LIMITS["shoulder_lift"][0], DEFAULT_LIMITS["shoulder_lift"][1]))
        target_elbow = float(np.clip(target_elbow, DEFAULT_LIMITS["elbow_flex"][0], DEFAULT_LIMITS["elbow_flex"][1]))

        # === PAN AXIS: hand X position in box ===
        t_x = float(np.clip((wrist[0] - bx1) / (bx2 - bx1), 0.0, 1.0))
        target_pan = self.pan_left_deg + t_x * (self.pan_right_deg - self.pan_left_deg)
        target_pan = float(np.clip(target_pan, DEFAULT_LIMITS["shoulder_pan"][0], DEFAULT_LIMITS["shoulder_pan"][1]))

        pos_targets = {
            "shoulder_pan": target_pan,
            "shoulder_lift": target_lift,
            "elbow_flex": target_elbow,
        }

        # Apply EMA smoothing
        for k in pos_targets:
            self.smoothed_joints[k] = self.ema_alpha * pos_targets[k] + (1.0 - self.ema_alpha) * self.smoothed_joints[k]

        # Compute normalized positions for HUD display
        norm_y = (t_h - 0.5) * 2.0  # -1.0 (top/UP) to +1.0 (bottom/DOWN)
        if self.invert_lift:
            norm_y = -norm_y

        metrics = {
            "wrist_x": wrist[0],
            "wrist_y": wrist[1],
            "scale": scale,
            "delta_x": wrist[0] - (bx1 + bx2) / 2.0,
            "delta_y": wrist[1] - (by1 + by2) / 2.0,
            "norm_x": (t_x - 0.5) * 2.0,
            "norm_y": norm_y,
            "depth_ratio": (scale - self.pos_neutral_scale) / max(self.pos_neutral_scale, 0.01) if self.scale_calibrated else 0.0,
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


# ==============================================================================
# UI Rendering
# ==============================================================================

def draw_hud(panel: np.ndarray, joints: Dict[str, float], is_clutched: bool,
             left_detected: bool, right_detected: bool,
             left_in_box: bool, right_in_box: bool,
             swap_roles: bool, invert_lift: bool, fps: float,
             robot_connected: bool, is_limits_calibrated: bool = False,
             voice_active: bool = False, voice_available: bool = False) -> None:
    """Renders the dark control dashboard on the LEFT side of the window (380px)."""
    h, w, _ = panel.shape
    panel[:] = (20, 20, 24)

    # Title & FPS
    cv2.putText(panel, "SO-101 DUAL-HAND TELEOP", (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"FPS: {fps:4.1f}", (w - 88, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(panel, (18, 42), (w - 18, 42), (70, 70, 75), 1)

    # Hand Roles & Tracking Status
    y = 64
    left_role_name = "POSITION (3D)" if not swap_roles else "WRIST & GRIP"
    right_role_name = "WRIST & GRIP" if not swap_roles else "POSITION (3D)"

    # Left Hand Badge
    l_color = (0, 255, 120) if left_detected else (0, 140, 255)
    l_status = "TRACKED" if left_detected else "SEARCHING..."
    cv2.circle(panel, (26, y - 5), 5, l_color, -1)
    cv2.putText(panel, f"LEFT: {left_role_name}", (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, l_status, (w - 105, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, l_color, 1, cv2.LINE_AA)

    y += 22
    # Right Hand Badge
    r_color = (0, 255, 120) if right_detected else (0, 140, 255)
    r_status = "TRACKED" if right_detected else "SEARCHING..."
    cv2.circle(panel, (26, y - 5), 5, r_color, -1)
    cv2.putText(panel, f"RIGHT: {right_role_name}", (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, r_status, (w - 105, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, r_color, 1, cv2.LINE_AA)

    # Clutch Status Banner
    y += 26
    if is_clutched:
        cv2.rectangle(panel, (18, y - 16), (w - 18, y + 8), (0, 130, 0), -1)
        clutch_text = "[ CLUTCH ENGAGED - ACTIVE ]"
        cv2.putText(panel, clutch_text, (35, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(panel, (18, y - 16), (w - 18, y + 8), (45, 45, 80), -1)
        cv2.putText(panel, "[ DISENGAGED - FROZEN ]", (45, y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (200, 200, 255), 1, cv2.LINE_AA)

    # Voice Command Indicator
    y += 26
    if voice_available:
        if voice_active:
            cv2.rectangle(panel, (18, y - 14), (w - 18, y + 6), (0, 90, 130), -1)
            cv2.putText(panel, "VOICE: 'ON' DETECTED (ACTIVE)", (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(panel, (18, y - 14), (w - 18, y + 6), (30, 30, 40), -1)
            cv2.putText(panel, "VOICE: READY (Say 'On' / 'Off')", (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "VOICE: UNAVAILABLE", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 130), 1, cv2.LINE_AA)

    y += 24
    # Hardware Status & Limits indicator
    if robot_connected:
        cv2.putText(panel, "ARM: CONNECTED (/dev/ttyACM1)", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 255, 60), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "ARM: SIMULATION (SAFE)", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 50), 1, cv2.LINE_AA)

    limits_str = "LIMITS: VERIFIED" if is_limits_calibrated else "LIMITS: DEFAULT"
    limits_col = (0, 255, 120) if is_limits_calibrated else (160, 160, 160)
    cv2.putText(panel, limits_str, (w - 135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, limits_col, 1, cv2.LINE_AA)

    # Operation Guide
    y += 18
    cv2.line(panel, (18, y), (w - 18, y), (70, 70, 75), 1)
    y += 16
    cv2.putText(panel, "OPERATION GUIDE:", (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    y += 16
    step1_ready = left_in_box and right_in_box
    step1_color = (0, 255, 120) if step1_ready else (160, 160, 160)
    cv2.putText(panel, "1. Position hands in Left & Right boxes", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, step1_color, 1, cv2.LINE_AA)
    y += 15
    step2_color = (0, 255, 120) if is_clutched else (180, 180, 180)
    cv2.putText(panel, "2. Say 'On' or hold [SPACE] to drive arm", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, step2_color, 1, cv2.LINE_AA)
    y += 15
    cv2.putText(panel, "3. [C] re-zero tool wrist orientation", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1, cv2.LINE_AA)

    # Joint Gauges - Grouped by Role
    y += 20
    cv2.line(panel, (18, y), (w - 18, y), (70, 70, 75), 1)

    # Section 1: Position Hand
    y += 18
    pos_header = f"--- {'LEFT' if not swap_roles else 'RIGHT'} HAND: ARM POSITION ---"
    cv2.putText(panel, pos_header, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.39, (255, 200, 80), 1, cv2.LINE_AA)
    y += 8

    pos_joints = [
        ("shoulder_pan", "Pan (Base Yaw)", "deg"),
        ("shoulder_lift", "Elevation (Lift)", "deg"),
        ("elbow_flex", "Reach (Elbow)", "deg"),
    ]

    for key, label, unit in pos_joints:
        y += 20
        extra_tag = ""
        val = joints.get(key, 0.0)
        if key == "shoulder_lift":
            if val <= 35.0:
                extra_tag = " [UP]"
            elif val >= 75.0:
                extra_tag = " [DOWN]"
            else:
                extra_tag = " [MID]"
        _draw_gauge_bar(panel, key, label, unit, val, y, w, color=(255, 190, 40), suffix=extra_tag)

    # Section 2: Tool Hand
    y += 26
    tool_header = f"--- {'RIGHT' if not swap_roles else 'LEFT'} HAND: WRIST & GRIP ---"
    cv2.putText(panel, tool_header, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.39, (80, 230, 120), 1, cv2.LINE_AA)
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

    # Keyboard & Voice Controls Footer
    cv2.line(panel, (18, h - 145), (w - 18, h - 145), (70, 70, 75), 1)
    controls = [
        "Voice: 'On' to Engage | 'Off' to Freeze",
        "[SPACE] Hold: Clutch | [T] Toggle Clutch",
        "[C] Calibrate Neutral Zero References",
        "[I] Invert Lift (Up <-> Down)",
        f"[S] Swap Roles ({'L:Pos, R:Tool' if not swap_roles else 'R:Pos, L:Tool'})",
        "[Q] or [ESC] Exit",
    ]
    cy = h - 125
    for c in controls:
        cv2.putText(panel, c, (18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 170), 1, cv2.LINE_AA)
        cy += 18


def _draw_gauge_bar(panel: np.ndarray, key: str, label: str, unit: str,
                    val: float, y: int, w: int, color: Tuple[int, int, int], is_grip: bool = False, suffix: str = "") -> None:
    """Helper to draw a single joint gauge bar with limits and center zero marker."""
    lim_min, lim_max = DEFAULT_LIMITS[key]
    ratio = np.clip((val - lim_min) / (lim_max - lim_min + 1e-6), 0.0, 1.0)

    cv2.putText(panel, f"{label}:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    val_str = f"{val:5.1f} {unit}{suffix}"
    cv2.putText(panel, val_str, (w - 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    bar_x = 18
    bar_y = y + 4
    bar_w = w - 36
    bar_h = 6
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 45), -1)

    fill_w = int(bar_w * ratio)
    if is_grip:
        bar_color = (int(255 * (1.0 - ratio)), int(255 * ratio), 40)
    else:
        bar_color = color
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    if lim_min < 0 < lim_max:
        center_x = int(bar_x + bar_w * (-lim_min / (lim_max - lim_min)))
        cv2.line(panel, (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), (180, 180, 180), 1)


def draw_target_box(cam_view: np.ndarray, norm_box: Tuple[float, float, float, float],
                    title: str, subtext: str, in_box: bool, base_color: Tuple[int, int, int],
                    is_pos_box: bool = False, current_norm_y: Optional[float] = None) -> None:
    """Draws a themed corner-bracketed target box with crosshair, elevation marks, and title."""
    h, w, _ = cam_view.shape
    x1, y1 = int(norm_box[0] * w), int(norm_box[1] * h)
    x2, y2 = int(norm_box[2] * w), int(norm_box[3] * h)

    color = (0, 255, 120) if in_box else base_color
    corner_len = 24

    cv2.line(cam_view, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)

    cv2.line(cam_view, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)

    cv2.line(cam_view, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)

    cv2.line(cam_view, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(cam_view, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(cam_view, (cx - 10, cy), (cx + 10, cy), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(cam_view, (cx, cy - 10), (cx, cy + 10), (120, 120, 120), 1, cv2.LINE_AA)

    if is_pos_box:
        cv2.putText(cam_view, "^ HIGH (UP)", (cx - 40, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 220, 255), 1, cv2.LINE_AA)
        cv2.line(cam_view, (x1 + 15, cy), (cx - 18, cy), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(cam_view, (cx + 18, cy), (x2 - 15, cy), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(cam_view, "NEUTRAL", (cx - 26, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1, cv2.LINE_AA)
        cv2.putText(cam_view, "v LOW (DOWN)", (cx - 44, y2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 120), 1, cv2.LINE_AA)

        if current_norm_y is not None and in_box:
            if current_norm_y < -0.15:
                pct = int(abs(current_norm_y) * 100)
                status_label = f"ELEVATION: UP ({pct}%)"
            elif current_norm_y > 0.15:
                pct = int(current_norm_y * 100)
                status_label = f"ELEVATION: DOWN ({pct}%)"
            else:
                status_label = "ELEVATION: NEUTRAL (READY)"
        else:
            status_label = "READY: Neutral Zone" if in_box else subtext
    else:
        status_label = "READY: Neutral Zone" if in_box else subtext

    cv2.putText(cam_view, title, (x1 + 8, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.44, color, 1, cv2.LINE_AA)
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

    for start, end in connections:
        cv2.line(cam_view, pts_px[start], pts_px[end], bone_color, 2, cv2.LINE_AA)

    for i, pt in enumerate(pts_px):
        if is_tool_hand and i in [4, 8]:
            cv2.circle(cam_view, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(cam_view, pt, 4, (0, 220, 120), -1, cv2.LINE_AA)

    wrist_pt = pts_px[0]
    tag = f"[{label.upper()}: {role_title}]"
    cv2.putText(cam_view, tag, (wrist_pt[0] - 50, wrist_pt[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)

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


# ==============================================================================
# Main Teleoperation Loop
# ==============================================================================

def run_visualizer(camera_id: int = 0, connect_robot: bool = False,
                   robot_port: str = "/dev/ttyACM1", default_swap: bool = False,
                   default_invert_lift: bool = False, limits_file: Path = DEFAULT_LIMITS_FILE,
                   do_test_sweep: bool = False, ask_calibrate: bool = True):
    print("=" * 65)
    print("SO-101 DUAL-HAND REAL-TIME TELEOPERATION")
    print("=" * 65)
    print("Division of Labor:")
    print("  • LEFT HAND  -> 3D Arm Position (Pan, Lift: Hand UP->Arm UP, Reach)")
    print("  • RIGHT HAND -> Tool Orientation & Gripper (Pitch, Roll, Pinch)")
    print("=" * 65)

    robot = None

    if connect_robot:
        # Prompt user if they wish to calibrate limits first
        if ask_calibrate:
            print("\nYou can verify and customize your arm workspace limits (table height, ceiling reach).")
            try:
                ans = input("Do you wish to calibrate the arm limits first? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans in ["y", "yes"]:
                calibrate_arm_limits(robot_port=robot_port, save_path=limits_file)

        # Load limits after possible calibration
        arm_limits = load_arm_limits(limits_file)
        if arm_limits:
            print(f"\n✓ Loaded verified workspace limits from: {limits_file}")
            up = arm_limits.get("shoulder_lift", {}).get("up", 15.0)
            mid = arm_limits.get("shoulder_lift", {}).get("neutral", 55.0)
            down = arm_limits.get("shoulder_lift", {}).get("down", 95.0)
            p_l = arm_limits.get("shoulder_pan", {}).get("left", -60.0)
            p_r = arm_limits.get("shoulder_pan", {}).get("right", 60.0)
            print(f"  • Elevation: [UP: {up:.1f}°] -> [NEUTRAL: {mid:.1f}°] -> [DOWN: {down:.1f}°]")
            print(f"  • Base Pan : [{p_l:.1f}° (Left) .. {p_r:.1f}° (Right)]")
        else:
            arm_limits = None
            print("Note: Custom arm limits not found. Using standard defaults.")

        try:
            from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
            print(f"Connecting to SO-101 Follower arm on {robot_port} (id: matti_follower_arm)...")
            config = SOFollowerRobotConfig(port=robot_port, id="matti_follower_arm", use_degrees=True)
            robot = SOFollower(config)
            robot.connect()
            print("Follower arm connected successfully with torque enabled!")

            # Run test sweep if requested
            if do_test_sweep and arm_limits:
                run_test_sweep(robot, arm_limits)

        except Exception as e:
            print(f"Warning: Failed to connect to robot arm ({e}). Continuing in simulation mode.")
            robot = None
    else:
        arm_limits = load_arm_limits(limits_file)

    mid_lift_init = arm_limits.get("shoulder_lift", {}).get("neutral", 55.0) if arm_limits else 55.0
    mid_elbow_init = arm_limits.get("elbow_flex", {}).get("neutral", 65.0) if arm_limits else 65.0

    current_robot_cmd: Dict[str, float] = {
        "shoulder_pan": 0.0,
        "shoulder_lift": mid_lift_init,
        "elbow_flex": mid_elbow_init,
        "wrist_flex": -30.0,
        "wrist_roll": 0.0,
        "gripper": 100.0,
    }

    if robot is not None and robot.is_connected:
        present_pos = robot.bus.sync_read("Present_Position")
        for k, v in present_pos.items():
            if k in current_robot_cmd:
                current_robot_cmd[k] = float(v)
        print(f"Loaded physical starting pose: {current_robot_cmd}")

    # Start Offline Voice Command Listener
    voice_listener = VoiceCommandListener()
    voice_listener.start()

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

    # Position box depends on hand role assignment
    left_box_norm = (0.08, 0.18, 0.46, 0.82)
    right_box_norm = (0.54, 0.18, 0.92, 0.82)
    pos_box = left_box_norm if not default_swap else right_box_norm

    mapper = DualHandPoseMapper(ema_alpha=0.30, invert_lift=default_invert_lift,
                                limits=arm_limits, pos_box=pos_box)
    mapper.swap_roles = default_swap

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

    print("\nVisualizer is running!")
    print("HOW TO OPERATE:")
    print("  1. Place your LEFT hand in the left box, and RIGHT hand in the right box.")
    print("  2. Say 'On' (or hold [SPACE] or press [T]) to activate the arm!")
    print("  3. Move LEFT hand: UP=arm up, DOWN=arm down, LEFT/RIGHT=pan.")
    print("  4. RIGHT hand: tilt wrist to pitch, roll palm to roll, pinch to grip.")
    print("  5. Press [C] to re-zero tool hand orientation. Press [Q] to quit.\n")

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
                    lms_mirrored = [(1.0 - lm.x, lm.y, lm.z) for lm in result.hand_landmarks[idx]]
                    if detected_label == "Left" and left_hand_lms is None:
                        left_hand_lms = lms_mirrored
                    elif detected_label == "Right" and right_hand_lms is None:
                        right_hand_lms = lms_mirrored

            left_detected = left_hand_lms is not None
            right_detected = right_hand_lms is not None

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

            pos_metrics = None
            if pos_lms is not None:
                _, pos_metrics = mapper.compute_position(pos_lms)

            tool_metrics = None
            if tool_lms is not None:
                is_right = (tool_hand_label == "Right")
                _, tool_metrics = mapper.compute_tool(tool_lms, is_right_hand=is_right)

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

            current_pos_norm_y = pos_metrics["norm_y"] if pos_metrics is not None else None
            left_is_pos = not mapper.swap_roles

            draw_target_box(cam_view, left_box_norm, left_title, "Hold Left Hand Here", left_in_box,
                            base_color=(180, 140, 60), is_pos_box=left_is_pos,
                            current_norm_y=current_pos_norm_y if left_is_pos else None)

            draw_target_box(cam_view, right_box_norm, right_title, "Hold Right Hand Here", right_in_box,
                            base_color=(60, 160, 180), is_pos_box=(not left_is_pos),
                            current_norm_y=current_pos_norm_y if (not left_is_pos) else None)

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

            # Clutch State: active when Space is held, Toggle is ON, OR Voice is 'ON'
            voice_clutch_active = voice_listener.available and voice_listener.is_active
            any_hand_detected = left_detected or right_detected
            active_clutch = (is_clutched or toggle_clutch or voice_clutch_active) and any_hand_detected

            # Smooth Bumpless Dispatch to Robot Arm
            target_joints = mapper.smoothed_joints.copy()

            if active_clutch:
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
            is_calibrated = arm_limits.get("calibrated", False) if arm_limits else False
            draw_hud(
                panel=hud_panel,
                joints=current_robot_cmd if active_clutch else target_joints,
                is_clutched=active_clutch,
                left_detected=left_detected,
                right_detected=right_detected,
                left_in_box=left_in_box,
                right_in_box=right_in_box,
                swap_roles=mapper.swap_roles,
                invert_lift=mapper.invert_lift,
                fps=fps,
                robot_connected=(robot is not None and robot.is_connected),
                is_limits_calibrated=is_calibrated,
                voice_active=voice_clutch_active,
                voice_available=voice_listener.available
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
            elif key in [ord('i'), ord('I')]:
                mapper.invert_lift = not mapper.invert_lift
                dir_label = "INVERTED (Up=Down)" if mapper.invert_lift else "NORMAL (Up=Up)"
                print(f"Toggled Lift Direction -> {dir_label}")
            elif key in [ord('s'), ord('S')]:
                mapper.swap_roles = not mapper.swap_roles
                # Update position box to match new role assignment
                mapper.pos_box = left_box_norm if not mapper.swap_roles else right_box_norm
                role_str = "Left=Position, Right=Tool" if not mapper.swap_roles else "Right=Position, Left=Tool"
                print(f"Swapped Hand Roles -> {role_str}")
            elif key in [ord('c'), ord('C')]:
                # [C] re-zeros the tool hand orientation (roll & pitch neutral)
                # Position hand uses absolute box mapping and doesn't need re-zeroing
                if tool_lms is not None and tool_metrics is not None:
                    mapper.set_tool_neutral(tool_metrics.get("roll_deg", 0.0) * math.pi / 180.0,
                                            tool_metrics.get("pitch_deg", -70.0))
                    # Also re-capture palm scale for elbow depth mapping
                    if pos_lms is not None and pos_metrics is not None:
                        mapper.pos_neutral_scale = pos_metrics["scale"]
                        mapper.scale_calibrated = True
                    print(f"Calibrated: Tool hand ({tool_hand_label}) neutral zeroed!")
                else:
                    print("Calibration note: Tool hand not detected.")
            else:
                is_clutched = False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        voice_listener.stop()
        if robot is not None:
            print("Disconnecting robot arm...")
            robot.disconnect()
        print("Visualizer shutdown cleanly.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 Dual-Hand Real-Time Teleoperation Visualizer")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--swap", action="store_true", help="Swap hand roles (Right=Position, Left=Tool)")
    parser.add_argument("--invert-lift", action="store_true", help="Invert vertical shoulder lift direction")
    parser.add_argument("--robot", action="store_true", help="Connect to physical SO-101 follower arm")
    parser.add_argument("--port", type=str, default="/dev/ttyACM1", help="Follower robot serial port (default: /dev/ttyACM1)")
    parser.add_argument("--calibrate-limits", action="store_true", help="Launch interactive workspace limits calibration")
    parser.add_argument("--test-sweep", action="store_true", help="Run a gentle test sweep through workspace limits before teleoperating")
    parser.add_argument("--limits-file", type=str, default="arm_limits.json", help="Path to arm limits JSON file (default: arm_limits.json)")
    parser.add_argument("--no-prompt", action="store_true", help="Skip the startup calibration question")
    args = parser.parse_args()

    limits_path = Path(args.limits_file)
    if not limits_path.is_absolute():
        limits_path = Path(__file__).parent / args.limits_file

    if args.calibrate_limits:
        calibrate_arm_limits(robot_port=args.port, save_path=limits_path)
        return

    run_visualizer(
        camera_id=args.camera,
        connect_robot=args.robot,
        robot_port=args.port,
        default_swap=args.swap,
        default_invert_lift=args.invert_lift,
        limits_file=limits_path,
        do_test_sweep=args.test_sweep,
        ask_calibrate=(not args.no_prompt and args.robot)
    )


if __name__ == "__main__":
    main()
