# SO-101 Dual-Hand Real-Time Teleoperation (Vision-Based)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Robot: SO-101 / SO-100](https://img.shields.io/badge/Robot-SO--101%20%7C%20SO--100-orange.svg)](https://github.com/huggingface/lerobot)

A real-time computer vision teleoperation system for the **SO-101 (and SO-100) robotic arm**. Controls the 6-DOF arm and gripper using **TWO hands** tracked simultaneously via your laptop webcam—no physical leader arm or expensive wearable sensors required.

---

## Why Dual-Hand Teleoperation?

Controlling all 6 degrees of freedom on a single hand introduces severe mechanical coupling: rotating your wrist or squeezing your fingers unintentionally shifts your palm in space, nudging the arm off-target right when grasping.

By dividing the robot's degrees of freedom naturally between two hands, control becomes intuitive and completely decoupled:
- **Left Hand ("3D Arm Joystick")**: Focuses entirely on gross 3D spatial positioning:
  - **Move Hand Up / Down** $\rightarrow$ **Arm moves UP / DOWN** (calibrated full range of motion that fits completely inside the on-screen target box).
  - **Move Hand Left / Right** $\rightarrow$ **Arm pans Left / Right**.
  - **Push / Pull Hand** $\rightarrow$ **Arm reaches forward / retracts back**.
- **Right Hand ("End-Effector Wand")**: Focuses entirely on tool orientation and grasping:
  - **Wrist Tilt / Pitch** $\rightarrow$ **Gripper tilts Up / Down**.
  - **Wrist Roll** $\rightarrow$ **Gripper rotates CW / CCW**.
  - **Pinch Thumb & Index** $\rightarrow$ **Gripper closes (0% pinch) to open (100%)**.

*(Note: Left-handed or alternate operator preference? Press **`[S]`** to swap hand roles instantly!)*

---

## Features

- **Intuitive Elevation Mapping**: Moving your hand **UP** commands the arm **UP**; moving your hand **DOWN** commands the arm **DOWN**.
- **Full On-Screen Range of Motion**: Scaled so that moving between the visual markers (`▲ UP` and `▼ DOWN`) inside the target box commands the arm's **entire useful vertical travel** without awkward off-screen reaching.
- **Decoupled Dual-Hand Tracking**: Simultaneous 30+ FPS landmark tracking of both hands via Google MediaPipe Tasks.
- **Side-by-Side Dual-Pane UI**: Dark HUD control dashboard on the left (380px), completely unobstructed live camera view on the right (960px).
- **Dual Guided Target Boxes**: On-screen corner brackets with crosshairs, elevation guides (`▲ HIGH (UP)`, `── NEUTRAL ──`, `▼ LOW (DOWN)`), and real-time color feedback.
- **Bumpless Soft Engagement**: Software slew-rate limiter gently glides the arm from its resting position to your hand targets (~42°/sec max), preventing sudden jerks.
- **Calibrated 2D Pinch Gripper**: Ultra-sensitive pinch detection (0% fully closed to 100% open) normalized by hand scale.
- **Runtime Direction & Role Toggles**:
  - Press **`[I]`** (or `--invert-lift`) to invert lift direction if desired.
  - Press **`[S]`** (or `--swap`) to swap hand roles (`Left=Position, Right=Tool` $\leftrightarrow$ `Right=Position, Left=Tool`).
- **Independent Calibration**: Press **`[C]`** to zero both hands simultaneously at your comfortable resting height.
- **Live Hardware Integration**: Seamlessly interfaces with LeRobot's `SOFollower` and Feetech STS3215 bus.

---

## System Architecture

```mermaid
flowchart LR
    A["Webcam (/dev/video0)"] --> B["MediaPipe HandLandmarker\n(num_hands=2)"]
    B --> C["Left Hand: Position\n(X -> Pan, Y -> Up/Down, Scale -> Reach)"]
    B --> D["Right Hand: Tool & Gripper\n(Pitch -> Flex, Roll -> Roll, Pinch -> Gripper)"]
    C --> E["Safety Pipeline\n(Slew-Rate Limiter, Calibrated Limits, Clutch)"]
    D --> E
    E --> F["SO-101 Follower Arm\n(/dev/ttyACM1)"]
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- An SO-101 or SO-100 follower arm powered and connected via USB
- A standard laptop webcam or external USB camera

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/MattiArlo/so101-hand-teleop.git
cd so101-hand-teleop

# Create or activate your virtual environment (e.g. conda or uv)
conda create -n so101-teleop python=3.12 -y
conda activate so101-teleop

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### 1. Simulation Mode (Safe Test without Robot)
Run the visualizer to test tracking and familiarize yourself with two-handed control:
```bash
python visualizer.py
```

### 2. Live Follower Arm Control
When ready to control the physical robot, add `--robot`:
```bash
python visualizer.py --robot --port /dev/ttyACM1
```

*(Note: `python teleoperate.py` is also available as a convenience launcher)*

---

## Control Mapping

### Left Hand: Arm Position (3-DOF)
| Hand Gesture | Controlled Joint | Description |
|---|---|---|
| **Move Hand Up / Down** | `shoulder_lift` | **Lifts arm UP (hand up) or lowers arm DOWN (hand down)** |
| **Move Hand Left / Right** | `shoulder_pan` | Rotates base yaw left or right ($\pm 60^\circ$) |
| **Push Hand Closer to Camera** | `elbow_flex` | Extends arm reach forward |
| **Pull Hand Back Away from Camera** | `elbow_flex` | Retracts arm back towards base |

### Right Hand: Wrist Orientation & Gripper (3-DOF)
| Hand Gesture | Controlled Joint | Description |
|---|---|---|
| **Tilt Wrist Down / Up** | `wrist_flex` | Gripper pitch tilt ($\pm 75^\circ$) |
| **Rotate Palm Clockwise / CCW** | `wrist_roll` | Gripper roll rotation ($\pm 170^\circ$) |
| **Pinch Thumb & Index Tips Together** | `gripper` | Closes gripper (0% pinch) |
| **Separate Thumb & Index Apart** | `gripper` | Opens gripper (100% open) |

---

## How to Operate

1. **Launch the visualizer** with `--robot --port /dev/ttyACM1` (or in simulation mode first).
2. **Hold your hands in front of the camera**:
   - Place your **Left Hand** inside the left target box on screen.
   - Place your **Right Hand** inside the right target box on screen.
3. Observe both target boxes turn **glowing green** (`READY: Neutral Zone`).
4. **Press `[C]`** on your keyboard:
   - Calibrates your current hand height and position as the neutral reference.
5. **Engage Clutch**:
   - **Hold `[SPACE]`** for momentary clutch (arm moves while held, freezes when released).
   - Or **press `[T]`** to toggle continuous teleoperation mode.
   - The arm smoothly glides into alignment via the slew-rate limiter.
6. **Move hand up and down**:
   - Move your hand up towards `▲ HIGH (UP)` to lift the arm into the air.
   - Move your hand down towards `▼ LOW (DOWN)` to reach down to the desk.
   - The entire range of motion is achieved comfortably inside the target box!

---

## Keyboard Shortcuts

| Key | Function |
|---|---|
| **`[SPACE]`** (Hold) | Engage clutch (streams motion to follower arm) |
| **`[T]`** | Toggle continuous tracking ON / OFF |
| **`[C]`** | Calibrate / zero neutral reference poses for both hands |
| **`[I]`** | Invert vertical lift direction (`Up` $\leftrightarrow$ `Down`) |
| **`[S]`** | Swap hand roles (`Left=Position, Right=Tool` $\leftrightarrow$ `Right=Position, Left=Tool`) |
| **`[Q]`** or **`[ESC]`** | Exit application and safely disable arm torque |

---

## License

This project is licensed under the Apache 2.0 License.
