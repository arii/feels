# Hardware & Viam Setup

## Robot Hardware

| Component | Details |
|-----------|---------|
| Arm | UFactory xArm x850 |
| Camera | Intel RealSense D435 |
| Robot name | sfhack12 |
| OS | linux/amd64 |
| viam-server | 1.14.3 |
| viam-agent | 1.2.0 |

---

## Motion Controllers & Arm Support in Viam

Arms are abstracted via the **Arm Component API** and orchestrated via the **Motion Service**.

### Built-in & Registry Driver Modules

- **UFactory xArm** — `viam-ufactory-xarm` module (covers xArm 5/6/7 and Lite 6), communicates over Ethernet/TCP with the control box.
- **Universal Robots (UR e-Series)** — C++ `viam:universal-robots` module (ur3e, ur5e, ur10e, ur16e, ur20), interacts via UR's primary/secondary/RTDE interfaces.
- **Custom / Open-Source Arms (SO-100 / SO-101)** — Hugging Face / TheRobotStudio follower/leader arm via serial bus servo protocols (e.g., Feetech STS3215 via USB bus boards), community modules like `devrel:so101:arm` or custom Python/C++ drivers.
- **Grippers** — Integrated as a child component (`rdk:component:gripper`) parented to the arm end-effector frame; exposes open, close, and grab commands.

### Motion Control Modes

| Mode | API | Notes |
|------|-----|-------|
| Direct Joint Control | `MoveToJointPositions`, `MoveThroughJointPositions` | Joint-space interpolation, bypasses global planner |
| Cartesian Pose Control | `MoveToPosition` | Moves TCP in straight Cartesian trajectory via local IK |
| Trajectory Streaming | `MoveThroughJointPositionsStreamed` | Streams timed waypoint batches for real-time servoing |
| Viam Motion Service (Planner) | Global | Collision-aware planning with cBiRRT fallback |

---

## Collectable Sensor Data

Sensor streams can be polled or streamed using Viam's **Arm API**, standard **Sensor API** (`GetReadings()`), or the **Data Management Service**.

### Joint Positions & Kinematics
- Joint angles (degrees or radians) across all arm DOF via `GetJointPositions()`
- End-effector 6-DOF Cartesian pose (X, Y, Z, θx, θy, θz) via `GetEndPosition()`
- Joint velocities and motor drive temperatures (available on UR RTDE, xArm, and Feetech STS smart servos)

### Force, Torque & Effort
- **UR e-Series** — Integrated 6-axis TCP force-torque sensor at the wrist (calibrated Fx, Fy, Fz, τx, τy, τz) plus per-joint estimated torques and motor currents
- **xArm** — Joint-level motor currents/efforts (and 6-axis F/T wrench if paired with external wrist load cells or the xArm torque sensor)
- **SO-101** — Per-servo present load/current from Feetech telemetry (approximates joint resistance, lacks raw wrist F/T)

### Gripper & Tool Telemetry
- Position / stroke percentage (0–100% or absolute mm)
- Gripper load/current (used for grip detection and stall protection)

### Perception & Environmental Data (if attached)
- Wrist/eye-in-hand cameras (color streams, depth maps, point clouds)
- External digital/analog I/O states (accessible via Viam's Board API)

---

## Viam Agent Skills (Developer-Side)

Install via: `/plugin marketplace add viam-devrel/agent-skills`

### `viam-python`
- Accelerates writing async robot clients (`viam.robot.client.RobotClient`) for concurrent sensor polling
- Handles boilerplate for polling UR e-Series RTDE / xArm current feeds while streaming Cartesian trajectories

### `viam-ml`
- Scaffolds data capture pipelines hookable into Viam's Data Management Service
- Provides patterns for deploying lightweight feature extractors (ONNX / TFLite modules) to the local Linux compute

### `viam-machine-config` & `viam-modules-fleet`
- Enables programmatic updates to the machine JSON configuration directly from CLI or agent (e.g., swapping arm configs, camera frame systems, tuning velocity caps)

### `viam-typescript`
- Fast-tracks building the browser control dashboard and HMI using `@viamrobotics/sdk` to render live telemetry charts and uncertainty indicators

---

## Data Collection with Viam

- **Arm Telemetry** (`viam:universal-robots` / `viam-ufactory-xarm`): Poll `GetJointPositions()`, motor currents, and the 6-axis wrist F/T sensor (`GetReadings()`) at 50–100 Hz during standardized trajectory playback via `MoveToPosition()` or `MoveThroughJointPositionsStreamed()`
- **Visual Verification** (`viam:camera`): Stream RGB-D frames from the RealSense D435 to track container slippage, tilt angles, and grasp stability
- **Sync to Cloud:** Use Viam's **Data Management Service** to package each trajectory execution into a structured multi-sensor dataset
