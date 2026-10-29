# Project Roadmap & Next Steps (TODO)

This document tracks immediate technical priorities and research directions for the **feels** physical AI robotics project.

---

## 1. Simulator Integration
- [ ] **Evaluate Simulation Environments:**
  - **PyBullet / URDF:** Load the official UFactory xArm850 URDF and gripper mesh to simulate forward/inverse kinematics, joint torque limits, and basic collision geometries.
  - **MuJoCo / Isaac Sim:** Evaluate for simulating dynamic contact mechanics, granular collisions (e.g. candy/screws rattling in a cup), and fluid-body sloshing.
  - **Viam Mock Hardware:** Use `rdk:builtin:arm` fake components to run the full software stack (dashboard, API, Atlas pipelines) offline without connecting to the physical machine.
- [ ] **Synthetic Data Generation:**
  - Build synthetic torque profiles for various fluid and solid items to pre-seed the Atlas vector search database before physical hardware runs.
- [ ] **Trajectory Validation:**
  - Test all new motion paths and safety boundaries ($Z \ge 15\,\text{mm}$) in simulation before executing on physical hardware.

---

## 2. On-Robot High-Rate Data Collection Tools
- [ ] **Deploy On-Host Telemetry Daemon on `sfhack12`:**
  - Current WebRTC cloud polling over the internet incurs $\approx 158\,\text{ms}$ RTT latency and caps sampling at $4\text{--}5\,\text{Hz}$ to avoid gRPC deadline timeouts.
  - Deploy a lightweight Python daemon directly on the host machine (`sfhack12`) communicating with `localhost:8080` or direct socket.
- [ ] **High-Frequency Socket Stream (Port 30000 / 100–200 Hz):**
  - Read direct actuator states at $100\text{--}200\,\text{Hz}$ to capture high-frequency physical dynamics (specifically the 5–25 Hz rattle chatter from loose solids like candy or hardware).
- [ ] **Automated Phase Annotation:**
  - Inject synchronized phase marks directly into recorded telemetry arrays:
    `APPROACH` $\to$ `GRASP` $\to$ `LIFT` $\to$ `STEADY_HOLD` $\to$ `SHAKE_SINE` $\to$ `TILT` $\to$ `RELEASE`.
- [ ] **Atlas Telemetry Sync:**
  - Stream completed runs directly into the MongoDB Atlas `runs` collection via `atlas.py`.

---

## 3. Posture Optimization for Enhanced Signal Sensitivity ("Moving to a Better Location")
- [ ] **Kinematic Sensitivity Analysis:**
  - *Current Pose:* Joint 2 at $86.42^\circ$ (nearly vertical). Vertical weight acts in axial column compression, yielding $\Delta \tau_2 \approx 0.00\,\text{N}\cdot\text{m}$.
  - *Target Posture:* Re-orient the arm into a cantilever configuration where Link 2 and Link 3 share the perpendicular moment arm ($d_\perp$).
  - Recommended Joint Configuration:
    - Joint 2 (Shoulder Pitch): $45^\circ \text{ to } 60^\circ$
    - Joint 3 (Elbow Pitch): $45^\circ \text{ to } 90^\circ$
- [ ] **Signal-to-Noise Ratio (SNR) Multiplication:**
  - Moving to the optimal moment arm multiplies the torque reaction by $2.5\times\text{--}3.5\times$ for both Joint 2 and Joint 3, making sub-50g objects clearly distinguishable above sensor noise.
- [ ] **Calibration & Tare Profile:**
  - Store baseline resting torques for the optimized pose across both joints.

---

## 4. Dynamic Trajectory Sensing ("Moving While Capturing")
- [ ] **Move Beyond Static Mass Detection:**
  - Static holding only senses gravitational mass ($F = mg$). Dynamic motion unlocks internal physical properties (inertia, viscosity, phase lag, and damping).
- [ ] **Trajectory 1 — Sinusoidal Shake (2–4 Hz):**
  - Command a continuous horizontal or vertical sinusoidal shake.
  - Measure torque phase lag ($\phi$) to separate viscous fluids (slosh lag) from rigid masses (zero phase lag).
- [ ] **Trajectory 2 — Step Impulse & Freeze:**
  - Accelerate rapidly and command an instantaneous freeze.
  - Measure post-stop decay time ($\tau_{\text{decay}}$) and oscillation settling time to distinguish water from soap and solid items.
- [ ] **Trajectory 3 — Slow Tilt (25°):**
  - Tilt the end-effector $25^\circ$ and return to level.
  - Measure static center-of-mass shift and torque hysteresis curve.

---

## 5. Alternative Controllers & Control Modes
- [ ] **Evaluate xArm Controller Modes:**
  - **Joint Velocity Control:** Test velocity-mode streaming for sinusoidal motions to eliminate jerky pause artifacts between discrete position waypoints.
  - **Cartesian Impedance / Compliance Mode:** Test compliance mode so the arm complies slightly under external loads, enabling spring/damper contact sensing.
- [ ] **Controller Tuning & Gain Adjustments:**
  - Tune motor stiffness and damping parameters to eliminate the high-frequency dither observed near joint limit boundaries ($q_3 = 2.17^\circ$).
  - Implement a digital low-pass Butterworth or Kalman filter on the controller side before computing derivative signals.
- [ ] **Viam Motion Planning vs. Direct Execution:**
  - Compare Viam's `MotionClient.move()` with direct trajectory interpolation to ensure real-time command delivery with minimal jitter.

---

## 6. Addressing Latency via On-Robot Generic Service Module

### Architecture & Overview
Remote cloud polling over WebRTC incurs $\sim 158\,\text{ms}$ RTT latency and caps sampling at $4\text{--}5\,\text{Hz}$, starving high-frequency signals (such as 5–25 Hz rattle chatter, slosh lag, and impact transients).

To solve this, we deploy an **on-robot custom Viam Generic Service module** (`hackathons:control_logic:generic-service`) directly onto `viam-server` on `sfhack12`.

```
┌──────────────────────────────────────── sfhack12 (Robot Host) ────────────────────────────────────────┐
│                                                                                                        │
│   ┌───────────────────────────────────┐    Local IPC (<0.5ms)     ┌────────────────────────────────┐  │
│   │        viam-server                │ ◄──────────────────────►  │  FeelsControlService (Module)  │  │
│   │  • arm (xArm850)                  │                           │  • Model: generic-service      │  │
│   │  • gripper (UFactory)             │                           │  • 100-200 Hz Sampling Loop    │  │
│   │  • cam (RealSense)                │                           │  • In-Memory Ring Buffer       │  │
│   └───────────────────────────────────┘                           │  • Onboard FFT & Features      │  │
│                     ▲                                             └────────────────────────────────┘  │
└─────────────────────┼─────────────────────────────────────────────────────────────────────────────────┘
                      │
                      │ WebRTC / gRPC (Cloud do_command)
                      ▼
       ┌───────────────────────────────┐
       │   Remote UI / Agent Client    │
       │   • Calls do_command(...)     │
       │   • Receives 10-dim vector    │
       └───────────────────────────────┘
```

### Module Template Implementation (`control_service.py`)
- **Base Architecture:** Subclasses `GenericService, EasyResource` with model `hackathons:control_logic:generic-service`.
- **Hardware Dependencies:** Resolves `arm`, `gripper`, and `camera` directly via `new(config, dependencies)`.
- **Onboard Polling Loop:** In `new()`, starts an unblocked `asyncio.create_task` running at 50–200 Hz, storing samples in a 2000-element circular buffer.
- **Onboard Feature Extraction:** In `do_command({"action": "compute_features"})`, runs NumPy FFT directly on the robot to compute:
  1. Mean delta torque $\Delta \bar{\tau}$ (mass proxy)
  2. Peak delta torque $\Delta \tau_{\text{peak}}$
  3. Torque variance $\sigma_\tau^2$
  4. Dominant FFT peak frequency $f_{\text{max}}$
  5. Dominant FFT peak power $P_{\text{max}}$
  6. Rattle band energy (5–25 Hz)
  7. Low-frequency slosh energy (0.5–3.0 Hz)
  8. Spectral energy ratio (rattle / slosh)
  9. Steady-state stability percentage
  10. Estimated payload mass (grams)
- **Zero-Drop Motion Macro:** In `do_command({"action": "run_macro"})`, commands steady hold, sine shake, and tilt while capturing 100 Hz data synchronously in memory.

### Viam App Machine Configuration JSON
Add this service entry to the robot configuration in the Viam app:
```json
{
  "name": "control-logic",
  "model": "hackathons:control_logic:generic-service",
  "type": "generic",
  "attributes": {
    "arm": "arm",
    "gripper": "gripper",
    "camera": "cam",
    "sampling_rate_hz": 100.0,
    "lever_arm_m": 0.45,
    "tare_j3_nm": -16.821
  }
}
```

### Implementation Checklist
- [x] **Scaffold Service:** Implemented complete on-robot service in [`control_service.py`](control_service.py).
- [x] **Client Test Script:** Created [`client_control_service.py`](client_control_service.py) for testing `do_command` actions over gRPC.
- [ ] **Deploy to `sfhack12`:** Upload module to Viam registry or register as a local process module on the machine.
- [ ] **Live Macro Test:** Execute `run_macro` on physical hardware and verify 100 Hz feature vector extraction.

---

## 7. Repeatable Grasping Workflow (Arm Saver Positions)
- [x] **Discover Static Waypoints:** Identified all 5 `erh:vmodutils:arm-position-saver` switches (`home-pose`, `approach`, `grasp`, `travel`, `place`).
- [x] **Implement Pick-and-Place Automation Script:** Developed [`repeatable_grasp.py`](repeatable_grasp.py) supporting full cycles, individual step execution, dry-run mode, and gripper actuation.
- [x] **Verify Home Pose Navigation:** Successfully validated `home-pose` execution and gripper state verification over Viam gRPC client.
- [ ] **Run Live Full Repeatable Grasp Cycle:** Execute physical pick-and-place with an object and verify grip/holding status.

