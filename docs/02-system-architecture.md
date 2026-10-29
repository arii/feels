# System Architecture

## Tech Stack Overview

| Layer | Tool / Hardware | Responsibility |
|-------|----------------|----------------|
| Hardware | xArm x850 + RealSense D435 | Actuation, wrist F/T sensing, RGB-D container localization |
| Control | Viam Arm API & Motion Service | Path planning, Cartesian trajectory streaming (`MoveThroughJointPositionsStreamed`) |
| Telemetry | Viam Sensor API & Data Management | Capturing joint torques, motor currents, and gripper feedback at high rates |
| Feature / Modeling | Python (scipy.fft, numpy, PyTorch) | Dynamic signal processing (FFT, resonance, rattle energy) to generate embeddings |
| Database & Vector Search | MongoDB Atlas | Storing rollout logs, historical metadata, and running `$vectorSearch` for k-NN identification |
| Interactive UX | Web Dashboard (Vite / React / Node.js) | UI for triggering calibration motions, inspecting model certainty, and registering new items |

---

## System Architecture Components

### 1. Hardware & Driver Abstraction (Viam RDK)
- **Arm & Gripper Components:** Configure arm drivers (`viam:universal-robots` or `viam-ufactory-xarm`) and parallel-jaw gripper child component (`rdk:component:gripper`)
- **Sensor Interfaces:** Expose 6-axis wrist F/T sensor (UR RTDE) or motor current streams via Viam Sensor API (`GetReadings()`)
- **Vision Service:** Eye-in-hand or overhead RGB-D camera (RealSense D435 / Orbbec Astra 2) for initial cup localization and grasp-pose detection

### 2. Motion Execution & Probing Controller
- **Grasp & Lift Primitive:** Standardized approach, gripper stall-force closure, and vertical lift to fixed test altitude
- **Excitation Trajectory Generator:** Parameterized Cartesian primitives executed via `MoveThroughJointPositionsStreamed` or straight-line pose control:
  - *Static Hold:* Pure gravity offset
  - *Axial Sine Shake:* High-frequency vertical/horizontal vibration (slosh / loose-item rattle)
  - *Figure-8 / Tilt:* Continuous orientation sweep for fluid inertia and CoM shift
  - *Impedance Push:* Compliant horizontal table slide for friction/tipping dynamics

### 3. High-Rate Telemetry & Sync Pipeline
- **Telemetry Recorder:** Python background process polling arm pose, joint torques, motor currents, and F/T wrench at 50–100 Hz synchronized to trajectory start/stop events
- **Viam Data Management Service:** Local JSON/BSON buffering and batch synchronization to cloud storage

### 4. Signal Processing & Feature Extraction Engine
- **Pre-processing:** Baseline tare subtraction using empty cup calibration profile
- **Spectral & Kinematic Feature Transforms:**
  - Fast Fourier Transform (FFT) / Power Spectral Density (PSD) on residual torques
  - Phase lag Δφ relative to commanded acceleration (viscosity metric)
  - High-frequency transient energy (5–25 Hz rattle/chatter detector)
  - Grip deformation metric Δg (stiffness/fill proxy)
- **Embedding Model:** Lightweight encoder (1D-CNN or MLP) projecting extracted dynamic features into a metric space R^d

### 5. Storage & Vector Retrieval (MongoDB Atlas)
- **Document Store:** Collections for raw trajectory runs, extracted time-series metrics, and ground-truth metadata
- **Atlas Vector Search Index:** Index on normalized trajectory embedding vectors using cosine/Euclidean distance
- **Online Registry API:** Endpoint to query top-k nearest neighbors for classification or upsert new labeled items on the fly

### 6. Interactive Decision & Active Policy Engine
- **Belief State Tracker:** Aggregates multi-action likelihoods to maintain Bayesian uncertainty over payload classes
- **Next-Best Action Selector:** Chooses the next probing motion based on maximum information gain if prediction confidence is below threshold

### 7. Operator UI & Demonstration Dashboard
- **Control Panel:** Buttons to trigger predefined routines (Tare, Pick, Calibrate, Run Probe Suite, Register New Item)
- **Live Telemetry & Uncertainty Display:** Visualizations of current joint effort graphs, FFT power spectra, and live class probability/confidence bars
- **Atlas Sync View:** Real-time log of runs, matched items from Vector Search, and ground-truth validation forms

---

## End-to-End Interaction Storyboard

| Step | User Action | Robot Physical Action | UI Screen State | Data & Backend State |
|------|-------------|----------------------|-----------------|----------------------|
| 1. Standby & Tare | Places mystery cup on marked table grid; clicks "Tare & Prepare" | Stays in home position; reads baseline sensor noise and confirms camera sightline | **Home Screen**: Status "Idle", camera feed showing cup bounding box, mode toggle ("Classify" vs. "Register") | Zeroes baseline offsets; confirms Atlas connection |
| 2. Grasp & Weigh | Clicks "Identify Mystery Object" | Approaches, executes side/top grasp, lifts to 15 cm, holds for 2 seconds | **Inspection View**: Gripper force gauge spikes; real-time torque graph streams; estimated mass appears | Computes steady-state gravity torque difference Δτg → m̂ |
| 3. Dynamic Excitations | Observes robot motion | Runs sequence: horizontal sine shake, then 30° roll/tilt, then figure-8 sweep | **Live Telemetry Panel**: FFT power spectrum pulses; phase-lag curve overlays commanded vs. measured torque | Buffers time-series; runs FFT; calculates viscous phase lag Δφ and rattle score |
| 4. Vector Match & Verdict | Views dynamic match results | Holds cup steady above table; returns to standby pose | **Decision Card**: Highlights top match with confidence bar, category badge, and nearest Atlas neighbors | Python encoder generates embedding vector e; executes `$vectorSearch` in Atlas |
| 5. Online Teaching (Branch) | *If item is new:* Types label (e.g., "Full Dish Soap") and clicks "Teach Robot" | Gently sets cup down on table; releases gripper and retracts | **Success Toast**: "Signature saved to Atlas memory! Vector indexed." | Upserts embedding vector + metadata into MongoDB Atlas cluster |

---

## Motion & Control Strategy

- Use off-the-shelf pick-up or approach-type tool for grasping (joint control, don't collect data)
- Then use Cartesian controller for next movements since this will likely give highest sensor signals

### Required Viam Integration & Tools
- Vision detection
- Motion planning to approach pose (top, side grasp, etc.)

---

## Diagrams to Create

### User Interface & Web Interaction
- Selecting the demo
- Training a new item
- Instructing to do some actions
- Understanding the model's understanding of the object (ideally we can see its uncertainty)
