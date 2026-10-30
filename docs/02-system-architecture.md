# System Architecture

> [!IMPORTANT]
> **Key Architectural Scoping Decisions:**
> - **Motion Primitive:** Single deterministic sequence: **Pick Up $\to$ Swirl $\to$ Place $\to$ Back-off**.
> - **No Pushing or Impedance Modes:** Table pushing and cup sliding are eliminated to avoid non-holonomic slippage and tipping.
> - **No Shake or Tilt:** Replaced with smooth horizontal orbital swirling to safely excite fluid and solid dynamic signatures.
> - **No Vision-Based Grasping:** Replaced by deterministic taped-square arm waypoints for 100% repeatable physical trials.
> - **No Deep Learning Training:** Replaced by hand-crafted 10D physical features mapped to a normalized unit embedding.
> - **Core Focus:** High-precision feature extraction + rich operator UX for detection reporting, live telemetry, and online recording.

---

## Tech Stack Overview

| Layer | Tool / Hardware | Responsibility | Status |
|-------|----------------|----------------|--------|
| Hardware | xArm x850 + Gripper | Fixed waypoint pick-and-place, swirl excitation, actuator load sensing | **ACTIVE** |
| Vision | RealSense D435 | Overhead staging / dashboard verification (YOLO-3D grasp synthesis archived) | **CONTEXTUAL** |
| Motion Control | Viam Arm API & Position Saver | Deterministic trajectory execution (home $\to$ approach $\to$ grasp $\to$ travel swirl $\to$ place) | **ACTIVE** |
| Telemetry Sync | Viam Python SDK / TelemetryRecorder | High-rate sampling of 6-joint torques, Cartesian pose, gripper load, and phase annotations | **ACTIVE** |
| Feature Extraction | Python (`feature_extractor.py`) | Deterministic temporal & spectral feature extraction (mass, swirl variance, FFT rattle energy, lag) | **ACTIVE** |
| Vector Memory | MongoDB Atlas `$vectorSearch` | Non-parametric k-NN retrieval on normalized 10D embeddings; online item registration | **ACTIVE** |
| Operator UX | Web Dashboard (`realtime_plot.py`) | 10s rolling telemetry visualizer, active phase display, detection reporting card, online recording | **ACTIVE** |
| Active Policy / Deep Learning | PyTorch / Bayesian Search | Neural net training and dynamic trajectory policies | **ARCHIVED** |

---

## System Architecture Components

### 1. Hardware & Motion Execution (Fixed Pick $\to$ Swirl $\to$ Place $\to$ Back-off)
- **Taped Square Workspace:** Cup is placed on a designated taped target, guaranteeing fixed Cartesian coordinates.
- **Arm-Position-Saver Waypoints:** Standardized positions (`home-pose`, `approach`, `grasp`, `travel`, `place`, `back-off`).
- **Fixed Trajectory Pipeline:**
  1. `APPROACH` $\to$ Align gripper above cup.
  2. `GRASP` $\to$ Descend and close gripper jaws.
  3. `LIFT` $\to$ Vertical lift to elevated travel altitude.
  4. `SWIRL` $\to$ Smooth horizontal circular/orbital swirl at the travel pose (exciting fluid vortex lag, solid chatter, and centripetal torque reactions).
  5. `PLACE` $\to$ Descend to table square, open jaws.
  6. `BACK_OFF` $\to$ Retreat clear of the placed cup.
  7. `RETRACT` $\to$ Return to travel altitude and home pose.

*(Archived: Cartesian impedance pushing, table shuffles, aggressive sine shakes, and tilt sweeps.)*

### 2. High-Rate Telemetry Synchronization
- **TelemetryRecorder (`repeatable_grasp.py`):**
  - Continuous asynchronous polling of 6 actuator torques ($\tau_1 \dots \tau_6$), joint angles ($q_1 \dots q_6$), Cartesian TCP coordinates, and gripper feedback.
  - Synchronized phase tagging: `INIT`, `APPROACH`, `REACH_GRASP`, `GRIPPER_CLOSE`, `LIFT_TO_TRAVEL`, `SWIRL`, `DESCEND_PLACE`, `GRIPPER_OPEN`, `BACK_OFF`, `RETRACT_TRAVEL`, `RETURN_HOME`.
  - Automatic standardization and export to `recordings/recording_{timestamp}_{label}.csv`.

### 3. Signal Processing & Feature Extraction Engine (`feature_extractor.py`)
- Direct algorithmic transformation of the multi-phase rollout into the real
  10-dimensional physical feature vector $z$ (`FEATURE_NAMES` in code):
  `mass_j2`, `mass_j3` (static gravity offsets — baseline-dominated, see the audit),
  `swirl_var_j2`, `swirl_var_j3` (torque variance during swirl),
  `swirl_ptp_j3` (peak-to-peak swing), `phase_lag_s` (xcorr lag of TCP speed
  vs J3 residual, ±2 s), `rattle_e` (observable band only — 5–25 Hz **not**
  reachable at 4–5 Hz sampling, flagged `rattle_band_limited`), `grip_comp`,
  `lift_dyn_j3`, `swirl_speed`. Per-feature scaled, L2-normalized to the
  embedding $e$. Full table in `05-ml-pipeline.md`.

### 4. Database & Vector Memory (MongoDB Atlas)
- **`signatures` Collection:** Stores document embeddings with vector search index `cup_dynamics_index` (numDimensions: 10, similarity: cosine).
- **`runs` Collection:** Raw rollout telemetry arrays for auditability and retrospective analysis.
- **`items` Collection:** Item metadata, display names, and ground-truth notes.
- **Non-Parametric Classification:** Fast `$vectorSearch` pipeline performs k-NN classification with mean similarity scoring across nearest neighbors.

### 5. Operator Web Dashboard UX (`realtime_plot.py`)
- **Real-Time Rolling Visualizer:** 6 native HTML5 canvas sparklines for individual joint torques (10 s rolling, fixed ±N·m axes).
- **Detection Reporting Panel (Mode 1):** Verdict card with predicted label, confidence bar, top-3 Atlas `$vectorSearch` neighbors with similarities; 10-D feature-space panel; "why" physics (swirl torque residuals + phase lag, honest observable-band spectrum); persists across refreshes.
- **Vector space card:** 2-D PCA of the Atlas signature library with hover tooltips, per-label counts, and gold/silver/bronze rings on the top-3 neighbor labels after a prediction.
- **Online Recording & Teaching (Mode 2):**
  - One-click "RUN SWIRL MACRO & PREDICT" / "RUN SWIRL & SAVE TO ATLAS" with elapsed-second tickers; "■ Stop" for the macro subprocess.
  - Preset label chips (confound set + basics); `teach_csv.py` batch-teaches labeled CSVs without the robot.
  - Manual start/stop recording with CSV download.

---

## End-to-End Interaction Storyboard

| Step | User Action | Robot Action | UI Screen State | Backend / Atlas State |
|------|-------------|--------------|-----------------|----------------------|
| 1. Standby & Tare | Places cup on taped square | Holds home pose | **Status**: live torque sparklines | Telemetry streaming at 4–5 Hz |
| 2. Run Routine | Clicks "RUN SWIRL MACRO & PREDICT" | Executes pick up → swirl → place → back-off | **Status**: elapsed-second ticker | Buffers telemetry with phase tags |
| 3. Swirl Excitation | Observes smooth circular swirl | Sweeps circular orbit in horizontal plane | Live joint-torque sparklines react | Buffers telemetry with phase tags |
| 4. Placement & Back-off | Robot places cup, opens gripper, backs off clear | Descends, opens gripper, retreats to back-off, returns home | Macro completes | Finalizes CSV dataset in `recordings/` |
| 5. Detection Report | Inspects verdict card on dashboard | Stationary at home pose | **Verdict**: predicted label, confidence, top-3 neighbors, feature-space bars, "why" physics, query in vector-space plot | `feature_extractor.py` generates $e$; Atlas `$vectorSearch` returns k-NN results |
| 6. Online Teaching (Optional) | Picks a preset chip or types a label, clicks "RUN SWIRL & SAVE TO ATLAS" | Stationary | **Status**: "Signature saved — indexed in MongoDB" | Inserts embedding into `signatures` |
