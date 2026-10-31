# System Architecture

The motion primitive is a single deterministic sequence: Pick Up $\to$ Swirl $\to$ Place.

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

### 1. Hardware & Motion Execution (Fixed Pick $\to$ Swirl $\to$ Place)
- **Taped Square Workspace:** The operator places the cup on a designated taped target, fixing Cartesian coordinates.
- **Table contact:** none. The cup stays in the gripper for the whole trajectory, so the signal carries gripper-held dynamics only.
- **Arm-Position-Saver Waypoints:** Standardized positions (`home-pose`, `approach`, `grasp`, `travel`, `place`).
- **Fixed Trajectory Pipeline:**
  1. `APPROACH` $\to$ Align gripper above cup.
  2. `GRASP` $\to$ Descend and close gripper jaws.
  3. `LIFT` $\to$ Vertical lift to elevated travel altitude.
  4. `STATIC_HOLD` $\to$ Measure steady-state gravitational torque offset $\Delta \tau_{\text{gravity}}$ for net mass.
  5. `SWIRL` $\to$ Execute a smooth horizontal circular/orbital swirl motion (exciting fluid vortex lag, solid chatter, and centripetal torque reactions).
  6. `PLACE` $\to$ Descend back to table square.
  7. `RELEASE` $\to$ Open jaws and retract arm to home pose.

*(Archived: Cartesian impedance pushing, table shuffles, aggressive sine shakes, and tilt sweeps).*

### 2. High-Rate Telemetry Synchronization
- **TelemetryRecorder (`repeatable_grasp.py`):**
  - Continuous asynchronous polling of 6 actuator torques ($\tau_1 \dots \tau_6$), joint angles ($q_1 \dots q_6$), Cartesian TCP coordinates, and gripper feedback.
  - Synchronized phase tagging: `INIT`, `APPROACH`, `REACH_GRASP`, `GRIPPER_CLOSE`, `LIFT_TO_TRAVEL`, `TRAVEL_STABLE_HOLD`, `SWIRL_EXCITATION`, `DESCEND_PLACE`, `GRIPPER_OPEN`, `RETURN_HOME`.
  - Automatic standardization and export to `recordings/recording_{timestamp}_{label}.csv`.

### 3. Signal Processing & Feature Extraction (`feature_extractor.py`)

The multi-phase rollout becomes a 10-dimensional physical feature vector $z$, normalized to the unit embedding $e = z / \|z\|_2 \in \mathbb{S}^9$ for cosine retrieval. `05-ml-pipeline.md` defines the features and the embedding.

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
  - Preset label chips (demo classes + basics); `teach_csv.py` batch-teaches labeled CSVs without the robot.
  - Manual start/stop recording with CSV download.

---

## End-to-End Interaction Storyboard

| Step | User Action | Robot Action | UI Screen State | Backend / Atlas State |
|------|-------------|--------------|-----------------|----------------------|
| 1. Standby & Baseline | Places cup on taped square | Holds home pose | **Status**: live torque sparklines | Telemetry streaming at 4–5 Hz |
| 2. Run Routine | Clicks "RUN SWIRL MACRO & PREDICT" | Executes pick up → swirl → place | **Status**: elapsed-second ticker | Buffers telemetry with phase tags |
| 3. Swirl Excitation | Observes smooth circular swirl | Sweeps circular orbit in horizontal plane | Live joint-torque sparklines react | Buffers telemetry with phase tags |
| 4. Placement & Settle | Robot places cup on square | Descends, opens gripper, retracts to home | Macro completes | Finalizes CSV dataset in `recordings/` |
| 5. Detection Report | Inspects verdict card on dashboard | Stationary at home pose | **Verdict**: predicted label, confidence, top-3 neighbors, feature-space bars, "why" physics, query in vector-space plot | `feature_extractor.py` generates $e$; Atlas `$vectorSearch` returns k-NN results |
| 6. Online Teaching (Optional) | Picks a preset chip or types a label, clicks "RUN SWIRL & SAVE TO ATLAS" | Stationary | **Status**: "Signature saved — indexed in MongoDB" | Inserts embedding into `signatures` |
