# System Architecture

> [!IMPORTANT]
> **Key Architectural Scoping Decisions:**
> - **Motion Primitive:** Single deterministic sequence: **Pick Up $\to$ Swirl $\to$ Place**.
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

### 1. Hardware & Motion Execution (Fixed Pick $\to$ Swirl $\to$ Place)
- **Taped Square Workspace:** Cup is placed on a designated taped target, guaranteeing fixed Cartesian coordinates.
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

### 3. Signal Processing & Feature Extraction Engine (`feature_extractor.py`)
- Direct algorithmic transformation of the multi-phase rollout into a 10-dimensional physical feature vector $z$:
  1. **Static Lift & Hold:** Net mass offset $\Delta \tau_{\text{gravity}} = \bar{\tau}_{3,\text{hold}} - \tau_{\text{tare}}$.
  2. **Swirl Dynamic Variance:** Torque variance $\text{Var}(\tau_3)_{\text{swirl}}$ across orbital motion.
  3. **FFT Peak Frequency & Power:** Dominant resonant frequencies $f_{\text{peak}}$, $P_{\text{peak}}$.
  4. **High-Frequency Rattle Energy:** Integrated spectral power in the 5–25 Hz band ($E_{\text{rattle}}$) to flag candy and loose solids.
  5. **Swirl Phase Lag ($\Delta \phi_{\text{lag}}$):** Cross-correlation phase shift between commanded orbit and measured torque response to detect fluid slosh/vortex vs. rigid mass.
  6. **Centripetal / Tangential Coupling:** Ratio of lateral to longitudinal torque reactions.
  7. **Gripper Compliance ($\Delta g_{\text{grip}}$):** Jaw stroke change during motion.
- **Normalized Metric Embedding:** $e = z / \|z\|_2 \in \mathbb{S}^9$ (unit-norm cosine vector).

### 4. Database & Vector Memory (MongoDB Atlas)
- **`signatures` Collection:** Stores document embeddings with vector search index `cup_dynamics_index` (numDimensions: 10, similarity: cosine).
- **`runs` Collection:** Raw rollout telemetry arrays for auditability and retrospective analysis.
- **`items` Collection:** Item metadata, display names, and ground-truth notes.
- **Non-Parametric Classification:** Fast `$vectorSearch` pipeline performs k-NN classification with mean similarity scoring across nearest neighbors.

### 5. Operator Web Dashboard UX (`realtime_plot.py`)
- **Real-Time Rolling Visualizer:** 10-second high-efficiency rolling chart of effective weight (g) and vertical force (N) alongside 6 native HTML5 canvas sparklines for individual joint torques.
- **Live Phase Timeline:** Real-time badge tracking the active step of the pick-swirl-place routine.
- **Detection Reporting Panel:** Prominent verdict card presenting predicted item label, confidence bar (0–100%), and top-3 k-NN matches.
- **Online Recording & Teaching:**
  - One-click "Teach / Save to Atlas" to register newly calibrated items on the fly.
  - Manual start/stop recording with automatic feature extraction.
  - Saved datasets browser with 1-click CSV download.

---

## End-to-End Interaction Storyboard

| Step | User Action | Robot Action | UI Screen State | Backend / Atlas State |
|------|-------------|--------------|-----------------|----------------------|
| 1. Standby & Tare | Places cup on taped square; clicks "Tare" | Holds home pose; samples resting torques | **Status**: "IDLE", baseline calibrated | Computes $\tau_{\text{tare}}$ baseline |
| 2. Run Routine | Clicks "▶ Run Pick & Place Cycle" | Executes fixed pick up $\to$ lift to travel | **Phase**: `APPROACH` $\to$ `GRASP` $\to$ `LIFT` | Telemetry streaming at 4–5 Hz |
| 3. Swirl Excitation | Observes smooth circular swirl | Sweeps circular orbit in horizontal plane | **Phase**: `SWIRL_EXCITATION`, live torque waveforms | Buffers telemetry with phase tags |
| 4. Placement & Settle | Robot places cup on square | Descends, opens gripper, retracts to home | **Phase**: `DESCEND_PLACE` $\to$ `RETURN_HOME` | Finalizes CSV dataset in `recordings/` |
| 5. Detection Report | Inspects verdict card on dashboard | Stationary at home pose | **Verdict**: Displays detected object, confidence score, and top matches | `feature_extractor.py` generates $e$; Atlas `$vectorSearch` returns k-NN results |
| 6. Online Teaching (Optional) | Selects label and clicks "Save to Atlas" | Stationary | **Toast**: "Signature indexed in MongoDB Atlas!" | Inserts embedding into `signatures` |
