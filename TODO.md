# Project Roadmap & Active Focus (TODO)

> [!IMPORTANT]
> **Key Decisions & Active Focus:**
> - **Motion Primitive:** Fixed **Pick Up $\to$ Swirl $\to$ Place** sequence from a calibrated taped table square.
> - **Current Focus:** **MongoDB Atlas feature extraction** (`feature_extractor.py`) + **Web UX** (`realtime_plot.py`) for live detection reporting, real-time telemetry, and online recording.
> - **Archived Approaches:** Pushing / impedance sliding, violent linear shake, tilt sweeps, vision-based grasping, and on-site deep learning training are officially ARCHIVED.

---

## 1. High-Priority Active Deliverables

### A. Dynamic Feature Extraction Engine (`feature_extractor.py`)
- [ ] Create standalone `feature_extractor.py` with `DynamicFeatureExtractor`:
  - Static lift mass offset ($\Delta \tau_{\text{gravity}}$)
  - Swirl residual torque variance ($\text{Var}(\tau_{\text{swirl}})$)
  - FFT spectral peak frequency & power ($f_{\text{peak}}$, $P_{\text{peak}}$)
  - High-frequency rattle energy in 5–25 Hz band ($E_{\text{rattle}}$)
  - Orbital swirl cross-correlation phase lag ($\Delta \phi_{\text{lag}}$)
  - L2-normalized 10D metric embedding $e \in \mathbb{S}^9$
- [ ] Unit tests running extractor on saved CSV recordings in `recordings/`.

### B. Repeatable Grasp Pipeline Refactoring (`repeatable_grasp.py`)
- [ ] Implement smooth horizontal orbital swirl excitation $a_{\text{swirl}}$ in travel position:
  - Small circular radius ($R \approx 20\text{--}30\,\text{mm}$) at $f \approx 1.5\text{--}2.0\,\text{Hz}$ to induce fluid swirl and granular chatter without spillage.
- [ ] Decouple `RepeatableGraspController` and `TelemetryRecorder` for programmatic import.
- [ ] Support phase-synchronized callback hooks for web dashboard status updates.
- [ ] Automatic end-of-run feature extraction and CSV export.

### C. Web Visualization Dashboard & Online Recording UX (`realtime_plot.py`)
- [ ] **Live Telemetry & Rolling Sparklines:**
  - 10-second rolling Chart.js (Weight + Force Fz).
  - 6 independent native HTML5 canvas sparklines for individual joint torques with zero lines.
- [ ] **Active Phase Status Badge:**
  - Real-time indicator (`IDLE` $\to$ `APPROACH` $\to$ `GRASP` $\to$ `LIFT` $\to$ `SWIRL` $\to$ `PLACE`).
- [ ] **Detection Reporting Card:**
  - Prominent verdict banner with predicted item emoji and title.
  - Confidence percentage bar (0–100%).
  - Top-3 k-NN matches from MongoDB Atlas `$vectorSearch` with similarity scores.
- [ ] **Online Recording & Instant Teaching:**
  - "▶ Run Pick & Place Cycle" one-click button.
  - "⏹ Abort" button for safety.
  - "Teach / Save to Atlas" one-click button to register current feature vector into Atlas memory.
  - Dataset recordings drawer with 1-click CSV download.

### D. MongoDB Atlas Calibration & Seed Dataset
- [ ] Calibrate 5 distinct baseline items via the Pick Up $\to$ Swirl $\to$ Place routine:
  1. `empty_cup` (Tare baseline, zero mass, zero swirl variance)
  2. `water_half` (Fluid vortex, dynamic swirl phase lag)
  3. `dish_soap_half` (Viscous damping, delayed peak torque)
  4. `candy_mixed` (High-frequency impact chatter in 5–25 Hz rattle band)
  5. `solid_weight` (Rigid mass, zero swirl lag, zero rattle)
- [ ] Verify Atlas `$vectorSearch` accuracy across all 5 classes.

---

## 2. Archived / Deprecated Exploration Avenues

The following avenues have been evaluated and deliberately scoped out for the hackathon:

- [x] **[ARCHIVED] Impedance Pushing & Multi-Cup Games:**
  - *Reason for Archiving:* Pushing or shuffling cups on table surfaces introduces unpredictable Coulomb friction, table seam catches, and cup tipping. Sticking strictly to a single vertical pick-and-place routine.
- [x] **[ARCHIVED] Violent Linear Shake & Tilt Sweeps:**
  - *Reason for Archiving:* Aggressive linear shakes cause motor dithering and fluid splash risks; tilt sweeps risk cup drops. Replaced with smooth horizontal circular swirl.
- [x] **[ARCHIVED] Dynamic Vision-Based Grasping & YOLO-3D:**
  - *Reason for Archiving:* Visual servoing and depth cloud estimation introduce lighting/occlusion sensitivity. Fixed taped table squares provide 100% mechanical repeatability for comparing dynamic torque signals.
- [x] **[ARCHIVED] On-Site Deep Learning Training (CNNs / Transformers):**
  - *Reason for Archiving:* Limited hackathon samples risk overfitting. Hand-crafted 10D physical features mapped to Atlas vector search provide robust, explainable zero-shot classification without training delays.
- [x] **[ARCHIVED] Dynamic Bayesian Policy Planning:**
  - *Reason for Archiving:* Real-time action selection branches add unnecessary latency. A single, deterministic diagnostic macro achieves superior signal-to-noise ratio.
