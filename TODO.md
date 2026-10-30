# Project Roadmap & Active Focus (TODO)

> [!IMPORTANT]
> **Key Decisions & Active Focus:**
> - **Motion Primitive:** Fixed **Pick Up $\to$ Swirl $\to$ Place** sequence from a calibrated taped table square.
> - **Current Focus:** **MongoDB Atlas feature extraction** (`feature_extractor.py`) + **Web UX** (`realtime_plot.py`) for live detection reporting, real-time telemetry, and online recording.
> - **Archived Approaches:** Pushing / impedance sliding, violent linear shake, tilt sweeps, vision-based grasping, and on-site deep learning training are officially ARCHIVED.

---

## 1. High-Priority Active Deliverables

### A. Dynamic Feature Extraction Engine (`feature_extractor.py`)
- [x] Standalone `feature_extractor.py` (done): the real 10-D `z` is
  `mass_j2`/`mass_j3` (static gravity offsets), `swirl_var_j2`/`swirl_var_j3`,
  `swirl_ptp_j3`, `phase_lag_s` (xcorr lag of TCP speed vs J3 residual, ±2 s),
  `rattle_e` (observable band only — 5–25 Hz **not** reachable at 4–5 Hz
  sampling, flagged `rattle_band_limited`), `grip_comp`, `lift_dyn_j3`,
  `swirl_speed` → per-feature scaled (`FEATURE_SCALES`, `--calibrate`), then
  L2-normalized to the embedding `e`. Ships `FEATURE_DESCRIPTIONS`
  (plain-language meanings for the dashboard) and `swirl_window_series()`
  (detrended residuals + phase lag for the "why" charts).
- [x] Unit sanity: extractor runs on saved CSV recordings in `recordings/`
  (`predict_mystery.py --eval` / `--qc`).

### B. Repeatable Grasp Pipeline Refactoring (`repeatable_grasp.py`)
- [ ] Implement smooth horizontal orbital swirl excitation $a_{\text{swirl}}$ in travel position:
  - Small circular radius ($R \approx 20\text{--}30\,\text{mm}$) at $f \approx 1.5\text{--}2.0\,\text{Hz}$ to induce fluid swirl and granular chatter without spillage.
- [ ] Decouple `RepeatableGraspController` and `TelemetryRecorder` for programmatic import.
- [ ] Support phase-synchronized callback hooks for web dashboard status updates.
- [ ] Automatic end-of-run feature extraction and CSV export.

### C. Web Visualization Dashboard & Online Recording UX (`realtime_plot.py`)
- [x] **Live Telemetry & Rolling Sparklines:**
  - 6 independent native HTML5 canvas sparklines for individual joint torques,
    fixed ±N·m axes (a still joint reads as a flat line, not stretched noise).
  - (Removed: derived payload-weight / force-Fz KPI cards — those numbers were
    tare-contaminated per the audit.)
- [x] **Detection Reporting Card (Mode 1):**
  - Predicted class with confidence bar, top-3 Atlas `$vectorSearch` neighbors
    with similarity scores; persists across refreshes via `/api/status`.
  - 10-D feature-space panel (grouped bars + plain-language hover descriptions).
  - "Why" physics: swirl torque-residual traces with measured phase lag, and an
    honest power spectrum (observable band only — the 5–25 Hz band is labeled
    as requiring 100 Hz sampling, never faked).
- [x] **Vector space card:** 2-D PCA of the Atlas signature library, hover
  tooltips, gold/silver/bronze rings on the top-3 neighbor labels after a
  prediction, per-label counts; refreshes on load, every 5 s, after teach/predict.
- [x] **Online Recording & Instant Teaching (Mode 2):**
  - "RUN SWIRL MACRO & PREDICT" / "RUN SWIRL & SAVE TO ATLAS" one-click buttons
    with elapsed-second tickers; "■ Stop" for the macro subprocess.
  - Preset label chips: confound set (`solid_150g`/`water_150g`/`candy_150g`/`soap_150g`)
    + basics.
  - `teach_csv.py` batch-teaches labeled CSVs without the robot
    (`--require-swirl` enforces protocol consistency).
  - Footer shows the running `build <sha>` so venue pulls are verifiable.
- [x] **Mock mode** (`FEELS_MOCK=1`): full 2-mode UI from recorded data with a
  purple MOCK badge — the robot/network failure fallback.

### D. MongoDB Atlas Calibration & Seed Dataset
- [ ] Calibrate baseline items via the Pick Up $\to$ Swirl $\to$ Place routine
  (**swirl protocol only** — mixed protocols confound the embedding):
  1. `empty_cup` (tare baseline)
  2. `water_half` (fluid slosh, phase lag)
  3. `dish_soap_half` (viscous damping, larger lag)
  4. `candy_mixed` (loose-impact variance; 5–25 Hz band *not* observable at 4–5 Hz)
  5. `solid_150g` / `water_150g` / `candy_150g` / `soap_150g` (same-mass confound set)
- [ ] Verify Atlas `$vectorSearch` accuracy across classes.
- [x] Atlas reachable from venue (was: TLS handshake failure — fixed via IP access list).

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
