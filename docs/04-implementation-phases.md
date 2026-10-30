# Implementation Phases & Roadmap

> [!IMPORTANT]
> **Core Decisions & Scope Refinement:**
> - **Motion Routine:** Fixed **Pick Up $\to$ Swirl $\to$ Place** sequence from a taped table square.
> - **Explicitly Excluded:**
>   - *No pushing / impedance manipulation:* Pushing or sliding cups leads to unpredictable friction slippage and tipping.
>   - *No shake or tilt:* Violent linear shaking and tilt sweeps are eliminated; replaced with gentle, continuous horizontal swirling.
>   - *No dynamic vision grasping:* Static waypoints provide 100% mechanical repeatability.
>   - *No deep learning training:* Hand-crafted signal features mapped to Atlas vector search provide robust, explainable metric classification without training delays.
> - **Primary Focus:** **MongoDB Atlas feature extraction** (`feature_extractor.py`) + **Operator Web UX** (`realtime_plot.py`) for live detection reporting, real-time telemetry, and online recording.

---

## The Minimal Viable Demo — "Pick Up $\to$ Swirl $\to$ Place"

One fixed, highly repeatable trajectory:

1. **Approach & Grasp (1.5 s):** Align and close parallel-jaw gripper at taped square.
2. **Lift & Static Hold (2.0 s):** Lift vertically to travel altitude; measure steady-state torque $\Delta \tau_{\text{gravity}}$ for net mass.
3. **Orbital Swirl Excitation (3.0 s):** Smooth horizontal circular swirl in the XY plane to excite fluid vortex/slosh lag, solid candy chatter, and torque variance without risking spills.
4. **Controlled Place & Release (2.0 s):** Descend to target square, open gripper, and return to home staging pose.

---

## Streamlined Item Library (Distinct Physical Signatures)

5 calibrated items with distinctly separable dynamic signatures under the swirl routine:

| Item | Swirl & Mass Signature | Expected Atlas Match |
|------|------------------------|----------------------|
| **Empty Cup** | Zero mass offset ($\Delta \tau \approx 0$), zero swirl variance, zero rattle | `empty_cup` (Tare baseline) |
| **Water (half-full)** | Medium mass (~180g), dynamic swirl vortex phase lag, low high-frequency chatter | `water_half` |
| **Dish Soap (half-full)** | Medium mass (~210g), heavy viscous swirl damping, delayed response peak | `dish_soap_half` |
| **Candy (loose fun-size)** | Medium mass (~140g), loose-impact torque variance (the 5–25 Hz rattle band itself is *not* observable at 4–5 Hz sampling) | `candy_mixed` |
| **Solid Weight (rigid)** | High mass (~350g), rigid coupling, zero swirl lag, zero rattle | `solid_weight` |

---

## Phase 1: Feature Extraction & Swirl Integration (Immediate Focus)

- [x] Verified arm-position-saver waypoints (`home-pose`, `approach`, `grasp`, `travel`, `place`)
- [x] Implement `feature_extractor.py` (done — the real 10-D vector is
  `mass_j2/mass_j3`, `swirl_var_j2/j3`, `swirl_ptp_j3`, `phase_lag_s`,
  `rattle_e` (observable band only, flagged band-limited), `grip_comp`,
  `lift_dyn_j3`, `swirl_speed` → per-feature scaled, L2-normalized embedding):
  - Static mass offsets (gravity torque at J2/J3)
  - Swirl residual torque variances
  - Swirl peak-to-peak swing
  - Swirl phase lag (xcorr lag, ±2 s window)
  - High-frequency rattle energy (observable band only — 5–25 Hz **not** reachable at 4–5 Hz sampling)
  - Per-feature scaled, L2-normalized 10D embedding vector $e \in \mathbb{S}^9$
- [ ] Implement smooth horizontal swirl orbital trajectory in `repeatable_grasp.py`
- [ ] Automatic end-of-run feature extraction and CSV logging

---

## Phase 2: Web Dashboard UX & Live Detection Reporting (Done)

- [x] `realtime_plot.py` dashboard:
  - **Live Telemetry:** 6-joint canvas sparklines, 10 s rolling, fixed ±N·m axes
    (derived weight/force KPI cards removed — tare-contaminated per audit)
  - **Detection Reporting Card:** predicted label, confidence bar, top-3 Atlas
    `$vectorSearch` neighbors with similarities; feature-space panel; "why"
    physics (torque residuals + phase lag, honest observable-band spectrum)
  - **Vector space card:** 2-D PCA of the signature library, hover tooltips,
    neighbor rings, per-label counts
  - **Online Recording & Teaching:**
    - "RUN SWIRL MACRO & PREDICT" / "RUN SWIRL & SAVE TO ATLAS" one-click buttons
      with elapsed tickers
    - "■ Stop" button for the macro subprocess
    - Preset label chips (confound set + basics)
    - `teach_csv.py` batch-teach from CSVs (`--require-swirl`)
    - Manual start/stop recording with CSV download
    - `FEELS_MOCK=1` fallback mode with MOCK badge

---

## What to Cut vs. What to Keep

| Cut (Archived) | Keep (Active Architecture) |
|----------------|----------------------------|
| Dynamic vision tracking / visual servoing / YOLO-3D grasp planning | Fixed taped table square with arm-position-saver waypoints |
| Cartesian impedance pushing & cup shuffling | Single vertical pick-and-place routine |
| Violent linear sine shaking & tilt sweeps | Smooth horizontal circular swirl excitation |
| Deep learning & neural network training on-site (CNNs/Transformers) | Hand-crafted physical signal features mapped to a normalized 10D vector |
| Dynamic Bayesian policy exploration | Deterministic pick $\to$ swirl $\to$ place execution with Atlas k-NN retrieval |
