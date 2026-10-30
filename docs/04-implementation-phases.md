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
| **Candy (loose fun-size)** | Medium mass (~140g), sharp impact chatter in 5–25 Hz rattle band | `candy_mixed` |
| **Solid Weight (rigid)** | High mass (~350g), rigid coupling, zero swirl lag, zero rattle | `solid_weight` |

---

## Phase 1: Feature Extraction & Swirl Integration (Immediate Focus)

- [x] Verified arm-position-saver waypoints (`home-pose`, `approach`, `grasp`, `travel`, `place`)
- [ ] Implement `DynamicFeatureExtractor` in `feature_extractor.py`:
  - Static mass offset ($\Delta \tau_{\text{gravity}}$)
  - Swirl residual torque variance ($\text{Var}(\tau_{\text{swirl}})$)
  - Dominant FFT frequency and power ($f_{\text{peak}}$, $P_{\text{peak}}$)
  - High-frequency rattle energy ($E_{\text{rattle}}$ 5–25 Hz)
  - Swirl phase lag ($\Delta \phi_{\text{lag}}$)
  - L2-normalized 10D embedding vector $e \in \mathbb{S}^9$
- [ ] Implement smooth horizontal swirl orbital trajectory in `repeatable_grasp.py`
- [ ] Automatic end-of-run feature extraction and CSV logging

---

## Phase 2: Web Dashboard UX & Live Detection Reporting (Immediate Focus)

- [ ] Upgrade `realtime_plot.py` dashboard:
  - **Live Telemetry:** 10s rolling Chart.js (Weight + Force Fz) and 6-joint canvas sparklines
  - **Active Phase Display:** Live badge tracking (`IDLE` $\to$ `APPROACH` $\to$ `GRASP` $\to$ `LIFT` $\to$ `SWIRL` $\to$ `PLACE`)
  - **Detection Reporting Card:** Prominent verdict displaying predicted label with emoji, confidence score, and top-3 k-NN matches from Atlas `$vectorSearch`
  - **Online Recording & Teaching:**
    - "▶ Run Pick & Place Cycle" one-click button
    - "⏹ Abort" button
    - "Teach / Save to Atlas" one-click button to register new calibrated items
    - CSV dataset drawer with download links

---

## What to Cut vs. What to Keep

| Cut (Archived) | Keep (Active Architecture) |
|----------------|----------------------------|
| Dynamic vision tracking / visual servoing / YOLO-3D grasp planning | Fixed taped table square with arm-position-saver waypoints |
| Cartesian impedance pushing & cup shuffling | Single vertical pick-and-place routine |
| Violent linear sine shaking & tilt sweeps | Smooth horizontal circular swirl excitation |
| Deep learning & neural network training on-site (CNNs/Transformers) | Hand-crafted physical signal features mapped to a normalized 10D vector |
| Dynamic Bayesian policy exploration | Deterministic pick $\to$ swirl $\to$ place execution with Atlas k-NN retrieval |
