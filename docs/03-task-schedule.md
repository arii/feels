# Hackathon Task Schedule

> [!IMPORTANT]
> **Refined Direction & Priority Focus:**
> - **Motion Routine:** Deterministic **Pick Up $\to$ Swirl $\to$ Place** routine from fixed taped square.
> - **Archived Concepts:** No pushing / impedance sliding, no violent linear shaking, no tilt sweeps, no dynamic YOLO-3D grasping, no neural network training.
> - **Core Delivery:** High-reliability **MongoDB Atlas feature extraction** (`feature_extractor.py`) paired with an intuitive **Web UX** (`realtime_plot.py`) for live detection reporting, telemetry visualization, and online recording.

---

## Day 1: Thursday — Foundations, Swirl Routine & Feature Pipeline

**Goal:** Establish the end-to-end loop: pick up $\to$ swirl $\to$ place $\to$ extract 10D features $\to$ index in Atlas $\to$ display detection report on the website.

---

### Phase 1: Robotics & Fixed Swirl Routine
- [x] Connect via Viam Python SDK; verify arm, gripper, and joint state telemetry
- [x] Launch high-performance rolling visualizer (`realtime_plot.py`)
- [x] Verify torque sensitivity on Joint 2 and Joint 3 ($\Delta \tau \approx -6.2\,\text{N}\cdot\text{m}$)
- [x] Configure fixed arm waypoints (`home-pose`, `approach`, `grasp`, `travel`, `place`) on taped square
- [ ] Implement smooth horizontal swirl orbital trajectory in travel position
- [ ] Wire synchronized phase tagging: `INIT` $\to$ `APPROACH` $\to$ `GRASP` $\to$ `LIFT` $\to$ `HOLD` $\to$ `SWIRL` $\to$ `PLACE` $\to$ `HOME`

### Phase 2: Signal Processing & MongoDB Atlas Memory
- [x] Create Atlas collections (`runs`, `signatures`, `items`) and vector index `cup_dynamics_index`
- [x] Verify `atlas.py` `teach()` and `identify()` methods
- [ ] Implement `feature_extractor.py`:
  - Static mass offset ($\Delta \tau_{\text{gravity}}$)
  - Swirl torque variance ($\text{Var}(\tau_{\text{swirl}})$)
  - FFT dominant frequency and power
  - High-frequency rattle energy ($E_{\text{rattle}}$ 5–25 Hz)
  - Orbital swirl phase lag ($\Delta \phi_{\text{lag}}$)
  - L2-normalized 10D embedding vector $e \in \mathbb{S}^9$
- [ ] Record reference datasets for calibration set: `empty_cup`, `water_half`, `dish_soap_half`, `candy_mixed`, `solid_weight`
- [ ] Teach baseline signatures into MongoDB Atlas memory

### Phase 3: Web Dashboard UX & Online Recording
- [ ] Integrate `RepeatableGraspController` into `realtime_plot.py`:
  - "▶ Run Pick & Place Cycle" button
  - "⏹ Abort" button
  - Real-time active phase badge
- [ ] Add live Detection Report Card:
  - Winning label banner with emoji and confidence percentage
  - Top-3 k-NN matches with cosine similarity scores
- [ ] Add Online Recording & Teaching controls:
  - "Teach / Save to Atlas" 1-click button
  - Dataset recordings drawer with instant CSV download links

---

## Day 2: Friday — Calibration, Rehearsals & Live Demos

**Goal:** Polish the operator web dashboard for live judging, record clean validation runs, and execute seamless live demonstrations.

---

### Morning Check (8:30 AM – 10:30 AM)
- [ ] Empty cup tare recalibration
- [ ] Check MongoDB Atlas connectivity and index status
- [ ] Run 5 automated validation cycles (one per item); verify 100% classification accuracy on detection report card

### The Show Layer & Demo Polish (10:30 AM – 2:00 PM)
- [ ] Polish web visualizer CSS for judging display (high-contrast dark mode, clear typography)
- [ ] Verify 1-click online teaching: demonstrate teaching a new mystery item live during the demo presentation
- [ ] Code freeze at 2:00 PM; run full rehearsal runs before 3:30 PM presentations
