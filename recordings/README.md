# Physical Arm Telemetry Recordings & Force Datasets

This directory contains experimental datasets captured from the physical **UFactory xArm850** robotic arm connected via the Viam Python SDK to `sfhack12-main.ty1cywh7tq.viam.cloud`.

---

## 1. Experimental Setup & Hardware Configuration

- **Machine:** `sfhack12-main.ty1cywh7tq.viam.cloud` (`sfhack12`, Linux amd64)
- **Arm Model:** UFactory xArm850 (6-DOF)
- **End-Effector:** UFactory Gripper
- **Tested Posture:**
  - Joint Angles: $q = [11.11^\circ, 86.42^\circ, 2.17^\circ, -0.46^\circ, -0.35^\circ, 9.86^\circ]$
  - End-Effector Position: $X = 136.2\,\text{mm}, Y = 26.8\,\text{mm}, Z = 744.8\,\text{mm}$
  - Horizontal Reach ($R_{xy}$): $\approx 139\,\text{mm}$
- **Collection Mechanism:** Live streaming via WebRTC with `realtime_plot.py` recording to timestamped CSV files.

---

## 2. Dataset Catalogue

| File | Type | Duration | Samples | Ground Truth State | Key Observations |
|---|---|---|---|---|---|
| [`recording_20261029_122337.csv`](recording_20261029_122337.csv) | Test Run 1 | ~5.0 s | 24 | **Empty / Unloaded** | Initial baseline calibration run. |
| [`recording_20261029_122355.csv`](recording_20261029_122355.csv) | Test Run 2 | ~5.8 s | 28 | **Holding Payload** | Preliminary holding test showing Joint 3 drop to $-22.14\,\text{N}\cdot\text{m}$. |
| [`recording_20261029_123957.csv`](recording_20261029_123957.csv) | Extended Run 3 | 12.7 s | 34 | **Empty / Unloaded** | Settles firmly into resting torque $-16.821\,\text{N}\cdot\text{m}$ for 76.5% of duration. |
| [`recording_20261029_124044.csv`](recording_20261029_124044.csv) | Extended Run 4 | 16.4 s | 71 | **Holding Payload with Release** | Sustained loaded state ($-22.0$ to $-23.02\,\text{N}\cdot\text{m}$) for 13.8 s (83.1% of duration), followed by release step back to $-16.821\,\text{N}\cdot\text{m}$ at $t=13.8\,\text{s}$. |

---

## 3. Telemetry Schema & Column Dictionary

Every CSV row records the robot's physical state at a sample instant:

| Column | Unit | Description |
|---|---|---|
| `elapsed_sec` | seconds | Time elapsed since visualizer initialization |
| `effective_force_n` | $\text{N}$ | Calculated vertical force $F_z$ derived from shoulder pitch delta |
| `payload_weight_g` | grams | Estimated payload mass derived from $F_z / g$ |
| `delta_j2_nm` | $\text{N}\cdot\text{m}$ | Shoulder pitch torque deflection relative to tare baseline |
| `r_xy_m` | meters | Planar distance from arm base origin to end-effector ($0.139\,\text{m}$) |
| `j1_raw` – `j6_raw` | $\text{N}\cdot\text{m}$ | Instantaneous actuator torque read directly from `arm.do_command({"load": ""})` |
| `j1_smooth` – `j6_smooth` | $\text{N}\cdot\text{m}$ | Low-pass exponentially smoothed joint torques ($\alpha = 0.35$) |

---

## 4. Analysis Figures

### Comparison of Extended Datasets (Run 3 vs Run 4)
![Recordings 3 vs 4 Comparison](figures/recordings_3_vs_4.png)
*Figure 1: Extended comparison showing resting baseline in Run 3 vs sustained holding plateau and clean release event at $t=13.8\,\text{s}$ in Run 4.*

### Initial Comparative Session (Run 1 vs Run 2)
![Session Comparison](figures/session_comparison.png)
*Figure 2: Initial short trials comparing 6-joint torques between unloaded and holding states.*

### Kinematic Sensitivity & Effective Force
![Effective Force Graph](figures/effective_force_graph.png)
*Figure 3: 15-second sensitivity analysis across all joints.*

---

## 5. Physical Insights & Calculation

1. **Load Distribution Mechanics:**
   - At Joint 2 angle $q_2 = 86.42^\circ$ (nearly vertical upper arm), vertical downward load acts in pure axial column compression along Link 2. As a result, Joint 2 torque shows near zero deflection ($\Delta \tau_2 \approx 0.00\,\text{N}\cdot\text{m}$).
   - At Joint 3 angle $q_3 = 2.17^\circ$, Link 3 (the forearm) extends horizontally as a cantilever beam. All downward force applies a bending moment directly to the **Joint 3 (Elbow Pitch)** actuator.

2. **Quantified Signal Difference:**
   - Unloaded resting torque: $\tau_{3,\text{unloaded}} = -16.821\,\text{N}\cdot\text{m}$
   - Loaded holding torque: $\tau_{3,\text{loaded}} = -23.020\,\text{N}\cdot\text{m}$
   - Torque delta: $\Delta \tau_3 = -6.199\,\text{N}\cdot\text{m}$
   - Forearm lever arm: $L_3 \approx 0.45\,\text{m}$
   - Estimated payload force:
     $$F_z = \frac{|\Delta \tau_3|}{L_3} = \frac{6.20\,\text{N}\cdot\text{m}}{0.45\,\text{m}} \approx 13.78\,\text{N}$$
   - Estimated payload mass:
     $$m = \frac{13.78\,\text{N}}{9.81\,\text{m/s}^2} \approx 1.40\,\text{kg} \ (1400\,\text{g})$$

3. **Release Signature:**
   - In `recording_20261029_124044.csv` at sample index 60 ($t = 13.8\,\text{s}$), the payload was released.
   - The torque immediately transitioned from $-20.73\,\text{N}\cdot\text{m} \to -16.821\,\text{N}\cdot\text{m}$, returning precisely to the baseline resting state observed in Recording 3.
