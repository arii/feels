# feels

**Proprioceptive Dynamic Identification of Sealed Container Payloads**  
Powered by the [Viam Python SDK](https://github.com/viamrobotics/viam-python-sdk) and [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search).

---

> [!IMPORTANT]
> **Key Architecture Decisions & Operational Focus:**
> - **Motion Routine:** Fixed **Pick Up $\to$ Swirl $\to$ Place** sequence from a calibrated taped table square.
> - **Operational Focus:** **MongoDB Atlas feature extraction** (`feature_extractor.py`) paired with an intuitive **Web UX** (`realtime_plot.py`) for live detection reporting, real-time telemetry streaming, and online recording.
> - **Archived Approaches:** Pushing / table sliding, violent linear shakes, tilt sweeps, dynamic vision grasping, and on-site deep learning training are officially ARCHIVED.

---

## Overview

The **feels** system enables an industrial robotic arm (UFactory xArm850) to determine the contents of an opaque, covered coffee cup (water, viscous soap, loose candy, rigid mass, or empty tare) using **proprioceptive torque telemetry** without needing line-of-sight vision into the container.

### End-to-End Pipeline
1. **Fixed Pick & Place:** Approaches a taped table square, grasps the cup, lifts to travel altitude, executes a smooth horizontal swirl, and places the cup back down.
2. **Dynamic Swirl Excitation:** Horizontal circular orbital swirl induces fluid vortex lag, loose candy chatter, and rotational inertia without spillage.
3. **Deterministic Feature Extraction:** Extracts a 10-dimensional explicit physical feature vector $z$ and calculates a unit-norm embedding $e \in \mathbb{S}^9$.
4. **MongoDB Atlas Vector Memory:** Queries the Atlas `cup_dynamics_index` via `$vectorSearch` for instant k-NN classification and online item registration.
5. **Real-Time Operator UX:** A high-performance web dashboard displaying 10-second rolling force/weight plots, 6-joint sparklines, active cycle phase badges, and live classification verdict cards.

---

## Hardware & Environment

- **Target Machine:** `sfhack12` (`sfhack12-main.ty1cywh7tq.viam.cloud`)
- **Manipulator:** UFactory xArm850 (6-DOF)
- **Gripper:** UFactory parallel-jaw gripper
- **Sensors:** Actuator strain/current load sensing via `arm.do_command({"load": ""})`
- **Reference Camera:** Intel RealSense D435 (contextual / monitoring)

---

## Getting Started

### 1. Virtual Environment Setup

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

Ensure `.env` contains your Viam and MongoDB Atlas credentials:

```env
VIAM_API_KEY=your_api_key_here
VIAM_API_KEY_ID=your_api_key_id_here
VIAM_MACHINE_ADDRESS=sfhack12-main.ty1cywh7tq.viam.cloud
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
MONGODB_DATABASE=physical_ai_db
```

### 3. Launch the Operator Web Dashboard & Visualizer

Launch the high-performance visualizer:

```bash
python realtime_plot.py --port 8080
```

Open **`http://localhost:8080`** in your browser to:
- Monitor live payload weight and vertical force in real time.
- View individual 6-joint actuator torque sparklines.
- Click **"▶ Run Pick & Place Cycle"** to trigger the automated pick $\to$ swirl $\to$ place sequence.
- Inspect the **Detection Reporting Card** to see the predicted object, confidence score, and top-3 Atlas matches.
- Click **"Teach / Save to Atlas"** to register a new calibrated item into Atlas memory on the fly.
- Download recorded CSV datasets from the dataset drawer.

### 4. CLI Headless Execution

To run the repeatable pick-and-place swirl sequence from the terminal:

```bash
# Run one full pick up -> swirl -> place cycle with labeled telemetry recording
python repeatable_grasp.py --record --object water_half
```
### 6. Force & Weight Sensitivity Analysis

```bash
python effective_force.py --duration 15.0
```

### 7. Zero-Latency On-Robot Control Service (`control_service.py`)

To eliminate the ~158ms WebRTC cloud latency and sample at 50–200 Hz, deploy [`control_service.py`](control_service.py) (`hackathons:control_logic:generic-service`) directly onto the robot host (`sfhack12`). Test with:

```bash
python client_control_service.py
```

### 8. Swirl Feature Extractor + 2-Mode Dashboard

The demo loop is pick-up → swirl → place, driven by `repeatable_grasp.py --swirl-smooth` (locked in, don't touch it). Everything below consumes its CSVs.

**Feature extractor:**

```bash
# Extract the 10-D vector + normalized embedding from a recording
python feature_extractor.py recordings/recording_20261029_180332_empty_cup.csv

# Recalibrate per-feature scales on your dataset (run once on P0 data, then freeze)
python feature_extractor.py --calibrate
```

Outputs `z` (10 interpretable features: mass offsets, swirl variances, peak-to-peak, phase lag, rattle energy, grip compliance, lift transient, swirl speed), `e` (L2-normalized embedding matching the Atlas `cup_dynamics_index`: 10 dims, cosine), and `diagnostics` (sampling rate, tare QC flags, rattle-band limits). The 5–25 Hz rattle band is **not** observable at 4–5 Hz cloud sampling — the extractor flags `rattle_band_limited` honestly; the real band needs the 100 Hz on-robot module (step 7).

**Dashboard API (2 modes):**

```bash
python realtime_plot.py --port 8080   # open http://localhost:8080
```

- **Mode 1 — Identify:** `POST /api/cycle/predict` runs the swirl macro on a mystery cup, extracts features, queries Atlas `$vectorSearch`, returns the label + confidence + top 3 neighbors.
- **Mode 2 — Teach:** `POST /api/cycle/teach` with `{"item_name": "..."}` runs the macro and inserts the signature into Atlas — instantly queryable, no retraining.
- `POST /api/cycle/abort` stops a running macro; `GET /api/status` reports cycle state, Viam address, and Atlas config.

**Mock mode (no robot / no Atlas):**

```bash
FEELS_MOCK=1 python realtime_plot.py --port 8080
```

Serves the full 2-mode UI from recorded data: predict runs the real extractor + a local cosine k-NN over the reference recordings; teach simulates the save. A purple **MOCK** badge is always shown — never demo it as live hardware.

**Offline classifier:**

```bash
python predict_mystery.py                        # classify mystery recordings
python predict_mystery.py recordings/foo.csv     # classify one CSV
python predict_mystery.py --eval                 # leave-one-recording-out accuracy
python predict_mystery.py --qc                   # tare-baseline quality gate per run
```

See [`BASELINE_FINDINGS.md`](BASELINE_FINDINGS.md) for the audit behind the QC checks.

---

## Experimental Findings: Weight Detection via Joint Torques

Even without a dedicated 6-axis F/T flange sensor, the xArm850 reports high-resolution joint loads via `arm.do_command({"load": ""})`.

1. **Posture Dynamics:**
   - In upright postures ($q_2 \approx 86^\circ$), the upper arm acts as an axial compression column, so vertical weight creates minimal angular moment on Joint 2 (Shoulder).
   - **Joint 3 (Elbow Pitch)** acts as the cantilever reaction pivot and absorbs the downward bending moment.
2. **Experimental Verification (Recordings 3 vs 4):**
   - **Empty / Resting State:** Joint 3 settles firmly to `-16.82 N·m`.
   - **Holding Payload:** Joint 3 deflects to **`-22.0` to `-23.02 N·m`** (a **`-6.20 N·m` shift**, equivalent to $\approx 1.4\,\text{kg}$ downward force).
   - When released, torque crisply snaps back from $-20.7 \to -16.82\,\text{N}\cdot\text{m}$.
   - CSV datasets, schema dictionary, and comparative figures are documented in [`recordings/README.md`](recordings/README.md).

![Recordings 3 vs 4 Comparison](recordings/figures/recordings_3_vs_4.png)

---

## Baseline Code Sample

The baseline connection pattern in [`main.py`](main.py):

```python
import asyncio
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService

async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='<API-KEY>',
        api_key_id='<API-KEY-ID>'
    )
    return await RobotClient.at_address('sfhack12-main.ty1cywh7tq.viam.cloud', opts)

async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        # arm
        arm: Arm = Arm.from_robot(machine, "arm")
        arm_return_value = await arm.get_end_position()
        print(f"arm get_end_position return value: {arm_return_value}")

        # cam
        cam: Camera = Camera.from_robot(machine, "cam")
        cam_return_value = await cam.get_images()
        print(f"cam get_images return value: {cam_return_value}")

        # gripper
        gripper: Gripper = Gripper.from_robot(machine, "gripper")
        gripper_return_value = await gripper.is_moving()
        print(f"gripper is_moving return value: {gripper_return_value}")

        # table
        table: Gripper = Gripper.from_robot(machine, "table")
        table_return_value = await table.is_moving()
        print(f"table is_moving return value: {table_return_value}")

        # ceiling
        ceiling: Gripper = Gripper.from_robot(machine, "ceiling")
        ceiling_return_value = await ceiling.is_moving()
        print(f"ceiling is_moving return value: {ceiling_return_value}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Repeatable Grasping Workflow (Arm Saver Positions)

Using the newly configured `erh:vmodutils:arm-position-saver` switches on `sfhack12`, automated pick-and-place trajectories can be reliably commanded with synchronized gripper open/close operations:

### Configured Static Waypoints

| Waypoint Switch | Description | X (mm) | Y (mm) | Z (mm) | Orientation |
|---|---|---|---|---|---|
| **`home-pose`** | Ready / staging position | 466.27 | 383.55 | 250.57 | $\theta=178.4^\circ, (0.421, -0.904, -0.072)$ |
| **`approach`** | Pre-grasp alignment above object | 370.96 | 320.94 | 66.45 | $\theta=174.0^\circ, (0.548, -0.834, -0.059)$ |
| **`grasp`** | Target contact / grip position | 446.58 | 239.78 | 70.42 | $\theta=175.7^\circ, (0.381, -0.922, -0.068)$ |
| **`travel`** | High transit clearance waypoint | 562.62 | 331.21 | 196.26 | $\theta=179.5^\circ, (0.345, -0.938, 0.022)$ |
| **`place`** | Deposit / placement target | 647.13 | 193.54 | 96.17 | $\theta=177.8^\circ, (0.715, -0.684, -0.147)$ |

### Sequence Flow

```
[home-pose] (gripper open) 
     │
     ▼
 [approach] (pre-grasp alignment)
     │
     ▼
  [grasp]   ──► gripper.grab() (close jaws)
     │
     ▼
  [travel]  (lift & translate with payload at Z=196mm)
     │
     ▼
  [place]   ──► gripper.open() (deposit object at Z=96mm)
     │
     ▼
  [travel]  (retract upwards to avoid collision)
     │
     ▼
[home-pose] (gripper open, ready for next cycle)
```

### Usage

```bash
# Run 1 full pick-and-place cycle
python repeatable_grasp.py

# Run multiple repeated cycles
python repeatable_grasp.py --cycles 3 --delay 1.5

# Step to a specific static position
python repeatable_grasp.py --step home-pose
python repeatable_grasp.py --step approach
python repeatable_grasp.py --step grasp
python repeatable_grasp.py --step travel
python repeatable_grasp.py --step place

# Manual gripper actuation
python repeatable_grasp.py --gripper open
python repeatable_grasp.py --gripper grab

# Inspect waypoints and status without moving (dry-run)
python repeatable_grasp.py --dry-run
```

---

## Calibrated Object Library

| Object Label | Dynamic Physical Signature | Atlas Verdict |
|--------------|----------------------------|---------------|
| `empty_cup` | Zero mass offset ($\Delta \tau \approx 0$), zero swirl variance | 👻 Empty cup (tare) |
| `water_half` | Medium mass (~180g), dynamic swirl vortex lag | 💧 Water (fluid slosh) |
| `dish_soap_half` | Medium mass (~210g), heavy viscous swirl damping | 🫧 Dish soap (viscous fluid) |
| `candy_mixed` | Sharp impact chatter in 5–25 Hz rattle band | 🍬 Candy (trick-or-treat rattle) |
| `solid_weight` | High mass (~350g), rigid coupling, zero swirl lag | 🪨 Solid weight (rigid mass) |

---

## Project Documentation Index

- [**00-Overview**](docs/00-overview.md) — Problem statement, core scope, and architecture decisions.
- [**01-Hardware & Setup**](docs/01-hardware-and-viam-setup.md) — Robot hardware specifications and Viam configuration.
- [**02-System Architecture**](docs/02-system-architecture.md) — End-to-end tech stack, telemetry sync, and storyboard.
- [**03-Task Schedule**](docs/03-task-schedule.md) — Hackathon roadmap and execution checkpoints.
- [**04-Implementation Phases**](docs/04-implementation-phases.md) — Minimal viable demo, swirl routine, and what was cut.
- [**05-ML Pipeline**](docs/05-ml-pipeline.md) — 10D feature extraction, metric embedding, and Atlas Vector Search.
- [**06-Formal Problem Definition**](docs/06-formal-problem-definition.md) — Mathematical state, action space, and non-parametric metric retrieval.
- [**PLAN.md**](PLAN.md) — Concrete finish plan and verification steps.
- [**TODO.md**](TODO.md) — Active task backlog and archived exploration avenues.
