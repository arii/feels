# feels

**Proprioceptive Dynamic Identification of Sealed Container Payloads**
Powered by the [Viam Python SDK](https://github.com/viamrobotics/viam-python-sdk) and [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search).

> [!IMPORTANT]
> **Key Architecture Decisions & Operational Focus:**
> - **Motion Routine:** Fixed **Pick Up $\to$ Swirl $\to$ Place** sequence from a calibrated taped table square.
> - **Operational Focus:** **MongoDB Atlas feature extraction** (`feature_extractor.py`) paired with an intuitive **Web UX** (`realtime_plot.py`) for live detection reporting, real-time telemetry streaming, and online recording.
> - **Archived Approaches:** Pushing / table sliding, violent linear shakes, tilt sweeps, dynamic vision grasping, and on-site deep learning training are officially ARCHIVED.

---

## Quickstart

No hardware needed for the first run — mock mode serves the full UI on bundled recordings.

```bash
bash setup.sh                  # venv + deps + .env + sanity check (run as `bash`, not `./`)
source .venv/bin/activate      # every new shell

FEELS_MOCK=1 python realtime_plot.py --port 8080   # open http://localhost:8080
```

Mode 1 identifies a mystery cup (real extractor + local k-NN); Mode 2 teaches. A purple **MOCK** badge is always shown — never demo it as live hardware.

With hardware: fill in `.env` (Viam + Atlas credentials; the Atlas IP allowlist must include your machine), then:

```bash
python realtime_plot.py --port 8080
```

## Repo map

```
setup.sh               # one-command dev setup (venv, deps, .env, sanity check)
realtime_plot.py       # dashboard: 2-mode web UI + /docs presentation site
feature_extractor.py   # 10-D swirl signature z + unit embedding e
atlas.py               # Atlas teach/identify, predict_lda decision head, --smoke setup
teach_csv.py           # batch-teach recording CSVs into Atlas (no robot needed)
predict_mystery.py     # offline classifier + tare QC (--eval, --qc)
repeatable_grasp.py    # data-collection macro: pick → swirl → place (LOCKED — do not modify)
control_service.py     # 100 Hz on-robot module (eliminates ~158 ms cloud latency)
static/index.html      # dashboard frontend
static/docs/           # presentation site (served at /docs — this IS the demo material)
recordings/            # telemetry CSVs
docs/                  # long-form docs: architecture, plan, findings
```

Every script documents itself: `python <script>.py --help` prints examples.

## The loop

1. **Record** — `python repeatable_grasp.py --object water_075pct --record --swirl-smooth`
   (one continuous pick → swirl → place; see `/docs/recording` for the protocol)
2. **Teach** — live in Mode 2 (preset chips), or batch with no robot:
   ```bash
   python teach_csv.py recordings/ --require-swirl --dry-run   # preview first
   python teach_csv.py recordings/ --require-swirl --session fri
   ```
3. **Predict** — Mode 1 runs the macro on a mystery cup: predicted category, material,
   closest taught item, feature-space panel, and torque-residual "why" charts.
4. **Correct** — "Wrong? Teach the correction" in the result card reuses the just-recorded
   CSV under the true label. A wrong prediction is the most valuable training signal.

Swirl-protocol consistency matters: only recordings with a true SWIRL phase enter the
library (`--require-swirl` enforces this). The embedding encodes the motion as much as
the contents — the same cup scores ~0.4 cosine across protocols vs ~0.9 for different
items within one protocol.

## Configuration

| Variable | Used by | Default |
|---|---|---|
| `MONGODB_URI` | live Atlas access | none — clear error if missing |
| `MONGODB_DATABASE` | live Atlas access | `physical_ai_db` |
| `FEELS_MOCK=1` | mock dashboard (no robot/Atlas) | off |
| `FEELS_MACRO_CMD` | custom data-collection macro | `repeatable_grasp.py --object {label} --record --cycles 1 --swirl-smooth` |
| `VIAM_*` | robot connection | see `.env.example` |

## Troubleshooting

- **`MONGODB_URI not set`** — `cp .env.example .env` and fill in credentials (`setup.sh` does the copy).
- **Atlas connection hangs / timeouts** — the cluster's IP allowlist must include your machine's public IP. This is the most common gotcha on new machines.
- **`teach_csv.py` skips everything** — `--require-swirl` only accepts recordings with a true SWIRL phase; run `--dry-run` to see per-file tare flags and skip reasons.
- **Dashboard shows "vector space unavailable"** — Atlas isn't reachable; check `MONGODB_URI` and the allowlist.
- **Predictions all look alike (~0.97)** — stale feature scales; recalibrate: `python feature_extractor.py --calibrate --require-swirl`, paste into `FEATURE_SCALES`, and re-teach with `--force`.
- **`./setup.sh` → permission denied** — run `bash setup.sh`; the exec bit doesn't always survive syncing.

---

## Overview

The **feels** system enables an industrial robotic arm (UFactory xArm850) to determine the contents of an opaque, covered coffee cup (water, viscous soap, loose candy, rigid mass, or empty tare) using **proprioceptive torque telemetry** without needing line-of-sight vision into the container.

### End-to-End Pipeline
1. **Fixed Pick & Place:** Approaches a taped table square, grasps the cup, lifts to travel altitude, executes a smooth horizontal swirl, and places the cup back down.
2. **Dynamic Swirl Excitation:** Horizontal circular orbital swirl induces fluid vortex lag, loose candy chatter, and rotational inertia without spillage.
3. **Deterministic Feature Extraction:** Extracts a 10-dimensional explicit physical feature vector $z$ and calculates a unit-norm embedding $e \in \mathbb{S}^9$.
4. **MongoDB Atlas Vector Memory:** Queries the Atlas `cup_dynamics_index` via `$vectorSearch` for instant k-NN classification and online item registration. A shrinkage-LDA head (refit in milliseconds per prediction) makes the final call — see `/docs/approach`.
5. **Real-Time Operator UX:** Web dashboard with 6-joint torque sparklines, the 2-mode predict/teach workflow, verdict cards with feature-space and "why" physics panels, and a 2-D PCA vector-space view of the signature library.

## Hardware & Environment

- **Target Machine:** `sfhack12` (`sfhack12-main.ty1cywh7tq.viam.cloud`)
- **Manipulator:** UFactory xArm850 (6-DOF)
- **Gripper:** UFactory parallel-jaw gripper
- **Sensors:** Actuator strain/current load sensing via `arm.do_command({"load": ""})`
- **Reference Camera:** Intel RealSense D435 (contextual / monitoring)

The Viam connection pattern is in [`main.py`](main.py) (`RobotClient.at_address` with API key).

## Zero-Latency On-Robot Control Service

To eliminate the ~158 ms WebRTC cloud latency and sample at 50–200 Hz, deploy [`control_service.py`](control_service.py) (`hackathons:control_logic:generic-service`) directly onto the robot host (`sfhack12`). Test with:

```bash
python client_control_service.py
```

## Experimental Findings: Weight Detection via Joint Torques

> **Note (2026-10-29 audit):** static hold torques were later found identical
> across contents to 3 decimals — the tare baseline dominates these numbers,
> not the payload. Treat the mass estimates below as pre-audit observations;
> the discriminative signal in the current pipeline comes from swirl *dynamics*,
> not statics. See [`BASELINE_FINDINGS.md`](BASELINE_FINDINGS.md).

Even without a dedicated 6-axis F/T flange sensor, the xArm850 reports high-resolution joint loads via `arm.do_command({"load": ""})`.

1. **Posture Dynamics:**
   - In upright postures ($q_2 \approx 86^\circ$), the upper arm acts as an axial compression column, so vertical weight creates minimal angular moment on Joint 2 (Shoulder).
   - **Joint 3 (Elbow Pitch)** acts as the cantilever reaction pivot and absorbs the downward bending moment.
2. **Experimental Verification (Recordings 3 vs 4):**
   - **Empty / Resting State:** Joint 3 settles firmly to `-16.82 N·m`.
   - **Holding Payload:** Joint 3 deflects to **`-22.0` to `-23.02 N·m`** (a **`-6.20 N·m` shift**, equivalent to $\approx 1.4\,\text{kg}$ downward force).
   - When released, torque crisply snaps back from $-20.7 \to -16.82\,\text{N}\cdot\text{m}$.
   - CSV datasets, schema dictionary, and comparative figures are documented in [`recordings/README.md`](recordings/README.md).

## Repeatable Grasping Workflow (Arm Saver Positions)

Using the configured `erh:vmodutils:arm-position-saver` switches on `sfhack12`, automated pick-and-place trajectories can be reliably commanded with synchronized gripper open/close operations:

| Waypoint Switch | Description | X (mm) | Y (mm) | Z (mm) | Orientation |
|---|---|---|---|---|---|
| **`home-pose`** | Ready / staging position | 466.27 | 383.55 | 250.57 | $\theta=178.4^\circ, (0.421, -0.904, -0.072)$ |
| **`approach`** | Pre-grasp alignment above object | 370.96 | 320.94 | 66.45 | $\theta=174.0^\circ, (0.548, -0.834, -0.059)$ |
| **`grasp`** | Target contact / grip position | 446.58 | 239.78 | 70.42 | $\theta=175.7^\circ, (0.381, -0.922, -0.068)$ |
| **`travel`** | High transit clearance waypoint | 562.62 | 331.21 | 196.26 | $\theta=179.5^\circ, (0.345, -0.938, 0.022)$ |
| **`place`** | Deposit / placement target | 647.13 | 193.54 | 96.17 | $\theta=177.8^\circ, (0.715, -0.684, -0.147)$ |

```bash
python repeatable_grasp.py --cycles 3 --delay 1.5   # repeated cycles
python repeatable_grasp.py --step grasp             # step to a waypoint (dry-run with --dry-run)
python repeatable_grasp.py --gripper grab           # manual gripper actuation
```

## Signature Library

Signatures are taught from swirl-protocol recordings only
(`teach_csv.py --require-swirl`; the dashboard records with `--swirl-smooth`).
The confound set is the key demo: cups at the same mass, separable only
by dynamics — mass features alone cannot distinguish them.

| Object Label | Expected dynamic cue |
|--------------|----------------------|
| `empty_cup` | Baseline tare; minimal swirl variance |
| `water_075pct` | Fluid slosh; strongest at 3/4 fill (most separable item in the dataset) |
| `solid_020g` / `solid_050g` | Rigid, quiet; reliably classified |
| `screws_100g` | Dense shift-and-rattle dynamics |
| `solid_150g` / `water_150g` / `candy_150g` / `soap_150g` | Same-mass confound set — mass features can't separate these; dynamics must |

## Project Documentation Index

- **`/docs`** (served by the dashboard) — the presentation site: problem, approach, force, relevance, figures, recording guide. This is the demo material.
- [**docs/00-overview.md**](docs/00-overview.md) — Problem statement, core scope, and architecture decisions.
- [**docs/01-hardware-and-viam-setup.md**](docs/01-hardware-and-viam-setup.md) — Robot hardware specifications and Viam configuration.
- [**docs/02-system-architecture.md**](docs/02-system-architecture.md) — End-to-end tech stack, telemetry sync, and storyboard.
- [**docs/05-ml-pipeline.md**](docs/05-ml-pipeline.md) — 10D feature extraction, metric embedding, and Atlas Vector Search.
- [**BASELINE_FINDINGS.md**](BASELINE_FINDINGS.md) — The tare-baseline audit: why statics don't discriminate.
- [**ARCHITECTURE.md**](ARCHITECTURE.md) — Module map and the CSV schema contract (the reuse interface).
- [**PLAN.md**](PLAN.md) — Concrete finish plan and verification steps.
- [**TODO.md**](TODO.md) — Active task backlog and archived exploration avenues.
