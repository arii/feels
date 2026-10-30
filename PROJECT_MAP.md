# 🎃 feels — Project Map

*What's in the cup?* — a robot picks up a lidded cup, swirls it, and infers the
contents from proprioception + an Atlas vector memory. This file is the map of the
codebase: every module, grouped by purpose, with how to run it and how to reuse it.

```
feels/
├─ src/            all application modules (grouped by purpose below)
│    ├─ Robot control & motion   repeatable_grasp, safe_pick, vision_grasp, control_service, main, sensors
│    ├─ ML / classification      feature_extractor, train_classifier, predict_content, state_spectrum, predict_mystery
│    ├─ Atlas vector memory      atlas, openset, teach_csv, push_to_atlas
│    ├─ Dashboard                realtime_plot (+ ../static/)
│    └─ Data capture             record_dataset, split_dataset
├─ tools/          one-off probes & utilities (standalone, no deps on src/)
├─ static/         dashboard UI (index.html)
├─ recordings/     115 labeled telemetry CSVs + figures
├─ models/         trained cascade + archived exploration (see models/README.md)
├─ notebooks/      honest leak-free analysis notebooks
├─ analysis/       offline analysis scripts & plots
└─ docs/           design docs
```

Run everything **from the repo root** as `python src/<file>.py` (or `python
tools/<file>.py`). The `src/` modules import each other by name (`import atlas`, …)
so they live together in one package dir; self-contained one-offs are in `tools/`.

---

## 🤖 Robot control & motion
| module | what it does | reuse |
|---|---|---|
| `repeatable_grasp.py` | Pick → swirl → place macro via Viam arm-position-saver switches, with labeled telemetry recording + optional smooth swirl | the **core robot routine**; `--record --object <label>` builds the dataset |
| `safe_pick.py` | Vision-gated, grasp-verified pickup (wraps `repeatable_grasp` primitives) | drop-in safe pick on top of the switch waypoints |
| `vision_grasp.py` | Vision-guided 3-D side grasp with contact detection | reusable grasp planner (`CupDetector`, `SideGraspPlanner`) |
| `control_service.py` | On-robot Viam Generic Service for 100 Hz telemetry + onboard features | deploy to the robot to beat cloud latency |
| `client_control_service.py` | Test client for the on-robot service over gRPC | — |
| `main.py` | CLI: `identify` / `teach` / `tare` / `status` | top-level entry point |
| `sensors.py` | Full component inventory + live sensor readout | `python src/sensors.py` to inspect any machine |

```bash
python src/repeatable_grasp.py --record --object water_050pct   # one labeled cycle
python src/repeatable_grasp.py --swirl --swirl-smooth           # smooth swirl agitation
python src/repeatable_grasp.py --step swirl                     # test a single waypoint
```

## 🧠 ML / classification
| module | what it does | reuse |
|---|---|---|
| `feature_extractor.py` | 10-D swirl feature vector + L2-normalized embedding for Atlas | the shared embedding used everywhere |
| `train_classifier.py` | 2-stage **cascade** (empty→loaded, solid↔liquid), **leak-free** eval | `--save` writes `models/cascade.joblib` |
| `predict_content.py` | Run the saved cascade on one recording → class + confidence | one-shot inference |
| `state_spectrum.py` | Represent contents as a 2-D **state spectrum** (loadedness × rigidity) + class-filtered Atlas retrieval | the honest, interpretable output |
| `predict_mystery.py` | Offline k-NN classifier (used by the dashboard's mock mode) | runs with no robot/Atlas |

```bash
python src/train_classifier.py            # leak-free evaluation of every stage
python src/train_classifier.py --save     # refit on all data → models/cascade.joblib
python src/predict_content.py recordings/recording_..._water_025pct.csv
python src/state_spectrum.py <recording.csv> --neighbors
```

## 🔮 Atlas vector memory (open-set learning)
| module | what it does | reuse |
|---|---|---|
| `atlas.py` | Mongo Atlas connection, collections, `teach()` / `identify()` (material-first + ambiguity guard) | the vector-memory API |
| `openset.py` | **Teach-and-recognize** loop: novelty detection + one-shot teach (no retrain) | add new items live |
| `teach_csv.py` | Batch-teach recording CSVs into Atlas (no robot needed) | seed the memory offline |
| `push_to_atlas.py` | Upload raw telemetry recordings into the Atlas `runs` collection | raw-data archive |

```bash
python src/teach_csv.py recordings/ --session thu    # seed signatures
python src/openset.py demo <recording.csv> --label oat_milk   # novelty → teach → recognize
python src/openset.py stats                           # what the memory knows
```

## 📊 Dashboard (operator UI)
| module | what it does |
|---|---|
| `realtime_plot.py` + `static/index.html` | 🎃 Halloween-themed live dashboard: all-6-joint torque graph, coarse→fine classify (infer/teach), Atlas feature-space PCA, existing-classes reference |

```bash
python src/realtime_plot.py --port 8080     # → http://localhost:8080
```

## 🎞️ Data capture & recording
| module | what it does |
|---|---|
| `record_dataset.py` | Multi-channel, multi-rate dataset recorder |
| `split_dataset.py` | Stratified train/test split (writes `recordings/split/`) |
| `recordings/` | 115 labeled CSVs + figures |

## 🛠️ tools/ (standalone, no local deps)
| tool | what it does |
|---|---|
| `tools/probe_vision.py` | Read-only: inspect a vision service's detections |
| `tools/probe_switches.py` | Read-only: inspect the arm-position-saver switches |
| `tools/capture_cup.py` | Capture camera frames into a Viam dataset (for detector fine-tuning) |
| `tools/telemetry_move.py` | Laptop-side single-joint telemetry movement (no module deploy) |
| `tools/fake_run.py` | Offline synthetic run generator (feature pipeline, no hardware) |

```bash
python tools/probe_vision.py --vision yolo
python tools/fake_run.py identify --label water_half    # fully offline
```

---

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in Viam + MongoDB creds
python src/realtime_plot.py --port 8080
```

## Reusing this for your own robot
- **Swap the arm:** the motion routine uses Viam's arm + arm-position-saver
  switches — re-record the waypoints and `repeatable_grasp.py` works unchanged.
- **Swap the embedding:** `feature_extractor.extract_features(df)` is the single
  seam — return any vector and the Atlas memory + classifier flow through it.
- **Swap the memory:** `atlas.teach()/identify()` is the only Mongo surface.
- **Offline first:** `tools/fake_run.py` + `predict_mystery.py` run the whole ML
  path with no robot or cloud, so you can develop without hardware.

## More docs
- `README.md` — full setup + experimental findings
- `ARCHITECTURE.md` — system architecture
- `BASELINE_FINDINGS.md` — the honest result (what proprioception can/can't resolve)
- `models/README.md` — the classifier + archived modeling exploration
