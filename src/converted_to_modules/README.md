# Converted to Viam modules

These standalone scripts have been **superseded by deployed Viam modules**. They're
kept here for reference/history; the live functionality now runs on-robot as a
module and is used over `do_command` (see `src/use_modules.py`).

| archived script | replaced by (module model) | code |
|---|---|---|
| `control_service.py` — pre-module on-robot telemetry + onboard features (Generic Service `hackathons:control_logic:generic-service`) | **`braveorangerover:sensing:torque-features`** | `src/torque_service.py` |
| `client_control_service.py` — test client for the above | **`src/use_modules.py`** (client for all 3 modules) | `src/use_modules.py` |

## The current module set (one Viam module, 3 models)
Defined under `src/` and packaged via `meta.json` + `run.sh` + `src/module_main.py`:

| model | code | role |
|---|---|---|
| `braveorangerover:pick_logic:yolo3d-pick` | `src/pick_service.py` (wraps `src/vision_grasp.py`) | vision object picking |
| `braveorangerover:sensing:torque-features` | `src/torque_service.py` | on-robot torque sampling + pose-normalized force/weight + FFT |
| `braveorangerover:content:classifier` | `src/content_service.py` (wraps `src/train_classifier.py`) | solid / liquid / empty from the torque cascade |

Use them:
```bash
python src/use_modules.py all                 # status of all three
python src/use_modules.py telemetry           # torque-features
python src/use_modules.py predict <rec.csv>   # content:classifier
python src/use_modules.py detect              # yolo3d-pick (no motion)
```

> Note: `vision_grasp.py`, `train_classifier.py`, `feature_extractor.py` etc. are
> **not** archived — the modules *wrap* them (and the dashboard/CLI still use them),
> so they stay in `src/`.
