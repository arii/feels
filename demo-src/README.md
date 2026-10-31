# demo-src

Alternative approaches explored during the hackathon. The primary vector-search
approach lives at the repo root (`atlas.py`, `feature_extractor.py`, `embedder.py`).

## Contents

### `../src/` (Sree's reorg)

Sree Dhyuti Nimmagadda's restructured version with Viam modules:
- `src/torque_service.py`, `src/pick_service.py` — Viam module implementations
- `src/predict_content.py` — classifier-based prediction (vs vector retrieval)
- `src/main.py` — alternative orchestration

This was the second demo shown at the hackathon. Preserved as-is for reference.

### Viam modules

See `../sree_modules/` on the `feat/swirl-extractor-dashboard` branch for the
original Viam module code (torque polling at 50 Hz, grasp service).

## Note

These are hackathon explorations, not the maintained path. For the vector
approach, see the root README.
