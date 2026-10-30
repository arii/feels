# Archived scripts

Exploration and one-off scripts from the hackathon, superseded by the
swirl-macro pipeline. Kept for reference; not part of the demo loop and not
imported by any active module.

| Script | What it was |
|---|---|
| `capture_cup.py` | Early cup-capture helper |
| `fake_run.py` | Synthetic fake-run generator (pre-hardware) |
| `probe_switches.py` / `probe_vision.py` | Viam switch / vision probing |
| `record_dataset.py` | Early dataset collection (added auto-discovered channels; superseded by `repeatable_grasp.py --record`) |
| `safe_pick.py` | Pick experiments |
| `sensors.py` | Sensor polling experiments |
| `telemetry_move.py` | Telemetry + motion experiments |

The active pipeline is: `repeatable_grasp.py` → `feature_extractor.py` →
`atlas.py` / `teach_csv.py` → `realtime_plot.py`. See the repo README.
