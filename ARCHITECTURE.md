# ARCHITECTURE — how the pieces fit and how to reuse them

```
┌──────────┐   CSV    ┌───────────────────┐  10-D e  ┌───────────┐
│  robot   │ ───────► │ feature_extractor │ ───────► │   atlas   │
│  macro   │  schema  │  (pure function)  │  cosine  │  ($vector  │
└──────────┘  below   └───────────────────┘          │  Search)  │
     ▲                                              └─────┬─────┘
     │                                                    │ query
     │              ┌────────────────┐                    │
     └──────────────│ realtime_plot  ◄────────────────────┘
         run        │  (web server   │   teach / identify
                    │   + dashboard)│
                    └────────────────┘
```

## Modules

| File | Role | Reuse story |
|------|------|-------------|
| `feature_extractor.py` | `DataFrame → {z, e, diagnostics}`. Pure functions, no I/O, no robot. | Import it. Any arm that logs joint torques can use it — only the CSV columns matter. |
| `atlas.py` | `teach(label, vector)` / `identify(vector)` over MongoDB `$vectorSearch`. | Import it. The vector-memory pattern works for any embedding, not just cups. |
| `teach_csv.py` | Batch-ingest labeled CSVs into Atlas (`--require-swirl`, `--dry-run`, `--list`). | Point it at any directory of labeled recordings. |
| `predict_mystery.py` | Offline classifier + `--eval` / `--qc` (tare-quality gate). | Audit any batch of recordings without the robot. |
| `realtime_plot.py` | aiohttp server: SSE telemetry, `/api/cycle/*` endpoints, serves `static/index.html`. | Run it; swap the macro via `FEELS_MACRO_CMD`; edit the UI in `static/` without touching Python. |
| `static/index.html` | The full dashboard (HTML/CSS/JS). | Restyle or embed freely — no Python inside. |
| `repeatable_grasp.py` | The xArm pick → swirl → place → back-off macro. **Locked for the hackathon** — don't modify. | Replace wholesale via `FEELS_MACRO_CMD` (below). |

## The CSV schema contract (the real interface)

Any macro — ours or yours — plugs into the pipeline by appending a
`recording_*.csv` with these columns:

```
elapsed_sec, phase, object_label,
tcp_x_mm, tcp_y_mm, tcp_z_mm,
q1_deg … q6_deg, tau1_nm … tau6_nm,
gripper_pos, gripper_holding
```

`phase` tags each row (`INIT`, `APPROACH`, `GRASP`, `LIFT_TO_TRAVEL`, `SWIRL`,
`DESCEND_PLACE`, `GRIPPER_OPEN`, `BACK_OFF`, `RETRACT_TRAVEL`, `RETURN_HOME`, …).
The macro is pick → swirl → place → back-off (retreat clear of the cup, then
return home). The extractor needs a dynamic window: rows tagged `SWIRL`, or it
falls back to the travel phases. **Keep the protocol fixed** — the
embedding encodes the motion as much as the contents (same item scores ~0.4
cosine across protocols vs ~0.9 for different items within one).

## Reuse recipes

**Your own arm:** set `FEELS_MACRO_CMD` — a shell-style template with
`{label}`, `{cycles}`, `{here}` placeholders, e.g.
`FEELS_MACRO_CMD='python my_arm_macro.py --label {label} --record'`.
Your macro just needs to write a schema-conformant CSV; the dashboard,
extractor, and Atlas flow stay untouched.

**Your own features:** edit `FEATURE_NAMES` / `FEATURE_DESCRIPTIONS` /
`FEATURE_SCALES` in `feature_extractor.py`, then `python
feature_extractor.py --calibrate` on your data and freeze the scales
(rescaling invalidates taught signatures).

**Dashboard without our robot:** `FEELS_MOCK=1 python realtime_plot.py`
runs the full UI from recorded CSVs (purple MOCK badge; never demo as live).

**Vector memory standalone:** `atlas.teach("label", embedding_vector)` and
`atlas.identify(query_vector, top_k=3)` — needs `MONGODB_URI` and the
`cup_dynamics_index` (10 dims, cosine) from `python atlas.py`.
