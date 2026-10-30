# feels

A 6-DOF arm holding a sealed cup uses proprioception to classify its contents
(empty, water, loose candy, or a rigid weight). The robot swirls the cup; the
system maps joint-torque dynamics to a 10-D signature and retrieves the nearest
taught example from MongoDB Atlas.

## TL;DR

```bash
git clone https://github.com/arii/feels && cd feels
bash setup.sh && source .venv/bin/activate
FEELS_MOCK=1 python realtime_plot.py --port 8080   # open http://localhost:8080
```

No hardware needed — mock mode runs the full UI on bundled recordings.
For the robot + Atlas, fill in `.env` (see Setup) and drop `FEELS_MOCK=1`.

**Dataset:** 106 labeled swirl recordings on Hugging Face —
[`dhyuti-n/feels-cup-dynamics`](https://huggingface.co/datasets/dhyuti-n/feels-cup-dynamics).
Pull and teach in one step:

```bash
pip install huggingface_hub
python tools/pull_from_hf.py --teach
```

## Setup

```bash
git clone https://github.com/arii/feels && cd feels
bash setup.sh                  # venv + deps + .env + sanity check (run as `bash`, not `./`)
source .venv/bin/activate      # every new shell
```

`setup.sh` creates `.venv/`, installs `requirements.txt`, and copies
`.env.example` to `.env`. Fill in `.env` for hardware + Atlas access:

| Variable | Purpose |
|---|---|
| `MONGODB_URI` | Atlas connection string (IP allowlist must include your machine) |
| `VIAM_API_KEY_ID` / `VIAM_API_KEY` | robot connection |

## Run the dashboard

No hardware needed — mock mode serves the full UI on bundled recordings:

```bash
FEELS_MOCK=1 python realtime_plot.py --port 8080   # open http://localhost:8080
```

A purple **MOCK** badge marks mock mode. With hardware and `.env` filled in:

```bash
python realtime_plot.py --port 8080
```

## Teach (Mode 2)

Pick Name + Amount from the dropdowns (the dashboard composes the label,
e.g. water + half → `water_half`), then **RUN SWIRL & SAVE TO ATLAS**. The
dropdowns replace hand-typed labels; a custom-label field sits behind a
disclosure as an escape hatch. Each teach runs the fixed swirl macro and
stores one 10-D signature. Teaching takes effect on the next prediction —
teaching is the training step.

Acceptance: the `SWIRL` phase is present and the baseline flags are clean.
The tool refuses the rest.

## Predict (Mode 1)

**RUN SWIRL MACRO & PREDICT** runs the macro on the mystery cup. The decision
card shows the predicted category, material, closest taught item, the 10-D
feature panel, and torque-residual "why" charts. No robot handy: switch to
**Simulated replay** to stream a recording through the same pipeline.

"Wrong? Teach the correction" reuses the just-recorded run under the true
label — a wrong prediction is the most valuable training signal.

## CLI reference

Batch and offline tools live on the [API page](/docs/api) (served by the
dashboard at `/docs/api`): `teach_csv.py` flags, `feature_extractor.py`,
`atlas.py` functions, and the HTTP endpoints. Every script also documents
itself: `python <script>.py --help`.

## Repo map

```
setup.sh               # one-command dev setup (venv, deps, .env, sanity check)
realtime_plot.py       # dashboard: 2-mode web UI + /docs presentation site
feature_extractor.py   # 10-D swirl signature z + unit embedding e
atlas.py               # Atlas teach/identify, predict_lda decision head
teach_csv.py           # batch-teach recording CSVs into Atlas (no robot needed)
predict_mystery.py     # offline classifier + baseline QC (--eval, --qc)
repeatable_grasp.py    # data-collection macro: pick → swirl → place → back-off (LOCKED)
static/index.html      # dashboard frontend
static/docs/           # presentation site (served at /docs)
recordings/            # telemetry CSVs
docs/                  # long-form docs: architecture, plan, findings
```

## Troubleshooting

- **`MONGODB_URI not set`** — fill in `.env` (`setup.sh` creates it from `.env.example`).
- **Atlas connection hangs** — the cluster's IP allowlist must include your machine's public IP.
- **`teach_csv.py` skips everything** — `--require-swirl` only accepts recordings with a true SWIRL phase; `--dry-run` shows per-file skip reasons.
- **Dashboard shows "vector space unavailable"** — Atlas isn't reachable; check `MONGODB_URI` and the allowlist.
- **`./setup.sh` → permission denied** — run `bash setup.sh`; the exec bit doesn't always survive syncing.
