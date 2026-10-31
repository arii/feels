# feels

A robot arm holding a sealed cup identifies the contents through touch. The arm swirls the cup; joint-torque dynamics map to a 10-D signature; MongoDB Atlas vector search retrieves the nearest taught example.

Built in 2 days at the Viam + MongoDB Robot Hackathon (Oct 29–30, 2026) by Ariel Anders and Sree Dhyuti Nimmagadda.

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

**Docs:** [arii.github.io/feels](https://arii.github.io/feels) — problem, approach, prediction, API.

## How it works

1. **Swirl:** The arm picks up the lidded cup, swirls it in a horizontal circle, and places it back. The motion is fixed and deterministic.
2. **Extract:** Joint torques during the swirl map to a 10-D physical signature (slosh variance, phase lag, rattle energy, etc.).
3. **Retrieve:** Atlas `$vectorSearch` finds the nearest taught signature. Teaching is the training step — one swirl per example.

Five classes: water, empty (gripper holding nothing), empty cup, screws, candy.

## Setup

```bash
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

```bash
FEELS_MOCK=1 python realtime_plot.py --port 8080   # mock mode, no hardware
python realtime_plot.py --port 8080                # live, needs .env + robot
```

## Teach (Mode 2)

Pick a class from the dropdown, then **Run swirl & save to Atlas**. Each teach runs the fixed swirl macro and stores one 10-D signature. Teaching takes effect on the next prediction.

## Predict (Mode 1)

**Run swirl macro & predict** runs the macro on the mystery cup. The decision card shows the predicted class, nearest neighbors, the 10-D feature panel, and torque-residual "why" charts. No robot handy: switch to **Simulated replay** to stream a recording through the same pipeline.

## Embeddings

`FEELS_EMBEDDER` selects the embedding (default `hand`):

| Value | Description |
|---|---|
| `hand` | 10-D hand-engineered physics features |
| `pca` | 10-D PCA on raw torque trajectories (no new deps) |
| `chronos` | 2304-D Chronos foundation model (`pip install torch chronos-forecasting`) |

The Atlas library is embedder-specific. Switching embedders requires rebuilding:
```bash
FEELS_EMBEDDER=pca python teach_csv.py --force recordings/
FEELS_EMBEDDER=pca python realtime_plot.py --port 8080
```

## Repo layout

| Path | Contents |
|---|---|
| `atlas.py`, `feature_extractor.py`, `embedder.py` | Core: vector memory, 10-D features, embeddings |
| `realtime_plot.py`, `repeatable_grasp.py` | Demo: dashboard + fixed swirl macro |
| `tools/` | HF pull/push, library prune, 5-class teach |
| `docs/` | Static site (GitHub Pages) |
| `demo-src/` | Alternative approaches from the hackathon (Viam modules, classifier) |

## License

MIT — see [LICENSE](LICENSE).
