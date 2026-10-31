# Feels: Teaching a robot to feel

Sree Dhyuti Nimmagadda, Ariel Anders

"Feels" is an online data collection and training platform designed to teach a robot how to classify the contents of a sealed cup based on physical interaction. The robot picks up a cup, moves it in a swirling motion, and predicts whether the contents are liquid, candy, or screws. The challenge is that our robot does not have a force-torque sensor; instead, we use the applied joint torques for prediction and leverage MongoDB Atlas Vector Search to support the online classification of both new and existing categories.

## Who will use this

Our product is designed for any company deploying robotic manipulation in unstructured or generalized environments, such as warehouses, households, and hospitals. Recent advances in computer vision have expanded robots' manipulation capabilities; however, they can still make mistakes in the absence of contextual information. To fix this, businesses often add more sensors—such as force-torque, weight, or X-ray sensors—even when this information could be accessed without the expense of additional hardware. Our product provides a cheaper, faster alternative that enables robots to cultivate tactile "common sense" using their existing hardware.

## Tech stack

Viam Python SDK, gRPC, WebRTC, aiohttp, streaming telemetry via Server-Sent Events (SSE), MongoDB Atlas Vector Search, Hugging Face, pytorch, and scipy.

## Quickstart

```bash
git clone https://github.com/arii/feels && cd feels
bash setup.sh && source .venv/bin/activate
FEELS_MOCK=1 python realtime_plot.py --port 8080   # open http://localhost:8080
```

No hardware needed — mock mode runs the full UI on bundled recordings.
For the robot + Atlas, fill in `.env` (see Setup) and drop `FEELS_MOCK=1`.

**Dataset:** 106 labeled swirl recordings on Hugging Face —
[`dhyuti-n/feels-cup-dynamics`](https://huggingface.co/datasets/dhyuti-n/feels-cup-dynamics).

```bash
pip install huggingface_hub
python tools/pull_from_hf.py --teach
```

**Docs:** [arii.github.io/feels](https://arii.github.io/feels)

## How it works

1. **Swirl:** The robot picks up the cup, moves it in a swirling motion, and places it back.
2. **Extract:** Applied joint torques during the swirl map to a physical signature.
3. **Retrieve:** MongoDB Atlas Vector Search finds the nearest taught example, supporting online classification of both new and existing categories.

## Setup

```bash
bash setup.sh                  # venv + deps + .env + sanity check (run as `bash`, not `./`)
source .venv/bin/activate      # every new shell
```

`setup.sh` creates `.venv/`, installs `requirements.txt`, and copies
`.env.example` to `.env`. Fill in `.env` for hardware + Atlas access.

## Repo layout

| Path | Contents |
|---|---|
| `atlas.py`, `feature_extractor.py`, `embedder.py` | Core: vector memory, physics features, embeddings |
| `realtime_plot.py`, `repeatable_grasp.py` | Demo: dashboard + swirl macro |
| `tools/` | HF pull/push, library prune, teaching scripts |
| `docs/` | Static site (GitHub Pages) |
| `demo-src/` | Alternative approaches from the hackathon |

## License

MIT — see [LICENSE](LICENSE).
