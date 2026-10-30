#!/usr/bin/env python3
"""
realtime_plot.py — Ultra-performant 10-second rolling realtime visualizer
for xArm850 effective force, payload weight, and 6-joint telemetry.

Optimizations:
- Strict 10-second rolling window (discards old data to prevent memory/CPU bloat)
- requestAnimationFrame throttled rendering (zero UI lag)
- High-efficiency HTML5 Canvas polyline rendering for the 6 joints (<0.1ms per frame)
- Chart.js optimized (animation: false, tension: 0, pointRadius: 0)
- Start / Stop Session Recording with 1-click CSV download
"""

import asyncio
import os
import re
import time
import math
import json
import csv
import glob
import argparse
import subprocess
import sys
from datetime import datetime
from typing import Set, List, Optional, Dict, Any
from dotenv import load_dotenv

from aiohttp import web
from viam.robot.client import RobotClient
from viam.components.arm import Arm

import pandas as pd
import feature_extractor

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
GRAVITY = 9.80665
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))

def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=HERE, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"

GIT_SHA = _git_sha()

# ── pick-swirl-place cycle machinery ──────────────────────────────────────
_cycle_lock = asyncio.Lock()
_cycle_proc: Optional[subprocess.Popen] = None
_cycle_status: str = "idle"   # idle | running | done | error | aborted
_last_prediction: Optional[dict] = None   # last successful Mode-1 result (survives refresh)

# MOCK MODE: FEELS_MOCK=1 serves the 2-mode UI from recorded data instead of
# the physical robot + Atlas. The mock predict runs the real feature
# extractor on a recorded CSV and a local cosine k-NN over the reference
# recordings (same math Atlas $vectorSearch would do). The UI shows a MOCK
# badge whenever this is active — never demo it as live hardware.
MOCK_MODE = os.getenv("FEELS_MOCK", "0") == "1"
MOCK_DELAY_SEC = float(os.getenv("FEELS_MOCK_DELAY", "2.0"))


def _atlas():
    """Import atlas lazily so the dashboard boots without Mongo creds."""
    import atlas as _a
    if not os.getenv("MONGODB_URI"):
        raise RuntimeError("MONGODB_URI not set — Atlas teach/identify unavailable")
    return _a


def _vspace_2d(items):
    """2-D PCA projection of (label, 10-D vector) items via SVD. JSON-safe."""
    import numpy as np
    if not items:
        return []
    X = np.array([v for _, v in items], dtype=float)
    X = X - X.mean(axis=0)
    if len(items) < 2 or np.allclose(X, 0):
        return [{"label": l, "x": 0.0, "y": 0.0} for l, _ in items]
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return [{"label": l, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4)}
            for (l, _), p in zip(items, coords)]


async def handle_vspace(request):
    """2-D PCA of the Atlas signature library, optionally with the query
    vector projected in the same basis (POST {"query": [10 floats]})."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    query = data.get("query")
    try:
        a = _atlas()
        docs = list(a.get_db().signatures.find({}, {"label": 1, "feature_vector": 1}))
        items = [(d.get("label", "?"), d["feature_vector"])
                 for d in docs if d.get("feature_vector")]
        qidx = -1
        if isinstance(query, list) and len(query) == 10:
            items.append(("QUERY", [float(v) for v in query]))
            qidx = len(items) - 1
        return web.json_response({
            "ok": True, "points": _vspace_2d(items),
            "query_index": qidx, "n": len(items) - (1 if qidx >= 0 else 0),
        })
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)[:300]}, status=500)


def _newest_recording(after_ts: float) -> Optional[str]:
    cands = [p for p in glob.glob(os.path.join(RECORDINGS_DIR, "recording_*.csv"))
             if os.path.getmtime(p) >= after_ts - 2]
    return max(cands, key=os.path.getmtime) if cands else None


def _sanitize_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "item"


def _macro_cmd(object_label: str) -> list:
    """Build the data-collection macro command for a labeled run.

    Default: the locked-in repeatable_grasp.py swirl macro. Another robot
    builder swaps it without touching code via FEELS_MACRO_CMD, a shell-style
    template with {label}, {cycles}, {here} placeholders, e.g.:
        FEELS_MACRO_CMD='python my_arm_macro.py --label {label} --record'
    The macro must append one recording_*.csv to recordings/ (see
    ARCHITECTURE.md for the CSV schema contract).
    """
    tpl = os.getenv("FEELS_MACRO_CMD")
    if tpl:
        import shlex
        return [a.format(label=object_label, cycles=1, here=HERE)
                for a in shlex.split(tpl)]
    return [sys.executable, os.path.join(HERE, "repeatable_grasp.py"),
            "--object", object_label, "--record", "--cycles", "1", "--swirl-smooth"]


async def _run_swirl_macro(object_label: str) -> str:
    """Run one pick-up → swirl → place cycle via repeatable_grasp.py.

    Returns the path of the freshly recorded CSV. Raises on failure/abort.
    Only one cycle may run at a time (guarded by _cycle_lock by callers).
    """
    global _cycle_proc, _cycle_status
    _cycle_status = "running"
    cmd = _macro_cmd(object_label)
    started = time.time()
    try:
        _cycle_proc = subprocess.Popen(
            cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        loop = asyncio.get_event_loop()
        # communicate() drains stdout while waiting — wait()+read() would
        # deadlock once the pipe buffer (~64KB) fills on a chatty macro.
        out, _ = await loop.run_in_executor(None, _cycle_proc.communicate)
        rc = _cycle_proc.returncode
        _cycle_proc = None
        if rc != 0:
            _cycle_status = "error"
            raise RuntimeError(f"swirl macro exited rc={rc}\n{out[-2000:]}")
        csv_path = _newest_recording(started)
        if not csv_path:
            _cycle_status = "error"
            raise RuntimeError("macro finished but no new recording CSV appeared")
        _cycle_status = "done"
        return csv_path
    except Exception:
        _cycle_status = "error"
        raise
    finally:
        _cycle_proc = None


class RobotForceState:
    def __init__(self):
        self.baseline_loads: List[float] = [0.0, -16.0489, -16.8209, 0.778, 0.3728, 0.0]
        self.is_calibrated: bool = False
        self.r_xy_m: float = 0.1391
        self.pose = {"x": 133.7, "y": 38.5, "z": 485.4}
        self.connected_to_viam: bool = False

        # EMA smoothed torques
        self.ema_loads: Optional[List[float]] = None
        self.filter_alpha: float = 0.25

        # Session Recording
        self.is_recording: bool = False
        self.recording_start_time: float = 0.0
        self.recorded_samples: List[Dict[str, Any]] = []
        self.current_recording_file: Optional[str] = None

        self.latest_sample: Dict[str, Any] = {
            "t": 0.0,
            "fz": 0.0,
            "grams": 0.0,
            "delta_tau_j2": 0.0,
            "raw": [0.0, -16.05, -16.82, 0.78, 0.37, 0.0],
            "smooth": [0.0, -16.05, -16.82, 0.78, 0.37, 0.0],
            "deltas": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "r_xy": 0.1391
        }
        self.subscribers: Set[asyncio.Queue] = set()

    def tare(self, loads: List[float]):
        if loads and len(loads) >= 6:
            self.baseline_loads = list(loads)
            self.is_calibrated = True
            print(f"[TARE] Baseline recalibrated: J2={loads[1]:.4f} Nm")

    def start_recording(self, object_label: str = ""):
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recorded_samples = []
        clean_lbl = "".join(c for c in object_label if c.isalnum() or c in ("-", "_")).lower()
        self.current_object_label = clean_lbl or "unlabeled"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_recording_file = f"recording_{timestamp}_{self.current_object_label}.csv"
        print(f"[RECORD] Started session: {self.current_recording_file} (object: {self.current_object_label})")

    def stop_recording(self) -> Optional[str]:
        if not self.is_recording:
            return None
        self.is_recording = False
        filename = self.current_recording_file
        if filename and self.recorded_samples:
            filepath = os.path.join(RECORDINGS_DIR, filename)
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "elapsed_sec", "object_label", "effective_force_n", "payload_weight_g",
                    "delta_j2_nm", "r_xy_m",
                    "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw",
                    "j1_smooth", "j2_smooth", "j3_smooth", "j4_smooth", "j5_smooth", "j6_smooth"
                ])
                for s in self.recorded_samples:
                    writer.writerow([
                        s["t"], self.current_object_label, s["fz"], s["grams"],
                        s["delta_tau_j2"], s["r_xy"],
                        s["raw"][0], s["raw"][1], s["raw"][2], s["raw"][3], s["raw"][4], s["raw"][5],
                        s["smooth"][0], s["smooth"][1], s["smooth"][2], s["smooth"][3], s["smooth"][4], s["smooth"][5]
                    ])
            print(f"[RECORD] Saved {len(self.recorded_samples)} samples to {filepath}")
            return filename
        return None


state = RobotForceState()


async def connect_robot():
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def robot_polling_loop(target_delay_ms: int = 200):
    """
    Steady 4-5 Hz sequential polling to maintain 100% stable gRPC tunnel.
    """
    print(f"Connecting to Viam machine at {VIAM_MACHINE_ADDRESS}...")
    t0 = time.time()

    while True:
        try:
            state.connected_to_viam = False
            async with await connect_robot() as machine:
                arm = Arm.from_robot(machine, "arm")
                state.connected_to_viam = True
                print("[✓] Connected to arm. Telemetry streaming active...")

                try:
                    p = await arm.get_end_position()
                    state.pose = {"x": p.x, "y": p.y, "z": p.z}
                    state.r_xy_m = max(0.05, math.sqrt(p.x**2 + p.y**2) / 1000.0)
                except Exception as e:
                    print(f"[!] Warning reading arm pose: {e}")

                consecutive_errors = 0

                while True:
                    loop_start = time.time()
                    try:
                        query_start = time.time()
                        res = await arm.do_command({"load": ""}, timeout=3.0)
                        rtt_ms = round((time.time() - query_start) * 1000.0, 1)
                        loads = res.get("load", [])
                        consecutive_errors = 0

                        if loads and len(loads) >= 6:
                            now = time.time()
                            elapsed = round(now - t0, 2)

                            # EMA low-pass filtering on all 6 joints
                            alpha = state.filter_alpha
                            if state.ema_loads is None:
                                state.ema_loads = list(loads)
                            else:
                                for i in range(len(loads)):
                                    state.ema_loads[i] = alpha * loads[i] + (1.0 - alpha) * state.ema_loads[i]

                            # Auto-calibration on initial startup if not yet set
                            if not state.is_calibrated:
                                state.baseline_loads = list(state.ema_loads)
                                state.is_calibrated = True
                                print(f"[✓] Baseline calibrated. Shoulder: {state.baseline_loads[1]:.4f} Nm")

                            # Compute delta from baseline for all 6 joints
                            deltas = [round(state.ema_loads[i] - state.baseline_loads[i], 4) for i in range(6)]

                            # Stable Weight & Force estimation:
                            # Joint 2 (Shoulder) is the clean reference with zero dithering.
                            # We apply deadband filtering to reject sub-30g idle noise.
                            j2_smooth = state.ema_loads[1]
                            delta_tau_j2 = j2_smooth - state.baseline_loads[1]
                            
                            # Deadband to reject idle jitter
                            if abs(delta_tau_j2) < 0.02:
                                delta_tau_j2 = 0.0

                            effective_fz = -delta_tau_j2 / state.r_xy_m
                            grams = (effective_fz / GRAVITY) * 1000.0

                            sample = {
                                "t": elapsed,
                                "rtt_ms": rtt_ms,
                                "fz": round(effective_fz, 2),
                                "grams": round(grams, 1),
                                "delta_tau_j2": round(delta_tau_j2, 4),
                                "raw": [round(x, 3) for x in loads[:6]],
                                "smooth": [round(x, 3) for x in state.ema_loads[:6]],
                                "deltas": deltas,
                                "r_xy": round(state.r_xy_m, 3),
                                "is_recording": state.is_recording,
                                "rec_count": len(state.recorded_samples),
                                "connected": True
                            }
                            state.latest_sample = sample

                            if state.is_recording:
                                state.recorded_samples.append(sample)

                            # Push to SSE subscriber queues
                            for q in list(state.subscribers):
                                try:
                                    q.put_nowait(sample)
                                except asyncio.QueueFull:
                                    pass

                    except Exception as e:
                        consecutive_errors += 1
                        print(f"[!] Polling hiccup ({consecutive_errors}): {e}")
                        if consecutive_errors >= 3:
                            print("[!] Multiple timeouts, cycling gRPC client connection...")
                            break
                        await asyncio.sleep(0.5)

                    elapsed_turn = (time.time() - loop_start) * 1000.0
                    sleep_ms = max(50.0, target_delay_ms - elapsed_turn)
                    await asyncio.sleep(sleep_ms / 1000.0)

        except Exception as e:
            state.connected_to_viam = False
            print(f"[!] Reconnecting to Viam machine in 2.0s: {e}")
            await asyncio.sleep(2.0)


# ==============================================================================
# Web Server Routes (Ultra-lightweight HTML Dashboard + SSE Stream + CSV API)
# ==============================================================================

# Dashboard frontend lives in static/index.html (extracted for modularity —
# edit the HTML/JS/CSS there without touching this server).
def _load_index_html():
    p = os.path.join(HERE, "static", "index.html")
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return "<h1>dashboard missing: static/index.html not found</h1>"


HTML_PAGE = _load_index_html()


async def handle_index(request):
    return web.Response(text=HTML_PAGE, content_type="text/html")


async def handle_tare(request):
    loads = list(state.latest_sample["smooth"])
    state.tare(loads)
    return web.json_response({"ok": True, "baseline_j2": state.baseline_loads[1]})


async def handle_record_start(request):
    obj = request.query.get("object", "")
    if not obj and request.can_read_body:
        try:
            data = await request.json()
            obj = data.get("object", "")
        except Exception:
            pass
    state.start_recording(object_label=obj)
    return web.json_response({
        "ok": True,
        "file": state.current_recording_file,
        "object": state.current_object_label
    })


async def handle_record_stop(request):
    filename = state.stop_recording()
    return web.json_response({"ok": True, "file": filename, "samples": len(state.recorded_samples)})


async def handle_record_download(request):
    filename = request.query.get("file")
    if not filename:
        return web.Response(status=400, text="Missing file param")
    safe_name = os.path.basename(filename)
    filepath = os.path.join(RECORDINGS_DIR, safe_name)
    if not os.path.isfile(filepath):
        return web.Response(status=404, text="File not found")
    return web.FileResponse(filepath, headers={"Content-Disposition": f'attachment; filename="{safe_name}"'})


# ── 2-mode ML API ─────────────────────────────────────────────────────────
async def _mock_predict():
    """Mock Mode 1: real extractor + local cosine k-NN on recorded data."""
    import numpy as np
    import predict_mystery as pm
    await asyncio.sleep(MOCK_DELAY_SEC)
    recs = pm.load_real_recordings()
    if not recs:
        raise RuntimeError("mock mode: no recordings under recordings/")
    mystery = [(n, df) for n, df in recs.items()
               if str(df["object_label"].iloc[0]) in pm.KNOWN_LABEL_SKIP]
    name, df = mystery[0] if mystery else sorted(recs.items())[-1]
    ref, scaler = pm.build_reference(recs)
    k = min(3, len(ref))
    pred, conf, q, neigh, ndist = pm.predict(df, ref, scaler, k=k)
    e_q = np.array(feature_extractor.extract_features(df)["e"])
    matches = []
    for (_, row) in neigh.iterrows():
        e_r = np.array(feature_extractor.extract_features(recs[row["file"]])["e"])
        sim = float(np.dot(e_q, e_r))  # both L2-normalized → cosine similarity
        matches.append({"label": row["label"], "similarity": round(max(0.0, sim), 3)})
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    margin = (matches[0]["similarity"] - matches[1]["similarity"]) if len(matches) > 1 else 1.0
    ambiguous = margin < 0.02
    import atlas as _a
    _mat = {}
    for m in matches:
        _mat.setdefault(_a.material_of(m["label"]), []).append(m["similarity"])
    _mat_ranked = sorted(((k, sum(v) / len(v)) for k, v in _mat.items()),
                         key=lambda t: t[1], reverse=True)
    _best_mat, _best_mat_conf = _mat_ranked[0]
    _mat_margin = _best_mat_conf - _mat_ranked[1][1] if len(_mat_ranked) > 1 else 1.0
    _mat_ambiguous = _mat_margin < 0.02
    feats = feature_extractor.extract_features(df)
    ts = feature_extractor.swirl_window_series(df)
    return {
        "ok": True, "mock": True,
        "csv": os.path.basename(name),
        "prediction": str(pred),
        "confidence": float(round(float(conf), 3)),
        "low_confidence": bool(float(conf) < 0.82) or ambiguous,
        "ambiguous": ambiguous,
        "margin": round(margin, 4),
        "material": _best_mat,
        "material_confidence": round(_best_mat_conf, 3),
        "material_low_confidence": bool(_best_mat_conf < 0.82) or _mat_ambiguous,
        "material_ambiguous": _mat_ambiguous,
        "material_margin": round(_mat_margin, 4),
        "material_matches": [{"material": k, "score": round(s, 4)} for k, s in _mat_ranked],
        "matches": matches,
        "features": {"names": feature_extractor.FEATURE_NAMES,
                     "z": [round(float(v), 4) for v in feats["z"]],
                     "descriptions": feature_extractor.FEATURE_DESCRIPTIONS},
        "e": [float(v) for v in feats["e"]],
        "timeseries": ts,
        "diagnostics": feats["diagnostics"],
    }


async def _mock_teach(label: str):
    """Mock Mode 2: pretend to save a signature (nothing touches Atlas)."""
    import uuid
    await asyncio.sleep(MOCK_DELAY_SEC)
    return {"ok": True, "mock": True, "label": label, "csv": None,
            "sig_id": f"mock-{uuid.uuid4().hex[:8]}",
            "note": "mock mode — not saved to Atlas"}


async def handle_cycle_predict(request):
    """Mode 1: run the swirl macro on a mystery cup → features → Atlas $vectorSearch."""
    global _last_prediction
    if _cycle_lock.locked():
        return web.json_response({"ok": False, "error": "a cycle is already running"}, status=409)
    async with _cycle_lock:
        try:
            if MOCK_MODE:
                result = await _mock_predict()
            else:
                csv_path = await _run_swirl_macro("mystery")
                df = pd.read_csv(csv_path)
                feats = feature_extractor.extract_features(df)
                ts = feature_extractor.swirl_window_series(df)
                a = _atlas()
                res = a.identify(feats["e"], top_k=3)
                result = {
                    "ok": True,
                    "csv": os.path.basename(csv_path),
                    "prediction": res["label"],
                    "confidence": round(res["confidence"], 3),
                    "low_confidence": res["low_confidence"],
                    "ambiguous": res.get("ambiguous", False),
                    "margin": res.get("margin", 1.0),
                    "material": res["material"],
                    "material_confidence": round(res["material_confidence"], 3),
                    "material_low_confidence": res["material_low_confidence"],
                    "material_ambiguous": res.get("material_ambiguous", False),
                    "material_margin": res.get("material_margin", 1.0),
                    "material_matches": res.get("material_matches", []),
                    "matches": [{"label": m["label"], "similarity": round(m["score"], 3)}
                                for m in res["matches"]],
                    "features": {"names": feature_extractor.FEATURE_NAMES,
                                 "z": [round(float(v), 4) for v in feats["z"]],
                                 "descriptions": feature_extractor.FEATURE_DESCRIPTIONS},
                    "e": [float(v) for v in feats["e"]],
                    "timeseries": ts,
                    "diagnostics": feats["diagnostics"],
                }
            _last_prediction = result
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)[:500]}, status=500)


async def handle_cycle_teach(request):
    """Mode 2: run the swirl macro on a labeled item → save signature to Atlas."""
    try:
        data = await request.json()
    except Exception:
        data = {}
    label = _sanitize_label(str(data.get("item_name", "")))
    if not label or label == "item":
        return web.json_response({"ok": False, "error": "provide item_name"}, status=400)
    if _cycle_lock.locked():
        return web.json_response({"ok": False, "error": "a cycle is already running"}, status=409)
    async with _cycle_lock:
        try:
            if MOCK_MODE:
                return web.json_response(await _mock_teach(label))
            csv_path = await _run_swirl_macro(label)
            feats = feature_extractor.extract_features(pd.read_csv(csv_path))
            a = _atlas()
            saved = a.teach(
                label, feats["e"], motion_type="swirl_macro", session="thu",
                properties={"tare_flags": feats["diagnostics"]["tare_flags"],
                            "phase_source": feats["diagnostics"]["phase_source"],
                            "csv": os.path.basename(csv_path)})
            return web.json_response({
                "ok": True, "label": label, "csv": os.path.basename(csv_path),
                "sig_id": saved.get("sig_id"),
                "diagnostics": feats["diagnostics"],
            })
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)[:500]}, status=500)


async def _mock_correct(label: str):
    """Mock correction: pretend to save the true label (nothing touches Atlas)."""
    import uuid
    await asyncio.sleep(MOCK_DELAY_SEC)
    return {"ok": True, "mock": True, "label": label, "was": None,
            "csv": None, "sig_id": f"mock-{uuid.uuid4().hex[:8]}",
            "note": "mock mode — not saved to Atlas"}


async def handle_cycle_correct(request):
    """Teach the true label for the most recent prediction's recording.

    A wrong prediction is the most valuable teaching signal: instead of
    re-running the macro, reuse the just-recorded mystery CSV and store it
    under the true label. Recorded as a correction (correction_of) so
    confusions can be audited later.
    """
    try:
        data = await request.json()
    except Exception:
        data = {}
    label = _sanitize_label(str(data.get("true_label", "")))
    if not label or label == "item":
        return web.json_response({"ok": False, "error": "provide true_label"}, status=400)
    if _cycle_lock.locked():
        return web.json_response({"ok": False, "error": "a cycle is already running"}, status=409)
    async with _cycle_lock:
        try:
            if MOCK_MODE:
                return web.json_response(await _mock_correct(label))
            last = _last_prediction or {}
            csv_name = last.get("csv")
            if not csv_name:
                return web.json_response(
                    {"ok": False, "error": "no prediction yet — run Mode 1 first"}, status=400)
            csv_path = os.path.join(RECORDINGS_DIR, os.path.basename(csv_name))
            if not os.path.isfile(csv_path):
                return web.json_response(
                    {"ok": False, "error": f"recording not found: {csv_name}"}, status=404)
            feats = feature_extractor.extract_features(pd.read_csv(csv_path))
            a = _atlas()
            saved = a.teach(
                label, feats["e"], motion_type="swirl_macro", session="thu",
                properties={"tare_flags": feats["diagnostics"]["tare_flags"],
                            "phase_source": feats["diagnostics"]["phase_source"],
                            "csv": os.path.basename(csv_path),
                            "correction_of": last.get("prediction")})
            return web.json_response({
                "ok": True, "label": label, "was": last.get("prediction"),
                "csv": os.path.basename(csv_path),
                "sig_id": saved.get("sig_id"),
                "diagnostics": feats["diagnostics"],
            })
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)[:500]}, status=500)


async def handle_cycle_abort(request):
    """Stop the running macro subprocess (physical e-stop remains at the venue)."""
    global _cycle_proc, _cycle_status
    proc, _cycle_proc = _cycle_proc, None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        _cycle_status = "aborted"
        return web.json_response({"ok": True, "aborted": True})
    return web.json_response({"ok": True, "aborted": False, "note": "no cycle running"})


async def handle_api_status(request):
    return web.json_response({
        "cycle": _cycle_status,
        "cycle_busy": _cycle_lock.locked(),
        "viam_address": VIAM_MACHINE_ADDRESS,
        "atlas_configured": bool(os.getenv("MONGODB_URI")),
        "mock": MOCK_MODE,
        "git_sha": GIT_SHA,
        "last_prediction": _last_prediction,
    })


async def handle_stream(request):
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)
    queue = asyncio.Queue(maxsize=30)
    state.subscribers.add(queue)

    try:
        while True:
            sample = await queue.get()
            msg = f"data: {json.dumps(sample)}\n\n"
            await response.write(msg.encode("utf-8"))
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        state.subscribers.discard(queue)

    return response


_DOC_PAGES = {"index", "problem", "approach", "force", "relevance"}


async def handle_docs(request):
    """Serve the built-in docs pages (the presentation). /docs and /docs/{page}."""
    page = request.match_info.get("page", "index")
    if page not in _DOC_PAGES:
        raise web.HTTPNotFound()
    return web.FileResponse(os.path.join(HERE, "static", "docs", f"{page}.html"))


async def start_web_server(host: str = "0.0.0.0", port: int = 8080):
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/docs", handle_docs)
    app.router.add_get("/docs/{page}", handle_docs)
    app.router.add_get("/stream", handle_stream)
    app.router.add_post("/tare", handle_tare)
    app.router.add_post("/record/start", handle_record_start)
    app.router.add_post("/record/stop", handle_record_stop)
    app.router.add_get("/record/download", handle_record_download)
    app.router.add_post("/api/cycle/predict", handle_cycle_predict)
    app.router.add_post("/api/cycle/teach", handle_cycle_teach)
    app.router.add_post("/api/cycle/abort", handle_cycle_abort)
    app.router.add_post("/api/cycle/correct", handle_cycle_correct)
    app.router.add_get("/api/status", handle_api_status)
    app.router.add_post("/api/vectorspace", handle_vspace)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"\n" + "=" * 60)
    print(f"🚀 ULTRA-PERFORMANT 10s DASHBOARD RUNNING:")
    print(f"   -> Open in your browser:  http://localhost:{port}")
    print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Ultra-performant 10s rolling realtime visualizer.")
    parser.add_argument("--port", type=int, default=8080, help="Web dashboard port (default: 8080)")
    parser.add_argument("--delay-ms", type=int, default=200, help="Target delay between polls in ms (default: 200)")
    args = parser.parse_args()

    await start_web_server(port=args.port)
    await robot_polling_loop(target_delay_ms=args.delay_ms)


if __name__ == "__main__":
    asyncio.run(main())
