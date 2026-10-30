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


def _newest_recording(after_ts: float) -> Optional[str]:
    cands = [p for p in glob.glob(os.path.join(RECORDINGS_DIR, "recording_*.csv"))
             if os.path.getmtime(p) >= after_ts - 2]
    return max(cands, key=os.path.getmtime) if cands else None


def _sanitize_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "item"


async def _run_swirl_macro(object_label: str) -> str:
    """Run one pick-up → swirl → place cycle via repeatable_grasp.py.

    Returns the path of the freshly recorded CSV. Raises on failure/abort.
    Only one cycle may run at a time (guarded by _cycle_lock by callers).
    """
    global _cycle_proc, _cycle_status
    _cycle_status = "running"
    cmd = [sys.executable, os.path.join(HERE, "repeatable_grasp.py"),
           "--object", object_label, "--record", "--cycles", "1", "--swirl-smooth"]
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

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Viam xArm850 — 10s Rolling Real-Time Visualizer</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-bright: #ffffff;
      --accent: #58a6ff;
      --red: #ff7b72;
      --green: #3fb950;
      --yellow: #d29922;
      --purple: #bc8cff;
      --orange: #f0883e;
      --cyan: #39c5cf;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 20px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 1.35rem; color: var(--text-bright); display: flex; align-items: center; gap: 10px; }
    .badge {
      font-size: 0.75rem;
      background: #238636;
      color: #fff;
      padding: 3px 8px;
      border-radius: 12px;
      font-weight: bold;
    }
    .badge.disconnected { background: #da3633; }
    .badge-rec {
      font-size: 0.75rem;
      background: var(--red);
      color: #fff;
      padding: 3px 8px;
      border-radius: 12px;
      font-weight: bold;
      animation: pulse 1.5s infinite;
      display: none;
    }
    @keyframes pulse {
      0% { opacity: 1; }
      50% { opacity: 0.3; }
      100% { opacity: 1; }
    }
    .grid-kpi {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .kpi-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 16px;
    }
    .kpi-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-val { font-size: 1.8rem; font-weight: 700; color: var(--text-bright); margin-top: 4px; font-variant-numeric: tabular-nums; }
    .kpi-sub { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }
    .highlight-red { color: var(--red); }
    .highlight-blue { color: var(--accent); }
    .highlight-green { color: var(--green); }
    .controls {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      background: #21262d;
      border: 1px solid var(--border);
      color: var(--text-bright);
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.85rem;
      transition: all 0.15s ease;
    }
    button:hover { background: #30363d; border-color: #8b949e; }
    button.tare-btn { background: #1f6feb; border-color: #388bfd; }
    button.tare-btn:hover { background: #388bfd; }
    button.rec-start { background: #da3633; border-color: #f85149; }
    button.rec-start:hover { background: #f85149; }
    button.rec-stop { background: #6e7681; border-color: #8b949e; }
    button.active-toggle { background: #238636; border-color: #3fb950; }
    .chart-container {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-weight: 600;
      font-size: 0.95rem;
      color: var(--text-bright);
    }
    .section-title {
      font-size: 1.05rem;
      color: var(--text-bright);
      margin: 20px 0 12px 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .grid-joints {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .joint-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 14px;
    }
    .joint-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 6px;
    }
    .joint-name { font-size: 0.85rem; font-weight: 600; color: var(--text-bright); }
    .joint-val { font-size: 1.05rem; font-weight: 700; font-variant-numeric: tabular-nums; }
    .canvas-wrap {
      position: relative;
      width: 100%;
      height: 90px;
      background: #0d1117;
      border-radius: 4px;
      border: 1px solid #21262d;
    }
    canvas.sparkline {
      display: block;
      width: 100%;
      height: 100%;
    }
    .download-banner {
      display: none;
      background: rgba(35, 134, 54, 0.15);
      border: 1px solid #238636;
      border-radius: 6px;
      padding: 10px 14px;
      margin-bottom: 16px;
      align-items: center;
      justify-content: space-between;
    }
    .feat-grid { display:grid; grid-template-columns: 1fr auto; gap:2px 10px; font-family:monospace; font-size:0.72rem; margin-top:4px; }
    .feat-name { color:#6e7681; }
    .feat-val { color:#e6edf3; text-align:right; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>
        PHYSICAL AI CUP INSPECTOR
        <span class="badge" id="status-badge">CONNECTING...</span>
        <span class="badge" id="atlas-badge" style="background:#6e7681;">ATLAS: ...</span>
        <span class="badge" id="mock-badge" style="background:#8957e5; display:none;">MOCK</span>
      </h1>
      <p style="color: #8b949e; font-size: 0.8rem; margin-top: 2px;">Pick up &rarr; swirl &rarr; place &mdash; identify what's inside the lidded cup</p>
    </div>
    <div class="controls">
      <button class="rec-stop" id="abort-btn" onclick="abortCycle()" title="Stop the running macro">■ Stop</button>
    </div>
  </header>

  <div class="grid-kpi" style="grid-template-columns: 1fr 1fr;">
    <div class="kpi-card">
      <div class="kpi-label">Mode 1 &mdash; Identify mystery object</div>
      <button class="tare-btn" id="predict-btn" onclick="runPredict()" style="font-size:1rem; padding:10px 20px; margin:8px 0;">RUN SWIRL MACRO &amp; PREDICT</button>
      <div id="decision-card" style="display:none; margin-top:8px; border-top:1px solid var(--border); padding-top:8px;">
        <div class="kpi-label">Predicted class</div>
        <div class="kpi-val" id="pred-label" style="font-size:1.5rem;">—</div>
        <div style="background:#21262d; border-radius:6px; height:14px; margin:6px 0; overflow:hidden;">
          <div id="pred-conf-bar" style="height:100%; width:0%; background:#3fb950;"></div>
        </div>
        <div class="kpi-sub" id="pred-conf-text">confidence —</div>
        <div class="kpi-label" style="margin-top:8px;">Nearest Atlas neighbors</div>
        <div id="pred-neighbors" style="font-size:0.85rem; margin-top:4px;"></div>
        <div class="kpi-label" style="margin-top:8px;">10-D swirl signature</div>
        <div class="feat-grid" id="pred-features"></div>
      </div>
      <div class="kpi-sub" id="predict-status"></div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Mode 2 &mdash; Teach new item (online dataset builder)</div>
      <div style="display:flex; gap:8px; margin:8px 0;">
        <input type="text" id="teach-label-input" placeholder="Item name, e.g. Loose Pens" style="flex:1; background:#21262d; border:1px solid var(--border); color:#fff; padding:10px; border-radius:6px; font-size:0.9rem;" />
        <button class="rec-start" id="teach-btn" onclick="teachItem()" style="font-size:0.9rem; padding:10px 16px;">RUN SWIRL &amp; SAVE TO ATLAS</button>
      </div>
      <div class="kpi-sub" id="teach-status">Signature saved → vector indexed in MongoDB, instantly queryable.</div>
    </div>
  </div>

  <!-- Live joint torques (native canvas sparklines) -->
  <div class="section-title">
    <span>Live Joint Torques (10s Rolling Windows)</span>
    <span style="font-size: 0.8rem; color: #8b949e;">Ultra-fast native canvas (&lt;0.1ms render)</span>
  </div>

  <div class="grid-joints">
    <div class="joint-card">
      <div class="joint-header">
        <span class="joint-name" style="color: #39c5cf;">Joint 1 — Base (Yaw)</span>
        <span class="joint-val" id="val-j1" style="color: #39c5cf;">0.000</span>
      </div>
      <div class="canvas-wrap"><canvas id="c-j1" class="sparkline"></canvas></div>
    </div>

    <div class="joint-card">
      <div class="joint-header">
        <span class="joint-name" style="color: #3fb950;">Joint 2 — Shoulder (Pitch) [Weight]</span>
        <span class="joint-val" id="val-j2" style="color: #3fb950;">-16.05</span>
      </div>
      <div class="canvas-wrap"><canvas id="c-j2" class="sparkline"></canvas></div>
    </div>

    <div class="joint-card">
      <div class="joint-header">
        <span class="joint-name" style="color: #d29922;">Joint 3 — Elbow (Pitch)</span>
        <span class="joint-val" id="val-j3" style="color: #d29922;">-16.82</span>
      </div>
      <div class="canvas-wrap"><canvas id="c-j3" class="sparkline"></canvas></div>
    </div>

    <div class="joint-card">
      <div class="joint-header">
        <span class="joint-name" style="color: #f0883e;">Joint 4 — Forearm (Roll)</span>
        <span class="joint-val" id="val-j4" style="color: #f0883e;">0.778</span>
      </div>
      <div class="canvas-wrap"><canvas id="c-j4" class="sparkline"></canvas></div>
    </div>

    <div class="joint-card">
      <div class="joint-header">
        <span class="joint-name" style="color: #bc8cff;">Joint 5 — Wrist (Pitch)</span>
        <span class="joint-val" id="val-j5" style="color: #bc8cff;">0.373</span>
      </div>
      <div class="canvas-wrap"><canvas id="c-j5" class="sparkline"></canvas></div>
    </div>

    <div class="joint-card">
      <div class="joint-header">
        <span class="joint-name" style="color: #58a6ff;">Joint 6 — Flange (Roll)</span>
        <span class="joint-val" id="val-j6" style="color: #58a6ff;">0.001</span>
      </div>
      <div class="canvas-wrap"><canvas id="c-j6" class="sparkline"></canvas></div>
    </div>
  </div>

  <script>
    // Strict 10-second rolling window (~45-50 points at 4.5Hz)
    const MAX_POINTS = 45;
    let useFilter = true;
    let viewModeDelta = true;  // Default to delta view so all joints center on 0
    let isRecording = false;
    let lastRecordedFile = null;

    // Rolling data buffers
    const history = {
      t: [],
      grams: [],
      fz: [],
      joints: [[], [], [], [], [], []]
    };

    // (force/weight chart removed — derived mass was tare-contaminated; see BASELINE_FINDINGS.md)

    // Native High-Performance Sparkline Renderer (<0.1ms per canvas)
    const sparkCanvases = [
      document.getElementById('c-j1'),
      document.getElementById('c-j2'),
      document.getElementById('c-j3'),
      document.getElementById('c-j4'),
      document.getElementById('c-j5'),
      document.getElementById('c-j6')
    ];
    const jointColors = ['#39c5cf', '#3fb950', '#d29922', '#f0883e', '#bc8cff', '#58a6ff'];

    function resizeCanvases() {
      sparkCanvases.forEach(c => {
        c.width = c.clientWidth * window.devicePixelRatio;
        c.height = c.clientHeight * window.devicePixelRatio;
      });
    }
    window.addEventListener('resize', resizeCanvases);
    resizeCanvases();

    function renderSparkline(canvas, data, color) {
      if (!data || data.length < 2) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      let min = Math.min(...data);
      let max = Math.max(...data);
      if (max - min < 0.02) {
        min -= 0.05;
        max += 0.05;
      }
      const padding = 6;
      const range = max - min;

      // Draw zero line if visible in range
      if (min <= 0 && max >= 0) {
        const zeroY = h - padding - ((0 - min) / range) * (h - 2 * padding);
        ctx.strokeStyle = '#2d333b';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, zeroY);
        ctx.lineTo(w, zeroY);
        ctx.stroke();
      }

      // Draw sparkline polyline
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * window.devicePixelRatio;
      ctx.beginPath();

      const step = w / (data.length - 1);
      data.forEach((val, i) => {
        const x = i * step;
        const y = h - padding - ((val - min) / range) * (h - 2 * padding);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // High performance render loop via requestAnimationFrame
    let needsRender = false;
    function animLoop() {
      if (needsRender) {
        // Render 6 native sparklines
        for (let i = 0; i < 6; i++) {
          renderSparkline(sparkCanvases[i], history.joints[i], jointColors[i]);
        }
        needsRender = false;
      }
      requestAnimationFrame(animLoop);
    }
    requestAnimationFrame(animLoop);

    // Resilient SSE Connection
    function connectSSE() {
      const badge = document.getElementById('status-badge');
      const eventSource = new EventSource('/stream');

      eventSource.onopen = () => {
        badge.innerText = 'LIVE 10s';
        badge.className = 'badge';
      };

      eventSource.onmessage = (event) => {
        const d = JSON.parse(event.data);

        // Manage rolling 10-second buffer
        history.t.push(`${d.t}s`);

        const activeVals = viewModeDelta ? d.deltas : (useFilter ? d.smooth : d.raw);
        for (let i = 0; i < 6; i++) {
          history.joints[i].push(activeVals[i]);
          const unit = viewModeDelta ? 'Δ' : 'Nm';
          const jSign = activeVals[i] >= 0 ? '+' : '';
          document.getElementById(`val-j${i+1}`).innerText = `${jSign}${activeVals[i].toFixed(3)} ${unit}`;
        }

        // Drop oldest data to strictly preserve ONLY 10 seconds
        if (history.t.length > MAX_POINTS) {
          history.t.shift();
          history.grams.shift();
          history.fz.shift();
          for (let i = 0; i < 6; i++) {
            history.joints[i].shift();
          }
        }

        needsRender = true;
      };

      eventSource.onerror = () => {
        badge.innerText = 'RECONNECTING...';
        badge.className = 'badge disconnected';
        eventSource.close();
        setTimeout(connectSSE, 1500);
      };
    }

    connectSSE();
    pollStatus();
    setInterval(pollStatus, 5000);

    function renderDecision(data) {
      document.getElementById('decision-card').style.display = 'block';
      document.getElementById('pred-label').innerText = data.prediction || 'unknown';
      const pct = Math.round((data.confidence || 0) * 100);
      document.getElementById('pred-conf-bar').style.width = pct + '%';
      document.getElementById('pred-conf-bar').style.background = data.low_confidence ? '#d29922' : '#3fb950';
      document.getElementById('pred-conf-text').innerText =
        `confidence ${pct}%${data.low_confidence ? ' — LOW, consider teaching this item' : ''}` +
        (data.mock ? '  (MOCK)' : '');
      setNeighbors(document.getElementById('pred-neighbors'), data.matches || []);
      const fn = (data.features && data.features.names) || [];
      const fv = (data.features && data.features.z) || [];
      document.getElementById('pred-features').innerHTML = fn.map((n, i) =>
        `<div class="feat-name">${n}</div><div class="feat-val">${(fv[i] ?? 0).toFixed(3)}</div>`).join('');
      const dq = data.diagnostics || {};
      document.getElementById('predict-status').innerText = `Done — ${data.csv}` +
        (data.mock ? '  (MOCK data, not live hardware)' : '') +
        (dq.tare_flags && dq.tare_flags.length ? `  ⚠ tare flags: ${dq.tare_flags.join(',')}` : '');
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        const s = await res.json();
        const b = document.getElementById('atlas-badge');
        b.innerText = s.atlas_configured ? 'ATLAS: SYNCED' : 'ATLAS: NO URI';
        b.style.background = s.atlas_configured ? '#238636' : '#da3633';
        document.getElementById('mock-badge').style.display = s.mock ? 'inline-block' : 'none';
        if (s.last_prediction && s.last_prediction.ok) renderDecision(s.last_prediction);
        const busy = s.cycle_busy;
        for (const id of ['predict-btn', 'teach-btn']) {
          const btn = document.getElementById(id);
          if (btn) btn.disabled = busy;
        }
        if (busy) {
          document.getElementById('predict-status').innerText = 'Swirl macro running on the robot…';
        }
      } catch (e) { /* dashboard stays usable offline */ }
    }

    function setNeighbors(el, matches) {
      el.innerHTML = matches.map((m, i) =>
        `<div>${i + 1}. ${m.label} — ${(m.similarity * 100).toFixed(1)}% similarity</div>`
      ).join('');
    }

    async function runPredict() {
      const btn = document.getElementById('predict-btn');
      const status = document.getElementById('predict-status');
      btn.disabled = true;
      document.getElementById('decision-card').style.display = 'none';
      status.innerText = 'Running pick-up → swirl → place… (this takes ~1 min)';
      try {
        const res = await fetch('/api/cycle/predict', { method: 'POST' });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'prediction failed');
        renderDecision(data);
      } catch (e) {
        status.innerText = 'Error: ' + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    async function teachItem() {
      const input = document.getElementById('teach-label-input');
      const btn = document.getElementById('teach-btn');
      const status = document.getElementById('teach-status');
      const name = input.value.trim();
      if (!name) { status.innerText = 'Enter an item name first.'; return; }
      btn.disabled = true;
      status.innerText = `Teaching "${name}" — running swirl macro…`;
      try {
        const res = await fetch('/api/cycle/teach', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ item_name: name }),
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'teach failed');
        status.innerText = data.mock
          ? `✓ (MOCK) "${data.label}" recorded — not saved to Atlas.`
          : `✓ Signature saved! "${data.label}" indexed in MongoDB (sig ${String(data.sig_id).slice(-6)}).`;
        input.value = '';
      } catch (e) {
        status.innerText = 'Error: ' + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    async function abortCycle() {
      const res = await fetch('/api/cycle/abort', { method: 'POST' });
      const data = await res.json();
      document.getElementById('predict-status').innerText =
        data.aborted ? 'Cycle aborted.' : 'No cycle running.';
    }
  </script>
</body>
</html>
"""


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
    feats = feature_extractor.extract_features(df)
    return {
        "ok": True, "mock": True,
        "csv": os.path.basename(name),
        "prediction": str(pred),
        "confidence": float(round(float(conf), 3)),
        "low_confidence": bool(float(conf) < 0.82),
        "matches": matches,
        "features": {"names": feature_extractor.FEATURE_NAMES,
                     "z": [round(float(v), 4) for v in feats["z"]]},
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
                feats = feature_extractor.extract_features(pd.read_csv(csv_path))
                a = _atlas()
                res = a.identify(feats["e"], top_k=3)
                result = {
                    "ok": True,
                    "csv": os.path.basename(csv_path),
                    "prediction": res["label"],
                    "confidence": round(res["confidence"], 3),
                    "low_confidence": res["low_confidence"],
                    "matches": [{"label": m["label"], "similarity": round(m["score"], 3)}
                                for m in res["matches"]],
                    "features": {"names": feature_extractor.FEATURE_NAMES,
                                 "z": [round(float(v), 4) for v in feats["z"]]},
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


async def start_web_server(host: str = "0.0.0.0", port: int = 8080):
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/stream", handle_stream)
    app.router.add_post("/tare", handle_tare)
    app.router.add_post("/record/start", handle_record_start)
    app.router.add_post("/record/stop", handle_record_stop)
    app.router.add_get("/record/download", handle_record_download)
    app.router.add_post("/api/cycle/predict", handle_cycle_predict)
    app.router.add_post("/api/cycle/teach", handle_cycle_teach)
    app.router.add_post("/api/cycle/abort", handle_cycle_abort)
    app.router.add_get("/api/status", handle_api_status)

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
