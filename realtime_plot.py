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
import time
import math
import json
import csv
import argparse
from datetime import datetime, timezone
from typing import Set, List, Optional, Dict, Any
from dotenv import load_dotenv

from aiohttp import web
from viam.robot.client import RobotClient
from viam.components.arm import Arm

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
GRAVITY = 9.80665
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)


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

    def start_recording(self):
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recorded_samples = []
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.current_recording_file = f"recording_{timestamp}.csv"
        print(f"[RECORD] Started session: {self.current_recording_file}")

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
                    "elapsed_sec", "effective_force_n", "payload_weight_g",
                    "delta_j2_nm", "r_xy_m",
                    "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw",
                    "j1_smooth", "j2_smooth", "j3_smooth", "j4_smooth", "j5_smooth", "j6_smooth"
                ])
                for s in self.recorded_samples:
                    writer.writerow([
                        s["t"], s["fz"], s["grams"],
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
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
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
  </style>
</head>
<body>
  <header>
    <div>
      <h1>
        xArm850 Real-Time Telemetry
        <span class="badge" id="status-badge">CONNECTING...</span>
        <span class="badge-rec" id="rec-badge">REC</span>
      </h1>
      <p style="color: #8b949e; font-size: 0.8rem; margin-top: 2px;">10-Second High-Performance Rolling Window</p>
    </div>
    <div class="controls">
      <button class="rec-start" id="rec-btn" onclick="toggleRecording()">● Record Session</button>
      <button class="tare-btn" onclick="tare()">Tare (Zero)</button>
      <button class="active-toggle" id="filter-btn" onclick="toggleFilter()">EMA Filter: ON</button>
      <button id="mode-btn" onclick="toggleMode()">View: Delta (Δ Nm)</button>
    </div>
  </header>

  <div class="download-banner" id="dl-banner">
    <span id="dl-msg">Session recorded successfully.</span>
    <button onclick="downloadCSV()" style="background: #238636; border-color: #3fb950;">⬇ Download CSV Dataset</button>
  </div>

  <div class="grid-kpi">
    <div class="kpi-card">
      <div class="kpi-label">Effective Payload Weight</div>
      <div class="kpi-val highlight-red" id="kpi-weight">+0.0 g</div>
      <div class="kpi-sub">End-Effector Mass</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Vertical Force Fz</div>
      <div class="kpi-val highlight-blue" id="kpi-force">0.00 N</div>
      <div class="kpi-sub">External Load</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Shoulder Torque (J2)</div>
      <div class="kpi-val highlight-green" id="kpi-j2">-16.05 Nm</div>
      <div class="kpi-sub" id="kpi-delta-j2">Δ: 0.0000 Nm</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Elbow Torque (J3)</div>
      <div class="kpi-val" id="kpi-j3">-16.82 Nm</div>
      <div class="kpi-sub" id="kpi-j3-sub">Cantilever Moment</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Recording Status</div>
      <div class="kpi-val" id="kpi-rec" style="font-size: 1.25rem;">IDLE</div>
      <div class="kpi-sub" id="kpi-rec-sub">0 samples</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Cloud Latency (RTT)</div>
      <div class="kpi-val" id="kpi-rtt" style="color: #39c5cf;">-- ms</div>
      <div class="kpi-sub">Round-trip Query Time</div>
    </div>
  </div>

  <!-- Primary Force & Weight Chart (Optimized 10s Window) -->
  <div class="chart-container">
    <div class="chart-header">
      <span>Effective End-Effector Force & Weight (Rolling 10s Window)</span>
      <span style="font-size: 0.8rem; color: #8b949e;">Derived from Shoulder Pitch Δτ</span>
    </div>
    <div style="height: 220px;">
      <canvas id="forceChart"></canvas>
    </div>
  </div>

  <!-- High-Performance Separated 6-Joint Sparklines (Pure Native Canvas) -->
  <div class="section-title">
    <span>Individual Actuator Torques (Independent Auto-Scaling 10s Windows)</span>
    <span style="font-size: 0.8rem; color: #8b949e;">Ultra-fast native canvas (<0.1ms render)</span>
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

    // Primary Chart.js optimized (zero animations, fast line)
    const ctxForce = document.getElementById('forceChart').getContext('2d');
    const forceChart = new Chart(ctxForce, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Weight (g)',
            data: [],
            borderColor: '#ff7b72',
            backgroundColor: 'rgba(255, 123, 114, 0.08)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0,
            fill: true,
            yAxisID: 'yWeight'
          },
          {
            label: 'Force Fz (N)',
            data: [],
            borderColor: '#58a6ff',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0,
            yAxisID: 'yForce'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        elements: { line: { tension: 0 } },
        scales: {
          x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 6 } },
          yWeight: {
            position: 'left',
            grid: { color: '#21262d' },
            ticks: { color: '#ff7b72', font: { size: 10 } },
            title: { display: true, text: 'Weight (g)', color: '#ff7b72' }
          },
          yForce: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#58a6ff', font: { size: 10 } },
            title: { display: true, text: 'Force (N)', color: '#58a6ff' }
          }
        },
        plugins: { legend: { labels: { color: '#c9d1d9', boxWidth: 12 } } }
      }
    });

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
        // Update Chart.js (10s rolling window)
        forceChart.data.labels = history.t;
        forceChart.data.datasets[0].data = history.grams;
        forceChart.data.datasets[1].data = history.fz;
        forceChart.update();

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

        // Update KPIs immediately
        const sign = d.grams >= 0 ? '+' : '';
        document.getElementById('kpi-weight').innerText = `${sign}${d.grams.toFixed(1)} g`;
        document.getElementById('kpi-force').innerText = `${d.fz.toFixed(2)} N`;
        document.getElementById('kpi-j2').innerText = `${(useFilter ? d.smooth[1] : d.raw[1]).toFixed(3)} Nm`;
        document.getElementById('kpi-delta-j2').innerText = `Δ: ${(d.delta_tau_j2 >= 0 ? '+' : '') + d.delta_tau_j2.toFixed(4)} Nm`;
        document.getElementById('kpi-j3').innerText = `${(useFilter ? d.smooth[2] : d.raw[2]).toFixed(3)} Nm`;

        if (d.rtt_ms !== undefined) {
          const rttEl = document.getElementById('kpi-rtt');
          rttEl.innerText = `${d.rtt_ms.toFixed(0)} ms`;
          if (d.rtt_ms < 100) rttEl.style.color = '#3fb950';
          else if (d.rtt_ms < 300) rttEl.style.color = '#d29922';
          else rttEl.style.color = '#ff7b72';
        }

        if (d.is_recording) {
          document.getElementById('kpi-rec').innerText = 'REC ●';
          document.getElementById('kpi-rec').style.color = '#ff7b72';
          document.getElementById('kpi-rec-sub').innerText = `${d.rec_count} samples`;
        } else {
          document.getElementById('kpi-rec').innerText = 'IDLE';
          document.getElementById('kpi-rec').style.color = '#c9d1d9';
        }

        // Manage rolling 10-second buffer
        history.t.push(`${d.t}s`);
        history.grams.push(d.grams);
        history.fz.push(d.fz);

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

    async function tare() {
      await fetch('/tare', { method: 'POST' });
    }

    async function toggleRecording() {
      const btn = document.getElementById('rec-btn');
      const badge = document.getElementById('rec-badge');

      if (!isRecording) {
        const res = await fetch('/record/start', { method: 'POST' });
        const data = await res.json();
        isRecording = true;
        btn.innerText = '■ Stop Recording';
        btn.className = 'rec-stop';
        badge.style.display = 'inline-block';
        document.getElementById('dl-banner').style.display = 'none';
      } else {
        const res = await fetch('/record/stop', { method: 'POST' });
        const data = await res.json();
        isRecording = false;
        btn.innerText = '● Record Session';
        btn.className = 'rec-start';
        badge.style.display = 'none';
        if (data.file) {
          lastRecordedFile = data.file;
          document.getElementById('dl-msg').innerText = `Session saved: ${data.file} (${data.samples} samples)`;
          document.getElementById('dl-banner').style.display = 'flex';
        }
      }
    }

    function downloadCSV() {
      if (lastRecordedFile) {
        window.location.href = `/record/download?file=${encodeURIComponent(lastRecordedFile)}`;
      }
    }

    function toggleFilter() {
      useFilter = !useFilter;
      const btn = document.getElementById('filter-btn');
      btn.innerText = useFilter ? 'EMA Filter: ON' : 'EMA Filter: RAW';
      btn.className = useFilter ? 'active-toggle' : '';
    }

    function toggleMode() {
      viewModeDelta = !viewModeDelta;
      const btn = document.getElementById('mode-btn');
      btn.innerText = viewModeDelta ? 'View: Delta (Δ Nm)' : 'View: Absolute (N·m)';
      btn.className = viewModeDelta ? 'active-toggle' : '';
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
    state.start_recording()
    return web.json_response({"ok": True, "file": state.current_recording_file})


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
