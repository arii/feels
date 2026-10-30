#!/usr/bin/env python3
"""
record_dataset.py — Multi-channel, multi-rate dataset recorder for cup-content ID.

Captures far more of the robot's internal state than the original torque-only
recorder, at a configurable (and optionally *varying*) sample rate:

  Proprioception (every sample, as fast as the link allows):
    - Cartesian TCP pose (x/y/z + orientation)
    - 6 joint angles (deg)
    - 6 joint torques / loads (N·m)                arm.do_command({"load": ""})
    - DERIVED joint velocities (deg/s) + TCP speed (mm/s)  [finite difference]
    - Any extra read-only arm channels the driver exposes (currents, speeds,
      temperatures, following error, ...) — AUTO-DISCOVERED at startup.
    - Gripper holding state + position (+ any gripper force/current the
      driver exposes, auto-discovered)
    - Achieved instantaneous rate (inst_hz) + target rate (sample_hz)

  Vision (optional, slow loop, --camera):
    - RGB and/or depth frames from the RealSense, saved to a side folder,
      indexed by elapsed time in a companion *_camera.csv

  Static (saved once):
    - arm.get_kinematics() URDF, so you can map torque→wrench properly later.

Sampling rate modes:
    --hz 20                       single target rate (default 10 Hz)
    --hz-sweep "2,5,10,20"        record `--duration` s at EACH rate in turn,
                                  tagged phase=SWEEP_<hz>HZ and column sample_hz
    --cam-hz 2                    camera loop rate when --camera is set

SAFETY: discovery only issues clearly read-only *name* queries via do_command,
each wrapped in try/except with a short timeout. It never commands motion.
Still — this talks to a PHYSICAL arm. Keep the workspace clear / e-stop in reach.

Examples:
    python record_dataset.py --object water_half --duration 8 --hz 15
    python record_dataset.py --object candy_loose --hz-sweep "2,5,10,20" --duration 5
    python record_dataset.py --object empty_cup --camera --cam-hz 2 --duration 10
    python record_dataset.py --probe-only            # just list discovered channels
"""

import asyncio
import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.components.camera import Camera

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
CAM_NAME = os.getenv("VIAM_CAMERA_NAME", "cam")
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
GRAVITY = 9.80665

ITEM_PRESETS = {
    "1": ("candy_loose", "🍬 Loose Candy (5–25 Hz rattle chatter)"),
    "2": ("water_half", "💧 Water (fluid slosh & dynamic phase lag)"),
    "3": ("dish_soap_half", "🫧 Dish Soap (heavy viscous damping)"),
    "4": ("empty_cup", "👻 Empty Cup (baseline tare / zero mass)"),
    "5": ("solid_weight", "🪨 Solid Heavy Weight (rigid coupling, high mass)"),
}

# Read-only name queries we attempt on the arm/gripper do_command. Only keys that
# return numeric data are kept. These are passive state reads — NOT motion commands.
ARM_QUERY_CANDIDATES = [
    {"velocity": ""}, {"velocities": ""}, {"joint_velocity": ""}, {"speed": ""},
    {"current": ""}, {"currents": ""}, {"servo_current": ""}, {"motor_current": ""},
    {"temperature": ""}, {"temperatures": ""}, {"motor_temp": ""},
    {"following_error": ""}, {"position_error": ""},
    {"servo_angle": ""}, {"state": ""}, {"status": ""},
]
GRIPPER_QUERY_CANDIDATES = [
    {"force": ""}, {"current": ""}, {"state": ""}, {"status": ""}, {"get_state": ""},
]

# Columns written first (keeps backward compatibility with existing analysis).
CORE_ORDER = [
    "elapsed_sec", "phase", "object_label", "sample_hz", "inst_hz", "dt_s",
    "tcp_x_mm", "tcp_y_mm", "tcp_z_mm", "tcp_ox", "tcp_oy", "tcp_oz", "tcp_theta_deg",
    "tcp_speed_mm_s",
    "q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg",
    "qd1_dps", "qd2_dps", "qd3_dps", "qd4_dps", "qd5_dps", "qd6_dps",
    "tau1_nm", "tau2_nm", "tau3_nm", "tau4_nm", "tau5_nm", "tau6_nm",
    "gripper_holding", "gripper_pos",
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

async def connect_robot() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID,
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isnum(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _san(k: str) -> str:
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in str(k)).strip("_").lower()


def flatten_numeric(prefix: str, resp: Any) -> Dict[str, float]:
    """Turn a do_command response into flat numeric columns."""
    out: Dict[str, float] = {}
    if isinstance(resp, dict):
        for k, v in resp.items():
            key = _san(k)
            if isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    if _isnum(x):
                        out[f"{prefix}{key}{i+1}"] = round(float(x), 5)
            elif _isnum(v):
                out[f"{prefix}{key}"] = round(float(v), 5)
    elif isinstance(resp, (list, tuple)):
        for i, x in enumerate(resp):
            if _isnum(x):
                out[f"{prefix}{i+1}"] = round(float(x), 5)
    return out


async def _try_query(component, cmd: dict, timeout: float = 1.0) -> Optional[dict]:
    try:
        resp = await component.do_command(cmd, timeout=timeout)
        return resp
    except Exception:
        return None


async def discover_channels(arm: Arm, gripper: Gripper) -> Dict[str, List[dict]]:
    """Probe read-only do_command queries; keep those returning numeric data."""
    print("\n[discover] probing read-only driver channels (no motion issued)...")
    active = {"arm": [], "gripper": []}
    for cmd in ARM_QUERY_CANDIDATES:
        resp = await _try_query(arm, cmd)
        if resp and flatten_numeric("", resp):
            active["arm"].append(cmd)
            print(f"  [arm] ✓ {list(cmd)[0]:16s} -> {list(flatten_numeric('', resp))[:6]}")
    for cmd in GRIPPER_QUERY_CANDIDATES:
        resp = await _try_query(gripper, cmd)
        if resp and flatten_numeric("", resp):
            active["gripper"].append(cmd)
            print(f"  [grip] ✓ {list(cmd)[0]:16s} -> {list(flatten_numeric('', resp))[:6]}")
    if not active["arm"] and not active["gripper"]:
        print("  (no extra driver channels found — using torque + derived velocities only)")
    return active


# ---------------------------------------------------------------------------
# Proprioception polling loop (one rate segment)
# ---------------------------------------------------------------------------

async def _poll_proprio(
    arm: Arm, gripper: Gripper, active: Dict[str, List[dict]],
    samples: List[Dict[str, Any]], stop_at: float, t0: float,
    target_hz: float, object_label: str, phase: str,
):
    interval = 1.0 / max(0.5, target_hz)
    prev_t: Optional[float] = None
    prev_q: Optional[List[float]] = None
    prev_pos = None

    while time.time() < stop_at:
        loop_start = time.time()
        elapsed = loop_start - t0
        try:
            calls = [
                arm.get_end_position(),
                arm.get_joint_positions(),
                arm.do_command({"load": ""}),
                gripper.is_holding_something(),
            ]
            calls += [arm.do_command(c) for c in active["arm"]]
            calls += [gripper.do_command(c) for c in active["gripper"]]
            results = await asyncio.gather(*calls, return_exceptions=True)

            pos, joints, load_res, grip_status = results[0], results[1], results[2], results[3]
            extra_results = results[4:]
            if any(isinstance(r, Exception) for r in (pos, joints, load_res, grip_status)):
                raise RuntimeError("core read failed")

            loads = load_res.get("load", []) if isinstance(load_res, dict) else []
            j = joints.values if joints else []
            if not (loads and len(loads) >= 6 and len(j) >= 6):
                raise RuntimeError("incomplete core sample")

            dt = (loop_start - prev_t) if prev_t else 0.0
            # derived joint velocities (deg/s) and TCP speed (mm/s)
            qd = [0.0] * 6
            tcp_speed = 0.0
            if prev_q and dt > 1e-4:
                qd = [round((j[i] - prev_q[i]) / dt, 3) for i in range(6)]
            if prev_pos is not None and dt > 1e-4:
                tcp_speed = round((((pos.x - prev_pos.x) ** 2 + (pos.y - prev_pos.y) ** 2
                                    + (pos.z - prev_pos.z) ** 2) ** 0.5) / dt, 2)

            grip_meta = grip_status.meta if grip_status else {}
            s: Dict[str, Any] = {
                "elapsed_sec": round(elapsed, 4),
                "phase": phase,
                "object_label": object_label,
                "sample_hz": round(target_hz, 2),
                "inst_hz": round(1.0 / dt, 2) if dt > 1e-4 else 0.0,
                "dt_s": round(dt, 4),
                "tcp_x_mm": round(pos.x, 2), "tcp_y_mm": round(pos.y, 2), "tcp_z_mm": round(pos.z, 2),
                "tcp_ox": round(pos.o_x, 4), "tcp_oy": round(pos.o_y, 4), "tcp_oz": round(pos.o_z, 4),
                "tcp_theta_deg": round(pos.theta, 2), "tcp_speed_mm_s": tcp_speed,
                "gripper_holding": grip_status.is_holding_something,
                "gripper_pos": round(grip_meta.get("position", 0.0), 1),
            }
            for i in range(6):
                s[f"q{i+1}_deg"] = round(j[i], 3)
                s[f"qd{i+1}_dps"] = qd[i]
                s[f"tau{i+1}_nm"] = round(loads[i], 4)

            # flatten any discovered extra channels
            k = 4
            for cmd in active["arm"]:
                r = extra_results[k - 4]
                if isinstance(r, dict):
                    s.update(flatten_numeric("arm_", r))
                k += 1
            for cmd in active["gripper"]:
                r = extra_results[k - 4]
                if isinstance(r, dict):
                    s.update(flatten_numeric("grip_", r))
                k += 1

            samples.append(s)
            prev_t, prev_q, prev_pos = loop_start, list(j), pos
            sys.stdout.write("."); sys.stdout.flush()
        except Exception:
            pass

        sleep = interval - (time.time() - loop_start)
        await asyncio.sleep(max(0.001, sleep))


# ---------------------------------------------------------------------------
# Camera polling loop (slow, optional)
# ---------------------------------------------------------------------------

async def _poll_camera(
    cam: Camera, cam_rows: List[Dict[str, Any]], frames_dir: str,
    stop_at: float, t0: float, cam_hz: float, phase_ref: Dict[str, str],
):
    import numpy as np
    try:
        from viam.media.video import CameraMimeType
        depth_mime = CameraMimeType.VIAM_RAW_DEPTH
    except Exception:
        depth_mime = "image/vnd.viam.dep"

    interval = 1.0 / max(0.2, cam_hz)
    idx = 0
    os.makedirs(frames_dir, exist_ok=True)
    while time.time() < stop_at:
        loop_start = time.time()
        elapsed = round(loop_start - t0, 4)
        row = {"elapsed_sec": elapsed, "phase": phase_ref.get("phase", ""), "frame_idx": idx}
        try:
            images, _ = await cam.get_images()
            for img in images:
                mime = str(img.mime_type)
                if "depth" in mime.lower():
                    try:
                        arr = np.array(img.bytes_to_depth_array(), dtype=np.uint16)
                        fn = os.path.join(frames_dir, f"depth_{idx:04d}.npy")
                        np.save(fn, arr)
                        row["depth_file"] = os.path.basename(fn)
                        row["depth_mean_mm"] = int(arr[arr > 0].mean()) if (arr > 0).any() else 0
                    except Exception:
                        pass
                elif "jpeg" in mime.lower() or "png" in mime.lower():
                    ext = "jpg" if "jpeg" in mime.lower() else "png"
                    fn = os.path.join(frames_dir, f"rgb_{idx:04d}.{ext}")
                    with open(fn, "wb") as f:
                        f.write(img.data)
                    row["rgb_file"] = os.path.basename(fn)
            cam_rows.append(row)
            idx += 1
            sys.stdout.write("c"); sys.stdout.flush()
        except Exception:
            pass
        await asyncio.sleep(max(0.01, interval - (time.time() - loop_start)))


# ---------------------------------------------------------------------------
# Record one fixed-rate segment (proprio + optional camera, concurrent)
# ---------------------------------------------------------------------------

async def record_segment(
    arm: Arm, gripper: Gripper, cam: Optional[Camera], active: Dict[str, List[dict]],
    object_label: str, duration: float, target_hz: float, phase: str,
    t0: float, samples: List[Dict[str, Any]], cam_rows: List[Dict[str, Any]],
    frames_dir: Optional[str], cam_hz: float, phase_ref: Dict[str, str],
):
    phase_ref["phase"] = phase
    print(f"\n[seg] '{object_label}' {phase} — {duration:.1f}s @ target {target_hz:g} Hz", end="")
    stop_at = time.time() + duration
    tasks = [_poll_proprio(arm, gripper, active, samples, stop_at, t0, target_hz, object_label, phase)]
    if cam is not None and frames_dir is not None:
        tasks.append(_poll_camera(cam, cam_rows, frames_dir, stop_at, t0, cam_hz, phase_ref))
    await asyncio.gather(*tasks)
    print(" ✓")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_dataset(samples: List[Dict[str, Any]], cam_rows: List[Dict[str, Any]],
                 object_label: str, meta: Dict[str, Any]) -> Optional[str]:
    if not samples:
        print("\n[save] no samples captured — nothing written.")
        return None

    clean = "".join(c for c in object_label if c.isalnum() or c in ("-", "_")).lower() or "unlabeled"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"recording_{stamp}_{clean}"
    path = os.path.join(RECORDINGS_DIR, base + ".csv")

    # union of all keys, core columns first
    extra = sorted({k for s in samples for k in s} - set(CORE_ORDER))
    fields = [c for c in CORE_ORDER if any(c in s for s in samples)] + extra

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(samples)

    if cam_rows:
        cam_path = os.path.join(RECORDINGS_DIR, base + "_camera.csv")
        cam_fields = sorted({k for r in cam_rows for k in r},
                            key=lambda c: (c != "elapsed_sec", c))
        with open(cam_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cam_fields, restval="")
            w.writeheader(); w.writerows(cam_rows)
        meta["camera_csv"] = os.path.basename(cam_path)
        meta["camera_frames"] = len(cam_rows)

    meta_path = os.path.join(RECORDINGS_DIR, base + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # quick summary
    rates = sorted({s.get("sample_hz") for s in samples})
    tau3 = [s["tau3_nm"] for s in samples if "tau3_nm" in s]
    mean3 = sum(tau3) / len(tau3) if tau3 else 0.0
    var3 = sum((x - mean3) ** 2 for x in tau3) / len(tau3) if len(tau3) > 1 else 0.0

    print("\n" + "=" * 72)
    print(f"📊 DATASET SAVED — '{object_label}'")
    print("=" * 72)
    print(f"• CSV:              {path}")
    print(f"• Samples:          {len(samples)}   (columns: {len(fields)})")
    print(f"• Rate segments:    {', '.join(f'{r:g}Hz' for r in rates if r is not None)}")
    print(f"• Extra channels:   {extra if extra else '(none beyond torque + derived velocities)'}")
    print(f"• J3 torque:        mean={mean3:.4f} N·m  var={var3:.5f}")
    if cam_rows:
        print(f"• Camera frames:    {len(cam_rows)} → {meta.get('camera_csv')}")
    print(f"• Meta/kinematics:  {os.path.basename(meta_path)}")
    print("=" * 72 + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    p = argparse.ArgumentParser(description="Multi-channel, multi-rate robot telemetry recorder.")
    p.add_argument("--object", type=str, help="Object label (e.g. water_half, candy_loose, empty_cup)")
    p.add_argument("--duration", type=float, default=10.0, help="Seconds PER rate segment (default 10)")
    p.add_argument("--hz", type=float, default=10.0, help="Target sample rate (default 10)")
    p.add_argument("--hz-sweep", type=str, default=None,
                   help='Comma list of rates to record in sequence, e.g. "2,5,10,20"')
    p.add_argument("--camera", action="store_true", help="Also capture RealSense RGB/depth frames")
    p.add_argument("--cam-hz", type=float, default=2.0, help="Camera loop rate (default 2)")
    p.add_argument("--move-travel", action="store_true", help="Move to 'travel' waypoint before recording")
    p.add_argument("--no-probe", action="store_true", help="Skip driver channel discovery")
    p.add_argument("--probe-only", action="store_true", help="Discover channels, print, and exit (no recording)")
    args = p.parse_args()

    object_label = args.object
    if not object_label and not args.probe_only:
        print("\n" + "=" * 60 + "\n🎯 SELECT OR ENTER OBJECT TO RECORD\n" + "=" * 60)
        for k, v in ITEM_PRESETS.items():
            print(f"  [{k}] {v[0]:<16} — {v[1]}")
        print("  [0] Enter custom name\n" + "=" * 60)
        choice = input("Choice (1-5 or custom name): ").strip()
        if choice in ITEM_PRESETS:
            object_label = ITEM_PRESETS[choice][0]
        elif choice == "0" or not choice:
            object_label = input("Enter custom object name: ").strip() or "unlabeled_item"
        else:
            object_label = choice

    rates = [args.hz]
    if args.hz_sweep:
        rates = [float(x) for x in args.hz_sweep.split(",") if x.strip()]

    print(f"\nConnecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect_robot() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")
        cam = None
        if args.camera:
            try:
                cam = Camera.from_robot(machine, CAM_NAME)
            except Exception as e:
                print(f"[warn] camera '{CAM_NAME}' unavailable: {e}")

        pos = await arm.get_end_position()
        print(f"[✓] Connected. Pose X={pos.x:.1f} Y={pos.y:.1f} Z={pos.z:.1f}")

        active = {"arm": [], "gripper": []}
        if not args.no_probe:
            active = await discover_channels(arm, gripper)
        if args.probe_only:
            print("\n[probe-only] done.")
            return

        # kinematics (static) for correct torque->wrench mapping later
        meta: Dict[str, Any] = {
            "object_label": object_label, "machine": VIAM_MACHINE_ADDRESS,
            "rates_hz": rates, "duration_s_per_rate": args.duration,
            "discovered_arm_queries": [list(c)[0] for c in active["arm"]],
            "discovered_gripper_queries": [list(c)[0] for c in active["gripper"]],
            "camera": bool(cam),
        }
        try:
            fmt, kin = await arm.get_kinematics()
            kin_path = os.path.join(RECORDINGS_DIR, "arm_kinematics.bin")
            with open(kin_path, "wb") as f:
                f.write(kin)
            meta["kinematics_format"] = str(fmt)
            meta["kinematics_file"] = os.path.basename(kin_path)
            print(f"[✓] saved arm kinematics ({len(kin)} bytes)")
        except Exception as e:
            print(f"[warn] get_kinematics unavailable: {e}")

        if args.move_travel:
            print("Moving to travel waypoint...")
            try:
                await Switch.from_robot(machine, "travel").set_position(2, timeout=40.0)
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"[warn] travel move failed: {e}")

        print(f"\nReady: '{object_label}'  rates={rates} Hz  {args.duration}s each"
              f"{'  + camera' if cam else ''}.")
        input("Press Enter to START recording...")

        samples: List[Dict[str, Any]] = []
        cam_rows: List[Dict[str, Any]] = []
        phase_ref: Dict[str, str] = {"phase": ""}
        base_phase = "TRAVEL_STABLE_HOLD" if args.move_travel else "HOLD_STATIC"
        frames_dir = None
        if cam:
            frames_dir = os.path.join(
                RECORDINGS_DIR, f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{object_label}")

        t0 = time.time()
        for hz in rates:
            phase = f"SWEEP_{hz:g}HZ" if args.hz_sweep else base_phase
            await record_segment(arm, gripper, cam, active, object_label, args.duration,
                                  hz, phase, t0, samples, cam_rows, frames_dir,
                                  args.cam_hz, phase_ref)

        save_dataset(samples, cam_rows, object_label, meta)


if __name__ == "__main__":
    asyncio.run(main())
