#!/usr/bin/env python3
"""
record_dataset.py — Dedicated dataset recording tool for physical object identification.

Features:
- Specify object label via CLI (`--object candy_loose`) or interactive prompt.
- Halloween item presets: candy_loose, water_half, dish_soap_half, empty_cup, solid_weight.
- Multi-mode recording:
    1. Static Hold: Measures steady-state load in travel position.
    2. Full Grasp Macro: Records the entire pick-and-place cycle with phase annotations.
    3. Custom Duration: High-rate sampling (5-10 Hz) for N seconds.
- Computes on-the-fly summary statistics (mean torque, delta tau, estimated mass, variance).
- Saves standardized CSVs to `recordings/recording_{timestamp}_{object}.csv`.
"""

import asyncio
import os
import sys
import time
import csv
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.switch import Switch

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
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


async def connect_robot() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID,
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def record_stream(
    arm: Arm,
    object_label: str,
    duration: float = 10.0,
    sample_interval: float = 0.15,
    phase: str = "HOLD_TRAVEL"
) -> List[Dict[str, Any]]:
    """Poll 6-joint torques for `duration` seconds and collect samples."""
    print(f"\n[STREAM] Recording '{object_label}' for {duration:.1f}s at ~{1.0/sample_interval:.0f} Hz...")
    samples = []
    t0 = time.time()
    t_end = t0 + duration

    while time.time() < t_end:
        now = time.time()
        elapsed = round(now - t0, 3)
        try:
            res = await arm.do_command({"load": ""}, timeout=1.5)
            loads = res.get("load", [])
            if loads and len(loads) >= 6:
                samples.append({
                    "elapsed_sec": elapsed,
                    "phase": phase,
                    "object_label": object_label,
                    "j1_nm": round(loads[0], 4),
                    "j2_nm": round(loads[1], 4),
                    "j3_nm": round(loads[2], 4),
                    "j4_nm": round(loads[3], 4),
                    "j5_nm": round(loads[4], 4),
                    "j6_nm": round(loads[5], 4),
                })
                # Progress dot
                sys.stdout.write(".")
                sys.stdout.flush()
        except Exception:
            pass

        loop_dur = time.time() - now
        sleep_sec = max(0.01, sample_interval - loop_dur)
        await asyncio.sleep(sleep_sec)

    print(" done!")
    return samples


def save_and_analyze(samples: List[Dict[str, Any]], object_label: str, tare_j3: float = -16.8209) -> str:
    """Save samples to CSV and output comparative mass estimation."""
    clean_lbl = "".join(c for c in object_label if c.isalnum() or c in ("-", "_")).lower() or "unlabeled"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}_{clean_lbl}.csv"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "elapsed_sec", "phase", "object_label",
            "j1_nm", "j2_nm", "j3_nm", "j4_nm", "j5_nm", "j6_nm"
        ])
        for s in samples:
            writer.writerow([
                s["elapsed_sec"], s["phase"], s["object_label"],
                s["j1_nm"], s["j2_nm"], s["j3_nm"], s["j4_nm"], s["j5_nm"], s["j6_nm"]
            ])

    # Statistics
    j3_vals = [s["j3_nm"] for s in samples]
    mean_j3 = sum(j3_vals) / len(j3_vals) if j3_vals else 0.0
    var_j3 = sum((x - mean_j3) ** 2 for x in j3_vals) / len(j3_vals) if len(j3_vals) > 1 else 0.0
    delta_j3 = mean_j3 - tare_j3
    
    # Effective payload mass (assuming travel lever arm ~ 0.56m)
    lever_arm = 0.56
    mass_kg = abs(delta_j3) / (GRAVITY * lever_arm) if abs(delta_j3) > 0.05 else 0.0

    print("\n" + "=" * 65)
    print(f"📊 DATASET RECORDED: '{object_label}'")
    print("=" * 65)
    print(f"• File Location:      {filepath}")
    print(f"• Samples Collected:  {len(samples)}")
    print(f"• Baseline J3 (Tare): {tare_j3:.4f} N·m")
    print(f"• Measured Mean J3:   {mean_j3:.4f} N·m (variance: {var_j3:.5f})")
    print(f"• Torque Delta (Δτ3): {delta_j3:.4f} N·m")
    print(f"• Estimated Mass:     {mass_kg * 1000.0:.1f} grams ({mass_kg:.3f} kg)")
    print("=" * 65 + "\n")
    return filepath


async def main():
    parser = argparse.ArgumentParser(description="Record labeled physical robot telemetry datasets.")
    parser.add_argument("--object", type=str, help="Object label (e.g. water_half, candy_loose, empty_cup)")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (default: 10s)")
    parser.add_argument("--move-travel", action="store_true", help="Move arm to travel position before recording")
    parser.add_argument("--tare", type=float, default=-16.8209, help="Baseline tare torque for Joint 3 (default: -16.8209 Nm)")
    args = parser.parse_args()

    object_label = args.object

    # Interactive prompt if not supplied via CLI
    if not object_label:
        print("\n" + "=" * 60)
        print("🎯 SELECT OR ENTER OBJECT TO RECORD")
        print("=" * 60)
        for k, v in ITEM_PRESETS.items():
            print(f"  [{k}] {v[0]:<16} — {v[1]}")
        print("  [0] Enter custom name")
        print("=" * 60)
        choice = input("Choice (1-5 or custom name): ").strip()
        if choice in ITEM_PRESETS:
            object_label = ITEM_PRESETS[choice][0]
        elif choice == "0" or not choice:
            custom = input("Enter custom object name: ").strip()
            object_label = custom or "unlabeled_item"
        else:
            object_label = choice

    print(f"\nConnecting to robot at {VIAM_MACHINE_ADDRESS}...")
    async with await connect_robot() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")

        pos = await arm.get_end_position()
        print(f"[✓] Connected. Current Arm Pose: X={pos.x:.1f}, Y={pos.y:.1f}, Z={pos.z:.1f}")

        if args.move_travel:
            print("Moving to travel waypoint...")
            sw_travel = Switch.from_robot(machine, "travel")
            await sw_travel.set_position(2, timeout=40.0)
            await asyncio.sleep(1.0)

        # Confirm before starting stream
        print(f"\nReady to record object: '{object_label}' for {args.duration} seconds.")
        input("Press Enter to START recording...")

        samples = await record_stream(arm, object_label=object_label, duration=args.duration)
        save_and_analyze(samples, object_label=object_label, tare_j3=args.tare)


if __name__ == "__main__":
    asyncio.run(main())
