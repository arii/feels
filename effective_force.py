#!/usr/bin/env python3
"""
effective_force.py — Computes effective end-effector force and weight change
directly from xArm850 joint torque sensors and kinematics.
"""

import asyncio
import os
import time
import math
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viam.robot.client import RobotClient
from viam.components.arm import Arm

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
GRAVITY = 9.80665  # m/s^2


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


def ascii_gauge(val_g: float, max_g: float = 1000.0, width: int = 16) -> str:
    """Render an ASCII bar indicating weight change."""
    fraction = min(1.0, max(0.0, abs(val_g) / max_g))
    filled = int(fraction * width)
    empty = width - filled
    sign = "+" if val_g >= 0 else "-"
    return f"{sign}[{'█' * filled}{'░' * empty}]"


async def monitor_effective_force(duration_sec: float = 15.0, rate_hz: float = 5.0, output_path: str = "effective_force_graph.png", baseline_sec: float = 2.0):
    interval = 1.0 / rate_hz

    timestamps: List[float] = []
    loads_history: List[List[float]] = []
    forces_z: List[float] = []
    weights_g: List[float] = []

    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")

        pose = await arm.get_end_position()
        joints = await arm.get_joint_positions()

        # Compute radial distance from base axis (meters)
        r_xy = math.sqrt(pose.x**2 + pose.y**2) / 1000.0
        # Prevent zero division if straight vertical
        r_xy = max(r_xy, 0.05)

        print("\n" + "=" * 60)
        print("          EFFECTIVE END-EFFECTOR FORCE SENSING          ")
        print("=" * 60)
        print(f"  Kinematic Pose:    X={pose.x:.1f} mm, Y={pose.y:.1f} mm, Z={pose.z:.1f} mm")
        print(f"  Radial Lever Arm:  R_xy = {r_xy * 100:.1f} cm ({r_xy:.4f} m)")
        print(f"  Sensitivity:       100g payload produces ~{0.1 * GRAVITY * r_xy:.4f} Nm on Shoulder")
        print("-" * 60)
        print(f"[+] Calibrating baseline for {baseline_sec:.1f}s, then sampling for {duration_sec:.1f}s...")
        print("    >> TIP: Press or place weight on the end effector during sampling!")
        print("-" * 60)

        t0 = time.time()
        baseline_samples: List[List[float]] = []
        baseline_j2 = 0.0
        is_calibrated = False

        while True:
            now = time.time()
            elapsed = now - t0
            if elapsed > duration_sec:
                break

            loop_start = time.time()
            try:
                res = await arm.do_command({"load": ""})
                loads = res.get("load", [])
                if loads and len(loads) >= 6:
                    j2 = loads[1]  # Shoulder pitch torque

                    # Calibration phase
                    if elapsed <= baseline_sec:
                        baseline_samples.append(loads)
                        print(f"  [Calibrating {elapsed:4.1f}s] Shoulder Torque: {j2:8.4f} Nm")
                    else:
                        if not is_calibrated:
                            baseline_arr = np.array(baseline_samples)
                            baseline_j2 = float(np.mean(baseline_arr[:, 1]))
                            is_calibrated = True
                            print(f"\n[✓] Calibration complete! Baseline Shoulder: {baseline_j2:.4f} Nm\n")
                            print("  Time    | Eff. Force F_z | Equiv. Weight | Visual Gauge")
                            print("  ------- | -------------- | ------------- | ------------")

                        # Compute torque delta relative to baseline
                        delta_tau_j2 = j2 - baseline_j2

                        # In this quadrant, downward external force makes shoulder torque more negative
                        # Delta torque = -F_ext_z * r_xy  =>  F_ext_z = -delta_tau / r_xy
                        effective_f_z = -delta_tau_j2 / r_xy
                        weight_change_g = (effective_f_z / GRAVITY) * 1000.0

                        timestamps.append(elapsed)
                        loads_history.append(loads)
                        forces_z.append(effective_f_z)
                        weights_g.append(weight_change_g)

                        gauge = ascii_gauge(weight_change_g, max_g=500.0)
                        print(f"  {elapsed:5.1f}s  | {effective_f_z:8.2f} N    | {weight_change_g:8.1f} g   | {gauge}")

            except Exception as e:
                print(f"[!] Sensor read error: {e}")

            took = time.time() - loop_start
            await asyncio.sleep(max(0.01, interval - took))

    # Generate Visualization Graph
    if len(timestamps) >= 3:
        plot_effective_force(
            t=np.array(timestamps),
            forces=np.array(forces_z),
            weights=np.array(weights_g),
            loads=np.array(loads_history),
            r_xy=r_xy,
            output_path=output_path
        )
    else:
        print("[!] Insufficient data points recorded to plot.")


def plot_effective_force(t: np.ndarray, forces: np.ndarray, weights: np.ndarray, loads: np.ndarray, r_xy: float, output_path: str):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.patch.set_facecolor("#181818")

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#222222")
        ax.tick_params(colors="#e0e0e0", labelsize=10)
        ax.xaxis.label.set_color("#e0e0e0")
        ax.yaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.35, color="#555555")

    # Plot 1: Effective Force (Newtons)
    ax1.plot(t, forces, color="#00e5ff", linewidth=2.2, label="Effective Vertical Force $F_z$ (N)")
    ax1.axhline(0, color="#888888", linestyle=":", alpha=0.8)
    ax1.set_title("Effective End-Effector Force $F_z$ Derived from Arm Joint Torques", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Force $F_z$ (N)", fontsize=11)
    ax1.legend(loc="upper right", facecolor="#181818", edgecolor="#444444", labelcolor="#e0e0e0")

    # Plot 2: Equivalent Weight / Mass Change (Grams)
    ax2.plot(t, weights, color="#ff5252", linewidth=2.5, label=r"Equivalent Weight Change $\Delta m$ (grams)")
    ax2.axhline(0, color="#888888", linestyle=":", alpha=0.8)
    # Noise tolerance band (+/- 30g)
    ax2.fill_between(t, -30, 30, color="#ff5252", alpha=0.12, label="Baseline Noise Floor (±30g)")
    ax2.set_title(f"End-Effector Payload / Weight Change (Lever Arm $R_{{xy}} = {r_xy*100:.1f}$ cm)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Weight Change (grams)", fontsize=11)
    ax2.legend(loc="upper right", facecolor="#181818", edgecolor="#444444", labelcolor="#e0e0e0")

    # Plot 3: Raw Joint Torques
    ax3.plot(t, loads[:, 1], color="#4caf50", linewidth=1.8, label="Joint 2 (Shoulder Pitch)")
    ax3.plot(t, loads[:, 2], color="#ffb74d", linewidth=1.8, label="Joint 3 (Elbow Pitch)")
    ax3.plot(t, loads[:, 4], color="#ba68c8", linewidth=1.8, label="Joint 5 (Wrist Pitch)")
    ax3.set_title("Raw Underlying Joint Torques (Nm)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax3.set_ylabel("Torque (Nm)", fontsize=11)
    ax3.legend(loc="upper right", facecolor="#181818", edgecolor="#444444", labelcolor="#e0e0e0")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"\n[✓] Effective force graph generated: {output_path}")

    # Summary
    peak_force = np.max(np.abs(forces))
    peak_weight = np.max(np.abs(weights))
    print("\n" + "=" * 60)
    print("                MEASUREMENT SUMMARY                ")
    print("=" * 60)
    print(f"  Peak Effective Force:       {peak_force:.2f} N")
    print(f"  Peak Weight Detected:       {peak_weight:.1f} grams")
    print(f"  Force Resolution / Noise:   ~{30.0 / 1000.0 * GRAVITY:.2f} N (±30 grams)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Estimate effective end-effector force from arm joint torques.")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration to record in seconds (default: 15.0)")
    parser.add_argument("--hz", type=float, default=5.0, help="Sampling frequency (default: 5.0)")
    parser.add_argument("--baseline", type=float, default=2.0, help="Baseline calibration period in seconds (default: 2.0)")
    parser.add_argument("--output", type=str, default="effective_force_graph.png", help="Output plot filename")
    args = parser.parse_args()

    asyncio.run(monitor_effective_force(
        duration_sec=args.duration,
        rate_hz=args.hz,
        output_path=args.output,
        baseline_sec=args.baseline
    ))


if __name__ == "__main__":
    main()
