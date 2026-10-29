#!/usr/bin/env python3
"""
graph_weight.py — Record and graph arm joint torques (loads) to detect end-effector weight changes.

Even without a dedicated 6-axis F/T flange sensor, the xArm850 reports high-resolution
joint torques via `arm.do_command({"load": ""})`. Gravity acting on end-effector payload
produces measurable torque shifts on the pitch joints (Joint 2 Shoulder, Joint 3 Elbow, Joint 5 Wrist).
"""

import asyncio
import os
import time
import math
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def sample_torques(duration_sec: float = 15.0, rate_hz: float = 5.0) -> Dict[str, Any]:
    interval = 1.0 / rate_hz
    timestamps: List[float] = []
    loads_history: List[List[float]] = []

    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")

        # Get initial end-effector position to estimate lever arm
        pose = await arm.get_end_position()
        r_xy_m = math.sqrt(pose.x**2 + pose.y**2) / 1000.0  # Convert mm to meters
        print(f"Arm Pose: X={pose.x:.1f}mm, Y={pose.y:.1f}mm, Z={pose.z:.1f}mm | Lever arm R_xy = {r_xy_m*100:.1f} cm")

        print(f"\n[+] Recording joint torques for {duration_sec}s @ {rate_hz}Hz...")
        print("    >> TIP: Try placing or removing an object in the gripper to see the torque shift!")

        t0 = time.time()
        sample_idx = 0

        while True:
            now = time.time()
            elapsed = now - t0
            if elapsed > duration_sec:
                break

            loop_start = time.time()
            try:
                res = await arm.do_command({"load": ""})
                joint_loads = res.get("load", [])
                if joint_loads:
                    timestamps.append(elapsed)
                    loads_history.append(joint_loads)

                    if sample_idx % int(max(1, rate_hz)) == 0:
                        j2, j3, j5 = joint_loads[1], joint_loads[2], joint_loads[4]
                        print(f"  [{elapsed:5.1f}s] J2(Shoulder): {j2:8.3f} Nm | J3(Elbow): {j3:8.3f} Nm | J5(Wrist): {j5:7.3f} Nm")
                    sample_idx += 1
            except Exception as e:
                print(f"[!] Error querying load: {e}")

            took = time.time() - loop_start
            sleep_time = max(0.01, interval - took)
            await asyncio.sleep(sleep_time)

    return {
        "timestamps": np.array(timestamps),
        "loads": np.array(loads_history),
        "lever_arm_m": max(r_xy_m, 0.05),
        "pose": pose
    }


def generate_plot(data: Dict[str, Any], output_path: str, baseline_sec: float = 2.0):
    t = data["timestamps"]
    loads = data["loads"]
    r_xy = data["lever_arm_m"]

    if len(t) < 5:
        print("[!] Not enough samples collected to plot.")
        return

    # Baseline calculation: average of the first baseline_sec seconds
    baseline_mask = t <= baseline_sec
    if not np.any(baseline_mask):
        baseline_mask = np.zeros(len(t), dtype=bool)
        baseline_mask[:min(5, len(t))] = True

    baseline_loads = np.mean(loads[baseline_mask, :], axis=0)
    delta_loads = loads - baseline_loads

    # J2 (Shoulder) and J3 (Elbow) are primary gravity load bearers
    # Torque delta due to vertical mass delta: Δτ ≈ ΔF * r_xy = (Δm * 9.81) * r_xy
    # => Δm (grams) ≈ (Δτ_J2 / (r_xy * 9.81)) * 1000
    # Note: J2 torque is negative in this kinematic quadrant, so downward force makes it more negative.
    delta_tau_j2 = delta_loads[:, 1]
    estimated_mass_g = (-delta_tau_j2 / (r_xy * 9.81)) * 1000.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.patch.set_facecolor("#1e1e1e")

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#252526")
        ax.tick_params(colors="#d4d4d4")
        ax.xaxis.label.set_color("#d4d4d4")
        ax.yaxis.label.set_color("#d4d4d4")
        ax.title.set_color("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.3, color="#555555")

    # Plot 1: Absolute Torques
    ax1.plot(t, loads[:, 1], label="Joint 2 (Shoulder Pitch)", color="#4fc1ff", linewidth=2)
    ax1.plot(t, loads[:, 2], label="Joint 3 (Elbow Pitch)", color="#4ec9b0", linewidth=2)
    ax1.plot(t, loads[:, 4], label="Joint 5 (Wrist Pitch)", color="#dcdcaa", linewidth=2)
    ax1.set_title("Absolute Joint Torques (Nm) Over Time", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Torque (Nm)")
    ax1.legend(facecolor="#1e1e1e", edgecolor="#555555", labelcolor="#d4d4d4")

    # Plot 2: Delta Torque
    ax2.plot(t, delta_loads[:, 1], label="Δ Joint 2 (Shoulder)", color="#4fc1ff", linewidth=2)
    ax2.plot(t, delta_loads[:, 2], label="Δ Joint 3 (Elbow)", color="#4ec9b0", linewidth=2)
    ax2.plot(t, delta_loads[:, 4], label="Δ Joint 5 (Wrist)", color="#dcdcaa", linewidth=2)
    ax2.axhline(0, color="#888888", linestyle=":", alpha=0.7)
    ax2.axvspan(0, baseline_sec, color="#4fc1ff", alpha=0.1, label="Baseline Calibration Window")
    ax2.set_title("Torque Shift Relative to Baseline (Δ Nm)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Δ Torque (Nm)")
    ax2.legend(facecolor="#1e1e1e", edgecolor="#555555", labelcolor="#d4d4d4")

    # Plot 3: Estimated Payload Weight Change
    ax3.plot(t, estimated_mass_g, label="Estimated Payload Δ (grams)", color="#f44747", linewidth=2.5)
    ax3.axhline(0, color="#888888", linestyle=":", alpha=0.7)
    ax3.set_title(f"Estimated End-Effector Weight Change (Lever Arm r={r_xy*100:.1f} cm)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Δ Weight (grams)")
    ax3.legend(facecolor="#1e1e1e", edgecolor="#555555", labelcolor="#d4d4d4")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n[✓] Graph saved successfully to: {output_path}")

    # Summary Statistics
    peak_delta_j2 = np.max(np.abs(delta_loads[:, 1]))
    peak_mass_g = np.max(np.abs(estimated_mass_g))
    print("\n" + "=" * 50)
    print("           WEIGHT DETECTION SENSITIVITY           ")
    print("=" * 50)
    print(f"  Baseline Shoulder Torque: {baseline_loads[1]:.4f} Nm")
    print(f"  Baseline Elbow Torque:    {baseline_loads[2]:.4f} Nm")
    print(f"  Peak Shoulder Δ Torque:   {peak_delta_j2:.4f} Nm")
    print(f"  Peak Estimated Weight Δ:  {peak_mass_g:.1f} grams")
    if peak_mass_g > 30.0:
        print("  -> Significant weight/load variation DETECTED!")
    else:
        print("  -> Load remained stable / no significant weight change during window.")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Graph arm joint torques to detect weight changes.")
    parser.add_argument("--duration", type=float, default=15.0, help="Sampling duration in seconds (default: 15.0)")
    parser.add_argument("--hz", type=float, default=5.0, help="Sampling frequency in Hz (default: 5.0)")
    parser.add_argument("--output", type=str, default="weight_torque_graph.png", help="Output graph path (default: weight_torque_graph.png)")
    parser.add_argument("--baseline", type=float, default=2.0, help="Baseline calibration window in seconds (default: 2.0)")
    args = parser.parse_args()

    data = asyncio.run(sample_torques(duration_sec=args.duration, rate_hz=args.hz))
    generate_plot(data, args.output, baseline_sec=args.baseline)


if __name__ == "__main__":
    main()
