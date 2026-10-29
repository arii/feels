#!/usr/bin/env python3
"""
repeatable_grasp.py — Automated repeatable pick-and-place routine with labeled telemetry recording.

Supports:
- Pick-and-place macro using Viam arm-position-saver switches:
    home -> approach -> grasp -> travel -> place -> travel -> home
- Synchronized telemetry data recording with customizable object labels:
    `--record --object water_half` or `--record --object candy_loose`
- Per-phase ground truth tagging:
    INIT, APPROACH, REACH_GRASP, GRIPPER_CLOSE, LIFT_TO_TRAVEL, HOLD_TRAVEL, DESCEND_PLACE, GRIPPER_OPEN, RETRACT_TRAVEL, RETURN_HOME
- Automatic payload mass estimation & torque delta reporting saved directly to CSV.
"""

import asyncio
import os
import sys
import time
import math
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

# Switch components corresponding to arm-position-saver waypoints
WAYPOINTS = {
    "home-pose": "Home / Staging Position",
    "approach": "Pre-Grasp Approach Alignment",
    "grasp": "Object Contact / Grasp Position",
    "travel": "Safe Transit / Elevated Waypoint",
    "place": "Object Placement / Dropoff Target",
}


class TelemetryRecorder:
    """
    Background asynchronous telemetry recorder that tracks 6-joint torques,
    gripper state, arm coordinates, and execution phases.
    """
    def __init__(self, arm: Arm, gripper: Gripper, object_label: str = "unlabeled", sample_interval: float = 0.15):
        self.arm = arm
        self.gripper = gripper
        self.object_label = object_label
        self.sample_interval = sample_interval
        self.current_phase = "INIT"
        self.samples: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.t0 = 0.0

    def set_phase(self, phase: str):
        self.current_phase = phase
        print(f"[REC] Phase -> {phase}")

    async def _poll_loop(self):
        while self._running:
            t = time.time() - self.t0
            try:
                # Concurrent high-rate query: Cartesian Pose + Joint Angles + Joint Torques + Gripper
                pos, joints, load_res, grip_status = await asyncio.gather(
                    self.arm.get_end_position(),
                    self.arm.get_joint_positions(),
                    self.arm.do_command({"load": ""}),
                    self.gripper.is_holding_something()
                )
                loads = load_res.get("load", [])
                j_vals = joints.values if joints else []
                grip_meta = grip_status.meta if grip_status else {}
                grip_pos = grip_meta.get("position", 0.0)

                if loads and len(loads) >= 6 and len(j_vals) >= 6:
                    self.samples.append({
                        "elapsed_sec": round(t, 3),
                        "phase": self.current_phase,
                        "object_label": self.object_label,
                        # Cartesian Trajectory
                        "tcp_x_mm": round(pos.x, 2),
                        "tcp_y_mm": round(pos.y, 2),
                        "tcp_z_mm": round(pos.z, 2),
                        "tcp_ox": round(pos.o_x, 4),
                        "tcp_oy": round(pos.o_y, 4),
                        "tcp_oz": round(pos.o_z, 4),
                        "tcp_theta_deg": round(pos.theta, 2),
                        # Joint Angle Trajectory (deg)
                        "q1_deg": round(j_vals[0], 2),
                        "q2_deg": round(j_vals[1], 2),
                        "q3_deg": round(j_vals[2], 2),
                        "q4_deg": round(j_vals[3], 2),
                        "q5_deg": round(j_vals[4], 2),
                        "q6_deg": round(j_vals[5], 2),
                        # 6-Joint Loads / Torques (Nm)
                        "tau1_nm": round(loads[0], 4),
                        "tau2_nm": round(loads[1], 4),
                        "tau3_nm": round(loads[2], 4),
                        "tau4_nm": round(loads[3], 4),
                        "tau5_nm": round(loads[4], 4),
                        "tau6_nm": round(loads[5], 4),
                        # Gripper
                        "gripper_holding": grip_status.is_holding_something,
                        "gripper_pos": round(grip_pos, 1),
                    })
            except Exception:
                pass
            await asyncio.sleep(self.sample_interval)

    def start(self):
        self._running = True
        self.t0 = time.time()
        self._task = asyncio.create_task(self._poll_loop())
        print(f"[REC] Full trajectory & torque recording started for: '{self.object_label}'")

    async def stop(self) -> Optional[str]:
        self._running = False
        if self._task:
            try:
                await self._task
            except Exception:
                pass

        if not self.samples:
            print("[REC] Warning: No telemetry samples captured.")
            return None

        clean_lbl = "".join(c for c in self.object_label if c.isalnum() or c in ("-", "_")).lower() or "unlabeled"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_{clean_lbl}.csv"
        filepath = os.path.join(RECORDINGS_DIR, filename)

        headers = [
            "elapsed_sec", "phase", "object_label",
            "tcp_x_mm", "tcp_y_mm", "tcp_z_mm", "tcp_ox", "tcp_oy", "tcp_oz", "tcp_theta_deg",
            "q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg",
            "tau1_nm", "tau2_nm", "tau3_nm", "tau4_nm", "tau5_nm", "tau6_nm",
            "gripper_holding", "gripper_pos"
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for s in self.samples:
                writer.writerow([
                    s["elapsed_sec"], s["phase"], s["object_label"],
                    s["tcp_x_mm"], s["tcp_y_mm"], s["tcp_z_mm"],
                    s["tcp_ox"], s["tcp_oy"], s["tcp_oz"], s["tcp_theta_deg"],
                    s["q1_deg"], s["q2_deg"], s["q3_deg"], s["q4_deg"], s["q5_deg"], s["q6_deg"],
                    s["tau1_nm"], s["tau2_nm"], s["tau3_nm"], s["tau4_nm"], s["tau5_nm"], s["tau6_nm"],
                    s["gripper_holding"], s["gripper_pos"]
                ])

        print(f"\n[REC] Saved {len(self.samples)} trajectory+torque samples to: {filepath}")
        self._analyze_recording(filepath)
        return filepath

    def _analyze_recording(self, filepath: str):
        """Compute quick summary statistics from the recorded session."""
        init_samples = [s["tau3_nm"] for s in self.samples if s["phase"] in ("INIT", "APPROACH")]
        stable_samples = [s["tau3_nm"] for s in self.samples if s["phase"] == "TRAVEL_STABLE_HOLD"]
        travel_samples = stable_samples if stable_samples else [s["tau3_nm"] for s in self.samples if "TRAVEL" in s["phase"]]
        z_vals = [s["tcp_z_mm"] for s in self.samples]

        print("=" * 70)
        print(f"📊 TRAJECTORY & TORQUE DATASET SUMMARY — '{self.object_label}'")
        print("=" * 70)
        print(f"• Total Data Points:     {len(self.samples)}")
        print(f"• File Location:         {filepath}")
        if z_vals:
            print(f"• Z-Trajectory Range:    {min(z_vals):.1f} mm -> {max(z_vals):.1f} mm (span: {max(z_vals)-min(z_vals):.1f} mm)")

        if init_samples and travel_samples:
            tare_j3 = sum(init_samples) / len(init_samples)
            loaded_j3 = sum(travel_samples) / len(travel_samples)
            var_hold = sum((x - loaded_j3) ** 2 for x in travel_samples) / len(travel_samples) if len(travel_samples) > 1 else 0.0
            std_hold = math.sqrt(var_hold)
            delta_j3 = loaded_j3 - tare_j3
            r_xy = 0.56
            mass_kg = abs(delta_j3) / (GRAVITY * r_xy) if abs(delta_j3) > 0.05 else 0.0
            print(f"• Baseline J3 (Tare):    {tare_j3:.4f} N·m")
            print(f"• Fixed Stop J3 (Stable):{loaded_j3:.4f} N·m (±{std_hold:.4f} N·m, {len(travel_samples)} points)")
            print(f"• Delta Torque (Δτ3):    {delta_j3:.4f} N·m")
            print(f"• Estimated Mass:        {mass_kg * 1000.0:.1f} grams ({mass_kg:.3f} kg)")
        print("=" * 70 + "\n")


async def connect_robot() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID,
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def get_pose_info(sw: Switch) -> Dict[str, Any]:
    """Retrieve saved Cartesian coordinates from an arm-position-saver switch."""
    try:
        cfg = await sw.do_command({"cfg": True}, timeout=5.0)
        return {
            "point": cfg.get("point", {}),
            "orientation": cfg.get("orientation", {}),
        }
    except Exception as e:
        return {"error": str(e)}


async def move_to_waypoint(machine: RobotClient, arm: Arm, name: str, timeout: float = 45.0) -> bool:
    """Trigger an arm-position-saver switch (position 2 = 'go to') and await completion."""
    print(f"\n[MOVE] Commanding arm to '{name}'...")
    try:
        sw = Switch.from_robot(machine, name)
        t0 = time.time()
        await sw.set_position(2, timeout=timeout)
        duration = time.time() - t0

        try:
            pos = await arm.get_end_position()
            print(f"[✓] Arrived at '{name}' in {duration:.1f}s -> (X: {pos.x:.1f}, Y: {pos.y:.1f}, Z: {pos.z:.1f})")
        except Exception:
            print(f"[✓] Completed move to '{name}' in {duration:.1f}s")
        return True
    except Exception as e:
        print(f"[✗] Failed moving to '{name}': {e}")
        return False


async def open_gripper(gripper: Gripper) -> bool:
    """Open gripper jaws."""
    print("[GRIPPER] Opening gripper...")
    try:
        await gripper.open(timeout=10.0)
        await asyncio.sleep(0.5)
        status = await gripper.is_holding_something()
        print(f"[✓] Gripper opened (holding: {status.is_holding_something})")
        return True
    except Exception as e:
        print(f"[✗] Gripper open failed: {e}")
        return False


async def close_gripper(gripper: Gripper) -> bool:
    """Close gripper jaws (grab object)."""
    print("[GRIPPER] Closing gripper (grab)...")
    try:
        holding = await gripper.grab(timeout=10.0)
        await asyncio.sleep(0.5)
        print(f"[✓] Gripper closed (grab result: {holding})")
        return True
    except Exception as e:
        print(f"[✗] Gripper grab failed: {e}")
        return False


async def inspect_resources(machine: RobotClient):
    """Print all discovered arm-position-saver switches and current poses."""
    print("\n" + "=" * 70)
    print("📍 DISCOVERED VIAM ARM-SAVER WAYPOINTS & ROBOT STATUS")
    print("=" * 70)

    arm = Arm.from_robot(machine, "arm")
    cur_pos = await arm.get_end_position()
    print(f"Current Arm Pose: X={cur_pos.x:.2f}mm, Y={cur_pos.y:.2f}mm, Z={cur_pos.z:.2f}mm")

    gripper = Gripper.from_robot(machine, "gripper")
    grip_status = await gripper.is_holding_something()
    print(f"Gripper Status:   is_holding={grip_status.is_holding_something}, meta={grip_status.meta}")
    print("-" * 70)

    for name, desc in WAYPOINTS.items():
        try:
            sw = Switch.from_robot(machine, name)
            info = await get_pose_info(sw)
            pt = info.get("point", {})
            ori = info.get("orientation", {})
            print(f"• {name:<12} [{desc}]")
            print(f"    Point:       X: {pt.get('X', 0):.2f}, Y: {pt.get('Y', 0):.2f}, Z: {pt.get('Z', 0):.2f}")
            print(f"    Orientation: th: {ori.get('th', 0):.2f}°, (x: {ori.get('x', 0):.3f}, y: {ori.get('y', 0):.3f}, z: {ori.get('z', 0):.3f})")
        except Exception as e:
            print(f"• {name:<12} [ERROR: {e}]")
    print("=" * 70 + "\n")


async def wait_for_stability_in_travel(
    arm: Arm,
    recorder: Optional[TelemetryRecorder] = None,
    stable_duration: float = 3.0,
    max_settle_time: float = 3.0,
    variance_threshold: float = 0.04
) -> Dict[str, float]:
    """
    Monitors the arm in the travel pose until physical dynamic oscillations
    (fluid sloshing, rattling, mechanical vibrations) have settled, then
    performs a fixed stop dwell during which steady-state trajectory and torques
    are recorded.
    """
    if recorder:
        recorder.set_phase("TRAVEL_SETTLING")
    print(f"\n--- STEP 5b: TRAVEL POSE STABILIZATION (Threshold: σ < {variance_threshold} N·m) ---")

    settle_t0 = time.time()
    recent_j3 = []
    is_stable = False

    while (time.time() - settle_t0) < max_settle_time:
        try:
            res = await arm.do_command({"load": ""}, timeout=1.0)
            loads = res.get("load", [])
            if loads and len(loads) >= 6:
                recent_j3.append(loads[2])
                if len(recent_j3) > 5:
                    recent_j3.pop(0)
                if len(recent_j3) >= 4:
                    mean_val = sum(recent_j3) / len(recent_j3)
                    var_val = sum((x - mean_val) ** 2 for x in recent_j3) / len(recent_j3)
                    std_val = math.sqrt(var_val)
                    if std_val < variance_threshold:
                        is_stable = True
                        elapsed_settle = time.time() - settle_t0
                        print(f"[✓] Transients settled in {elapsed_settle:.2f}s (std: {std_val:.4f} N·m)")
                        break
        except Exception:
            pass
        await asyncio.sleep(0.1)

    if not is_stable:
        print(f"[!] Settling window reached {max_settle_time:.1f}s — proceeding to fixed stop dwell.")

    # Phase: Fixed stop dwell
    if recorder:
        recorder.set_phase("TRAVEL_STABLE_HOLD")
    print(f"\n--- STEP 5c: 🛑 FIXED STOP DWELL IN TRAVEL POSE ({stable_duration:.1f}s) ---")
    
    hold_samples = []
    hold_t0 = time.time()
    while (time.time() - hold_t0) < stable_duration:
        try:
            res = await arm.do_command({"load": ""}, timeout=1.0)
            loads = res.get("load", [])
            if loads and len(loads) >= 6:
                hold_samples.append(loads[2])
        except Exception:
            pass
        await asyncio.sleep(0.12)

    stats = {}
    if hold_samples:
        stats["mean_j3"] = sum(hold_samples) / len(hold_samples)
        stats["std_j3"] = math.sqrt(sum((x - stats["mean_j3"]) ** 2 for x in hold_samples) / len(hold_samples))
        print(f"[✓] Fixed stop complete: Mean J3={stats['mean_j3']:.4f} N·m (±{stats['std_j3']:.4f} N·m, {len(hold_samples)} samples)")

    return stats


async def run_grasp_cycle(
    machine: RobotClient,
    arm: Arm,
    gripper: Gripper,
    cycle_idx: int = 1,
    step_delay: float = 1.0,
    recorder: Optional[TelemetryRecorder] = None,
    stable_hold: float = 3.0,
    settle_threshold: float = 0.04,
) -> bool:
    """
    Executes one complete pick-and-place sequence with phase-synchronized recording:
    home -> approach -> grasp (close) -> travel (hold & weigh) -> place (open) -> travel -> home
    """
    print("\n" + "#" * 60)
    print(f"🚀 EXECUTING REPEATABLE GRASP CYCLE #{cycle_idx}")
    if recorder:
        print(f"📦 Object Under Test: '{recorder.object_label}'")
    print("#" * 60)

    # 1. Initial State: Gripper Open at Home
    if recorder:
        recorder.set_phase("INIT")
    print("\n--- STEP 1: INITIALIZE AT HOME ---")
    await open_gripper(gripper)
    if not await move_to_waypoint(machine, arm, "home-pose"):
        return False
    await asyncio.sleep(step_delay)

    # 2. Approach: Lower/Align towards object
    if recorder:
        recorder.set_phase("APPROACH")
    print("\n--- STEP 2: APPROACH OBJECT ---")
    if not await move_to_waypoint(machine, arm, "approach"):
        return False
    await asyncio.sleep(step_delay)

    # 3. Grasp: Move into contact position
    if recorder:
        recorder.set_phase("REACH_GRASP")
    print("\n--- STEP 3: REACH GRASP POSITION ---")
    if not await move_to_waypoint(machine, arm, "grasp"):
        return False
    await asyncio.sleep(step_delay)

    # 4. Actuate Gripper: Grab object
    if recorder:
        recorder.set_phase("GRIPPER_CLOSE")
    print("\n--- STEP 4: CLOSE GRIPPER ---")
    await close_gripper(gripper)
    await asyncio.sleep(step_delay)

    # 5. Travel: Lift payload and move to transit waypoint
    if recorder:
        recorder.set_phase("LIFT_TO_TRAVEL")
    print("\n--- STEP 5: TRAVEL WITH PAYLOAD ---")
    if not await move_to_waypoint(machine, arm, "travel"):
        return False

    # 5b. Fixed Stop at Travel Pose (Stability Gated)
    await wait_for_stability_in_travel(
        arm=arm,
        recorder=recorder,
        stable_duration=stable_hold,
        variance_threshold=settle_threshold
    )

    # 6. Place: Descend to target placement spot
    if recorder:
        recorder.set_phase("DESCEND_PLACE")
    print("\n--- STEP 6: DESCEND TO PLACE POSITION ---")
    if not await move_to_waypoint(machine, arm, "place"):
        return False
    await asyncio.sleep(step_delay)

    # 7. Release: Open gripper to deposit object
    if recorder:
        recorder.set_phase("GRIPPER_OPEN")
    print("\n--- STEP 7: RELEASE OBJECT ---")
    await open_gripper(gripper)
    await asyncio.sleep(step_delay)

    # 8. Retract: Ascend back to travel height to clear deposit zone safely
    if recorder:
        recorder.set_phase("RETRACT_TRAVEL")
    print("\n--- STEP 8: RETRACT TO TRAVEL HEIGHT ---")
    if not await move_to_waypoint(machine, arm, "travel"):
        return False
    await asyncio.sleep(step_delay)

    # 9. Return: Return to home pose ready for next run
    if recorder:
        recorder.set_phase("RETURN_HOME")
    print("\n--- STEP 9: RETURN TO HOME POSE ---")
    if not await move_to_waypoint(machine, arm, "home-pose"):
        return False
    await asyncio.sleep(step_delay)

    print("\n" + "=" * 60)
    print(f"✅ CYCLE #{cycle_idx} COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")
    return True


async def main():
    parser = argparse.ArgumentParser(
        description="Repeatable pick-and-place with labeled telemetry data recording."
    )
    parser.add_argument(
        "--object",
        type=str,
        help="Specify the object label being recorded (e.g. water_half, candy_loose, empty_cup, solid_weight)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record synchronized 6-joint telemetry data during the cycle",
    )
    parser.add_argument(
        "--stable-hold",
        type=float,
        default=3.0,
        help="Fixed stop dwell duration (seconds) once settled in travel position (default: 3.0s)",
    )
    parser.add_argument(
        "--settle-threshold",
        type=float,
        default=0.04,
        help="Torque standard deviation threshold in Nm to confirm stabilization (default: 0.04 Nm)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of pick-and-place cycles to execute (default: 1)",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["home-pose", "approach", "grasp", "travel", "place"],
        help="Execute a single arm waypoint move instead of the full cycle",
    )
    parser.add_argument(
        "--gripper",
        type=str,
        choices=["open", "grab", "close"],
        help="Actuate gripper manually ('open' or 'grab')",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Settling delay (seconds) between trajectory steps (default: 1.0s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect resources and planned trajectory without triggering motions",
    )

    args = parser.parse_args()

    # Determine object label if recording requested
    object_label = args.object
    if args.record and not object_label:
        print("\n" + "=" * 60)
        print("💡 PRESET ITEM LABELS:")
        print("   1. empty_cup       (baseline tare)")
        print("   2. water_half      (fluid slosh)")
        print("   3. dish_soap_half  (viscous fluid)")
        print("   4. candy_loose     (rattling chatter)")
        print("   5. solid_weight    (rigid heavy mass)")
        print("=" * 60)
        user_input = input("Enter object label to record: ").strip()
        object_label = user_input or "unlabeled"

    print("Connecting to Viam machine...")
    async with await connect_robot() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")

        await inspect_resources(machine)

        if args.dry_run:
            print("[DRY RUN] Motion execution skipped.")
            return

        # Single gripper command
        if args.gripper:
            if args.gripper == "open":
                await open_gripper(gripper)
            else:
                await close_gripper(gripper)
            return

        # Single waypoint step command
        if args.step:
            await move_to_waypoint(machine, arm, args.step)
            return

        # Full automated repeatable cycles
        for c in range(1, args.cycles + 1):
            recorder = None
            if args.record or object_label:
                recorder = TelemetryRecorder(arm=arm, gripper=gripper, object_label=object_label or "unlabeled")
                recorder.start()

            try:
                success = await run_grasp_cycle(
                    machine, arm, gripper,
                    cycle_idx=c,
                    step_delay=args.delay,
                    recorder=recorder,
                    stable_hold=args.stable_hold,
                    settle_threshold=args.settle_threshold,
                )
                if not success:
                    print(f"[!] Cycle #{c} aborted due to an error.")
                    sys.exit(1)
            finally:
                if recorder:
                    await recorder.stop()

            if c < args.cycles:
                print(f"Pausing 2s before next cycle...")
                await asyncio.sleep(2.0)


if __name__ == "__main__":
    asyncio.run(main())
