#!/usr/bin/env python3
"""
repeatable_grasp.py — Automated repeatable pick-and-place routine using Viam arm-position-saver switches.

Sequence:
  1. Home Pose     (`home-pose`)  — Gripper Open
  2. Approach      (`approach`)   — Move above / align with target object
  3. Grasp Pose    (`grasp`)      — Move down to object
  4. Grasp Actuate (`gripper`)    — Close / grab object
  5. Travel Pose   (`travel`)     — Elevate to safe transit height & move to destination
  6. Place Pose    (`place`)      — Move down to placement target
  7. Place Release (`gripper`)    — Open gripper to release object
  8. Retract       (`travel`)     — Lift up to clear placed object
  9. Home Pose     (`home-pose`)  — Return to ready state
"""

import asyncio
import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.switch import Switch

load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")

# Switch components corresponding to arm-position-saver waypoints
WAYPOINTS = {
    "home-pose": "Home / Staging Position",
    "approach": "Pre-Grasp Approach Alignment",
    "grasp": "Object Contact / Grasp Position",
    "travel": "Safe Transit / Elevated Waypoint",
    "place": "Object Placement / Dropoff Target",
}


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
        # Position 2 corresponds to 'go to' in erh:vmodutils:arm-position-saver
        t0 = time.time()
        await sw.set_position(2, timeout=timeout)
        duration = time.time() - t0

        # Verify reached position
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
        # Brief settling delay
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


async def run_grasp_cycle(
    machine: RobotClient,
    arm: Arm,
    gripper: Gripper,
    cycle_idx: int = 1,
    step_delay: float = 1.0,
) -> bool:
    """
    Executes one complete pick-and-place sequence:
    home -> approach -> grasp (close) -> travel -> place (open) -> travel -> home
    """
    print("\n" + "#" * 60)
    print(f"🚀 EXECUTING REPEATABLE GRASP CYCLE #{cycle_idx}")
    print("#" * 60)

    # 1. Initial State: Gripper Open at Home
    print("\n--- STEP 1: INITIALIZE AT HOME ---")
    await open_gripper(gripper)
    if not await move_to_waypoint(machine, arm, "home-pose"):
        return False
    await asyncio.sleep(step_delay)

    # 2. Approach: Lower/Align towards object
    print("\n--- STEP 2: APPROACH OBJECT ---")
    if not await move_to_waypoint(machine, arm, "approach"):
        return False
    await asyncio.sleep(step_delay)

    # 3. Grasp: Move into contact position
    print("\n--- STEP 3: REACH GRASP POSITION ---")
    if not await move_to_waypoint(machine, arm, "grasp"):
        return False
    await asyncio.sleep(step_delay)

    # 4. Actuate Gripper: Grab object
    print("\n--- STEP 4: CLOSE GRIPPER ---")
    await close_gripper(gripper)
    await asyncio.sleep(step_delay)

    # 5. Travel: Lift payload and move to transit waypoint
    print("\n--- STEP 5: TRAVEL WITH PAYLOAD ---")
    if not await move_to_waypoint(machine, arm, "travel"):
        return False
    await asyncio.sleep(step_delay)

    # 6. Place: Descend to target placement spot
    print("\n--- STEP 6: DESCEND TO PLACE POSITION ---")
    if not await move_to_waypoint(machine, arm, "place"):
        return False
    await asyncio.sleep(step_delay)

    # 7. Release: Open gripper to deposit object
    print("\n--- STEP 7: RELEASE OBJECT ---")
    await open_gripper(gripper)
    await asyncio.sleep(step_delay)

    # 8. Retract: Ascend back to travel height to clear deposit zone safely
    print("\n--- STEP 8: RETRACT TO TRAVEL HEIGHT ---")
    if not await move_to_waypoint(machine, arm, "travel"):
        return False
    await asyncio.sleep(step_delay)

    # 9. Return: Return to home pose ready for next run
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
        description="Repeatable pick-and-place using Viam arm-position-saver static positions."
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

    print("Connecting to Viam machine...")
    async with await connect_robot() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")

        # Always print resource overview
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
            success = await run_grasp_cycle(machine, arm, gripper, cycle_idx=c, step_delay=args.delay)
            if not success:
                print(f"[!] Cycle #{c} aborted due to an error.")
                sys.exit(1)
            if c < args.cycles:
                print(f"Pausing 2s before next cycle...")
                await asyncio.sleep(2.0)


if __name__ == "__main__":
    asyncio.run(main())
