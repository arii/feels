#!/usr/bin/env python3
"""
safe_pick.py — Vision-gated, grasp-verified cup pickup.

Thin SAFETY wrapper around the teammate's repeatable_grasp.py: it reuses the
proven arm-position-saver primitives (connect_robot, move_to_waypoint,
open_gripper, close_gripper) and adds the two interlocks a safe pick needs:

  1. VISION GATE  — before moving, confirm the YOLO detector actually sees a
                    `cup` (>= min confidence). If not, ABORT without moving.
  2. GRASP VERIFY — after closing the gripper, check is_holding_something().
                    If the grab missed, open, retreat to travel/home, ABORT.

Sequence (all moves are saved arm-position-saver waypoints, so Z/collision
safety is whatever was recorded into the switches):
    [vision gate] -> home -> approach -> grasp -> close -> [verify] -> travel
    (--place also does: place -> open -> travel -> home)

Usage:
    python safe_pick.py --dry-run          # vision gate + print plan, NO motion
    python safe_pick.py                    # gated pick, lift to travel, hold
    python safe_pick.py --place            # full pick-and-place
    python safe_pick.py --no-vision        # skip the vision gate (not recommended)
    python safe_pick.py --min-conf 0.35    # vision gate confidence threshold

⚠️  This drives the PHYSICAL arm. Clear the workspace, keep the e-stop in reach.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.proto.common import Pose

# Reuse the teammate's proven primitives — do not duplicate motion logic.
from repeatable_grasp import (
    connect_robot,
    move_to_waypoint,
    open_gripper,
    close_gripper,
    wait_for_stability_in_travel,
)

load_dotenv()

VISION_NAME = os.getenv("VIAM_VISION_NAME", "yolo")
CAM_NAME    = os.getenv("VIAM_CAMERA_NAME", "cam")
MIN_SAFE_Z_MM = 15.0


async def descend(arm: Arm, drop_mm: float, label: str = ""):
    """Lower the TCP straight down by drop_mm (keeps orientation), Z clamped >= MIN_SAFE_Z."""
    if drop_mm <= 0:
        return
    p = await arm.get_end_position()
    z = max(MIN_SAFE_Z_MM, p.z - drop_mm)
    print(f"[nudge{(' ' + label) if label else ''}] lowering {p.z:.1f} -> {z:.1f} mm")
    await arm.move_to_position(Pose(x=p.x, y=p.y, z=z,
                                    o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta))
    await asyncio.sleep(0.3)


async def cup_gate(machine, min_conf: float):
    """Return (ok, message, cups). ok=True iff a `cup` is detected >= min_conf."""
    try:
        vision = VisionClient.from_robot(machine, VISION_NAME)
        dets = await vision.get_detections_from_camera(CAM_NAME)
    except Exception as e:
        return False, f"vision service '{VISION_NAME}' error: {e}", []

    cups = sorted(
        [d for d in dets if d.class_name.lower() == "cup" and d.confidence >= min_conf],
        key=lambda d: d.confidence, reverse=True,
    )
    others = sorted({d.class_name for d in dets})
    if cups:
        best = cups[0]
        cx = (best.x_min + best.x_max) // 2
        cy = (best.y_min + best.y_max) // 2
        msg = (f"cup CENTER=({cx},{cy})  conf={best.confidence:.2f}  "
               f"bbox=({best.x_min},{best.y_min})-({best.x_max},{best.y_max})"
               f"  [{len(cups)} cup(s); frame also has: {', '.join(others) or 'none'}]")
        return True, msg, cups
    return False, f"NO cup >= {min_conf:.2f} in frame (saw: {', '.join(others) or 'nothing'})", []


async def verify_hold(gripper: Gripper) -> bool:
    try:
        status = await gripper.is_holding_something()
        return bool(status.is_holding_something)
    except Exception as e:
        print(f"[verify] is_holding_something error: {e}")
        return False


async def retreat(machine, arm):
    """Best-effort safe retreat after an aborted grasp."""
    print("[abort] retreating to travel, then home...")
    try:
        await move_to_waypoint(machine, arm, "travel")
        await move_to_waypoint(machine, arm, "home-pose")
    except Exception as e:
        print(f"[abort] retreat error: {e}")


async def run(args):
    print("Connecting to Viam machine...")
    async with await connect_robot() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")

        # ---- 1. VISION GATE ---------------------------------------------
        if args.no_vision:
            print("[gate] vision gate SKIPPED (--no-vision)")
        else:
            ok, msg, _ = await cup_gate(machine, args.min_conf)
            print(f"[gate] {msg}")
            if not ok:
                print("[ABORT] no cup confirmed — not moving the arm.")
                return

        plan = "home -> approach -> grasp -> close -> [verify] -> travel"
        if args.place:
            plan += " -> place -> open -> travel -> home"
        print(f"[plan] {plan}")

        if args.dry_run:
            print("[DRY RUN] vision gate passed; no motion executed.")
            return

        # ---- 2. PICK ----------------------------------------------------
        await open_gripper(gripper)
        for wp in ("home-pose", "approach", "grasp"):
            if not await move_to_waypoint(machine, arm, wp):
                print(f"[ABORT] move to '{wp}' failed.")
                await retreat(machine, arm)
                return

        await descend(arm, args.grasp_drop, "grasp")   # optionally grab a bit lower
        await close_gripper(gripper)
        await asyncio.sleep(0.5)

        # ---- 3. GRASP VERIFY --------------------------------------------
        if not await verify_hold(gripper):
            print("[ABORT] grasp missed — gripper not holding anything.")
            await open_gripper(gripper)
            await retreat(machine, arm)
            return
        print("[✓] grasp verified — holding the cup.")

        # ---- 4. LIFT (+ teammate's travel stabilization / weigh) --------
        if not await move_to_waypoint(machine, arm, "travel"):
            print("[ABORT] lift to travel failed.")
            return
        print("[✓] cup lifted to travel pose.")
        await wait_for_stability_in_travel(
            arm, recorder=None,
            stable_duration=args.stable_hold,
            variance_threshold=args.settle_threshold,
        )

        # ---- 5. OPTIONAL PLACE ------------------------------------------
        if args.place:
            if not await move_to_waypoint(machine, arm, "place"):
                print("[ABORT] move to 'place' failed (still holding).")
                return
            # Set the cup DOWN on the surface before releasing, otherwise it gets
            # carried back up with the gripper on retract.
            await descend(arm, args.place_drop, "place")
            await open_gripper(gripper)
            await asyncio.sleep(args.settle)          # let the cup settle, fully released
            await move_to_waypoint(machine, arm, "travel")
            await move_to_waypoint(machine, arm, "home-pose")
            print("[✓] placed and returned home.")
        else:
            print("[done] holding at travel pose. Re-run with --place to set it down.")


def main():
    p = argparse.ArgumentParser(description="Vision-gated, grasp-verified cup pickup")
    p.add_argument("--dry-run", action="store_true", help="vision gate + plan only, no motion")
    p.add_argument("--no-vision", action="store_true", help="skip the YOLO cup gate")
    p.add_argument("--place", action="store_true", help="full pick-and-place (else lift & hold)")
    p.add_argument("--min-conf", type=float, default=0.35, help="cup detection confidence gate")
    p.add_argument("--grasp-drop", type=float, default=15.0,
                   help="mm to lower at the grasp pose before closing — 'hold a little below' (default 15)")
    p.add_argument("--place-drop", type=float, default=20.0,
                   help="mm to lower at the place pose before opening (set cup down for clean release)")
    p.add_argument("--settle", type=float, default=0.7,
                   help="seconds to dwell after opening at place before retracting")
    p.add_argument("--stable-hold", type=float, default=3.0,
                   help="travel fixed-stop dwell seconds (matches teammate's cycle)")
    p.add_argument("--settle-threshold", type=float, default=0.04,
                   help="torque std threshold (Nm) to confirm settled in travel")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
