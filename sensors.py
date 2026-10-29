"""
sensors.py — Full robot component inventory for sfhack12.

Components on the machine:
  arm           — xArm x850 (6-DOF)
  gripper       — End-effector gripper
  cam           — RealSense D435 (RGB-D)
  yolo          — Vision service (viam-labs:vision:yolov8, COCO detect)
  wall-back     — Back boundary sensor (Gripper)
  wall-left     — Left boundary sensor (Gripper)
  home-pose     — Home position switch
  table         — Table surface sensor (Gripper)
  ceiling       — Ceiling boundary sensor (Gripper)
"""

import asyncio
import argparse
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.services.vision import VisionClient

load_dotenv()

VIAM_API_KEY         = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID      = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
VISION_NAME          = os.getenv("VIAM_VISION_NAME", "yolo")


async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID,
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def read_all_sensors(save_image: bool = True):
    if not VIAM_API_KEY or not VIAM_API_KEY_ID:
        print("[!] VIAM_API_KEY and VIAM_API_KEY_ID must be set in .env")
        return

    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        print("\n" + "=" * 55)
        print("         sfhack12 — FULL SENSOR READOUT")
        print("=" * 55)
        print(f"Resources: {[r.name for r in machine.resource_names]}\n")

        # ── Arm ─────────────────────────────────────────────────
        try:
            arm = Arm.from_robot(machine, "arm")
            joints  = await arm.get_joint_positions()
            pose    = await arm.get_end_position()
            moving  = await arm.is_moving()
            print("[ARM]")
            print(f"  Moving:  {'YES' if moving else 'NO'}")
            print(f"  Joints:  {[f'{v:.2f}°' for v in joints.values]}")
            print(f"  Pose:    X={pose.x:.1f}  Y={pose.y:.1f}  Z={pose.z:.1f} mm")
            print(f"           ox={pose.o_x:.3f}  oy={pose.o_y:.3f}  oz={pose.o_z:.3f}  θ={pose.theta:.2f}°")
        except Exception as e:
            print(f"[ARM] error: {e}")

        # ── Gripper ─────────────────────────────────────────────
        try:
            gripper = Gripper.from_robot(machine, "gripper")
            holding = await gripper.is_holding_something()
            moving  = await gripper.is_moving()
            print(f"\n[GRIPPER]")
            print(f"  Moving:  {'YES' if moving else 'NO'}")
            print(f"  Holding: {holding.is_holding_something}")
        except Exception as e:
            print(f"\n[GRIPPER] error: {e}")

        # ── Boundary sensors ────────────────────────────────────
        for name in ("wall-back", "wall-left", "table", "ceiling"):
            try:
                sensor = Gripper.from_robot(machine, name)
                moving = await sensor.is_moving()
                print(f"\n[{name.upper()}]  moving={moving}")
            except Exception as e:
                print(f"\n[{name.upper()}] error: {e}")

        # ── Home pose switch ────────────────────────────────────
        try:
            home = Switch.from_robot(machine, "home-pose")
            pos  = await home.get_position()
            print(f"\n[HOME-POSE]  position={pos}")
        except Exception as e:
            print(f"\n[HOME-POSE] error: {e}")

        # ── Camera ──────────────────────────────────────────────
        try:
            cam = Camera.from_robot(machine, "cam")
            images, metadata = await cam.get_images()
            print(f"\n[CAMERA]")
            print(f"  Captured at: {metadata.captured_at.seconds}s")
            for idx, img in enumerate(images):
                print(f"  Stream {idx+1}: name='{img.name}'  mime='{img.mime_type}'  size={len(img.data)} bytes")
                if save_image and img.mime_type == "image/jpeg":
                    fname = f"latest_{img.name}.jpg"
                    with open(fname, "wb") as f:
                        f.write(img.data)
                    print(f"    → saved {fname}")
        except Exception as e:
            print(f"\n[CAMERA] error: {e}")

        # ── Vision service (YOLOv8) ─────────────────────────────
        try:
            vision = VisionClient.from_robot(machine, VISION_NAME)
            props  = await vision.get_properties()
            print(f"\n[{VISION_NAME.upper()}]")
            print(f"  classifications_supported: {props.classifications_supported}")
            print(f"  detections_supported:      {props.detections_supported}")
            if props.detections_supported:
                dets = await vision.get_detections_from_camera("cam")
                print(f"  detections on 'cam': {len(dets)}")
                for d in dets[:10]:
                    print(f"    {d.class_name:15s} conf={d.confidence:.3f}")
        except Exception as e:
            print(f"\n[{VISION_NAME.upper()}] error: {e}")

        print("\n" + "=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read all sfhack12 sensors")
    parser.add_argument("--no-save-image", action="store_true")
    args = parser.parse_args()
    asyncio.run(read_all_sensors(save_image=not args.no_save_image))
