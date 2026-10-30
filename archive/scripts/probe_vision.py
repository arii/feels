"""
probe_vision.py — READ-ONLY: inspect what the machine's vision stack returns.

Lists resources, queries the vision service properties + detections + classifications
on the overhead camera, and saves the frame. No motion, no config changes.

Usage:
    python probe_vision.py
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.vision import VisionClient

load_dotenv()

VIAM_API_KEY         = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID      = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
CAM_NAME             = os.getenv("VIAM_CAMERA_NAME", "cam")

_p = argparse.ArgumentParser(description="READ-ONLY vision probe")
_p.add_argument("--vision", default=os.getenv("VIAM_VISION_NAME", "yolo"),
                help="vision service name (e.g. yolo)")
_p.add_argument("--camera", default=CAM_NAME, help="camera name")
_args = _p.parse_args()
VISION_NAME = _args.vision
CAM_NAME    = _args.camera


async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(api_key=VIAM_API_KEY, api_key_id=VIAM_API_KEY_ID)
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def main():
    if not VIAM_API_KEY or not VIAM_API_KEY_ID:
        raise SystemExit("[!] VIAM_API_KEY / VIAM_API_KEY_ID not set in .env")

    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        print("\n=== RESOURCES ===")
        for r in machine.resource_names:
            print(f"  {r.namespace}:{r.type}:{r.subtype}/{r.name}")

        print(f"\n=== VISION SERVICE '{VISION_NAME}' ===")
        try:
            vision = VisionClient.from_robot(machine, VISION_NAME)
            props = await vision.get_properties()
            print(f"  detections_supported:         {props.detections_supported}")
            print(f"  classifications_supported:    {props.classifications_supported}")
            print(f"  object_point_clouds_supported:{props.object_point_clouds_supported}")

            if props.detections_supported:
                dets = await vision.get_detections_from_camera(CAM_NAME)
                print(f"\n  detections on '{CAM_NAME}': {len(dets)}")
                for d in dets:
                    print(f"    {d.class_name:20s} conf={d.confidence:.3f} "
                          f"bbox=({d.x_min},{d.y_min})-({d.x_max},{d.y_max})")

            if props.classifications_supported:
                cls = await vision.get_classifications_from_camera(CAM_NAME, 5)
                print(f"\n  top classifications on '{CAM_NAME}':")
                for c in cls:
                    print(f"    {c.class_name:20s} conf={c.confidence:.3f}")
        except Exception as e:
            print(f"  [vision error] {e}")

        print(f"\n=== CAMERA '{CAM_NAME}' (saving frame) ===")
        try:
            cam = Camera.from_robot(machine, CAM_NAME)
            images, _ = await cam.get_images()
            for img in images:
                print(f"  stream name='{img.name}' mime='{img.mime_type}' size={len(img.data)}")
                if img.mime_type == "image/jpeg":
                    fn = f"probe_{img.name}.jpg"
                    with open(fn, "wb") as f:
                        f.write(img.data)
                    print(f"    saved {fn}")
        except Exception as e:
            print(f"  [camera error] {e}")


if __name__ == "__main__":
    asyncio.run(main())
