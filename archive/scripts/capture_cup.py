"""
capture_cup.py — Capture cup photos from the robot camera and push them into a
Viam cloud dataset, ready for bounding-box labeling + object-detection training.

Grabs color frames from `cam` at an interval, saves each locally (for review) AND
uploads it to a Viam dataset via the Data Client. Slowly reposition / re-angle the
cup (and vary lighting / add occlusions) while this runs to build a varied set.

After capture:
  1. Viam app → DATA → Datasets → <dataset> → draw a `cup` bounding box on each image
     (min 15 images / 10 boxes for detection; ~100 ideal — transfer learning).
  2. Train an Object Detection model on the dataset (cloud).
  3. Deploy it (ML model tflite_cpu + vision/ML model service), then re-probe.

Usage:
    python capture_cup.py --count 60 --interval 1.0 --dataset cup-detection
    python capture_cup.py --count 100 --interval 0.8          # more frames
    python capture_cup.py --local-only                        # save locally, no upload

Env (falls back to these sfhack12 values if unset):
    VIAM_ORG_ID, VIAM_PART_ID
"""

import argparse
import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions

load_dotenv()

VIAM_API_KEY         = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID      = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
CAM_NAME             = os.getenv("VIAM_CAMERA_NAME", "cam")

# From the sfhack12 machine config (CONFIGURE → Details)
ORG_ID  = os.getenv("VIAM_ORG_ID",  "6d6c7293-bc67-43e4-9c8b-31c76aac27d3")
PART_ID = os.getenv("VIAM_PART_ID", "6a76c77c-5ce3-4cfd-9d83-e9f25949de7f")

CAMERA_COMPONENT_TYPE = "rdk:component:camera"
CAPTURE_METHOD        = "ReadImage"


async def connect_robot() -> RobotClient:
    opts = RobotClient.Options.with_api_key(api_key=VIAM_API_KEY, api_key_id=VIAM_API_KEY_ID)
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def get_jpeg(cam: Camera) -> bytes | None:
    """Return the color JPEG bytes from the camera, or None."""
    images, _ = await cam.get_images()
    for img in images:
        if img.mime_type == "image/jpeg":
            return bytes(img.data)
    return None


async def run(count: int, interval: float, dataset_name: str, tag: str, local_only: bool):
    if not VIAM_API_KEY or not VIAM_API_KEY_ID:
        raise SystemExit("[!] VIAM_API_KEY / VIAM_API_KEY_ID not set in .env")

    out_dir = f"captures/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Cloud data client (skip if local-only)
    viam_client = None
    data_client = None
    dataset_id = None
    if not local_only:
        viam_client = await ViamClient.create_from_dial_options(
            DialOptions.with_api_key(VIAM_API_KEY, VIAM_API_KEY_ID)
        )
        data_client = viam_client.data_client
        dataset_id = await data_client.create_dataset(dataset_name, ORG_ID)
        print(f"[✓] Created dataset '{dataset_name}'  id={dataset_id}")
        print(f"    (if it already exists, re-run with a new --dataset name)")

    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    robot = await connect_robot()
    cam = Camera.from_robot(robot, CAM_NAME)

    print(f"\nCapturing {count} frames every {interval}s → {out_dir}"
          f"{'' if local_only else ' + Viam dataset'}")
    print("Slowly move / re-angle the cup while this runs. Ctrl-C to stop early.\n")

    uploaded = 0
    try:
        for i in range(count):
            jpeg = await get_jpeg(cam)
            if not jpeg:
                print(f"  [{i+1}/{count}] no JPEG frame, skipping")
                await asyncio.sleep(interval)
                continue

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            local_path = f"{out_dir}/{stamp}.jpg"
            with open(local_path, "wb") as f:
                f.write(jpeg)

            if not local_only:
                await data_client.binary_data_capture_upload(
                    binary_data=jpeg,
                    part_id=PART_ID,
                    component_type=CAMERA_COMPONENT_TYPE,
                    component_name=CAM_NAME,
                    method_name=CAPTURE_METHOD,
                    file_extension=".jpg",
                    mime_type="image/jpeg",
                    tags=[tag],
                    dataset_ids=[dataset_id],
                )
                uploaded += 1

            print(f"  [{i+1}/{count}] saved {local_path}"
                  f"{'  → uploaded' if not local_only else ''}")
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\n[stopped early]")
    finally:
        await robot.close()
        if viam_client:
            viam_client.close()

    print(f"\n[✓] Done. {uploaded} uploaded to dataset, frames saved in {out_dir}/")
    if not local_only:
        print("\nNext steps:")
        print("  1. Viam app → DATA → Datasets → "
              f"'{dataset_name}' → draw a 'cup' bounding box on each image")
        print("  2. Train model → Object Detection")
        print("  3. Deploy (tflite_cpu ML model + vision/ML model), then re-probe")


def main():
    p = argparse.ArgumentParser(description="Capture cup photos into a Viam dataset")
    p.add_argument("--count", type=int, default=60, help="number of frames to capture")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between frames")
    p.add_argument("--dataset", default="cup-detection", help="Viam dataset name to create")
    p.add_argument("--tag", default="cup", help="tag applied to uploaded images")
    p.add_argument("--local-only", action="store_true", help="save locally, do not upload")
    args = p.parse_args()

    asyncio.run(run(args.count, args.interval, args.dataset, args.tag, args.local_only))


if __name__ == "__main__":
    main()
