import asyncio
import os
import argparse
from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
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


async def read_all_sensors(save_image: bool = True):
    if not VIAM_API_KEY or not VIAM_API_KEY_ID:
        print("[!] VIAM_API_KEY and VIAM_API_KEY_ID must be set in .env")
        return

    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        print("\n" + "=" * 50)
        print("         LIVE ROBOT SENSOR & TELEMETRY         ")
        print("=" * 50)

        # 1. Arm Sensors
        try:
            arm = Arm.from_robot(machine, "arm")
            joints = await arm.get_joint_positions()
            pose = await arm.get_end_position()
            is_moving = await arm.is_moving()

            print("\n[ARM ENCODERS & KINEMATICS]")
            print(f"  Moving Status:     {'MOVING' if is_moving else 'STATIONARY'}")
            print("  Joint Angles (deg):")
            for idx, angle in enumerate(joints.values):
                print(f"    Joint {idx + 1}: {angle:8.2f}°")
            print("  End-Effector Pose:")
            print(f"    Position:    X={pose.x:.2f} mm, Y={pose.y:.2f} mm, Z={pose.z:.2f} mm")
            print(f"    Orientation: ({pose.o_x:.3f}, {pose.o_y:.3f}, {pose.o_z:.3f}), theta={pose.theta:.2f}°")
        except Exception as e:
            print(f"[!] Arm sensor error: {e}")

        # 2. Gripper Sensors
        for g_name in ["gripper", "table", "ceiling"]:
            try:
                g = Gripper.from_robot(machine, g_name)
                holding = await g.is_holding_something()
                moving = await g.is_moving()

                print(f"\n[GRIPPER: '{g_name}']")
                print(f"  Moving Status:     {'MOVING' if moving else 'IDLE'}")
                print(f"  Holding Object:    {holding.is_holding_something}")
                if holding.meta:
                    print(f"  Feedback Metadata: {dict(holding.meta)}")
            except Exception as e:
                print(f"[!] Gripper '{g_name}' sensor error: {e}")

        # 3. Camera Sensor
        try:
            cam = Camera.from_robot(machine, "cam")
            images, metadata = await cam.get_images()
            print(f"\n[CAMERA STREAMS: 'cam']")
            print(f"  Captured At:       {metadata.captured_at.seconds}s")
            for idx, img in enumerate(images):
                print(f"  Stream {idx + 1}:          name='{img.name}', mime='{img.mime_type}', size={len(img.data)} bytes")
                if save_image and img.mime_type == "image/jpeg":
                    filename = f"latest_{img.name}.jpg"
                    with open(filename, "wb") as f:
                        f.write(img.data)
                    print(f"    -> Saved frame to {filename}")
        except Exception as e:
            print(f"[!] Camera sensor error: {e}")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect live Viam robot sensors")
    parser.add_argument("--no-save-image", action="store_true", help="Do not write camera frames to disk")
    args = parser.parse_args()

    asyncio.run(read_all_sensors(save_image=not args.no_save_image))
