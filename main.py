import asyncio
import os
from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService

# Load environment variables from .env if present
load_dotenv()

VIAM_API_KEY = os.getenv("VIAM_API_KEY", "<API-KEY>")
VIAM_API_KEY_ID = os.getenv("VIAM_API_KEY_ID", "<API-KEY-ID>")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def main():
    if VIAM_API_KEY == "<API-KEY>" or VIAM_API_KEY_ID == "<API-KEY-ID>":
        print("[!] Warning: VIAM_API_KEY or VIAM_API_KEY_ID not configured.")
        print("    Please set VIAM_API_KEY and VIAM_API_KEY_ID in your .env file or environment.")
        return

    print(f"Connecting to machine at {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        print("Resources:")
        print(machine.resource_names)

        # arm
        try:
            arm: Arm = Arm.from_robot(machine, "arm")
            arm_return_value = await arm.get_end_position()
            print(f"arm get_end_position return value: {arm_return_value}")
        except Exception as e:
            print(f"arm check failed: {e}")

        # cam
        try:
            cam: Camera = Camera.from_robot(machine, "cam")
            cam_return_value = await cam.get_images()
            print(f"cam get_images return value: {cam_return_value}")
        except Exception as e:
            print(f"cam check failed: {e}")

        # gripper
        try:
            gripper: Gripper = Gripper.from_robot(machine, "gripper")
            gripper_return_value = await gripper.is_moving()
            print(f"gripper is_moving return value: {gripper_return_value}")
        except Exception as e:
            print(f"gripper check failed: {e}")

        # table
        try:
            table: Gripper = Gripper.from_robot(machine, "table")
            table_return_value = await table.is_moving()
            print(f"table is_moving return value: {table_return_value}")
        except Exception as e:
            print(f"table check failed: {e}")

        # ceiling
        try:
            ceiling: Gripper = Gripper.from_robot(machine, "ceiling")
            ceiling_return_value = await ceiling.is_moving()
            print(f"ceiling is_moving return value: {ceiling_return_value}")
        except Exception as e:
            print(f"ceiling check failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
