"""
Client script demonstrating how to interact with the on-robot FeelsControlService.

Calls the module over Viam gRPC via do_command to:
1. Query instant high-rate telemetry without network lag
2. Tare baseline load
3. Trigger the 3-motion diagnostic macro
4. Retrieve the 10-dimensional physical feature vector computed onboard
"""

import asyncio
import os
import pprint
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.services.generic import Generic as GenericService

load_dotenv()


async def connect():
    api_key = os.getenv("VIAM_API_KEY")
    api_key_id = os.getenv("VIAM_API_KEY_ID")
    address = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")

    opts = RobotClient.Options.with_api_key(
        api_key=api_key,
        api_key_id=api_key_id,
    )
    return await RobotClient.at_address(address, opts)


async def main():
    async with await connect() as machine:
        print("[✓] Connected to robot.")

        # Check if the on-robot service is deployed
        service_name = "control-logic"
        try:
            service = GenericService.from_robot(machine, service_name)
        except Exception:
            print(f"[!] Service '{service_name}' not yet deployed on machine.")
            print("    Deploy 'hackathons:control_logic:generic-service' to sfhack12 via Viam app.")
            return

        # 1. Query latest telemetry
        print("\n--- 1. Instant Onboard Telemetry ---")
        telemetry = await service.do_command({"action": "get_telemetry"})
        pprint.pprint(telemetry)

        # 2. Tare resting baseline
        print("\n--- 2. Tare Resting Baseline ---")
        tare_resp = await service.do_command({"action": "tare"})
        pprint.pprint(tare_resp)

        # 3. Run Macro & Retrieve Onboard Extracted Features
        print("\n--- 3. Run Motion Macro & Extract Physical Features ---")
        macro_resp = await service.do_command({"action": "run_macro", "duration": 6.0})
        pprint.pprint(macro_resp)


if __name__ == "__main__":
    asyncio.run(main())
