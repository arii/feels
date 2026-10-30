#!/usr/bin/env python3
"""
use_modules.py — client that uses the deployed feels Viam modules.

Calls the three on-machine services over do_command (replaces the old
client_control_service.py, which drove the pre-module control_service):

    braveorangerover:sensing:torque-features   → service "torque-features"
    braveorangerover:pick_logic:yolo3d-pick     → service "picker"
    braveorangerover:content:classifier         → service "content"

Usage:
    python src/use_modules.py telemetry                       # torque-features
    python src/use_modules.py features
    python src/use_modules.py detect                          # yolo3d-pick (no motion)
    python src/use_modules.py pick                            # yolo3d-pick (moves!)
    python src/use_modules.py predict <recording.csv>         # content:classifier
    python src/use_modules.py all                             # status of all three
"""
import argparse
import asyncio
import os
import pprint

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.services.generic import Generic

load_dotenv()

# service names as configured in the Viam app (override via env if you named them differently)
TORQUE = os.getenv("FEELS_TORQUE_SVC", "torque-features")
PICKER = os.getenv("FEELS_PICKER_SVC", "picker")
CONTENT = os.getenv("FEELS_CONTENT_SVC", "content")


async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=os.getenv("VIAM_API_KEY"), api_key_id=os.getenv("VIAM_API_KEY_ID"))
    return await RobotClient.at_address(
        os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud"), opts)


async def call(machine, service_name: str, command: dict):
    try:
        svc = Generic.from_robot(machine, service_name)
        return await svc.do_command(command)
    except Exception as e:
        return {"error": f"service '{service_name}' unavailable: {str(e)[:120]}"}


async def main():
    ap = argparse.ArgumentParser(description="Use the deployed feels Viam modules")
    ap.add_argument("action", choices=["telemetry", "features", "effective_force",
                                       "detect", "pick", "pick_place", "predict", "all"])
    ap.add_argument("recording", nargs="?", help="recording CSV (for 'predict')")
    args = ap.parse_args()

    async with await connect() as m:
        if args.action == "all":
            print("torque-features:"); pprint.pprint(await call(m, TORQUE, {"action": "status"}))
            print("\nyolo3d-pick:");    pprint.pprint(await call(m, PICKER, {"action": "status"}))
            print("\ncontent:");        pprint.pprint(await call(m, CONTENT, {"action": "status"}))
        elif args.action in ("telemetry", "features", "effective_force"):
            pprint.pprint(await call(m, TORQUE, {"action": args.action}))
        elif args.action in ("detect", "pick", "pick_place"):
            pprint.pprint(await call(m, PICKER, {"action": args.action}))
        elif args.action == "predict":
            if not args.recording:
                raise SystemExit("predict needs a recording CSV: python src/use_modules.py predict <csv>")
            pprint.pprint(await call(m, CONTENT, {"action": "predict", "recording": args.recording}))


if __name__ == "__main__":
    asyncio.run(main())
