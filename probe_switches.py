"""
probe_switches.py — READ-ONLY: inspect the Switch components.

Reads each switch's current position + number of positions (and tries a
do_command for any metadata). Does NOT set any position, so it triggers NO
motion. Helps decide whether grasp/approach/place/travel/home-pose are
prebuilt pose triggers we can use as a pick shortcut.

Usage:
    python probe_switches.py
"""

import asyncio
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.switch import Switch

load_dotenv()

VIAM_API_KEY         = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID      = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")

SWITCH_NAMES = ["home-pose", "approach", "grasp", "travel", "place"]


async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(api_key=VIAM_API_KEY, api_key_id=VIAM_API_KEY_ID)
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def main():
    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        names = [r.name for r in machine.resource_names]
        print(f"\nSwitch-like resources present: "
              f"{[n for n in names if n in SWITCH_NAMES]}\n")

        for name in SWITCH_NAMES:
            if name not in names:
                print(f"[{name}] not present")
                continue
            try:
                sw = Switch.from_robot(machine, name)
                pos = await sw.get_position()
                try:
                    n_pos = await sw.get_number_of_positions()
                except Exception:
                    n_pos = "?"
                info = ""
                try:
                    info = await sw.do_command({"": ""})
                except Exception:
                    pass
                print(f"[{name}]  position={pos}  num_positions={n_pos}"
                      f"{('  info=' + str(info)) if info else ''}")
            except Exception as e:
                print(f"[{name}] error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
