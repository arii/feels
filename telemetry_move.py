"""
telemetry_move.py — Laptop-side "telemetry movement" with no on-robot module.

Connects to sfhack12 over the Viam cloud, gently oscillates ONE joint with a
small sine, and records 6-joint loads (torques) throughout to a CSV in
recordings/. This is the quick path for seeing a telemetry movement while the
on-robot control-logic service is not yet deployed.

NOTE: this runs over the cloud (~158 ms RTT), so the effective sample rate is
only a few Hz — fine for eyeballing the torque response, not for capturing the
5–25 Hz rattle band (that needs the on-robot module).

Usage:
    python telemetry_move.py                 # safe default: ±5° sine on J1 (base), 3 cycles
    python telemetry_move.py --joint 2 --amplitude 8 --cycles 4 --freq 0.3
    python telemetry_move.py --dry-run       # read telemetry only, DO NOT move

Safety:
    - Clear the workspace and keep the e-stop in reach.
    - Starts from and restores the current pose. Small amplitude by default.
    - Joint index is 1-based (J1=base ... J6=wrist).
"""

import argparse
import asyncio
import csv
import math
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.proto.component.arm import JointPositions

load_dotenv()

VIAM_API_KEY         = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID      = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")

# Resting unloaded baseline loads (Nm) — same reference realtime_plot.py uses.
BASELINE_LOADS = [0.0, -16.0489, -16.8209, 0.778, 0.3728, 0.0]


async def connect() -> RobotClient:
    if not VIAM_API_KEY or not VIAM_API_KEY_ID:
        raise SystemExit("[!] VIAM_API_KEY / VIAM_API_KEY_ID not set in .env")
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID,
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


async def read_loads(arm: Arm) -> list[float]:
    """Read 6-joint actuator loads (Nm). Returns [] if unavailable."""
    try:
        res = await arm.do_command({"load": ""}, timeout=3.0)
        loads = res.get("load", []) if isinstance(res, dict) else []
        return [float(x) for x in loads[:6]]
    except Exception as e:
        print(f"  [load read error] {e}")
        return []


async def run(joint: int, amplitude: float, cycles: int, freq: float,
              steps_per_cycle: int, dry_run: bool):
    jidx = joint - 1  # to 0-based

    print(f"Connecting to {VIAM_MACHINE_ADDRESS}...")
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")

        start = await arm.get_joint_positions()
        start_vals = list(start.values)
        print(f"[✓] Connected. Start joints: {[f'{v:.2f}' for v in start_vals]}")

        if jidx < 0 or jidx >= len(start_vals):
            raise SystemExit(f"[!] joint {joint} out of range (1..{len(start_vals)})")

        # Prepare CSV
        os.makedirs("recordings", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"recordings/telemove_{stamp}.csv"
        f = open(path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(
            ["elapsed_sec", "cmd_joint", "cmd_angle_deg"]
            + [f"j{i+1}_raw" for i in range(6)]
            + [f"j{i+1}_delta" for i in range(6)]
        )

        def record(t0, cmd_angle, loads):
            padded = (loads + [0.0] * 6)[:6]
            deltas = [round(padded[i] - BASELINE_LOADS[i], 4) for i in range(6)]
            writer.writerow(
                [round(time.perf_counter() - t0, 3), joint, round(cmd_angle, 3)]
                + [round(x, 4) for x in padded]
                + deltas
            )

        t0 = time.perf_counter()

        if dry_run:
            print("[dry-run] Recording telemetry for 5 s, no motion...")
            for _ in range(25):
                loads = await read_loads(arm)
                if loads:
                    print(f"  J{joint} load = {loads[jidx]:+.3f} Nm   all = "
                          f"{[round(x,2) for x in loads]}")
                record(t0, start_vals[jidx], loads)
                await asyncio.sleep(0.2)
            f.close()
            print(f"[✓] Telemetry saved → {path}")
            return

        total_steps = cycles * steps_per_cycle
        period = 1.0 / freq
        step_dt = period / steps_per_cycle
        print(f"Oscillating J{joint}: ±{amplitude}° sine, {cycles} cycles @ {freq} Hz "
              f"({total_steps} waypoints). Recording → {path}")

        try:
            for s in range(total_steps + 1):
                phase = (s / steps_per_cycle) * 2.0 * math.pi
                offset = amplitude * math.sin(phase)
                target = list(start_vals)
                target[jidx] = start_vals[jidx] + offset

                await arm.move_to_joint_positions(JointPositions(values=target))
                loads = await read_loads(arm)
                record(t0, target[jidx], loads)

                if loads:
                    print(f"  t={time.perf_counter()-t0:5.2f}s  "
                          f"cmd J{joint}={target[jidx]:+7.2f}°  "
                          f"J{joint}load={loads[jidx]:+.3f} Nm")
                await asyncio.sleep(step_dt)

        finally:
            # Always restore the start pose
            print("Restoring start pose...")
            await arm.move_to_joint_positions(JointPositions(values=start_vals))
            f.close()

        print(f"[✓] Done. Telemetry saved → {path}")


def main():
    p = argparse.ArgumentParser(description="Gentle single-joint telemetry movement for sfhack12")
    p.add_argument("--joint", type=int, default=1, help="1-based joint index (J1=base ... J6=wrist)")
    p.add_argument("--amplitude", type=float, default=5.0, help="sine amplitude in degrees")
    p.add_argument("--cycles", type=int, default=3, help="number of sine cycles")
    p.add_argument("--freq", type=float, default=0.3, help="target oscillation freq (Hz); cloud caps real rate")
    p.add_argument("--steps-per-cycle", type=int, default=12, help="waypoints per cycle")
    p.add_argument("--dry-run", action="store_true", help="record telemetry only, do NOT move")
    args = p.parse_args()

    asyncio.run(run(args.joint, args.amplitude, args.cycles, args.freq,
                    args.steps_per_cycle, args.dry_run))


if __name__ == "__main__":
    main()
