"""
feels — Interactive Dynamic Identification of Hidden Cup Contents
================================================================
Robot picks up a mystery cup, shakes it, and tells you what's inside.

Usage:
    python main.py identify              # run the full pipeline on mystery cup
    python main.py teach <label>         # record + label a new item
    python main.py tare                  # recalibrate baseline (empty cup)
    python main.py status                # check robot + Atlas connection

Halloween special: can it tell candy from water from dish soap? 🎃
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.services.generic import Generic as GenericService

import atlas

load_dotenv()

VIAM_API_KEY         = os.getenv("VIAM_API_KEY")
VIAM_API_KEY_ID      = os.getenv("VIAM_API_KEY_ID")
VIAM_MACHINE_ADDRESS = os.getenv("VIAM_MACHINE_ADDRESS", "sfhack12-main.ty1cywh7tq.viam.cloud")
SERVICE_NAME         = "control-logic"

HALLOWEEN_LABELS = {
    "candy_loose":    "🍬 CANDY!  (trick-or-treat rattling detected)",
    "water_half":     "💧 Water   (sloshing phase lag detected)",
    "dish_soap_half": "🫧 Dish soap  (viscous damping detected)",
    "empty_cup":      "👻 Empty cup  (nothing... spooky)",
    "solid_weight":   "🪨 Solid weight  (rigid mass, zero rattle)",
}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

async def connect() -> RobotClient:
    if not VIAM_API_KEY or not VIAM_API_KEY_ID:
        print("[!] VIAM_API_KEY / VIAM_API_KEY_ID not set in .env")
        sys.exit(1)
    opts = RobotClient.Options.with_api_key(
        api_key=VIAM_API_KEY,
        api_key_id=VIAM_API_KEY_ID,
    )
    return await RobotClient.at_address(VIAM_MACHINE_ADDRESS, opts)


def get_service(machine: RobotClient) -> GenericService:
    try:
        return GenericService.from_robot(machine, SERVICE_NAME)
    except Exception:
        print(f"[!] On-robot service '{SERVICE_NAME}' not found.")
        print("    Deploy control_service.py to sfhack12 via Viam app first.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_status():
    """Check robot + Atlas connectivity."""
    print("Checking Atlas connection...")
    try:
        db = atlas.get_db()
        counts = {c: db[c].count_documents({}) for c in ("runs", "signatures", "items")}
        print(f"  Atlas OK  — {counts}")
    except Exception as e:
        print(f"  Atlas FAIL — {e}")

    print(f"\nChecking robot at {VIAM_MACHINE_ADDRESS}...")
    try:
        async with await connect() as machine:
            resources = machine.resource_names
            service = GenericService.from_robot(machine, SERVICE_NAME)
            telemetry = await service.do_command({"action": "get_telemetry"})
            print(f"  Robot OK  — {len(resources)} resources")
            print(f"  Service OK — sampling at {telemetry.get('sampling_rate_hz')} Hz")
            print(f"  Weight est: {telemetry.get('weight_grams', '?')} g")
    except Exception as e:
        print(f"  Robot FAIL — {e}")


async def cmd_tare():
    """Recalibrate baseline torques with an empty cup."""
    print("Place the EMPTY cup in the gripper, then press Enter...")
    input()
    async with await connect() as machine:
        service = get_service(machine)
        resp = await service.do_command({"action": "tare"})
        print(f"  Tare complete — J3 baseline: {resp.get('tare_j3_nm', '?'):.4f} Nm")


async def cmd_identify(session: str = "thu"):
    """Run the 3-motion macro and identify the cup contents."""
    _print_banner()
    print("Place the MYSTERY cup at the taped square and press Enter...")
    input()

    async with await connect() as machine:
        service = get_service(machine)

        print("\n[1/3] Running diagnostic macro (lift → shake → tilt)...")
        macro_resp = await service.do_command({"action": "run_macro", "duration": 6.0})

        if "features" not in macro_resp or macro_resp["features"].get("status") != "success":
            print(f"[!] Macro failed: {macro_resp}")
            return

        feature_vector = macro_resp["features"]["feature_vector"]
        metrics        = macro_resp["features"]["metrics"]

        print(f"[2/3] Feature vector extracted ({len(feature_vector)}-dim):")
        print(f"      mass={metrics['estimated_mass_grams']:.0f}g  "
              f"rattle={metrics['rattle_energy_5_25hz']:.3f}  "
              f"slosh={metrics['slosh_energy_05_3hz']:.3f}  "
              f"freq={metrics['dominant_freq_hz']:.1f}Hz")

        print("[3/3] Querying Atlas vector memory...")
        result = atlas.identify(feature_vector, top_k=5, session=session)

    _print_verdict(result)


async def cmd_teach(label: str, session: str = "thu"):
    """Record a run and save it to Atlas under the given label."""
    print(f"\nTeaching '{label}' — place the cup and press Enter...")
    input()

    async with await connect() as machine:
        service = get_service(machine)

        print("Running diagnostic macro...")
        macro_resp = await service.do_command({"action": "run_macro", "duration": 6.0})

        if "features" not in macro_resp or macro_resp["features"].get("status") != "success":
            print(f"[!] Macro failed: {macro_resp}")
            return

        feature_vector = macro_resp["features"]["feature_vector"]
        metrics        = macro_resp["features"]["metrics"]
        raw_data       = macro_resp["features"].get("raw_j3", [])

        ids = atlas.teach(
            label=label,
            feature_vector=feature_vector,
            raw_data=raw_data or None,
            motion_type="3in1_macro",
            properties={"estimated_mass_g": metrics["estimated_mass_grams"]},
            session=session,
        )
        print(f"  Saved to Atlas — sig_id: {ids['sig_id']}")
        print(f"  '{label}' is now queryable via identify()")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║          feels — What's in the cup?  🎃              ║
║    Interactive Dynamic Identification System         ║
╚══════════════════════════════════════════════════════╝""")


def _print_verdict(result: dict):
    label      = result["label"]
    confidence = result["confidence"]
    low_conf   = result["low_confidence"]

    display = HALLOWEEN_LABELS.get(label, f"❓ {label}")
    bar_len  = int(confidence * 30)
    bar      = "█" * bar_len + "░" * (30 - bar_len)

    print(f"""
┌──────────────────────────────────────────────────────┐
│  VERDICT                                             │
│                                                      │
│  {display:<50s}│
│                                                      │
│  Confidence  [{bar}] {confidence*100:.0f}%  │
│  Low signal? {'⚠ YES — try another shake axis' if low_conf else '✓ NO'}{'':>18s}│
└──────────────────────────────────────────────────────┘""")

    if result["matches"]:
        print("  Top matches:")
        for m in result["matches"][:3]:
            lbl = HALLOWEEN_LABELS.get(m["label"], m["label"])
            print(f"    {m['score']:.3f}  {lbl}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="feels — robot identifies hidden cup contents via proprioception"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status",   help="Check robot + Atlas connectivity")
    sub.add_parser("tare",     help="Calibrate baseline with empty cup")

    p_id = sub.add_parser("identify", help="Identify mystery cup contents")
    p_id.add_argument("--session", default="thu", help="Session tag for Atlas filter (thu/fri)")

    p_teach = sub.add_parser("teach", help="Record and label a known item")
    p_teach.add_argument("label", help="Item label, e.g. candy_loose")
    p_teach.add_argument("--session", default="thu", help="Session tag (thu/fri)")

    args = parser.parse_args()

    if args.command == "status":
        asyncio.run(cmd_status())
    elif args.command == "tare":
        asyncio.run(cmd_tare())
    elif args.command == "identify":
        asyncio.run(cmd_identify(session=args.session))
    elif args.command == "teach":
        asyncio.run(cmd_teach(label=args.label, session=args.session))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
