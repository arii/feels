#!/usr/bin/env python3
"""
push_to_atlas.py — Upload all telemetry recordings into MongoDB Atlas.

Ingests every recordings/recording_*.csv (the teammate's 24-column
trajectory+torque schema) as one document, stored column-oriented for
compact + queryable storage, with per-run summary stats (tare/hold J3,
delta torque, estimated mass). Idempotent: upserts keyed by source_file,
so re-running updates rather than duplicates.

Usage:
    python push_to_atlas.py --dry-run          # parse + report, no writes
    python push_to_atlas.py                    # upload into 'runs'
    python push_to_atlas.py --collection telemetry_runs --session oct29
"""

import argparse
import csv
import glob
import os
from datetime import datetime, timezone

import atlas

GRAVITY = 9.80665
R_XY = 0.56  # effective lever arm used in repeatable_grasp analysis

STR_COLS = {"phase", "object_label"}
BOOL_COLS = {"gripper_holding"}


def parse_recording(path: str) -> dict | None:
    fname = os.path.basename(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "elapsed_sec" not in rows[0]:
        return None  # not a telemetry recording (e.g. split_manifest.csv)

    headers = list(rows[0].keys())
    cols: dict[str, list] = {h: [] for h in headers}
    for r in rows:
        for h in headers:
            v = r[h]
            if h in STR_COLS:
                cols[h].append(v)
            elif h in BOOL_COLS:
                cols[h].append(str(v).strip().lower() == "true")
            else:
                try:
                    cols[h].append(float(v))
                except (ValueError, TypeError):
                    cols[h].append(None)

    # label: prefer the column, else parse from filename recording_<ts>_<label>.csv
    labels = [x for x in cols.get("object_label", []) if x]
    label = max(set(labels), key=labels.count) if labels else \
        fname.replace("recording_", "").split("_", 2)[-1].rsplit(".", 1)[0]

    # created_at from filename timestamp
    try:
        stamp = "_".join(fname.split("_")[1:3])
        created = datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        created = datetime.now(timezone.utc)

    phases = list(dict.fromkeys(cols.get("phase", [])))  # unique, order-preserving
    tau3 = cols.get("tau3_nm", [])
    ph = cols.get("phase", [])
    tare = [t for t, p in zip(tau3, ph) if p in ("INIT", "APPROACH") and t is not None]
    hold = [t for t, p in zip(tau3, ph) if p == "TRAVEL_STABLE_HOLD" and t is not None]
    if not hold:
        hold = [t for t, p in zip(tau3, ph) if "TRAVEL" in p and t is not None]
    z = [v for v in cols.get("tcp_z_mm", []) if v is not None]

    summary = {}
    if tare and hold:
        tare_j3 = sum(tare) / len(tare)
        hold_j3 = sum(hold) / len(hold)
        delta = hold_j3 - tare_j3
        summary = {
            "tare_j3_nm": round(tare_j3, 4),
            "hold_j3_nm": round(hold_j3, 4),
            "delta_j3_nm": round(delta, 4),
            "est_mass_g": round(abs(delta) / (GRAVITY * R_XY) * 1000.0, 1) if abs(delta) > 0.05 else 0.0,
        }
    if z:
        summary["z_min_mm"] = round(min(z), 1)
        summary["z_max_mm"] = round(max(z), 1)

    return {
        "source_file": fname,
        "kind": "telemetry_recording",
        "object_label": label,
        "motion_type": "pick_place",
        "num_samples": len(rows),
        "duration_s": round(cols["elapsed_sec"][-1] - cols["elapsed_sec"][0], 2) if cols.get("elapsed_sec") else 0.0,
        "phases": phases,
        "columns": headers,
        "data": cols,
        "summary": summary,
        "created_at": created,
    }


def main():
    p = argparse.ArgumentParser(description="Push recordings into MongoDB Atlas")
    p.add_argument("--collection", default="runs", help="target collection (default: runs)")
    p.add_argument("--session", default="oct29", help="session tag stored on each doc")
    p.add_argument("--glob", default="recordings/recording_*.csv", help="files to ingest")
    p.add_argument("--dry-run", action="store_true", help="parse + report, no writes")
    args = p.parse_args()

    files = sorted(glob.glob(args.glob))
    print(f"Found {len(files)} candidate files matching {args.glob!r}")

    docs = []
    for path in files:
        d = parse_recording(path)
        if d is None:
            print(f"  skip (not telemetry): {os.path.basename(path)}")
            continue
        d["session"] = args.session
        docs.append(d)

    # label histogram
    from collections import Counter
    hist = Counter(d["object_label"] for d in docs)
    print(f"\nParsed {len(docs)} recordings:")
    for lbl, n in sorted(hist.items()):
        print(f"  {lbl:16} {n}")

    if args.dry_run:
        print("\n[DRY RUN] no writes. Sample summary:")
        if docs:
            s = docs[0]
            print(f"  {s['source_file']}  label={s['object_label']}  "
                  f"samples={s['num_samples']}  summary={s['summary']}")
        return

    db = atlas.get_db()
    coll = db[args.collection]
    coll.create_index("source_file", unique=True)
    up = 0
    for d in docs:
        coll.replace_one({"source_file": d["source_file"]}, d, upsert=True)
        up += 1
    print(f"\n[✓] Upserted {up} recordings into '{args.collection}' "
          f"(now {coll.count_documents({})} docs). Session='{args.session}'.")


if __name__ == "__main__":
    main()
