#!/usr/bin/env python3
"""Batch-teach recording CSVs into Atlas (no robot needed).

Teaches labeled recordings collected earlier, e.g. after the P0 run:

    python teach_csv.py recordings/zzzold/recording_20261029_153038_water_half.csv
    python teach_csv.py recordings/ --session thu
    python teach_csv.py recordings/ --dry-run     # show what would be taught
    python teach_csv.py --list                    # signature counts per label

Label resolution: --label override > object_label column > filename.
Files labeled mystery/unlabeled are skipped unless --include-mystery.
Files already taught (matched by properties.csv) are skipped unless --force.
"""

import argparse
import glob
import os
import sys

import pandas as pd

import atlas
import feature_extractor

SKIP_LABELS = {"mystery", "unlabeled", "unknown"}
MOTION_TYPE = "swirl_macro"


def resolve_label(csv_path, df, override=None):
    """Determine the teach label for a CSV.

    Precedence: explicit override, then the object_label column, then the
    recording_YYYYMMDD_HHMMSS_<label>.csv filename.

    Args:
        csv_path: path to the recording CSV.
        df: the recording DataFrame.
        override: forced label, or None.

    Returns:
        Label string, or None when none can be determined.
    """
    if override:
        return override
    if "object_label" in df.columns:
        lbl = str(df["object_label"].iloc[0]).strip()
        if lbl and lbl.lower() != "nan":
            return lbl
    base = os.path.basename(csv_path)
    # recording_YYYYMMDD_HHMMSS_<label>.csv
    parts = base.replace(".csv", "").split("_")
    if len(parts) >= 4 and parts[0] == "recording":
        return "_".join(parts[3:])
    return None


def teach_file(db, csv_path, session, label_override=None, dry_run=False,
               force=False, include_mystery=False, require_swirl=False):
    """Extract features from one CSV and teach it to Atlas.

    Args:
        db: MongoDB database handle.
        csv_path: path to the recording CSV.
        session: session tag, e.g. "thu".
        label_override: forced label, or None.
        dry_run: report what would be taught without writing.
        force: re-teach files already in Atlas.
        include_mystery: also teach mystery-labeled files.
        require_swirl: skip recordings without a true SWIRL phase.

    Returns:
        Tuple (status, filename, detail) where status is one of
        "taught", "dry", "skip", "error".
    """
    name = os.path.basename(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return ("error", name, f"unreadable: {e}")
    if not {"phase", "tau2_nm", "tau3_nm"}.issubset(df.columns):
        return ("skip", name, "not a grasp recording")
    label = resolve_label(csv_path, df, label_override)
    if not label:
        return ("skip", name, "no label found")
    if label.lower() in SKIP_LABELS and not include_mystery:
        return ("skip", name, f"label '{label}' (use --include-mystery to teach it)")
    try:
        feats = feature_extractor.extract_features(df)
    except Exception as e:
        return ("error", name, f"extraction failed: {e}")
    if dry_run:
        dq = feats["diagnostics"]
        proto = "swirl" if dq["phase_source"] == "SWIRL" else "fallback"
        if require_swirl and dq["phase_source"] != "SWIRL":
            return ("skip", name, "no SWIRL phase (fallback protocol)")
        return ("dry", name,
                f"would teach '{label}' [{proto}] "
                f"({dq['n_samples']} samples, tare={dq['tare_flags'] or 'OK'})")
    if require_swirl and feats["diagnostics"]["phase_source"] != "SWIRL":
        return ("skip", name, "no SWIRL phase (fallback protocol)")
    if not force and db.signatures.count_documents({"properties.csv": name}, limit=1):
        return ("skip", name, "already in Atlas (use --force to re-teach)")
    dq = feats["diagnostics"]
    saved = atlas.teach(
        label, feats["e"], motion_type=MOTION_TYPE, session=session,
        properties={"csv": name, "source": "teach_csv.py",
                    "tare_flags": dq["tare_flags"],
                    "phase_source": dq["phase_source"],
                    "z": [round(float(v), 4) for v in feats["z"]],
                    "n_samples": dq["n_samples"], "fs_hz": dq["fs_hz"]})
    return ("taught", name, f"'{label}' sig_id={saved['sig_id']}")


def list_library():
    """Print signature counts per label in the Atlas library."""
    db = atlas.get_db()
    rows = list(db.signatures.aggregate([
        {"$group": {"_id": "$label", "n": {"$sum": 1},
                    "motion_types": {"$addToSet": "$motion_type"},
                    "sessions": {"$addToSet": "$session"}}},
        {"$sort": {"_id": 1}}]))
    if not rows:
        print("signatures collection is empty")
        return
    print(f"{'label':<24} {'n':>3}  motion_types  sessions")
    for r in rows:
        print(f"{r['_id']:<24} {r['n']:>3}  "
              f"{','.join(sorted(r['motion_types']))}  {','.join(sorted(r['sessions']))}")


def main():
    """CLI entry: batch-teach recordings to Atlas."""
    ap = argparse.ArgumentParser(
        description="Batch-teach recording CSVs into Atlas.",
        epilog="""examples:
  python teach_csv.py recordings/ --require-swirl --dry-run   # preview
  python teach_csv.py recordings/ --require-swirl --session fri
  python teach_csv.py --list                                  # library contents
  python teach_csv.py foo.csv --label water_075pct --force    # re-teach one file

Label resolution: --label > object_label column > filename.
Mystery/unlabeled and non-grasp CSVs are skipped; already-taught files are
skipped unless --force. Needs MONGODB_URI in .env, and the Atlas IP
allowlist must include this machine.""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="CSV files, globs, or dirs (default: recordings/)")
    ap.add_argument("--session", default="thu", help="session tag (default: thu)")
    ap.add_argument("--label", default=None, help="force one label for all files")
    ap.add_argument("--dry-run", action="store_true", help="show what would be taught")
    ap.add_argument("--force", action="store_true", help="re-teach files already in Atlas")
    ap.add_argument("--include-mystery", action="store_true")
    ap.add_argument("--require-swirl", action="store_true",
                    help="only teach recordings with a true SWIRL phase")
    ap.add_argument("--list", action="store_true", help="show signature counts per label")
    args = ap.parse_args()

    if args.list:
        list_library()
        return

    files = []
    for p in args.paths or ["recordings/"]:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True))
        elif os.path.isfile(p):
            files.append(p)
        else:
            files += sorted(glob.glob(p, recursive=True))
    if not files:
        sys.exit("no CSV files found")

    db = None if args.dry_run else atlas.get_db()
    counts = {}
    for f in files:
        status, name, detail = teach_file(
            db, f, args.session, label_override=args.label,
            dry_run=args.dry_run, force=args.force,
            include_mystery=args.include_mystery,
            require_swirl=args.require_swirl)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status:>7}] {name}: {detail}")
    print("\nsummary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
