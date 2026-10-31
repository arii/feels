#!/usr/bin/env python3
"""
teach_hf_5class.py — Populate Atlas with the 5-class demo taxonomy from HF.

Maps old HF labels to: water, empty_cup, screws, candy.
Skips solid/*, soap/* (not in demo taxonomy).

    python tools/teach_hf_5class.py              # pull + teach
    python tools/teach_hf_5class.py --dry-run    # show what would happen

Then teach 2-3 'empty' (gripper holding nothing) via the dashboard.
Requires: pip install huggingface_hub, MONGODB_URI set.
"""
import argparse
import glob
import os
import re
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "dhyuti-n/feels-cup-dynamics"


def map_label(old: str):
    """Old HF label -> new 5-class label, or None to skip."""
    old = old.strip().lower()
    if old.startswith("water"):
        return "water"
    if old == "empty_cup":
        return "empty_cup"
    if old == "empty":
        return "empty"
    if old.startswith("screw"):
        return "screws"
    if old.startswith("candy"):
        return "candy"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 1. Pull from HF
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    dest = os.path.join(HERE, "recordings", "hf_5class")
    if not args.dry_run:
        print(f"Pulling {REPO}...")
        src = snapshot_download(repo_id=REPO, repo_type="dataset")
        os.makedirs(dest, exist_ok=True)
        n = 0
        for root, _, files in os.walk(src):
            for f in files:
                if f.endswith(".csv"):
                    shutil.copy2(os.path.join(root, f), os.path.join(dest, f))
                    n += 1
        print(f"  {n} files -> {dest}")

    # 2. Map labels
    counts = Counter()
    to_teach = []
    for f in sorted(glob.glob(os.path.join(dest, "*.csv"))):
        base = os.path.basename(f)
        m = re.match(r'recording_\d+_\d+_(.+)\.csv', base)
        old_label = m.group(1) if m else base.replace(".csv", "")
        new_label = map_label(old_label)
        if new_label:
            to_teach.append((f, new_label))
            counts[new_label] += 1
        else:
            counts[f"SKIP:{old_label}"] += 1

    print("\nLabel mapping:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

    if args.dry_run or not to_teach:
        return

    # 3. Teach to Atlas
    sys.path.insert(0, HERE)
    import pandas as pd
    import feature_extractor
    import embedder
    from realtime_plot import _atlas
    at = _atlas()

    print(f"\nTeaching {len(to_teach)} to Atlas...")
    done = 0
    for fpath, label in to_teach:
        try:
            df = pd.read_csv(fpath)
            feats = feature_extractor.extract_features(df)
            emb = embedder.embed(df)
            at.teach(label, emb, motion_type="swirl_macro", session="hf_5class",
                     properties={"csv": os.path.basename(fpath),
                                 "z": [round(float(v), 4) for v in feats["z"]]})
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(to_teach)}")
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {str(e)[:80]}")

    print(f"\nDone: {done} signatures taught.")
    print("Next: teach 2-3 'empty' (gripper holding nothing) via dashboard Mode 2.")


if __name__ == "__main__":
    main()
