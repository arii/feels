#!/usr/bin/env python3
"""
split_dataset.py — stratified train/test split of the recorded trials.

Splits at the RECORDING level (never across rows — rows within a trial are
correlated and would leak). Stratifies by (object_label, motion_path) so every
class AND both motion paths (normal / swirl) are represented in train and test
in the same proportion.

Non-destructive: writes a manifest + file lists; use --copy to also copy files
into recordings/split/{train,test}/.

Usage:
    python split_dataset.py                       # 80/20, stratified, seed 42
    python split_dataset.py --test-size 0.25
    python split_dataset.py --by label            # stratify by label only
    python split_dataset.py --copy                # also copy files into split dirs
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import shutil
from collections import defaultdict

import pandas as pd

REC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings")


def scan_recordings():
    files = [f for f in sorted(glob.glob(os.path.join(REC_DIR, "recording_*.csv")))
             if not f.endswith("_camera.csv")]
    rows = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("object_label", "phase"))
            label = str(d["object_label"].iloc[0])
            motion = "swirl" if (d["phase"] == "SWIRL").any() else "normal"
            rows.append({"file": os.path.basename(f), "label": label,
                         "motion_path": motion, "n_rows": len(d)})
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")
    return rows


def stratified_split(rows, test_size, seed, by):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in rows:
        key = r["label"] if by == "label" else (r["label"], r["motion_path"])
        strata[key].append(r)

    for key, items in sorted(strata.items(), key=lambda kv: str(kv[0])):
        items.sort(key=lambda r: r["file"])   # deterministic before shuffle
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_size))
        if n >= 2:                              # keep >=1 in each side
            n_test = min(max(n_test, 1), n - 1)
        else:
            n_test = 0                          # singleton stratum -> train
        for i, r in enumerate(items):
            r["split"] = "test" if i < n_test else "train"
    return rows


def main():
    p = argparse.ArgumentParser(description="Stratified train/test split of recordings.")
    p.add_argument("--test-size", type=float, default=0.2, help="fraction held out (default 0.2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--by", choices=["label_motion", "label"], default="label_motion",
                   help="stratification key (default: label+motion_path)")
    p.add_argument("--copy", action="store_true", help="also copy files into recordings/split/{train,test}/")
    args = p.parse_args()

    rows = scan_recordings()
    if not rows:
        print("No recordings found."); return
    rows = stratified_split(rows, args.test_size, args.seed, args.by)

    # manifest
    man_path = os.path.join(REC_DIR, "split_manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "motion_path", "n_rows", "split"])
        w.writeheader(); w.writerows(rows)

    # plain file lists
    split_dir = os.path.join(REC_DIR, "split")
    os.makedirs(split_dir, exist_ok=True)
    for sp in ("train", "test"):
        with open(os.path.join(split_dir, f"{sp}.txt"), "w") as f:
            for r in rows:
                if r["split"] == sp:
                    f.write(os.path.join("recordings", r["file"]) + "\n")

    if args.copy:
        for sp in ("train", "test"):
            d = os.path.join(split_dir, sp); os.makedirs(d, exist_ok=True)
            for r in rows:
                if r["split"] == sp:
                    shutil.copy2(os.path.join(REC_DIR, r["file"]), os.path.join(d, r["file"]))

    # summary
    summary = defaultdict(lambda: defaultdict(int))
    for r in rows:
        summary[r["label"]][r["split"]] += 1
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")

    print(f"\nStratified by: {args.by}   test_size={args.test_size}   seed={args.seed}")
    print("=" * 56)
    print(f"{'label':22s} {'train':>6s} {'test':>6s} {'total':>6s}")
    print("-" * 56)
    for lab in sorted(summary):
        tr, te = summary[lab]["train"], summary[lab]["test"]
        print(f"{lab:22s} {tr:6d} {te:6d} {tr+te:6d}")
    print("-" * 56)
    print(f"{'TOTAL':22s} {n_train:6d} {n_test:6d} {n_train+n_test:6d}")
    print("=" * 56)
    print(f"\nmanifest: {os.path.relpath(man_path)}")
    print(f"lists:    recordings/split/train.txt, recordings/split/test.txt"
          + ("  (+ copied files)" if args.copy else ""))

    with open(os.path.join(split_dir, "split_meta.json"), "w") as f:
        json.dump({"test_size": args.test_size, "seed": args.seed, "by": args.by,
                   "n_train": n_train, "n_test": n_test, "n_total": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
