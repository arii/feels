#!/usr/bin/env python3
"""
push_to_hf.py — publish the labeled telemetry recordings to the Hugging Face Hub
as a train/test DatasetDict, one example per recording.

Reads recordings/split_manifest.csv (the stratified, recording-level split) and
emits one example per trial: trial metadata + each telemetry channel as a list
(nested time series). Grouping stays at the recording level so the train/test
split never leaks across rows.

Usage:
    python tools/push_to_hf.py                      # build + push (private)
    python tools/push_to_hf.py --repo-id user/name  # override repo
    python tools/push_to_hf.py --public             # push public
    python tools/push_to_hf.py --dry-run            # build only, don't push
"""

import argparse
import csv
import os

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_DIR = os.path.join(ROOT, "recordings")
MANIFEST = os.path.join(REC_DIR, "split_manifest.csv")

# telemetry channels stored as per-row lists (nested time series)
SEQ_FLOAT = [
    "elapsed_sec",
    "tcp_x_mm", "tcp_y_mm", "tcp_z_mm",
    "tcp_ox", "tcp_oy", "tcp_oz", "tcp_theta_deg",
    "q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg",
    "tau1_nm", "tau2_nm", "tau3_nm", "tau4_nm", "tau5_nm", "tau6_nm",
    "gripper_pos",
]
SEQ_STR = ["phase"]
SEQ_BOOL = ["gripper_holding"]

FEATURES = Features({
    "file": Value("string"),
    "label": Value("string"),
    "motion_path": Value("string"),
    "split": Value("string"),
    "n_rows": Value("int32"),
    **{c: Sequence(Value("float32")) for c in SEQ_FLOAT},
    **{c: Sequence(Value("string")) for c in SEQ_STR},
    **{c: Sequence(Value("bool")) for c in SEQ_BOOL},
})


def load_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def build_example(row):
    path = os.path.join(REC_DIR, row["file"])
    df = pd.read_csv(path)
    ex = {
        "file": row["file"],
        "label": row["label"],
        "motion_path": row["motion_path"],
        "split": row["split"],
        "n_rows": int(len(df)),
    }
    for c in SEQ_FLOAT:
        ex[c] = df[c].astype("float32").tolist()
    for c in SEQ_STR:
        ex[c] = df[c].astype(str).tolist()
    for c in SEQ_BOOL:
        ex[c] = df[c].astype(bool).tolist()
    return ex


def main():
    p = argparse.ArgumentParser(description="Push recordings to the HF Hub as train/test splits.")
    p.add_argument("--repo-id", default="dhyuti-n/feels-cup-dynamics")
    p.add_argument("--public", action="store_true", help="create a public repo (default: private)")
    p.add_argument("--dry-run", action="store_true", help="build the DatasetDict but don't push")
    args = p.parse_args()

    rows = load_manifest()
    by_split = {"train": [], "test": []}
    for r in rows:
        by_split.setdefault(r["split"], []).append(build_example(r))

    dd = DatasetDict({
        sp: Dataset.from_list(exs, features=FEATURES)
        for sp, exs in by_split.items() if exs
    })

    print(dd)
    for sp, ds in dd.items():
        labs = {}
        for lab in ds["label"]:
            labs[lab] = labs.get(lab, 0) + 1
        print(f"  {sp}: {len(ds)} recordings | " + ", ".join(f"{k}={v}" for k, v in sorted(labs.items())))

    if args.dry_run:
        print("\n[dry-run] not pushing.")
        return

    print(f"\nPushing to https://huggingface.co/datasets/{args.repo_id} (private={not args.public}) ...")
    dd.push_to_hub(args.repo_id, private=not args.public)
    print("Done.")


if __name__ == "__main__":
    main()
