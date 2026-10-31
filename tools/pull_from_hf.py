#!/usr/bin/env python3
"""
pull_from_hf.py — Download the feels dataset from Hugging Face.

    python tools/pull_from_hf.py                          # to recordings/hf/
    python tools/pull_from_hf.py --teach                   # + teach to Atlas
    python tools/pull_from_hf.py --repo dhyuti-n/feels-cup-dynamics

Requires: pip install huggingface_hub
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO = "dhyuti-n/feels-cup-dynamics"


def main():
    ap = argparse.ArgumentParser(description="Pull feels dataset from Hugging Face")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    ap.add_argument("--dest", default=os.path.join(HERE, "recordings", "hf"),
                    help="destination directory")
    ap.add_argument("--teach", action="store_true",
                    help="teach downloaded recordings to Atlas after pull")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing files")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {args.repo} ...")
    src = snapshot_download(repo_id=args.repo, repo_type="dataset")
    print(f"Downloaded to {src}")

    os.makedirs(args.dest, exist_ok=True)
    n = 0
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(".csv"):
                dst = os.path.join(args.dest, f)
                if os.path.exists(dst) and not args.force:
                    continue
                shutil.copy2(os.path.join(root, f), dst)
                n += 1
    print(f"Copied {n} recordings -> {args.dest}")

    if args.teach:
        import subprocess
        cmd = [sys.executable, os.path.join(HERE, "teach_csv.py"),
               "--require-swirl", "--force", args.dest]
        print(f"Teaching: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=HERE)


if __name__ == "__main__":
    main()
