#!/usr/bin/env python3
"""
state_spectrum.py — represent cup contents as a 2-D STATE SPECTRUM, not hard labels.

The cascade's two stage probabilities ARE two physically-grounded axes:
    loadedness = P(loaded)        from Stage 1  (empty  <-> full)   [mass]
    rigidity   = P(solid)         from Stage 2  (liquid <-> solid)  [swirl dynamics]

Plus CLASS-FILTERED neighbour retrieval: once the coarse state is decided, only
compare against neighbours of that state (don't let liquid signatures pollute a
solid match) — the Atlas `$vectorSearch` `filter` pattern, shown here locally
over the recordings so it's leak-free and offline-safe.

Honest note: on proprioception-only features the signal is weak, so these axes are
confidence *leanings*, not calibrated physical measurements. The representation is
correct and sharpens with a richer (vision/audio) embedding.

Usage:
    python state_spectrum.py recordings/recording_..._water_025pct.csv
    python state_spectrum.py <csv> --neighbors      # + class-filtered nearest items
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import joblib

import train_classifier as tc
import feature_extractor as fx

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "cascade.joblib")


def _p(pipe, x, cls):
    """P(class=cls) from a fitted pipeline."""
    proba = pipe.predict_proba(x)[0]
    return float(dict(zip(pipe.classes_, proba)).get(cls, 0.0))


def bar(v, width=24, lo="", hi=""):
    n = int(round(np.clip(v, 0, 1) * width))
    return f"{lo:>8} [{'█'*n}{'·'*(width-n)}] {hi:<8}  {v*100:4.0f}%"


def spectrum(csv_path):
    b = joblib.load(MODEL)
    s1, s2, order = b["stage1"], b["stage2_all"], b["feature_order"]
    df = pd.read_csv(csv_path)
    has_swirl = bool((df["phase"] == fx.SWIRL_PHASE).any())
    x = np.array([[tc.features(df)[k] for k in order]])

    loadedness = _p(s1, x, "loaded")       # 0 empty .. 1 full
    rigidity = _p(s2, x, "solid")          # 0 liquid .. 1 solid (only meaningful if loaded)

    label = str(df["object_label"].iloc[0]) if "object_label" in df else "?"
    print(f"file: {os.path.basename(csv_path)}   (true: {label}, swirl={has_swirl})")
    print("\n  STATE SPECTRUM")
    print("   loadedness  " + bar(loadedness, lo="empty", hi="full"))
    if loadedness < 0.5:
        print("   rigidity    (n/a — reads as empty)")
        coarse = "empty"
    else:
        tag = "  (low-confidence — no swirl)" if not has_swirl else ""
        print("   rigidity    " + bar(rigidity, lo="liquid", hi="solid") + tag)
        coarse = "solid" if rigidity >= 0.5 else "liquid"
    print(f"\n  reads as: {coarse.upper()}"
          f"  (loaded {loadedness*100:.0f}%" + (f", solid {rigidity*100:.0f}%)" if coarse != 'empty' else ")"))
    return x, coarse, loadedness, rigidity


def class_filtered_neighbors(x_feats, coarse, k=5):
    """Atlas-pattern: retrieve nearest items FILTERED to the predicted coarse state.
    Done locally over recordings (cosine kNN on fx embeddings = what Atlas stores)."""
    def emb(p):
        try:
            return np.array(fx.extract_features(pd.read_csv(p))["e"])
        except Exception:
            return None
    # the query embedding
    # (re-embed the queried file is cleaner; but we only have x_feats here, so
    #  recompute from the same file path passed by caller via globals)
    q = class_filtered_neighbors.query_emb
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    pool = []
    for p in sorted(glob.glob("recordings/recording_*.csv")):
        lbl = re.search(r"_\d{6}_(.*)\.csv", os.path.basename(p)).group(1)
        c = tc.to_3class(lbl)
        state = "empty" if c == "empty" else ("solid" if c in ("solid", "screws") else "liquid")
        if state != coarse:                      # <-- the filter: same state only
            continue
        e = emb(p)
        if e is not None and not np.array_equal(e, q):
            pool.append((cos(q, e), lbl, os.path.basename(p)))
    pool.sort(reverse=True)
    print(f"\n  class-filtered neighbours (state='{coarse}', Atlas $vectorSearch filter):")
    for s, lbl, fn in pool[:k]:
        print(f"    {s:.3f}  {lbl:16} {fn[-22:]}")


def main():
    ap = argparse.ArgumentParser(description="2-D state spectrum + class-filtered neighbours")
    ap.add_argument("csv")
    ap.add_argument("--neighbors", action="store_true", help="also show class-filtered nearest items")
    a = ap.parse_args()
    if not os.path.isfile(MODEL):
        sys.exit(f"no model at {MODEL} — run: python train_classifier.py --save")
    x, coarse, load, rig = spectrum(a.csv)
    if a.neighbors:
        class_filtered_neighbors.query_emb = np.array(fx.extract_features(pd.read_csv(a.csv))["e"])
        class_filtered_neighbors(x, coarse)


if __name__ == "__main__":
    main()
