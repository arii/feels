#!/usr/bin/env python3
"""
predict_content.py — run the saved cascade on one recording: empty -> solid/liquid.

    python predict_content.py recordings/recording_..._water_025pct.csv

Loads models/cascade.joblib (from `train_classifier.py --save`):
  Stage 1  empty vs loaded          (mass)
  Stage 2  solid vs liquid          (swirl dynamics; all-loaded model is the default)

Honest by construction: prints per-stage confidence and flags low-confidence /
"swirl recommended" cases rather than over-committing. Numbers are approximate
(small training folds) — see models/README.md.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

import train_classifier as tc
import feature_extractor as fx

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "cascade.joblib")


def _conf(pipe, x):
    """Return {class: prob} if available, else a decision-margin proxy."""
    clf = pipe.steps[-1][1]
    if hasattr(clf, "predict_proba"):
        return dict(zip(pipe.classes_, pipe.predict_proba(x)[0]))
    d = np.atleast_1d(pipe.decision_function(x)[0])
    if d.size == 1:  # binary decision_function -> signed distance
        p = 1 / (1 + np.exp(-d[0]))
        return {pipe.classes_[1]: float(p), pipe.classes_[0]: float(1 - p)}
    e = np.exp(d - d.max())
    return dict(zip(pipe.classes_, (e / e.sum()).tolist()))


def predict(csv_path):
    if not os.path.isfile(MODEL):
        sys.exit(f"no model at {MODEL} — run: python train_classifier.py --save")
    b = joblib.load(MODEL)
    s1, s2_all, s2_sw, order = b["stage1"], b["stage2_all"], b["stage2_swirl"], b["feature_order"]

    df = pd.read_csv(csv_path)
    has_swirl = bool((df["phase"] == fx.SWIRL_PHASE).any())
    x = np.array([[tc.features(df)[k] for k in order]])

    print(f"file: {os.path.basename(csv_path)}")
    if "object_label" in df:
        lbl = str(df["object_label"].iloc[0])
        print(f"true: {lbl} -> {tc.to_3class(lbl)}   (swirl in recording: {has_swirl})")

    # Stage 1
    c1 = _conf(s1, x); p1 = max(c1, key=c1.get)
    print(f"\n[stage 1] {p1}   ({', '.join(f'{k}={v:.2f}' for k,v in sorted(c1.items(), key=lambda t:-t[1]))})")
    if p1 == "empty":
        flag = "  ⚠ low confidence" if c1["empty"] < 0.6 else ""
        print(f"\nPREDICTION: EMPTY{flag}")
        return

    # Stage 2 (default = all-loaded model; swirl model shown when applicable)
    c2 = _conf(s2_all, x); p2 = max(c2, key=c2.get)
    print(f"[stage 2] {p2}   ({', '.join(f'{k}={v:.2f}' for k,v in sorted(c2.items(), key=lambda t:-t[1]))})")
    note = ""
    if not has_swirl:
        note = "  (no swirl in this recording — liquid/solid is low-confidence; a swirl is recommended)"
    elif abs(c2[p2] - 0.5) < 0.12:
        note = "  ⚠ borderline"
    else:
        c2s = _conf(s2_sw, x)
        note = f"  (swirl-model agrees: {max(c2s, key=c2s.get)})"
    print(f"\nPREDICTION: {p2.upper()}{note}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python predict_content.py <recording.csv>")
    predict(sys.argv[1])
