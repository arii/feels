#!/usr/bin/env python3
"""
predict_mystery.py — offline k-NN classifier for the feels cup-content study.

Takes a freshly recorded CSV (e.g. a mystery cup probed by
`repeatable_grasp.py --record`) and predicts its contents by comparing its
cleaned joint-torque signature against every labeled recording in
`recordings/` (real grasp-cycle recordings only; single-joint telemove sweeps
and synthetic data are skipped automatically).

Cleaning + feature pipeline mirrors `notebooks/content_discrimination.ipynb`:
  1. per-joint tare on the INIT-phase baseline,
  2. 2nd-order Butterworth low-pass (0.8 Hz) on the ~4-5 Hz telemetry,
  3. 12-D feature vector:
       hold_tau1..6  — mean cleaned Δτ in TRAVEL_STABLE_HOLD (steady payload)
       dyn_tau1..6   — std of raw Δτ in LIFT_TO_TRAVEL + DESCEND_PLACE (motion response)

Usage:
  python predict_mystery.py                       # classify recordings labeled 'mystery'
  python predict_mystery.py recordings/foo.csv    # classify a specific CSV file
  python predict_mystery.py --eval                # leave-one-recording-out accuracy
                                                  # on the labeled reference set
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "recordings")

JOINTS = list(range(1, 7))
HOLD_PHASE = "TRAVEL_STABLE_HOLD"
DYN_PHASES = ["LIFT_TO_TRAVEL", "DESCEND_PLACE"]
KNOWN_LABEL_SKIP = {"mystery", "unknown", ""}


# ── signal cleaning (mirrors the notebook) ──────────────────────────────
def tare_value(df, col="tau3_nm", phase="INIT"):
    init = df.loc[df["phase"] == phase, col]
    if len(init):
        return init.median()
    return df[col].iloc[:5].median()


def butter_lowpass(x, fs, cutoff_hz=0.8, order=2):
    nyq = fs / 2.0
    wc = min(cutoff_hz, 0.9 * nyq) / nyq
    b, a = butter(order, wc, btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x.astype(float)
    return filtfilt(b, a, x)


def add_clean(df):
    df = df.copy()
    diffs = df["elapsed_sec"].diff().median()
    fs = 1.0 / diffs if diffs and diffs > 0 else 5.0
    for j in JOINTS:
        col = f"tau{j}_nm"
        tare = tare_value(df, col=col)
        df[f"dtau{j}"] = df[col].values - tare
        df[f"dtau{j}_s"] = butter_lowpass(df[f"dtau{j}"].values, fs, 0.8)
    return df


# ── features (mirrors the notebook's 12-D joint vector) ─────────────────
def extract_features(df):
    feat = {"label": str(df["object_label"].iloc[0])}
    hold_mask = df["phase"] == HOLD_PHASE
    dyn_mask = df["phase"].isin(DYN_PHASES)
    for j in JOINTS:
        h = df.loc[hold_mask, f"dtau{j}_s"].values
        if len(h) < 3:
            h = df[f"dtau{j}_s"].values[len(df) // 2:]  # fallback: second half
        d = df.loc[dyn_mask, f"dtau{j}"].values
        feat[f"hold_tau{j}"] = round(float(np.mean(h)) if len(h) else 0.0, 3)
        feat[f"dyn_tau{j}"] = round(float(np.std(d)) if len(d) else 0.0, 4)
    return feat


FEATS = [f"hold_tau{j}" for j in JOINTS] + [f"dyn_tau{j}" for j in JOINTS]


# ── data loading ────────────────────────────────────────────────────────
def load_real_recordings():
    """All real grasp-cycle CSVs under recordings/ (incl. zzzold)."""
    recs = {}
    for path in sorted(glob.glob(os.path.join(REC_DIR, "**", "*.csv"), recursive=True)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        need = {"object_label", "phase", "elapsed_sec"} | {f"tau{j}_nm" for j in JOINTS}
        if not need.issubset(df.columns):
            continue  # telemove sweeps / other schemas
        name = os.path.relpath(path, REC_DIR)
        recs[name] = df
    return recs


def build_reference(recs):
    """Standardized 12-D reference set from labeled recordings."""
    rows = []
    for name, df in recs.items():
        label = str(df["object_label"].iloc[0])
        if label in KNOWN_LABEL_SKIP:
            continue
        rows.append({"file": name, **extract_features(add_clean(df))})
    if not rows:
        raise SystemExit("No labeled reference recordings found under recordings/.")
    ref = pd.DataFrame(rows)
    scaler = StandardScaler().fit(ref[FEATS])
    return ref, scaler


# ── prediction ──────────────────────────────────────────────────────────
def predict(df, ref, scaler, k=3):
    q = extract_features(add_clean(df))
    Xq = scaler.transform(pd.DataFrame([q])[FEATS])
    Xr = scaler.transform(ref[FEATS])
    dists = np.linalg.norm(Xr - Xq, axis=1)
    order = np.argsort(dists)
    k = min(k, len(ref))
    neigh = ref.iloc[order[:k]]
    ndist = dists[order[:k]]
    votes, counts = np.unique(neigh["label"], return_counts=True)
    pred = votes[int(np.argmax(counts))]
    conf = counts.max() / k
    return pred, conf, q, neigh, ndist


def print_report(name, pred, conf, q, neigh, ndist):
    print(f"\n=== {name} ===")
    print(f"  held Δτ₃ = {q['hold_tau3']:+.3f} N·m   dyn Δτ₃ = {q['dyn_tau3']:.4f} N·m")
    print("  nearest neighbours (12-D standardized joint-torque space):")
    for (_, row), d in zip(neigh.iterrows(), ndist):
        mark = " ★" if row["label"] == pred else ""
        print(f"    {row['label']:<14s} dist={d:6.3f}  (Δτ₃={row['hold_tau3']:+.3f}){mark}")
    print(f"  →  PREDICTION: {pred}   (confidence {conf:.0%} = {counts_of(neigh, pred)}/{len(neigh)} votes)")


def counts_of(neigh, pred):
    return int((neigh["label"] == pred).sum())


def run_eval(ref):
    print(f"\nLeave-one-recording-out on {len(ref)} labeled recordings:")
    correct = 0
    for i, held in ref.iterrows():
        tr = ref.drop(i)
        scaler = StandardScaler().fit(tr[FEATS])
        knn = KNeighborsClassifier(n_neighbors=min(3, len(tr)))
        knn.fit(scaler.transform(tr[FEATS]), tr["label"])
        pred = knn.predict(scaler.transform(held[FEATS].to_frame().T))[0]
        ok = pred == held["label"]
        correct += ok
        print(f"  {held['file']:<52s} true={held['label']:<12s} pred={pred:<12s} {'✓' if ok else '✗'}")
    print(f"\n  LORO accuracy: {correct}/{len(ref)} = {correct/len(ref):.0%}")


def run_qc(recs):
    """Flag recordings whose tare baseline is contaminated.

    Checks, per recording:
      1. INIT drift: max-min of tau2/tau3 within INIT (arm should be still).
      2. INIT pose: tcp during INIT vs the nominal home pose (motion during tare).
      3. Settle residual: |TRAVEL_STABLE_HOLD - TRAVEL_SETTLING| on tau3
         (large => baseline taken mid-transient).
    """
    print("\nRun quality check (tare-baseline diagnostics):")
    for name, df in recs.items():
        lab = str(df["object_label"].iloc[0])
        init = df[df["phase"] == "INIT"]
        hold = df[df["phase"] == HOLD_PHASE]
        settle = df[df["phase"] == "TRAVEL_SETTLING"]
        flags = []
        d2 = d3 = float("nan")
        if len(init) >= 2:
            d2 = init["tau2_nm"].max() - init["tau2_nm"].min()
            d3 = init["tau3_nm"].max() - init["tau3_nm"].min()
            if d2 > 3.0 or d3 > 1.0:
                flags.append(f"INIT-DRIFT tau2±{d2:.1f}/tau3±{d3:.1f} N·m (arm not settled)")
            q2_rng = init["q2_deg"].max() - init["q2_deg"].min()
            if q2_rng > 5.0:
                flags.append(f"INIT-MOTION q2 range {q2_rng:.1f}° (tare taken mid-motion)")
        else:
            flags.append("INIT too short for tare")
        if len(settle) and len(hold):
            res = abs(hold["tau3_nm"].median() - settle["tau3_nm"].median())
            if res > 1.0:
                flags.append(f"SETTLE-RESIDUAL {res:.1f} N·m on tau3 (transient bleed)")
        status = "OK " if not flags else "WARN"
        print(f"  [{status}] {name[:46]:46s} {lab:<11s}")
        for f in flags:
            print(f"           ⚠ {f}")
    print("\n  Thresholds: INIT drift >3 N·m (tau2) / >1 N·m (tau3), "
          "INIT q2 motion >5°, settle residual >1 N·m.")


def main():
    ap = argparse.ArgumentParser(
        description="k-NN cup-content classifier (offline).",
        epilog="""examples:
  python predict_mystery.py recordings/mystery_01.csv   # classify one recording
  python predict_mystery.py --eval                      # leave-one-recording-out accuracy
  python predict_mystery.py --qc                        # tare-baseline quality per recording

Runs entirely offline on CSVs — no robot, no Atlas. See BASELINE_FINDINGS.md
for what the QC flags mean.""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="CSV to classify (default: all 'mystery' recordings)")
    ap.add_argument("--eval", action="store_true", help="LORO accuracy on the reference set")
    ap.add_argument("--qc", action="store_true", help="tare-baseline quality diagnostics per recording")
    args = ap.parse_args()

    recs = load_real_recordings()
    if not recs:
        raise SystemExit("No real grasp-cycle recordings found under recordings/.")
    ref, scaler = build_reference(recs)
    print(f"Reference set: {len(ref)} labeled recordings "
          f"({', '.join(sorted(ref['label'].unique()))})")

    if args.eval:
        run_eval(ref)
        return

    if args.qc:
        run_qc(recs)
        return

    targets = []
    if args.csv:
        df = pd.read_csv(args.csv)
        targets.append((args.csv, df))
    else:
        for name, df in recs.items():
            if str(df["object_label"].iloc[0]) in KNOWN_LABEL_SKIP:
                targets.append((name, df))
        if not targets:
            raise SystemExit("No 'mystery' recordings found; pass a CSV path explicitly.")

    k = min(3, len(ref))
    for name, df in targets:
        pred, conf, q, neigh, ndist = predict(df, ref, scaler, k=k)
        print_report(name, pred, conf, q, neigh, ndist)


if __name__ == "__main__":
    main()
