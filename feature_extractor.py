#!/usr/bin/env python3
"""
feature_extractor.py — swirl-phase feature extraction for the feels cup-content study.

Consumes one telemetry CSV from repeatable_grasp.py (single continuous
pick-up → swirl → place macro) and outputs the explicit 10-D feature vector z
plus its L2-normalized embedding e in S^9, ready for MongoDB Atlas
$vectorSearch (index: cup_dynamics_index, 10 dims, cosine).

10-D feature vector z:
  1. mass_j2        Δτ2 at pre-swirl peak-Z vs INIT tare        (N·m)
  2. mass_j3        Δτ3 at pre-swirl peak-Z vs INIT tare        (N·m)
  3. swirl_var_j2   variance of detrended τ2 across SWIRL       (N·m²)
  4. swirl_var_j3   variance of detrended τ3 across SWIRL       (N·m²)
  5. swirl_ptp_j3   peak-to-peak τ3 across SWIRL                (N·m)
  6. phase_lag_s    xcorr lag: TCP speed vs τ3 residual (s;
                    + = torque lags the commanded motion)
  7. rattle_e       spectral energy of τ3 residual in the observable
                    high band [1.5 Hz, Nyquist]. NOTE: at 4-5 Hz cloud
                    sampling Nyquist ≈ 2.5 Hz, so the true 5-25 Hz rattle
                    band is NOT observable — see diagnostics
                    ["rattle_band_limited"]. Needs the 100 Hz on-robot
                    module (control_service.py) for the real band.
  8. grip_comp      std of gripper_pos across SWIRL             (encoder units)
  9. lift_dyn_j3    std of τ3 in LIFT_TO_TRAVEL                 (N·m)
 10. swirl_speed    mean TCP speed across SWIRL                 (mm/s)

Usage:
  python feature_extractor.py recordings/recording_20261029_180332_empty_cup.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

JOINTS = list(range(1, 7))
SWIRL_PHASE = "SWIRL"
FALLBACK_DYN_PHASES = ["LIFT_TO_TRAVEL", "DESCEND_PLACE", "TRAVEL_SETTLING"]
FEATURE_NAMES = [
    "mass_j2", "mass_j3",
    "swirl_var_j2", "swirl_var_j3", "swirl_ptp_j3",
    "phase_lag_s", "rattle_e",
    "grip_comp", "lift_dyn_j3", "swirl_speed",
]

# Per-feature scales so the L2-normalized embedding isn't dominated by the
# largest-magnitude raw feature (swirl_speed ~160 mm/s vs variances ~1-5).
# Calibrated as the 90th percentile of |z| over the recordings available on
# 2026-10-29 (n=8, only one true SWIRL recording). RECALIBRATE on the P0
# swirl dataset, then freeze — rescaling invalidates old Atlas signatures.
# Regenerate with:  python feature_extractor.py --calibrate
FEATURE_SCALES = {
    "mass_j2": 8.828,
    "mass_j3": 2.327,
    "swirl_var_j2": 3.813,
    "swirl_var_j3": 1.37,
    "swirl_ptp_j3": 10.93,
    "phase_lag_s": 1.366,
    "rattle_e": 165,
    "grip_comp": 0.7478,
    "lift_dyn_j3": 3.069,
    "swirl_speed": 82.66,
}


# ── signal helpers ──────────────────────────────────────────────────────
def _fs(df):
    diffs = df["elapsed_sec"].diff().median()
    return 1.0 / diffs if diffs and diffs > 0 else 5.0


def _lowpass(x, fs, cutoff_hz=0.8, order=2):
    nyq = fs / 2.0
    wc = min(cutoff_hz, 0.9 * nyq) / nyq
    b, a = butter(order, wc, btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x.astype(float)
    return filtfilt(b, a, x)


def _tare_with_qc(df):
    """Per-joint INIT tare + quality flags (see BASELINE_FINDINGS.md)."""
    tares, flags = {}, []
    for j in JOINTS:
        col = f"tau{j}_nm"
        init = df.loc[df["phase"] == "INIT", col]
        tares[j] = init.median() if len(init) else df[col].iloc[:5].median()
    if len(init) >= 2:
        d2 = df.loc[df["phase"] == "INIT", "tau2_nm"]
        d3 = df.loc[df["phase"] == "INIT", "tau3_nm"]
        if d2.max() - d2.min() > 3.0 or d3.max() - d3.min() > 1.0:
            flags.append("INIT_DRIFT")
        q2 = df.loc[df["phase"] == "INIT", "q2_deg"]
        if q2.max() - q2.min() > 5.0:
            flags.append("INIT_MOTION")
    return tares, flags


def _tcp_speed(df):
    t = df["elapsed_sec"].values
    x = df["tcp_x_mm"].values
    y = df["tcp_y_mm"].values
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    return np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y)) / dt])


# ── main extraction ─────────────────────────────────────────────────────
def extract_features(df):
    if not {"phase", "elapsed_sec", "tcp_x_mm", "tcp_y_mm", "tau2_nm", "tau3_nm"}.issubset(df.columns):
        raise ValueError("CSV missing required columns (phase, elapsed_sec, tcp_*, tau2/3_nm)")
    fs = _fs(df)
    tares, tare_flags = _tare_with_qc(df)
    dtau2 = df["tau2_nm"].values - tares[2]
    dtau3 = df["tau3_nm"].values - tares[3]
    speed = _tcp_speed(df)

    sw = df[df["phase"] == SWIRL_PHASE]
    phase_source = SWIRL_PHASE
    if len(sw) < 4:
        sw = df[df["phase"].isin(FALLBACK_DYN_PHASES)]
        phase_source = "fallback:" + ",".join(FALLBACK_DYN_PHASES)
    if len(sw) < 4:
        raise ValueError("No SWIRL (or fallback dynamic) phase samples in recording")

    idx = sw.index.values
    r2 = dtau2[idx] - _lowpass(dtau2[idx], fs)   # detrended residuals
    r3 = dtau3[idx] - _lowpass(dtau3[idx], fs)
    v = speed[idx]

    # 1-2. mass offset: pre-swirl peak-Z mean vs INIT tare
    pre = df.iloc[: idx[0]]
    peak = pre[pre["tcp_z_mm"] == pre["tcp_z_mm"].max()]
    mass_j2 = float(peak["tau2_nm"].mean() - tares[2]) if len(peak) else 0.0
    mass_j3 = float(peak["tau3_nm"].mean() - tares[3]) if len(peak) else 0.0

    # 3-5. swirl variance / peak-to-peak
    swirl_var_j2 = float(np.var(r2))
    swirl_var_j3 = float(np.var(r3))
    swirl_ptp_j3 = float(np.ptp(dtau3[idx]))

    # 6. phase lag: xcorr(TCP speed, τ3 residual), lag of max |corr| within ±2 s
    a = v - v.mean()
    b = r3 - r3.mean()
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        phase_lag_s = 0.0
    else:
        corr = np.correlate(a / (np.std(a) + 1e-12), b / (np.std(b) + 1e-12), mode="full")
        lags = (np.arange(len(corr)) - (len(a) - 1)) / fs
        m = np.abs(lags) <= 2.0
        phase_lag_s = float(lags[m][np.argmax(np.abs(corr[m]))]) if m.any() else 0.0

    # 7. rattle energy: observable high band only (honest about Nyquist)
    nyq = fs / 2.0
    lo, hi = 1.5, nyq
    rattle_band_limited = nyq < 5.0
    if len(r3) >= 8 and hi > lo:
        freqs = np.fft.rfftfreq(len(r3), d=1.0 / fs)
        psd = np.abs(np.fft.rfft(r3)) ** 2
        band = (freqs >= lo) & (freqs <= hi)
        rattle_e = float(np.sum(psd[band]))
    else:
        rattle_e = 0.0

    # 8. grip compliance
    grip_comp = float(sw["gripper_pos"].std()) if "gripper_pos" in sw else 0.0

    # 9. lift transient
    lift = df[df["phase"] == "LIFT_TO_TRAVEL"]
    lift_dyn_j3 = float(lift["tau3_nm"].std()) if len(lift) >= 2 else 0.0

    # 10. mean swirl speed
    swirl_speed = float(np.nanmean(v)) if len(v) else 0.0

    z = np.array([
        mass_j2, mass_j3,
        swirl_var_j2, swirl_var_j3, swirl_ptp_j3,
        phase_lag_s, rattle_e,
        grip_comp, lift_dyn_j3, swirl_speed,
    ], dtype=float)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    scales = np.array([FEATURE_SCALES[n] for n in FEATURE_NAMES], dtype=float)
    zs = z / np.where(scales > 0, scales, 1.0)   # balanced, unit-ish features
    norm = np.linalg.norm(zs)
    e = (zs / norm).tolist() if norm > 1e-12 else [0.0] * len(zs)

    diagnostics = {
        "fs_hz": round(float(fs), 2),
        "nyquist_hz": round(float(nyq), 2),
        "phase_source": phase_source,
        "n_swirl_samples": int(len(sw)),
        "tare_flags": tare_flags,
        "rattle_band_hz": [float(lo), round(float(hi), 2)],
        "rattle_band_limited": bool(rattle_band_limited),
        "label": str(df["object_label"].iloc[0]) if "object_label" in df else None,
    }
    return {"z": z.tolist(), "e": e, "diagnostics": diagnostics}


def _calibrate():
    """Recompute FEATURE_SCALES from all real recordings; print the dict."""
    import glob
    rows = []
    for f in sorted(glob.glob("recordings/**/*.csv", recursive=True)):
        d = pd.read_csv(f)
        if not {"phase", "tau3_nm"}.issubset(d.columns):
            continue
        try:
            rows.append(np.abs(extract_features(d)["z"]))
        except Exception as ex:
            print(f"skip {f}: {ex}", file=sys.stderr)
    if not rows:
        sys.exit("no usable recordings under recordings/")
    A = np.array(rows)
    print(f"# calibrated on {len(rows)} recordings")
    print("FEATURE_SCALES = {")
    for n, s in zip(FEATURE_NAMES, np.percentile(A, 90, axis=0)):
        print(f'    "{n}": {s:.4g},')
    print("}")


def main():
    ap = argparse.ArgumentParser(description="Extract 10-D swirl features + normalized embedding.")
    ap.add_argument("csv", nargs="?", help="Telemetry CSV from repeatable_grasp.py --record")
    ap.add_argument("--calibrate", action="store_true",
                    help="recompute FEATURE_SCALES from recordings/ and print them")
    args = ap.parse_args()
    if args.calibrate:
        _calibrate()
        return
    if not args.csv:
        ap.error("pass a CSV path (or --calibrate)")
    if not os.path.isfile(args.csv):
        sys.exit(f"file not found: {args.csv}")
    out = extract_features(pd.read_csv(args.csv))
    out["feature_names"] = FEATURE_NAMES
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
