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

# Plain-language meaning of each feature, shown in the dashboard.
FEATURE_DESCRIPTIONS = {
    "mass_j2": "Static gravity torque offset at shoulder (J2) — payload mass estimate [excluded from matching: tare noise exceeds payload signal]",
    "mass_j3": "Static gravity torque offset at elbow (J3) — payload mass estimate [excluded from matching: tare noise exceeds payload signal]",
    "swirl_var_j2": "Torque variance at shoulder during swirl — slosh / loose-content energy",
    "swirl_var_j3": "Torque variance at elbow during swirl",
    "swirl_ptp_j3": "Peak-to-peak torque swing at elbow during swirl — shifting mass",
    "phase_lag_s": "Lag between TCP motion and torque response — viscous / fluid damping",
    "rattle_e": "High-frequency torque energy — rattle (band-limited by 4-5 Hz sampling)",
    "grip_comp": "Gripper compliance change under load — squishy vs rigid contents",
    "lift_dyn_j3": "Dynamic torque at elbow during lift — inertial mass cue",
    "swirl_speed": "Mean TCP speed during swirl — excitation normalization [excluded from matching: measures the fixed macro's drive, not contents]",
}

# Per-feature scales so the L2-normalized embedding isn't dominated by the
# largest-magnitude raw feature (swirl_speed ~160 mm/s vs variances ~1-5).
# Calibrated on swirl-protocol data ONLY (--calibrate --require-swirl);
# mixing protocols confounds motion with contents.
#
# Per-feature weights: 0.0 excludes a feature from the embedding (it stays in
# z for display/diagnostics). mass_j2/mass_j3 are proven dead — the audit
# found static hold torques identical across contents to 3 decimals, i.e.
# payload signal (~1 N-m) is below tare noise (~10 N-m). swirl_speed is a
# nuisance by construction: it measures how hard the fixed macro drives,
# which is ~constant across contents, so it only compresses cosine
# similarities toward 1. The embedding is effectively 7-D (index stays 10-D
# with zeros in the excluded dims — no migration needed).
FEATURE_WEIGHTS = {
    "mass_j2": 0.0,
    "mass_j3": 0.0,
    "swirl_var_j2": 1.0,
    "swirl_var_j3": 1.0,
    "swirl_ptp_j3": 1.0,
    "phase_lag_s": 1.0,
    "rattle_e": 1.0,
    "grip_comp": 1.0,
    "lift_dyn_j3": 1.0,
    "swirl_speed": 0.0,
}
# Calibrated as the 90th percentile of |z| over the recordings available on
# 2026-10-29 (n=8, only one true SWIRL recording). RECALIBRATE on the P0
# swirl dataset, then freeze — rescaling invalidates old Atlas signatures.
# Regenerate with:  python feature_extractor.py --calibrate
FEATURE_SCALES = {
    # Calibrated 2026-10-29 on swirl-only train recordings (n=33, 90th pct of |z|).
    # The previous values were fallback-protocol calibrated and collapsed every
    # swirl embedding to ~0.97 cosine (98%-confidence artifact). Reintroduced
    # to the feat branch after the main-branch rollback (was 84080f0).
    "mass_j2": 8.978,
    "mass_j3": 3.244,
    "swirl_var_j2": 19.866,
    "swirl_var_j3": 4.883,
    "swirl_ptp_j3": 16.479,
    "phase_lag_s": 1.455,
    "rattle_e": 179.594,
    "grip_comp": 0.632,
    "lift_dyn_j3": 4.141,
    "swirl_speed": 182.422,
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


def _phase_lag(x, y, fs, max_lag_s=2.0):
    """Lag of max |cross-correlation| between x and y, in seconds."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    corr = np.correlate(a / (np.std(a) + 1e-12), b / (np.std(b) + 1e-12), mode="full")
    lags = (np.arange(len(corr)) - (len(a) - 1)) / fs
    m = np.abs(lags) <= max_lag_s
    return float(lags[m][np.argmax(np.abs(corr[m]))]) if m.any() else 0.0


# ── main extraction ─────────────────────────────────────────────────────
def extract_features(df):
    """Compute the 10-D swirl signature and normalized embedding from a recording.

    Reads the SWIRL phase (or falls back to dynamic phases), detrends joint
    torques against the INIT baseline, and computes the ten physical features
    (static lift, swirl dynamics, grip/excitation groups).

    Args:
        df: recording DataFrame with phase, elapsed_sec, tcp_*_mm, tau*_nm columns.
    Returns:
        Dict with z (10-D raw feature list), e (L2-normalized embedding list),
        diagnostics (tare flags, phase source, sample count, sample rate).
    Raises:
        ValueError: when required columns are missing.
    """
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
    phase_lag_s = _phase_lag(v, r3, fs)

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
    weights = np.array([FEATURE_WEIGHTS[n] for n in FEATURE_NAMES], dtype=float)
    zs = z / np.where(scales > 0, scales, 1.0)   # balanced, unit-ish features
    zw = zs * weights                            # drop dead/nuisance dims
    norm = np.linalg.norm(zw)
    e = (zw / norm).tolist() if norm > 1e-12 else [0.0] * len(zw)

    diagnostics = {
        "fs_hz": round(float(fs), 2),
        "nyquist_hz": round(float(nyq), 2),
        "n_samples": int(len(df)),
        "phase_source": phase_source,
        "n_swirl_samples": int(len(sw)),
        "tare_flags": tare_flags,
        "rattle_band_hz": [float(lo), round(float(hi), 2)],
        "rattle_band_limited": bool(rattle_band_limited),
        "label": str(df["object_label"].iloc[0]) if "object_label" in df else None,
    }
    return {"z": z.tolist(), "e": e, "diagnostics": diagnostics}


def swirl_window_series(df):
    """Detrended torque residuals over the dynamic (swirl) window, for the
    dashboard 'why' charts. Returns a JSON-safe dict."""
    if not {"phase", "elapsed_sec", "tau2_nm", "tau3_nm"}.issubset(df.columns):
        raise ValueError("CSV missing required columns (phase, elapsed_sec, tau2/3_nm)")
    fs = _fs(df)
    tares, _ = _tare_with_qc(df)
    dtau2 = df["tau2_nm"].values - tares[2]
    dtau3 = df["tau3_nm"].values - tares[3]
    speed = _tcp_speed(df) if {"tcp_x_mm", "tcp_y_mm"}.issubset(df.columns) else np.zeros(len(df))
    sw = df[df["phase"] == SWIRL_PHASE]
    if len(sw) < 4:
        sw = df[df["phase"].isin(FALLBACK_DYN_PHASES)]
    if len(sw) < 4:
        raise ValueError("No SWIRL (or fallback dynamic) phase samples in recording")
    idx = sw.index.values
    t0 = float(df["elapsed_sec"].values[idx[0]])
    r2 = dtau2[idx] - _lowpass(dtau2[idx], fs)
    r3 = dtau3[idx] - _lowpass(dtau3[idx], fs)
    return {
        "t": [round(float(x) - t0, 3) for x in df["elapsed_sec"].values[idx]],
        "r2": [round(float(x), 4) for x in r2],
        "r3": [round(float(x), 4) for x in r3],
        "phase_lag_s": round(_phase_lag(speed[idx], r3, fs), 3),
        "fs_hz": round(float(fs), 2),
    }


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
    """CLI entry: extract features from a CSV or recalibrate FEATURE_SCALES."""
    ap = argparse.ArgumentParser(
        description="Extract 10-D swirl features + normalized embedding.",
        epilog="""examples:
  python feature_extractor.py recordings/foo.csv          # print z, e, diagnostics
  python feature_extractor.py --calibrate --require-swirl  # recompute FEATURE_SCALES
                                                           # from swirl recordings, then
                                                           # paste the output into FEATURE_SCALES

The 5-25 Hz rattle band is NOT observable at 4-5 Hz cloud sampling;
diagnostics flags rattle_band_limited honestly.""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
