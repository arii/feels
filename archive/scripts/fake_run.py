"""
fake_run.py — Offline synthetic run generator (no robot, no power needed).

Synthesizes Joint-3 torque time-series with class-appropriate dynamics, then
extracts the SAME 10-dim feature vector as the on-robot service
(control_service._handle_compute_features), so synthetic runs are interchangeable
with real ones for developing/testing the Atlas teach→identify pipeline.

Per-class dynamics baked in:
  empty_cup     — low mass, no rattle, no slosh
  water_half    — medium mass, strong 1.5 Hz slosh (phase lag), low rattle
  dish_soap_half— medium mass, heavily damped slosh, low rattle
  candy_loose   — high 5-25 Hz rattle chatter
  solid_weight  — high mass, rigid, zero rattle / slosh

Usage:
    python fake_run.py gen --label candy_loose           # one run → features
    python fake_run.py gen --label water_half --csv      # + save a recording CSV
    python fake_run.py identify --label water_half       # offline self-classify
    python fake_run.py seed --n 8                         # write synthetic CSVs
    python fake_run.py seed --n 8 --atlas                 # + teach() into Atlas
    python fake_run.py identify --label candy_loose --atlas   # query real Atlas
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np

# Mirror control_service defaults
TARE_J3_NM = -16.821
LEVER_ARM_M = 0.45
G = 9.80665
FS = 100.0          # sampling rate (Hz)
DUR = 6.0           # macro duration (s): hold 1.5 + shake 2.5 + tilt 2.0

LABELS = ["empty_cup", "water_half", "dish_soap_half", "candy_loose", "solid_weight"]

# mass_offset: steady downward torque shift (Nm, negative = load on J3)
# slosh: (freq_hz, amplitude_nm, damping)   rattle: amplitude_nm in 5-25 Hz band
PROFILES = {
    "empty_cup":      dict(mass=-0.25, slosh=(0.0, 0.0,  0.0), rattle=0.00, tilt=0.05, noise=0.04),
    "water_half":     dict(mass=-1.00, slosh=(1.5, 0.55, 0.6), rattle=0.02, tilt=0.35, noise=0.06),
    "dish_soap_half": dict(mass=-1.05, slosh=(1.1, 0.18, 2.2), rattle=0.01, tilt=0.30, noise=0.05),
    "candy_loose":    dict(mass=-0.70, slosh=(0.0, 0.0,  0.0), rattle=0.45, tilt=0.10, noise=0.09),
    "solid_weight":   dict(mass=-1.60, slosh=(0.0, 0.0,  0.0), rattle=0.00, tilt=0.12, noise=0.05),
}


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synth_j3(label: str, seed: int | None = None):
    """Return (timestamps, j3_torque_series) for a synthetic macro run."""
    if label not in PROFILES:
        raise SystemExit(f"[!] unknown label '{label}'. choose from {LABELS}")
    rng = np.random.default_rng(seed)
    p = PROFILES[label]

    n = int(FS * DUR)
    t = np.arange(n) / FS
    j3 = np.full(n, TARE_J3_NM + p["mass"])   # tare + static payload torque

    # Phase windows
    shake = (t >= 1.5) & (t < 4.0)
    tilt = t >= 4.0
    t_shake = t - 1.5

    # Slosh: damped sinusoid during (and after) shake — liquids only
    sf, sa, sd = p["slosh"]
    if sa > 0:
        env = np.exp(-sd * np.clip(t_shake, 0, None))
        j3 += np.where(t >= 1.5, sa * env * np.sin(2 * np.pi * sf * t_shake), 0.0)

    # Rattle: band-limited high-freq chatter 5-25 Hz during shake — loose solids
    if p["rattle"] > 0:
        for f in (8.0, 12.5, 17.0, 21.0):
            ph = rng.uniform(0, 2 * np.pi)
            amp = p["rattle"] * rng.uniform(0.5, 1.0)
            j3 += np.where(shake, amp * np.sin(2 * np.pi * f * t + ph), 0.0)

    # Tilt: slow center-of-mass shift during the tilt window
    if np.any(tilt):
        tt = (t[tilt] - 4.0) / (DUR - 4.0)           # 0..1
        j3[tilt] += -p["tilt"] * np.sin(np.pi * tt)   # bump and return

    # Sensor noise
    j3 += rng.normal(0.0, p["noise"], n)

    ts = time.time() + t
    return ts, j3


# ---------------------------------------------------------------------------
# Feature extraction — identical formulas to control_service._handle_compute_features
# ---------------------------------------------------------------------------

def compute_features(ts: np.ndarray, j3: np.ndarray) -> dict:
    dt = float(np.mean(np.diff(ts)))
    fs = 1.0 / max(0.001, dt)
    delta_tau = j3 - TARE_J3_NM

    mean_torque = float(np.mean(delta_tau))
    peak_torque = float(np.min(delta_tau))
    torque_variance = float(np.var(j3))

    n = len(j3)
    fft_vals = np.abs(np.fft.rfft(j3 - np.mean(j3)))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(freqs) > 1:
        fft_vals, freqs = fft_vals[1:], freqs[1:]     # drop DC
        pk = int(np.argmax(fft_vals))
        dominant_freq_hz = float(freqs[pk])
        dominant_power = float(fft_vals[pk])
        rmask = (freqs >= 5.0) & (freqs <= 25.0)
        smask = (freqs >= 0.5) & (freqs <= 3.0)
        rattle_energy = float(np.sum(fft_vals[rmask])) if np.any(rmask) else 0.0
        slosh_energy = float(np.sum(fft_vals[smask])) if np.any(smask) else 0.0
        spectral_ratio = float(rattle_energy / max(0.001, slosh_energy))
    else:
        dominant_freq_hz = dominant_power = rattle_energy = slosh_energy = spectral_ratio = 0.0

    stability_pct = float(np.mean(np.abs(j3 - TARE_J3_NM) < 1.0) * 100.0)
    est_mass_g = abs(mean_torque) / LEVER_ARM_M / G * 1000.0

    feature_vector = [
        round(mean_torque, 4), round(peak_torque, 4), round(torque_variance, 4),
        round(dominant_freq_hz, 2), round(dominant_power, 4),
        round(rattle_energy, 4), round(slosh_energy, 4), round(spectral_ratio, 4),
        round(stability_pct, 1), round(est_mass_g, 1),
    ]
    metrics = {
        "mean_delta_torque_nm": round(mean_torque, 3),
        "peak_delta_torque_nm": round(peak_torque, 3),
        "torque_variance": round(torque_variance, 4),
        "dominant_freq_hz": round(dominant_freq_hz, 2),
        "rattle_energy_5_25hz": round(rattle_energy, 3),
        "slosh_energy_05_3hz": round(slosh_energy, 3),
        "spectral_ratio": round(spectral_ratio, 3),
        "steady_state_stability_pct": round(stability_pct, 1),
        "estimated_mass_grams": round(est_mass_g, 1),
    }
    return {"status": "success", "samples_analyzed": n, "sample_rate_fs": round(fs, 1),
            "feature_vector": feature_vector, "metrics": metrics}


def make_run(label: str, seed: int | None = None) -> dict:
    ts, j3 = synth_j3(label, seed)
    feats = compute_features(ts, j3)
    feats["raw_j3"] = [round(float(x), 4) for x in j3]
    feats["timestamps"] = ts
    feats["label"] = label
    return feats


def save_csv(label: str, ts: np.ndarray, j3: np.ndarray) -> str:
    os.makedirs("recordings/synthetic", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = f"recordings/synthetic/synth_{label}_{stamp}.csv"
    t0 = ts[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_sec", "j3_raw", "delta_j3_nm", "weight_grams"])
        for i in range(len(j3)):
            dj = j3[i] - TARE_J3_NM
            wt = abs(dj) / LEVER_ARM_M / G * 1000.0
            w.writerow([round(ts[i] - t0, 3), round(float(j3[i]), 4),
                        round(float(dj), 4), round(float(wt), 1)])
    return path


# ---------------------------------------------------------------------------
# Offline self-classifier (no Atlas): cosine on z-scored class centroids
# ---------------------------------------------------------------------------

def offline_identify(mystery_label: str, refs_per_class: int = 8):
    # Build reference centroids per class
    ref_vecs = {lbl: np.array([make_run(lbl, seed=1000 + i)["feature_vector"]
                               for i in range(refs_per_class)]).mean(axis=0)
                for lbl in LABELS}

    # z-score normalize across the reference matrix (features have wildly
    # different scales — raw cosine would be dominated by energy/variance terms)
    M = np.array([ref_vecs[l] for l in LABELS])
    mu, sigma = M.mean(axis=0), M.std(axis=0) + 1e-9

    def norm(v):
        z = (np.array(v) - mu) / sigma
        return z / (np.linalg.norm(z) + 1e-9)

    centroids = {l: norm(ref_vecs[l]) for l in LABELS}

    mystery = make_run(mystery_label, seed=42)
    q = norm(mystery["feature_vector"])
    scores = sorted(((l, float(np.dot(q, centroids[l]))) for l in LABELS),
                    key=lambda x: x[1], reverse=True)

    print(f"\n[offline identify]  true label = {mystery_label}")
    print(f"  metrics: mass={mystery['metrics']['estimated_mass_grams']}g  "
          f"rattle={mystery['metrics']['rattle_energy_5_25hz']}  "
          f"slosh={mystery['metrics']['slosh_energy_05_3hz']}  "
          f"freq={mystery['metrics']['dominant_freq_hz']}Hz")
    print("  ranked (cosine, z-scored):")
    for l, s in scores:
        mark = "  <-- predicted" + ("  ✓" if l == mystery_label else "  ✗") if l == scores[0][0] else ""
        print(f"    {l:16s} {s:+.3f}{mark}")
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_gen(args):
    run = make_run(args.label, seed=args.seed)
    print(f"\n=== synthetic run: {args.label} ===")
    print(f"feature_vector ({len(run['feature_vector'])}-dim): {run['feature_vector']}")
    for k, v in run["metrics"].items():
        print(f"  {k:28s} {v}")
    if args.csv:
        path = save_csv(args.label, run["timestamps"], np.array(run["raw_j3"]))
        print(f"\nsaved {path}")


def cmd_seed(args):
    use_atlas = args.atlas
    if use_atlas:
        import atlas
    total = 0
    for lbl in LABELS:
        for i in range(args.n):
            run = make_run(lbl, seed=i)
            save_csv(lbl, run["timestamps"], np.array(run["raw_j3"]))
            if use_atlas:
                atlas.teach(
                    label=lbl,
                    feature_vector=run["feature_vector"],
                    raw_data=run["raw_j3"],
                    motion_type="3in1_macro_synthetic",
                    properties={"estimated_mass_g": run["metrics"]["estimated_mass_grams"],
                                "synthetic": True},
                    session="sim",
                )
            total += 1
        print(f"  generated {args.n} × {lbl}")
    print(f"\n[✓] {total} synthetic runs written to recordings/synthetic/"
          f"{' + taught to Atlas (session=sim)' if use_atlas else ''}")


def cmd_identify(args):
    if args.atlas:
        import atlas
        run = make_run(args.label, seed=42)
        print(f"\n[atlas identify]  true label = {args.label}")
        result = atlas.identify(run["feature_vector"], top_k=5, session="sim")
        print(f"\n  predicted : {result['label']}  (conf {result['confidence']:.3f})"
              f"  {'✓' if result['label'] == args.label else '✗'}")
    else:
        offline_identify(args.label)


def main():
    p = argparse.ArgumentParser(description="Offline synthetic run generator for feels")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate one run and print its features")
    g.add_argument("--label", required=True, choices=LABELS)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--csv", action="store_true", help="also save a recording CSV")
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("seed", help="generate N runs per class (optionally into Atlas)")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--atlas", action="store_true", help="also teach() into Atlas")
    s.set_defaults(func=cmd_seed)

    i = sub.add_parser("identify", help="generate a mystery run and classify it")
    i.add_argument("--label", required=True, choices=LABELS)
    i.add_argument("--atlas", action="store_true", help="query Atlas instead of offline")
    i.set_defaults(func=cmd_identify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
