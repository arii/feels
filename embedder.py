#!/usr/bin/env python3
"""
embedder.py — swappable embedding backends for the feels vector search.

The Atlas cosine search + material aggregation + margin guard don't care where
the vector comes from. This module puts one interface in front of the options:

    FEELS_EMBEDDER=hand      10-D hand-engineered swirl features (default; current behavior)
    FEELS_EMBEDDER=pca       10-D PCA on raw 6-channel torque-delta trajectories,
                             fit on swirl-only train recordings. Unsupervised —
                             no hand tuning, no scales to calibrate.
    FEELS_EMBEDDER=chronos   2304-D zero-shot time-series foundation model
                             (amazon/chronos-t5-mini). Live-capable: Atlas picks
                             the vector index by embedding dim. Venue setup:
                               pip install torch chronos-forecasting
                               python -c "import atlas; atlas.ensure_vector_indexes('chronos')"
                               FEELS_EMBEDDER=chronos python teach_csv.py --force --require-swirl recordings/
                             then restart the dashboard with FEELS_EMBEDDER=chronos.
                             (First embed() call downloads the model from HuggingFace.)

Contract: embed(df) -> list[float], L2-normalized. The pca backend reuses the
hand pipeline's preprocessing (INIT-median tare via _tare_with_qc, SWIRL window
else fallback dynamic phases) so the only difference is the representation.

Switching embedders changes the vector SPACE, so the Atlas library must be
rebuilt under the same embedder or queries hit the wrong space:

    FEELS_EMBEDDER=pca python teach_csv.py --force --require-swirl recordings/

The live predict path refuses to query a library built with a different
embedder (explicit 409, not silent garbage). Taught signatures are tagged
with their embedder in properties.

Offline bake-off on the committed train/test split (swirl-only):

    python embedder.py --compare     # hand vs pca, atlas-style ranking
    python embedder.py --fit         # (re)fit the PCA cache from swirl train

2026-10-29 bake-off (swirl-only test, n=9, top-5 mean-cosine per material):
hand 22%, pca 33%, chronos 56%. Absolute numbers are modest for all three —
the 3.5 Hz signal is the bottleneck, not the features.
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

EMBEDDER = os.getenv("FEELS_EMBEDDER", "hand").strip().lower()
if EMBEDDER not in ("hand", "pca", "chronos"):
    raise ValueError(f"FEELS_EMBEDDER={EMBEDDER!r}: choose hand|pca|chronos")

DIM = {"hand": 10, "pca": 10, "chronos": 2304}
CACHE = os.path.join(_HERE, "pca_swirl.npz")
TAU_COLS = [f"tau{i}_nm" for i in range(1, 7)]

_MATERIAL_STEMS = {
    "water": "water", "soap": "soap", "dish_soap": "soap",
    "candy": "candy", "solid": "solid", "screws": "screws",
    "screw": "screws", "empty": "empty", "empty_cup": "empty",
}
_CAPACITY_SUFFIX = re.compile(r"_(\d+(pct|g|ml|oz)|half|full|mixed|empty)$")


def material_of(label: str) -> str:
    stem = _CAPACITY_SUFFIX.sub("", label.strip().lower())
    return _MATERIAL_STEMS.get(stem, stem)


def _l2(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _window_deltas(df):
    """(T, 6) torque deltas over the dynamic window, hand-pipeline preprocessing."""
    from feature_extractor import _tare_with_qc, SWIRL_PHASE, FALLBACK_DYN_PHASES
    tares, _ = _tare_with_qc(df)
    idx = df.index[df["phase"] == SWIRL_PHASE].values
    if len(idx) < 4:
        idx = df.index[df["phase"].isin(FALLBACK_DYN_PHASES)].values
    if len(idx) < 4:
        raise ValueError("no dynamic window")
    return np.stack([df[c].values[idx] - tares[i + 1] for i, c in enumerate(TAU_COLS)], axis=1)


def _resample(mat, T):
    n, ch = mat.shape
    if n == T:
        return mat
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, T)
    return np.stack([np.interp(x_new, x_old, mat[:, c]) for c in range(ch)], axis=1)


def _swirl_train_files():
    manifest = os.path.join(_HERE, "recordings", "split_manifest.csv")
    if not os.path.isfile(manifest):
        raise RuntimeError("recordings/split_manifest.csv not found — can't fit PCA")
    files = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            if r["split"].strip() == "train" and r["motion_path"].strip() == "swirl":
                files.append(os.path.join(_HERE, "recordings", r["file"].strip()))
    if not files:
        raise RuntimeError("no swirl train recordings in manifest")
    return files


def fit_pca(cache_path=CACHE):
    """Fit PCA-10 on swirl-only train recordings; persist components to .npz."""
    import pandas as pd
    from sklearn.decomposition import PCA
    mats = []
    for p in _swirl_train_files():
        try:
            df = pd.read_csv(p)
            mats.append(_window_deltas(df))
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
    T = int(np.median([m.shape[0] for m in mats]))
    X = np.stack([_resample(m, T).ravel() for m in mats])
    pca = PCA(n_components=10, random_state=0).fit(X)
    np.savez(cache_path, components=pca.components_, mean=pca.mean_,
             T=np.array(T), n_train=np.array(len(mats)))
    print(f"fit PCA on {len(mats)} swirl train recordings "
          f"(T={T}, explained var {pca.explained_variance_ratio_.sum():.3f}) -> {cache_path}")
    return cache_path


def _load_pca():
    if not os.path.isfile(CACHE):
        print("PCA cache missing — fitting from swirl train recordings...")
        fit_pca()
    z = np.load(CACHE)
    return z["components"], z["mean"], int(z["T"])


_chronos_pipe = None


def _chronos_embed(deltas):
    global _chronos_pipe
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError:
        raise RuntimeError(
            "FEELS_EMBEDDER=chronos needs: pip install torch chronos-forecasting "
            "(plus a HuggingFace model download on first use)")
    if _chronos_pipe is None:
        print("loading amazon/chronos-t5-mini ...", flush=True)
        _chronos_pipe = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-mini", device_map="cpu", torch_dtype=torch.float32)
    ctx = torch.tensor(deltas.T, dtype=torch.float32)  # (6, T)
    with torch.no_grad():
        emb, _ = _chronos_pipe.embed(ctx)  # (6, T', d)
    pooled = emb.float().mean(dim=1).numpy()  # mean over time -> (6, d)
    return _l2(pooled.ravel())


def embed(df):
    """10-D (hand/pca) or 2304-D (chronos) L2-normalized embedding for one recording."""
    if EMBEDDER == "hand":
        from feature_extractor import extract_features
        return [float(x) for x in extract_features(df)["e"]]
    if EMBEDDER == "pca":
        comp, mean, T = _load_pca()
        d = _window_deltas(df)
        v = (_resample(d, T).ravel()[None, :] - mean) @ comp.T
        return [float(x) for x in _l2(v[0])]
    if EMBEDDER == "chronos":
        return [float(x) for x in _chronos_embed(_window_deltas(df))]
    raise AssertionError("unreachable")


def _compare():
    """Offline bake-off: hand vs pca on the swirl-only test split, atlas-style ranking."""
    import pandas as pd
    manifest = os.path.join(_HERE, "recordings", "split_manifest.csv")
    rows = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            if r["motion_path"].strip() == "swirl":
                rows.append(r)
    recs = {}
    for r in rows:
        try:
            df = pd.read_csv(os.path.join(_HERE, "recordings", r["file"].strip()))
            from feature_extractor import extract_features
            e_hand = _l2(np.array(extract_features(df)["e"]))
            comp, mean, T = _load_pca()
            d = _window_deltas(df)
            e_pca = _l2(((_resample(d, T).ravel()[None, :] - mean) @ comp.T)[0])
            recs[r["file"]] = {"material": material_of(r["label"]),
                               "split": r["split"].strip(),
                               "hand": e_hand, "pca": e_pca}
        except Exception as e:
            print("  skip", r["file"].strip(), str(e)[:80])
    train = [v for v in recs.values() if v["split"] == "train"]
    test = [v for v in recs.values() if v["split"] == "test"]
    for name, key in [("hand", "hand"), ("pca", "pca")]:
        lib = np.stack([v[key] for v in train])
        mats = [v["material"] for v in train]
        correct = 0
        for v in test:
            sims = lib @ v[key]
            order = np.argsort(-sims)[:5]
            scores = {}
            for j in order:
                scores.setdefault(mats[j], []).append(float(sims[j]))
            ranked = sorted(scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
            correct += (ranked[0][0] == v["material"])
        print(f"{name}: {correct}/{len(test)} = {correct / len(test):.3f}  (swirl-only test)")


def main():
    ap = argparse.ArgumentParser(description="Swappable embedding backends for feels vector search.")
    ap.add_argument("--fit", action="store_true", help="(re)fit the PCA cache from swirl train recordings")
    ap.add_argument("--compare", action="store_true", help="offline bake-off: hand vs pca on swirl-only test")
    args = ap.parse_args()
    if args.fit:
        fit_pca()
    elif args.compare:
        _compare()
    else:
        print(f"FEELS_EMBEDDER={EMBEDDER}  dim={DIM[EMBEDDER]}")


if __name__ == "__main__":
    main()
