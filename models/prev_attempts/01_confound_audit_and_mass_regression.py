# ARCHIVED EXPLORATORY SCRIPT — historical record of a dead end (see models/README.md).
# Ran against the pre-cascade flat-3-class train_classifier.py; some helpers it
# imported were removed in the cascade refactor. Kept for the methodology + result,
# not as a maintained tool. Findings are summarized in models/README.md.

#!/usr/bin/env python3
"""1) confound audit  2) inertia + cross-joint features  3) mass regression."""
import os, sys, re, warnings
warnings.filterwarnings("ignore")
REPO = "/Applications/Projects/Hackathons/feels"
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
import feature_extractor as fx
from train_classifier import to_3class
def read_split(name):  # self-contained shim (recordings/split/*.txt)
    with open(os.path.join(REPO,"recordings","split",name)) as fh:
        return [os.path.join(REPO,l.strip()) for l in fh if l.strip()]
DYN_JOINTS = (2, 3, 4, 5, 6)  # self-contained (removed from train_classifier after cascade refactor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor)
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier


def meta_of(p):
    df = pd.read_csv(p)
    base = os.path.basename(p)
    t = re.search(r"_(\d{6})_", base)
    tsec = int(t.group(1)[:2]) * 3600 + int(t.group(1)[2:4]) * 60 + int(t.group(1)[4:]) if t else 0
    return df, {
        "cls": to_3class(str(df["object_label"].iloc[0])),
        "has_swirl": 1.0 if (df["phase"] == "SWIRL").any() else 0.0,
        "n_samples": float(len(df)),
        "duration": float(df["elapsed_sec"].iloc[-1] - df["elapsed_sec"].iloc[0]),
        "tsec": float(tsec), "label": str(df["object_label"].iloc[0]),
    }


def kitchensink(df, inertia=False):
    f = {}
    ph = lambda n, c: (df[df["phase"] == n][c].mean() if len(df[df["phase"] == n]) else np.nan)
    hold = df[df["phase"] == "TRAVEL_STABLE_HOLD"]
    if len(hold) < 2: hold = df[df["phase"].astype(str).str.contains("TRAVEL")]
    sw = df[df["phase"] == "SWIRL"]
    if len(sw) < 4: sw = df[df["phase"].isin(fx.FALLBACK_DYN_PHASES)]
    for j in (2, 3, 4, 5, 6):
        f[f"raw_hold_j{j}"] = hold[f"tau{j}_nm"].mean()
        f[f"gdelta_j{j}"] = ph("GRIPPER_CLOSE", f"tau{j}_nm") - ph("REACH_GRASP", f"tau{j}_nm")
        f[f"swstd_j{j}"] = sw[f"tau{j}_nm"].std() if len(sw) >= 4 else 0.0
    st = df[df["phase"] == "TRAVEL_SETTLING"]["tau3_nm"].values
    f["settle_std"] = np.std(st) if len(st) >= 3 else 0.0
    f["grip_dpos"] = ph("GRIPPER_CLOSE", "gripper_pos") - ph("REACH_GRASP", "gripper_pos")
    if inertia:
        lift = df[df["phase"] == "LIFT_TO_TRAVEL"]
        hold_mean = hold["tau3_nm"].mean()
        for j in (2, 3, 5):
            lj = lift[f"tau{j}_nm"].values
            f[f"lift_ptp_j{j}"] = float(np.ptp(lj)) if len(lj) >= 2 else 0.0
        f["lift_overshoot_j3"] = float(np.max(np.abs(lift["tau3_nm"].values - hold_mean))) if len(lift) else 0.0
        # cross-joint ratios / differences (center-of-mass distribution, pose-invariant-ish)
        f["ratio_j5_j3"] = f["raw_hold_j5"] / f["raw_hold_j3"] if abs(f["raw_hold_j3"]) > 1e-6 else 0.0
        f["diff_j3_j5"] = f["raw_hold_j3"] - f["raw_hold_j5"]
        f["diff_j3_j4"] = f["raw_hold_j3"] - f["raw_hold_j4"]
    return f


files_tr = [p for p in read_split("train.txt") if os.path.isfile(p)]
files_te = [p for p in read_split("test.txt") if os.path.isfile(p)]
meta_tr = [meta_of(p) for p in files_tr]
meta_te = [meta_of(p) for p in files_te]
ytr = np.array([m["cls"] for _, m in meta_tr]); yte = np.array([m["cls"] for _, m in meta_te])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
base = f1_score(yte, DummyClassifier(strategy="most_frequent").fit(np.zeros((len(ytr),1)), ytr).predict(np.zeros((len(yte),1))), average="macro")
print(f"baseline test macro-F1 = {base:.3f}\n")

# ---- 1. CONFOUND AUDIT ----
print("=" * 60); print("1) CONFOUND AUDIT — does PROTOCOL (not contents) predict class?")
PROT = ["has_swirl", "n_samples", "duration", "tsec"]
Xp_tr = np.array([[m[k] for k in PROT] for _, m in meta_tr])
Xp_te = np.array([[m[k] for k in PROT] for _, m in meta_te])
for name, cols in [("has_swirl only", ["has_swirl"]), ("recording-time only", ["tsec"]), ("all protocol", PROT)]:
    idx = [PROT.index(c) for c in cols]
    pipe = make_pipeline(StandardScaler(), RandomForestClassifier(200, max_depth=4, class_weight="balanced", random_state=0))
    cvf1 = cross_val_score(pipe, Xp_tr[:, idx], ytr, cv=cv, scoring="f1_macro").mean()
    pipe.fit(Xp_tr[:, idx], ytr); tef1 = f1_score(yte, pipe.predict(Xp_te[:, idx]), average="macro")
    flag = "  <-- CONFOUND" if tef1 > base + 0.05 else ""
    print(f"  {name:20} CV={cvf1:.3f}  TEST={tef1:.3f}{flag}")
# swirl coverage by class
import collections
cov = collections.Counter((m["cls"], int(m["has_swirl"])) for _, m in meta_tr)
print("  swirl coverage by class (train):",
      {c: f"{cov[(c,1)]}/{cov[(c,1)]+cov[(c,0)]}" for c in sorted(set(ytr))})

# ---- 2. INERTIA + CROSS-JOINT FEATURES ----
print("\n" + "=" * 60); print("2) INERTIA + CROSS-JOINT features — any NEW signal?")
def build(metas, inertia):
    return np.nan_to_num(np.array([list(kitchensink(df, inertia).values()) for df, _ in metas], float))
for tag, inertia in [("kitchen-sink (baseline set)", False), ("+ inertia + cross-joint", True)]:
    Xtr = build(meta_tr, inertia); Xte = build(meta_te, inertia)
    best = 0
    for clf in [GradientBoostingClassifier(random_state=42),
                RandomForestClassifier(300, max_depth=5, class_weight="balanced", random_state=42)]:
        pipe = make_pipeline(StandardScaler(), clf)
        cvf1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro").mean()
        pipe.fit(Xtr, ytr); tef1 = f1_score(yte, pipe.predict(Xte), average="macro")
        best = max(best, tef1)
        print(f"  {tag:28} {type(clf).__name__[:12]:12} CV={cvf1:.3f} TEST={tef1:.3f}")

# ---- 3. MASS REGRESSION (within material, ordinal) ----
print("\n" + "=" * 60); print("3) MASS REGRESSION — is there monotonic signal? (CV R^2)")
def target(label):
    m = re.search(r"(\d+)(g|pct)", label)
    return float(m.group(1)) if m else (0.0 if "empty" in label else np.nan)
allm = meta_tr + meta_te
for mat, key in [("solid (grams)", lambda l: ("solid" in l or "screw" in l)),
                 ("water (pct)", lambda l: "water" in l)]:
    rows = [(kitchensink(df, True), target(m["label"])) for df, m in allm if key(m["label"])]
    rows = [(f, t) for f, t in rows if not np.isnan(t)]
    X = np.nan_to_num(np.array([list(f.values()) for f, _ in rows], float))
    y = np.array([t for _, t in rows], float)
    r2 = cross_val_score(make_pipeline(StandardScaler(), RandomForestRegressor(200, random_state=0)),
                         X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
    print(f"  {mat:16} n={len(y):3d}  targets={sorted(set(y))}  CV R^2={r2:+.3f}"
          + ("  (>0 = real signal)" if r2 > 0.1 else "  (~0 = no monotonic signal)"))
