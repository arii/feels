# ARCHIVED EXPLORATORY SCRIPT — historical record of a dead end (see models/README.md).
# Ran against the pre-cascade flat-3-class train_classifier.py; some helpers it
# imported were removed in the cascade refactor. Kept for the methodology + result,
# not as a maintained tool. Findings are summarized in models/README.md.

#!/usr/bin/env python3
"""Data augmentation (jitter + class-balance + mixup) on leaky AND leak-free splits."""
import os, sys, re, glob, warnings
warnings.filterwarnings("ignore")
REPO = "/Applications/Projects/Hackathons/feels"; sys.path.insert(0, REPO)
import numpy as np, pandas as pd
import train_classifier as tc
from collections import Counter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
rng = np.random.default_rng(0)

def tsec(b):
    g = re.search(r"_(\d{6})_", b).group(1); return int(g[:2])*3600+int(g[2:4])*60+int(g[4:])

# all recordings -> features + time
recs = []
for p in sorted(glob.glob(f"{REPO}/recordings/recording_*.csv")):
    df = pd.read_csv(p); cls = tc.to_3class(str(df["object_label"].iloc[0]))
    if cls not in ("empty", "solid", "water"): continue
    f, _ = tc.robust_features(df, use_tare=False)
    recs.append({"t": tsec(os.path.basename(p)), "cls": cls,
                 "x": np.nan_to_num(np.array([f[k] for k in tc.FEATURE_ORDER], float))})

def augment(X, y, copies=8, noise=0.4, mixup=True):
    sd = X.std(0) + 1e-9
    Xa, ya = [X], [y]
    cnt = Counter(y); mx = max(cnt.values())
    for c in cnt:                                   # balance minority classes
        idx = np.where(y == c)[0]; need = mx - len(idx)
        if need > 0:
            pick = rng.choice(idx, need, replace=True)
            Xa.append(X[pick] + rng.standard_normal((need, X.shape[1])) * sd * noise); ya.append(y[pick])
    for _ in range(copies):                         # jitter copies
        Xa.append(X + rng.standard_normal(X.shape) * sd * noise); ya.append(y)
    if mixup:                                        # within-class interpolation
        for c in cnt:
            idx = np.where(y == c)[0]
            if len(idx) >= 2:
                a = rng.choice(idx, len(idx)); b = rng.choice(idx, len(idx))
                lam = rng.uniform(0.3, 0.7, (len(idx), 1))
                Xa.append(lam * X[a] + (1 - lam) * X[b]); ya.append(y[a])
    return np.vstack(Xa), np.concatenate(ya)

def run(Xtr, ytr, Xte, yte, tag):
    base = f1_score(yte, DummyClassifier(strategy="most_frequent").fit(Xtr, ytr).predict(Xte), average="macro")
    gb = lambda: make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))
    p0 = gb().fit(Xtr, ytr); f0 = f1_score(yte, p0.predict(Xte), average="macro")
    Xa, ya = augment(Xtr, ytr); p1 = gb().fit(Xa, ya); f1a = f1_score(yte, p1.predict(Xte), average="macro")
    print(f"{tag:22} baseline={base:.3f}   no-aug={f0:.3f}   +AUG={f1a:.3f}   "
          f"(train {len(ytr)}->{len(ya)} after aug)")

# leaky label-stratified split
Xtr, ytr, _, _ = tc.build_matrix(tc.read_split("train.txt"), use_tare=False)
Xte, yte, _, _ = tc.build_matrix(tc.read_split("test.txt"), use_tare=False)
print("=== LEAKY split (confounded) ===")
run(Xtr, ytr, Xte, yte, "leaky label-strat")

# leak-free: within-class earliest 70% train / latest 30% test
print("\n=== LEAK-FREE split (within-class time holdout) ===")
import collections
by = collections.defaultdict(list)
for r in recs: by[r["cls"]].append(r)
tr, te = [], []
for c in by:
    s = sorted(by[c], key=lambda r: r["t"]); k = int(round(len(s) * 0.7)); tr += s[:k]; te += s[k:]
Xtr2 = np.array([r["x"] for r in tr]); ytr2 = np.array([r["cls"] for r in tr])
Xte2 = np.array([r["x"] for r in te]); yte2 = np.array([r["cls"] for r in te])
run(Xtr2, ytr2, Xte2, yte2, "time-honest")
print("\nTakeaway: aug can only move the LEAKY number (more confound) — leak-free it stays ~baseline,")
print("because augmentation resamples existing points; it cannot add information that was never captured.")
