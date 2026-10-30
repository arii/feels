# ARCHIVED EXPLORATORY SCRIPT — historical record of a dead end (see models/README.md).
# Ran against the pre-cascade flat-3-class train_classifier.py; some helpers it
# imported were removed in the cascade refactor. Kept for the methodology + result,
# not as a maintained tool. Findings are summarized in models/README.md.

#!/usr/bin/env python3
"""Aggressive-denoise mass regression (is there ANY monotonic signal?) + KNN once."""
import os, sys, re, glob, warnings
warnings.filterwarnings("ignore")
REPO = "/Applications/Projects/Hackathons/feels"; sys.path.insert(0, REPO)
import numpy as np, pandas as pd, feature_extractor as fx
from scipy.stats import pearsonr
from train_classifier import to_3class
def read_split(name):  # self-contained shim (recordings/split/*.txt)
    with open(os.path.join(REPO,"recordings","split",name)) as fh:
        return [os.path.join(REPO,l.strip()) for l in fh if l.strip()]
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

def trimmed(x, p=0.1):
    x = np.sort(np.asarray(x, float)); k = int(len(x) * p)
    return np.mean(x[k:len(x) - k]) if len(x) - 2 * k > 0 else np.mean(x)

def grams(label):
    m = re.search(r"(\d+)(g|pct)", label)
    if "empty" in label: return 0.0
    return float(m.group(1)) if m else np.nan

rows = []
for p in sorted(glob.glob(f"{REPO}/recordings/recording_*.csv")):
    df = pd.read_csv(p); lbl = str(df["object_label"].iloc[0])
    hold = df[df["phase"] == "TRAVEL_STABLE_HOLD"]["tau3_nm"].values
    if len(hold) < 3: hold = df[df["phase"].astype(str).str.contains("TRAVEL")]["tau3_nm"].values
    rows.append({"lbl": lbl, "cls": to_3class(lbl), "g": grams(lbl),
                 "j3": trimmed(hold)})   # aggressively-denoised robust hold torque
d = pd.DataFrame(rows)

print("=" * 64)
print("AGGRESSIVE-DENOISE MASS REGRESSION — is there monotonic signal?")
print("=" * 64)
emp = d[d.cls == "empty"]["j3"].mean()
for mat, key in [("SOLID (grams)", d.cls == "solid"), ("WATER (pct)", d.cls == "water")]:
    sub = d[key].dropna(subset=["g"])
    print(f"\n{mat}:  robust J3_hold by level (minus empty ref {emp:.2f}):")
    gp = sub.groupby("g")["j3"].agg(["mean", "std", "count"])
    for g, r in gp.iterrows():
        print(f"   {g:5.0f}:  {r['mean']-emp:+7.3f} N·m  (±{r['std']:.2f}, n={int(r['count'])})")
    r, pval = pearsonr(sub["g"], sub["j3"])
    X = sub[["j3"]].values; y = sub["g"].values
    r2 = cross_val_score(make_pipeline(StandardScaler(), Ridge()), X, y,
                         cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
    sign = "monotonic signal EXISTS" if (abs(r) > 0.4 and r2 > 0.1) else "NO usable monotonic signal"
    print(f"   Pearson r(mass, J3) = {r:+.2f} (p={pval:.2f})   CV R² = {r2:+.3f}   -> {sign}")

print("\n" + "=" * 64)
print("KNN (k=5) 3-class — as requested, once")
print("=" * 64)
import train_classifier as tc
Xtr, ytr, _, _ = tc.build_matrix(read_split("train.txt"), use_tare=False)
Xte, yte, _, _ = tc.build_matrix(read_split("test.txt"), use_tare=False)
base = f1_score(yte, DummyClassifier(strategy="most_frequent").fit(Xtr, ytr).predict(Xte), average="macro")
for k in (5, 7, 9):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cvf1 = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="f1_macro").mean()
    pipe.fit(Xtr, ytr); tef1 = f1_score(yte, pipe.predict(Xte), average="macro")
    print(f"  KNN k={k}:  CV macro-F1={cvf1:.3f}  TEST macro-F1={tef1:.3f}  (leaky split; baseline {base:.3f})")
print("  NOTE: this is the leaky label-stratified split; leak-free it collapses to baseline like the others.")
