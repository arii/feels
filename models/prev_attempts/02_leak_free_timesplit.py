# ARCHIVED EXPLORATORY SCRIPT — historical record of a dead end (see models/README.md).
# Ran against the pre-cascade flat-3-class train_classifier.py; some helpers it
# imported were removed in the cascade refactor. Kept for the methodology + result,
# not as a maintained tool. Findings are summarized in models/README.md.

#!/usr/bin/env python3
"""Remove the temporal leak: within-class earliest->train / latest->test."""
import os, sys, re, glob, warnings
warnings.filterwarnings("ignore")
REPO = "/Applications/Projects/Hackathons/feels"; sys.path.insert(0, REPO)
import numpy as np, pandas as pd, feature_extractor as fx
from train_classifier import to_3class
DYN_JOINTS = (2, 3, 4, 5, 6)  # self-contained (removed from train_classifier after cascade refactor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier

def tsec(base):
    t = re.search(r"_(\d{6})_", base); g = t.group(1)
    return int(g[:2])*3600 + int(g[2:4])*60 + int(g[4:])

def kitchensink(df):
    f = {}; ph = lambda n,c:(df[df["phase"]==n][c].mean() if len(df[df["phase"]==n]) else np.nan)
    hold = df[df["phase"]=="TRAVEL_STABLE_HOLD"]
    if len(hold)<2: hold = df[df["phase"].astype(str).str.contains("TRAVEL")]
    sw = df[df["phase"]=="SWIRL"]
    if len(sw)<4: sw = df[df["phase"].isin(fx.FALLBACK_DYN_PHASES)]
    for j in (2,3,4,5,6):
        f[f"raw_hold_j{j}"]=hold[f"tau{j}_nm"].mean()
        f[f"gdelta_j{j}"]=ph("GRIPPER_CLOSE",f"tau{j}_nm")-ph("REACH_GRASP",f"tau{j}_nm")
        f[f"swstd_j{j}"]=sw[f"tau{j}_nm"].std() if len(sw)>=4 else 0.0
    f["settle_std"]=np.std(df[df["phase"]=="TRAVEL_SETTLING"]["tau3_nm"].values) if (df["phase"]=="TRAVEL_SETTLING").sum()>=3 else 0.0
    return f

recs=[]
for p in sorted(glob.glob(f"{REPO}/recordings/recording_*.csv")):
    df=pd.read_csv(p); lbl=str(df["object_label"].iloc[0])
    cls=to_3class(lbl)
    if cls in ("empty","solid","water"):
        recs.append({"t":tsec(os.path.basename(p)),"cls":cls,"f":kitchensink(df)})
recs.sort(key=lambda r:r["t"])

# ---- temporal structure ----
print("=== temporal layout per class (HH:MM min..max, n) ===")
import collections
by=collections.defaultdict(list)
for r in recs: by[r["cls"]].append(r["t"])
for c in sorted(by):
    ts=sorted(by[c]); fmt=lambda s:f"{s//3600:02d}:{(s%3600)//60:02d}"
    print(f"  {c:6} n={len(ts):2d}  {fmt(ts[0])}..{fmt(ts[-1])}")
# are classes interleaved in time? show the class sequence over time
seq="".join({"empty":"E","solid":"S","water":"W"}[r["cls"]] for r in recs)
print("  class order over time:", seq)

# ---- within-class time split: earliest 70% train, latest 30% test ----
tr, te = [], []
for c in by:
    cr=sorted([r for r in recs if r["cls"]==c], key=lambda r:r["t"])
    k=int(round(len(cr)*0.7)); tr+=cr[:k]; te+=cr[k:]
feats=list(recs[0]["f"].keys())
Xtr=np.nan_to_num(np.array([[r["f"][k] for k in feats] for r in tr],float)); ytr=np.array([r["cls"] for r in tr])
Xte=np.nan_to_num(np.array([[r["f"][k] for k in feats] for r in te],float)); yte=np.array([r["cls"] for r in te])
base=f1_score(yte, DummyClassifier(strategy="most_frequent").fit(Xtr,ytr).predict(Xte), average="macro")
print(f"\n=== TIME-HONEST split (within-class earliest->train, latest->test) ===")
print(f"train={len(ytr)} test={len(yte)}  baseline macro-F1={base:.3f}")
for clf in [GradientBoostingClassifier(random_state=42),
            RandomForestClassifier(300,max_depth=5,class_weight="balanced",random_state=42)]:
    pipe=make_pipeline(StandardScaler(),clf).fit(Xtr,ytr); pred=pipe.predict(Xte)
    print(f"  {type(clf).__name__[:16]:16} acc={accuracy_score(yte,pred):.3f}  macro-F1={f1_score(yte,pred,average='macro'):.3f}")
print("  (compare: leaky label-stratified split gave macro-F1 up to 0.33-0.47)")
