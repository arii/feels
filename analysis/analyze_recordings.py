import csv
import numpy as np

f1 = "/home/ari/feels/recordings/recording_20261029_122337.csv"
f2 = "/home/ari/feels/recordings/recording_20261029_122355.csv"

def load_data(filepath):
    data = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                data[k].append(float(v))
    return {k: np.array(v) for k, v in data.items()}

d1 = load_data(f1)
d2 = load_data(f2)

dur1 = d1["elapsed_sec"][-1] - d1["elapsed_sec"][0]
dur2 = d2["elapsed_sec"][-1] - d2["elapsed_sec"][0]

print("=" * 65)
print(f"SESSION 1 (NOT HOLDING ANYTHING): {len(d1['elapsed_sec'])} samples, {dur1:.1f}s")
print("=" * 65)
for col in ["payload_weight_g", "effective_force_n", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw"]:
    vals = d1[col]
    print(f"  {col:20s}: Mean = {vals.mean():8.3f} | Std = {vals.std():6.3f} | Min = {vals.min():8.3f} | Max = {vals.max():8.3f}")

print("\n" + "=" * 65)
print(f"SESSION 2 (HOLDING SOMETHING):     {len(d2['elapsed_sec'])} samples, {dur2:.1f}s")
print("=" * 65)
for col in ["payload_weight_g", "effective_force_n", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw"]:
    vals = d2[col]
    print(f"  {col:20s}: Mean = {vals.mean():8.3f} | Std = {vals.std():6.3f} | Min = {vals.min():8.3f} | Max = {vals.max():8.3f}")

print("\n" + "=" * 65)
print("COMPARISON & DETECTED DELTAS (Session 2 - Session 1)")
print("=" * 65)
for col in ["payload_weight_g", "effective_force_n", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw"]:
    diff = d2[col].mean() - d1[col].mean()
    p_diff = np.max(np.abs(d2[col] - d1[col].mean()))
    print(f"  Δ {col:18s}: Mean Shift = {diff:+8.3f} | Peak Delta = {p_diff:8.3f}")
