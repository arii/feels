import csv
import numpy as np

f3 = "/home/ari/feels/recordings/recording_20261029_123957.csv"
f4 = "/home/ari/feels/recordings/recording_20261029_124044.csv"

def load(fp):
    data = {}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in r.items():
                data.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in data.items()}

d3 = load(f3)
d4 = load(f4)

dur3 = d3["elapsed_sec"][-1] - d3["elapsed_sec"][0]
dur4 = d4["elapsed_sec"][-1] - d4["elapsed_sec"][0]

print("=" * 65)
print("RECORDING 3 (recording_20261029_123957.csv):", len(d3["elapsed_sec"]), "samples,", round(dur3, 1), "s")
print("=" * 65)
for col in ["payload_weight_g", "effective_force_n", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw"]:
    v = d3[col]
    print(f"  {col:18s}: mean={v.mean():8.3f}, std={v.std():6.3f}, min={v.min():8.3f}, max={v.max():8.3f}")

print("\n" + "=" * 65)
print("RECORDING 4 (recording_20261029_124044.csv):", len(d4["elapsed_sec"]), "samples,", round(dur4, 1), "s")
print("=" * 65)
for col in ["payload_weight_g", "effective_force_n", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw"]:
    v = d4[col]
    print(f"  {col:18s}: mean={v.mean():8.3f}, std={v.std():6.3f}, min={v.min():8.3f}, max={v.max():8.3f}")

print("\n" + "=" * 65)
print("DIFFERENCE (Rec 4 - Rec 3)")
print("=" * 65)
for col in ["payload_weight_g", "effective_force_n", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw", "j6_raw"]:
    diff = d4[col].mean() - d3[col].mean()
    p_diff = np.max(np.abs(d4[col] - d3[col].mean()))
    print(f"  Δ {col:18s}: Mean Shift = {diff:+8.3f} | Peak Delta = {p_diff:8.3f}")
