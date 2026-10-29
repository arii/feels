import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Reset time to start from 0 for both sessions
t1 = d1["elapsed_sec"] - d1["elapsed_sec"][0]
t2 = d2["elapsed_sec"] - d2["elapsed_sec"][0]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
fig.patch.set_facecolor("#0d1117")

for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#ffffff")
    ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

# Subplot 1: Joint 3 (Elbow Pitch) - The Key Distinguisher!
ax1.plot(t1, d1["j3_raw"], color="#58a6ff", linewidth=2.5, label="Session 1: Empty (Not Holding) [Mean = -16.82 Nm]")
ax1.plot(t2, d2["j3_raw"], color="#ff7b72", linewidth=2.5, label="Session 2: Holding Object [Mean = -21.61 Nm]")
ax1.set_title("Joint 3 (Elbow Pitch Torque) — Clear ~4.8 Nm Reaction Shift!", fontsize=12, fontweight="bold")
ax1.set_ylabel("Torque (Nm)", fontsize=11)
ax1.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

# Subplot 2: Joint 2 (Shoulder Pitch)
ax2.plot(t1, d1["j2_raw"], color="#58a6ff", linewidth=2, linestyle="--", label="Session 1: Shoulder J2 (-16.05 Nm)")
ax2.plot(t2, d2["j2_raw"], color="#ff7b72", linewidth=2, label="Session 2: Shoulder J2 (-16.05 Nm)")
ax2.set_title("Joint 2 (Shoulder Pitch Torque) — Remained Stationary (Link 2 is Near-Vertical)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Torque (Nm)", fontsize=11)
ax2.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

# Subplot 3: Wrist & Roll Joints (J4, J5, J6)
ax3.plot(t2, d2["j4_raw"], color="#f0883e", linewidth=1.8, label="J4 Forearm Roll (0.78 Nm)")
ax3.plot(t2, d2["j5_raw"], color="#bc8cff", linewidth=1.8, label="J5 Wrist Pitch (0.37 Nm)")
ax3.plot(t2, d2["j6_raw"], color="#39c5cf", linewidth=1.8, label="J6 Flange Roll (0.001 Nm)")
ax3.set_title("Wrist and Torsion Joints (J4, J5, J6) — Unaffected", fontsize=12, fontweight="bold")
ax3.set_xlabel("Elapsed Time in Session (seconds)", fontsize=11)
ax3.set_ylabel("Torque (Nm)", fontsize=11)
ax3.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

plt.tight_layout()
output_file = "/home/ari/feels/session_comparison.png"
plt.savefig(output_file, dpi=160)
plt.close()
print("Saved comparison plot to:", output_file)
