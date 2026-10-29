import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

t3 = d3["elapsed_sec"] - d3["elapsed_sec"][0]
t4 = d4["elapsed_sec"] - d4["elapsed_sec"][0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
fig.patch.set_facecolor("#0d1117")

for ax in (ax1, ax2):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#ffffff")
    ax.grid(True, linestyle="--", alpha=0.3, color="#30363d")

# Subplot 1: Recording 3 (Empty)
ax1.plot(t3, d3["j3_raw"], color="#58a6ff", linewidth=2.5, label="Joint 3 Elbow Torque (Empty Session)")
ax1.axhline(-16.82, color="#8b949e", linestyle=":", label="Unloaded Resting State (-16.82 Nm)")
ax1.set_title("Recording 3 (Empty / Not Holding): Quickly settles to -16.82 Nm resting state (76.5% of time)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Torque (Nm)", fontsize=10)
ax1.set_ylim(-24, -15)
ax1.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

# Subplot 2: Recording 4 (Holding)
ax2.plot(t4, d4["j3_raw"], color="#ff7b72", linewidth=2.5, label="Joint 3 Elbow Torque (Holding Object)")
ax2.axhline(-16.82, color="#8b949e", linestyle=":", label="Unloaded Resting State (-16.82 Nm)")
ax2.axhline(-23.02, color="#ff7b72", linestyle="--", alpha=0.6, label="Peak Loaded Torque (-23.02 Nm)")
ax2.set_title("Recording 4 (Holding Object): Sustained heavy torque (-22 to -23 Nm) for 13s, then released!", fontsize=11, fontweight="bold")
ax2.set_xlabel("Elapsed Time (seconds)", fontsize=10)
ax2.set_ylabel("Torque (Nm)", fontsize=10)
ax2.set_ylim(-24, -15)
ax2.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

plt.tight_layout()
output_img = "/home/ari/feels/recordings_3_vs_4.png"
plt.savefig(output_img, dpi=160)
plt.close()
print("Saved comparison plot to:", output_img)
