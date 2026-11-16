
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Loading aircraft engine sensor data...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train_FD001.txt")

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

data = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)

# Select Engine 1
engine = data[data["engine_id"] == 1]

# Plot all 21 sensors
fig, axes = plt.subplots(7, 3, figsize=(18, 22))

for i in range(1, 22):
    ax = axes[(i - 1) // 3, (i - 1) % 3]

    ax.plot(engine["cycle"], engine[f"sensor_{i}"])
    ax.set_title(f"Sensor {i}")
    ax.set_xlabel("Operating Cycles")
    ax.set_ylabel("Measurement")
    ax.grid(True)

plt.suptitle("Aircraft Engine 1 - All 21 Sensor Measurements", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.97])

OUTPUT_PATH = os.path.join(BASE_DIR, "all_sensors_analysis.png")

plt.savefig(OUTPUT_PATH, dpi=250, bbox_inches="tight")
plt.close()

print("All 21 sensor graphs saved:", OUTPUT_PATH)
