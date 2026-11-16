
import os
import pandas as pd
import matplotlib

# Use a reliable backend for saving graphs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Starting aircraft engine data analysis...", flush=True)

# Locate project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "train_FD001.txt"
)

# Load NASA C-MAPSS dataset
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

data = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset loaded:", data.shape, flush=True)

# Select Engine 1
engine = data[data["engine_id"] == 1]

# Plot Sensor 2 against operating cycles
plt.figure(figsize=(10, 5))
plt.plot(engine["cycle"], engine["sensor_2"])

plt.title("Aircraft Engine 1 - Sensor 2 Analysis")
plt.xlabel("Operating Cycles")
plt.ylabel("Sensor 2 Measurement")
plt.grid(True)

# Save graph
OUTPUT_PATH = os.path.join(BASE_DIR, "sensor_analysis.png")

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print("Graph saved successfully:", OUTPUT_PATH, flush=True)
