
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Starting Remaining Useful Life Analysis...", flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train_FD001.txt")

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load NASA dataset
data = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset loaded:", data.shape, flush=True)

# Calculate maximum cycle for every engine
max_cycles = data.groupby("engine_id")["cycle"].max()

# Calculate Remaining Useful Life
data["RUL"] = data["engine_id"].map(max_cycles) - data["cycle"]

print("\nRUL calculation completed!", flush=True)

# Display sample records
print("\nFirst 10 records:")
print(data[["engine_id", "cycle", "RUL"]].head(10))

# Display RUL statistics
print("\nRUL statistics:")
print(data["RUL"].describe())

# Analyze Engine 1
engine = data[data["engine_id"] == 1]

# Create RUL graph
plt.figure(figsize=(10, 5))

plt.plot(engine["cycle"], engine["RUL"], linewidth=2)

plt.title("Engine 1 - Remaining Useful Life")
plt.xlabel("Operating Cycles")
plt.ylabel("Remaining Useful Life (Cycles)")
plt.grid(True)

OUTPUT_PATH = os.path.join(BASE_DIR, "rul_analysis.png")

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph saved successfully:", OUTPUT_PATH, flush=True)
