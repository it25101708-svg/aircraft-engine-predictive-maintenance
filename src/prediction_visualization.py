
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Starting Prediction Visualization...", flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "prediction_results.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prediction results
data = pd.read_csv(DATA_PATH)

print("Prediction data loaded:", data.shape)

# Graph 1: Actual vs Predicted RUL
plt.figure(figsize=(8, 7))

plt.scatter(
    data["actual_RUL"],
    data["predicted_RUL"],
    alpha=0.7
)

plt.plot([0, 125], [0, 125], "r--", label="Perfect Prediction")

plt.title("Actual vs Predicted RUL")
plt.xlabel("Actual RUL (Cycles)")
plt.ylabel("Predicted RUL (Cycles)")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Graph 2: First 20 engines comparison
sample = data.head(20)

plt.figure(figsize=(12, 6))

plt.plot(
    sample["engine_id"],
    sample["actual_RUL"],
    marker="o",
    label="Actual RUL"
)

plt.plot(
    sample["engine_id"],
    sample["predicted_RUL"],
    marker="s",
    label="Predicted RUL"
)

plt.title("Actual vs Predicted RUL - First 20 Engines")
plt.xlabel("Engine ID")
plt.ylabel("RUL (Cycles)")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "engine_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nVISUALIZATION COMPLETED!")
print("Graphs saved inside:", OUTPUT_DIR)
