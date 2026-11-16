
import os
import joblib
import pandas as pd

print("AIRCRAFT ENGINE RUL PREDICTION SYSTEM")
print("=" * 45)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Load trained AI model
model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

# Load NASA test dataset
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

data = pd.read_csv(
    os.path.join(RAW_DIR, "test_FD001.txt"),
    sep=r"\s+",
    header=None,
    names=columns
)

print("\nAvailable engines: 1 - 100")

engine_id = int(input("\nEnter Engine ID: "))

if engine_id < 1 or engine_id > 100:
    print("Invalid Engine ID!")
    raise SystemExit

# Get latest available record for selected engine
engine = data[data["engine_id"] == engine_id]
latest = engine.sort_values("cycle").tail(1)

# Prepare input features
X = latest[feature_columns]

# Predict RUL
predicted_rul = float(model.predict(X)[0])

print("\n" + "=" * 45)
print("ENGINE PREDICTION RESULTS")
print("=" * 45)

print("Engine ID:", engine_id)
print("Current Operating Cycle:", int(latest["cycle"].iloc[0]))
print(f"Predicted Remaining Useful Life: {predicted_rul:.2f} cycles")

# Simple illustrative alert categories
if predicted_rul <= 20:
    status = "CRITICAL"
elif predicted_rul <= 50:
    status = "WARNING"
else:
    status = "NORMAL"

print("Predicted Status:", status)

print("=" * 45)
print("Prediction completed successfully!")

