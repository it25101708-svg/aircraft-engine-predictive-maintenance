
import os
import joblib
import pandas as pd

print("AIRCRAFT ENGINE MAINTENANCE ALERT SYSTEM")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained model
model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

# Dataset columns
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load NASA test dataset
data = pd.read_csv(
    os.path.join(DATA_DIR, "test_FD001.txt"),
    sep=r"\s+",
    header=None,
    names=columns
)

# Select latest record from each engine
latest_data = (
    data.sort_values("cycle")
    .groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
)

# Prepare model input
X = latest_data[feature_columns]

# Predict RUL
predictions = model.predict(X)

# Create report
report = pd.DataFrame({
    "engine_id": latest_data["engine_id"].values,
    "current_cycle": latest_data["cycle"].values,
    "predicted_RUL": predictions
})

# Keep predictions within model target range
report["predicted_RUL"] = report["predicted_RUL"].clip(0, 125)

# Assign illustrative alert categories
def get_status(rul):
    if rul <= 20:
        return "CRITICAL"
    elif rul <= 50:
        return "WARNING"
    else:
        return "NORMAL"

report["status"] = report["predicted_RUL"].apply(get_status)

# Save report
output_path = os.path.join(
    OUTPUT_DIR,
    "maintenance_alert_report.csv"
)

report.to_csv(output_path, index=False)

# Display summary
print("\nMAINTENANCE ALERT SUMMARY")
print("=" * 50)

print(report["status"].value_counts().to_string())

print("\nCRITICAL ENGINES:")
print(
    report[report["status"] == "CRITICAL"].to_string(index=False)
)

print("\nWARNING ENGINES:")
print(
    report[report["status"] == "WARNING"].to_string(index=False)
)

print("\nReport saved successfully:")
print(output_path)

print("\nMAINTENANCE ALERT SYSTEM COMPLETED!")
