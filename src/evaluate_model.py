
import os
import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Starting official test-set evaluation...", flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load processed test dataset
test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "test_processed.csv")
)

# Load saved model and feature names
model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

# Load NASA's actual RUL values
actual_rul = pd.read_csv(
    os.path.join(RAW_DIR, "RUL_FD001.txt"),
    sep=r"\s+",
    header=None,
    names=["RUL"]
)["RUL"]

# Select the last observed record for each test engine
last_records = (
    test.sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
)

# Actual RUL for the 100 test engines
y_true = actual_rul.clip(upper=125).reset_index(drop=True)

# Features for predictions
X_test = last_records[feature_columns]

# Predict RUL
predictions = model.predict(X_test)

# Evaluate predictions
mae = mean_absolute_error(y_true, predictions)
rmse = mean_squared_error(y_true, predictions) ** 0.5
r2 = r2_score(y_true, predictions)

print("\nOFFICIAL TEST-SET RESULTS")
print("-------------------------")
print(f"MAE:  {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")
print(f"R2:   {r2:.4f}")

# Create results table
results = pd.DataFrame({
    "engine_id": last_records["engine_id"].values,
    "actual_RUL": y_true.values,
    "predicted_RUL": predictions.round(2)
})

results["prediction_error"] = (
    results["predicted_RUL"] - results["actual_RUL"]
)

# Save results
output_path = os.path.join(
    PROCESSED_DIR,
    "prediction_results.csv"
)

results.to_csv(output_path, index=False)

print("\nFIRST 10 PREDICTIONS")
print(results.head(10).to_string(index=False))

print("\nEvaluation completed!")
print("Results saved:", output_path)
