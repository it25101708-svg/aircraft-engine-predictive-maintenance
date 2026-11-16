
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Starting Random Forest Model Training...", flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "train_processed.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load processed training data
data = pd.read_csv(DATA_PATH)

print("Dataset loaded:", data.shape, flush=True)

# Select features and target
feature_columns = [
    col for col in data.columns
    if col not in ["engine_id", "RUL"]
]

X = data[feature_columns]
y = data["RUL"]
groups = data["engine_id"]

print("Number of features:", len(feature_columns))

# Split by engine to avoid data leakage
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

print("Training records:", len(X_train))
print("Validation records:", len(X_val))

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining model...", flush=True)
model.fit(X_train, y_train)

# Validate model
predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = mean_squared_error(y_val, predictions) ** 0.5
r2 = r2_score(y_val, predictions)

print("\nMODEL VALIDATION RESULTS")
print("------------------------")
print(f"MAE:  {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")
print(f"R2:   {r2:.4f}")

# Train final model using all training data
print("\nTraining final model on all training engines...", flush=True)

model.fit(X, y)

# Save model and feature names
model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

joblib.dump(model, model_path)
joblib.dump(feature_columns, features_path)

print("\nMODEL TRAINING COMPLETED!")
print("Model saved:", model_path)
print("Features saved:", features_path)
