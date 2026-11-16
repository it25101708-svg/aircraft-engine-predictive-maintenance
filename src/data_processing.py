from pathlib import Path
import pandas as pd

# Locate the project and dataset folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# NASA FD001 column names
columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load the training and testing datasets
train_data = pd.read_csv(
    DATA_DIR / "train_FD001.txt",
    sep=r"\s+",
    header=None
)

test_data = pd.read_csv(
    DATA_DIR / "test_FD001.txt",
    sep=r"\s+",
    header=None
)

# Remove any extra empty columns and assign column names
train_data = train_data.iloc[:, :26]
test_data = test_data.iloc[:, :26]

train_data.columns = columns
test_data.columns = columns

# Load the test RUL labels
rul_data = pd.read_csv(
    DATA_DIR / "RUL_FD001.txt",
    sep=r"\s+",
    header=None,
    names=["RUL"]
)

# Display dataset information
print("\n========== NASA FD001 DATASET ==========")

print("\nTraining dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)
print("RUL dataset shape:", rul_data.shape)

print("\n========== FIRST 5 TRAINING ROWS ==========")
print(train_data.head())

print("\n========== DATASET INFORMATION ==========")
train_data.info()

print("\n========== ENGINE INFORMATION ==========")
print("Number of training engines:", train_data["engine_id"].nunique())
print("Number of testing engines:", test_data["engine_id"].nunique())

print("\n========== CYCLE INFORMATION ==========")
print("Maximum training cycle:", train_data["cycle"].max())
print("Minimum training cycle:", train_data["cycle"].min())

print("\nDataset loaded successfully!")