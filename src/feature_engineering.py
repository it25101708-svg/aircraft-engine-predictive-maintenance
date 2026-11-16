
import os
import pandas as pd

print("Starting Feature Engineering...", flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load training data
train = pd.read_csv(
    os.path.join(RAW_DIR, "train_FD001.txt"),
    sep=r"\s+",
    header=None,
    names=columns
)

# Calculate training RUL
train_max = train.groupby("engine_id")["cycle"].max()
train["RUL"] = train["engine_id"].map(train_max) - train["cycle"]

# Cap RUL at 125 cycles
train["RUL"] = train["RUL"].clip(upper=125)

print("Training RUL calculated.", flush=True)

# Load test data
test = pd.read_csv(
    os.path.join(RAW_DIR, "test_FD001.txt"),
    sep=r"\s+",
    header=None,
    names=columns
)

# Load actual RUL values for test engines
actual_rul = pd.read_csv(
    os.path.join(RAW_DIR, "RUL_FD001.txt"),
    sep=r"\s+",
    header=None,
    names=["RUL"]
)["RUL"]

# Calculate RUL for every test record
test_max = test.groupby("engine_id")["cycle"].max()

test["RUL"] = (
    test["engine_id"].map(test_max)
    - test["cycle"]
    + test["engine_id"].map(lambda x: actual_rul.iloc[x - 1])
)

test["RUL"] = test["RUL"].clip(upper=125)

print("Testing RUL calculated.", flush=True)

# Save processed datasets
train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
test_path = os.path.join(PROCESSED_DIR, "test_processed.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

print("\nFEATURE ENGINEERING COMPLETED!", flush=True)

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("\nTraining RUL statistics:")
print(train["RUL"].describe())

print("\nSaved files:")
print(train_path)
print(test_path)
