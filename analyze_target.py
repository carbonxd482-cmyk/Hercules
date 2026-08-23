from pathlib import Path
import pandas as pd


# Find the HERCULES project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

# Load dataset
data = pd.read_csv(DATA_PATH)


print("=" * 60)
print("              HERCULES TARGET ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# 1. TARGET DISTRIBUTION
# --------------------------------------------------

print("\nTARGET: koi_disposition")

print("\nNumber of objects in each class:")

print(data["koi_disposition"].value_counts())


# --------------------------------------------------
# 2. TARGET PERCENTAGES
# --------------------------------------------------

print("\nClass percentages:")

percentages = data["koi_disposition"].value_counts(normalize=True) * 100

for label, percentage in percentages.items():
    print(f"{label}: {percentage:.2f}%")


# --------------------------------------------------
# 3. MISSING TARGET VALUES
# --------------------------------------------------

print("\nMissing target values:")

print(data["koi_disposition"].isna().sum())


# --------------------------------------------------
# 4. POTENTIAL DATA LEAKAGE
# --------------------------------------------------

print("\nPotential leakage-related columns:")

leakage_columns = [
    "koi_disposition",
    "koi_pdisposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]

for column in leakage_columns:
    if column in data.columns:
        print(f"{column}: present")


print("\n" + "=" * 60)
print("TARGET ANALYSIS COMPLETE")
print("=" * 60)