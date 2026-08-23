from pathlib import Path
import pandas as pd


# Find the HERCULES project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate the NASA dataset
DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

# Load the dataset
data = pd.read_csv(DATA_PATH)


print("=" * 60)
print("           HERCULES DATA INSPECTION")
print("=" * 60)

print("\nDataset shape:")
print(data.shape)

print("\nNumber of rows:", len(data))
print("Number of columns:", len(data.columns))

print("\nFirst 5 observations:")
print(data.head())

print("\nColumn names:")
for column in data.columns:
    print(column)

print("\nMissing values:")
print(data.isnull().sum().sort_values(ascending=False).head(20))