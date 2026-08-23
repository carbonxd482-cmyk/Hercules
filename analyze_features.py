from pathlib import Path
import pandas as pd


# --------------------------------------------------
# HERCULES FEATURE ANALYSIS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# FEATURES HERCULES WILL USE
# --------------------------------------------------

FEATURES = [
    # Transit / orbital measurements
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_ror",

    # Planet / orbital properties
    "koi_prad",
    "koi_sma",
    "koi_incl",
    "koi_teq",
    "koi_insol",
    "koi_dor",

    # Signal quality
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_model_snr",
    "koi_num_transits",

    # Host star properties
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
    "koi_kepmag",
]


print("=" * 60)
print("             HERCULES FEATURE ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# Check that every feature exists
# --------------------------------------------------

print("\nChecking features...")

missing_features = [
    feature for feature in FEATURES
    if feature not in data.columns
]

if missing_features:
    print("\nERROR: These features were not found:")
    for feature in missing_features:
        print("-", feature)

else:
    print("\nAll selected features exist.")


# --------------------------------------------------
# Number of features
# --------------------------------------------------

print("\nNumber of selected features:")
print(len(FEATURES))


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\nMissing values in selected features:")

missing = data[FEATURES].isnull().sum()

for feature, count in missing.items():
    percentage = (count / len(data)) * 100
    print(f"{feature:25} {count:5} ({percentage:6.2f}%)")


# --------------------------------------------------
# Basic statistics
# --------------------------------------------------

print("\nBasic statistics:")

print(data[FEATURES].describe().T)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 60)