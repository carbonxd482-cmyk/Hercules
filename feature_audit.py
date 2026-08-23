from pathlib import Path

import pandas as pd


print("=" * 60)
print("          HERCULES V3 FEATURE AUDIT")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading NASA/Kepler dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Observations: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# ============================================================
# CURRENT FEATURES
# ============================================================

FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_ror",
    "koi_prad",
    "koi_sma",
    "koi_incl",
    "koi_teq",
    "koi_insol",
    "koi_dor",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_model_snr",
    "koi_num_transits",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
    "koi_kepmag",
]

# ============================================================
# TARGET / LEAKAGE FIELDS
# ============================================================

TARGET = "koi_disposition"

LEAKAGE_FIELDS = [
    "koi_disposition",
    "koi_pdisposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]


# ============================================================
# FEATURE CATEGORIES
# ============================================================

categories = {

    "TRANSIT / SIGNAL": [
        "koi_period",
        "koi_duration",
        "koi_depth",
        "koi_impact",
        "koi_ror",
        "koi_model_snr",
        "koi_num_transits",
        "koi_max_sngle_ev",
        "koi_max_mult_ev",
    ],

    "ORBITAL": [
        "koi_sma",
        "koi_incl",
        "koi_dor",
    ],

    "PLANETARY": [
        "koi_prad",
        "koi_teq",
        "koi_insol",
    ],

    "STELLAR": [
         "koi_steff",
        "koi_slogg",
        "koi_smet",
        "koi_srad",
        "koi_smass",
        "koi_kepmag",
    ],
}
# ============================================================
# CURRENT FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CURRENT HERCULES FEATURES")
print("=" * 60)


for category, features in categories.items():

    print(f"\n[{category}]")

    for feature in features:

        if feature in df.columns:

            missing = df[feature].isna().sum()

            print(
                f"  ✓ {feature:<25}"
                f" missing={missing}"
            )

        else:

            print(
                f"  ✗ {feature:<25}"
                f" NOT FOUND"
            )

# ============================================================
# POTENTIAL ADDITIONAL NUMERIC FEATURES
# ============================================================

print("\n" + "=" * 60)
print("POTENTIAL ADDITIONAL NUMERIC FEATURES")
print("=" * 60)


numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()


available = [
    column
    for column in numeric_columns
    if column not in FEATURES
    and column not in LEAKAGE_FIELDS
]


for column in available:

    missing_pct = (
        df[column].isna().mean() * 100
    )

    print(
        f"{column:<30}"
        f" missing={missing_pct:6.2f}%"
    )

# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE AUDIT")
print("=" * 60)


for field in LEAKAGE_FIELDS:

    status = (
        "PRESENT"
        if field in df.columns
        else "NOT FOUND"
    )

    print(
        f"{field:<25} {status}"
    )
print("\nCurrent HERCULES features containing leakage fields:")

bad_features = [
    feature
    for feature in FEATURES
    if feature in LEAKAGE_FIELDS
]


if bad_features:

    for feature in bad_features:
        print("  ❌", feature)

else:

    print(
        "  ✅ NONE — current feature set is clean."
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("          FEATURE AUDIT COMPLETE")
print("=" * 60)

