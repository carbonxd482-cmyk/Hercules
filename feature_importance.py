from pathlib import Path

import joblib
import pandas as pd

# HERCULES V1 — FEATURE IMPORTANCE

print("=" * 60)
print("          HERCULES V1 FEATURE IMPORTANCE")
print("=" * 60)

# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hercules_v1.pkl"

# 2. LOAD MODEL
# ------------------------------------------------------------

print("\nLoading HERCULES model...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# 3. FEATURE LIST
# ------------------------------------------------------------

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

# 4. GET RANDOM FOREST
# ------------------------------------------------------------

model = pipeline.named_steps["classifier"]


# ------------------------------------------------------------
# 5. EXTRACT IMPORTANCE
# ------------------------------------------------------------

importance = model.feature_importances_


importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": importance,
    }
)
# 6. SORT
# ------------------------------------------------------------

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False,
)

# 7. DISPLAY

print("\nFeature importance ranking:\n")

for index, row in importance_df.iterrows():

    print(
        f"{row['feature']:25} "
        f"{row['importance']:.6f}"
    )

# 8. TOP 10

print("\n" + "=" * 60)
print("TOP 10 FEATURES")
print("=" * 60)

print(
    importance_df.head(10).to_string(
        index=False
    )
)

# 9. FINISHED
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 60)

