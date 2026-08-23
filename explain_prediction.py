from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance


# ============================================================
# HERCULES V2 — PREDICTION EXPLAINER
# ============================================================

print("=" * 60)
print("          HERCULES V2 PREDICTION EXPLAINER")
print("=" * 60)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hercules_v2.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"


# ------------------------------------------------------------
# 2. FEATURES
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

TARGET = "koi_disposition"


# ------------------------------------------------------------
# 3. LOAD MODEL
# ------------------------------------------------------------

print("\nLoading HERCULES V2...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ------------------------------------------------------------
# 4. LOAD NASA DATA
# ------------------------------------------------------------

data = pd.read_csv(DATA_PATH)

print(
    f"Loaded {len(data)} NASA/Kepler observations."
)


# ------------------------------------------------------------
# 5. SELECT A REAL OBSERVATION
# ------------------------------------------------------------

sample_index = 0

sample = data[FEATURES].iloc[[sample_index]]

actual_label = data[
    TARGET
].iloc[sample_index]


# ------------------------------------------------------------
# 6. PREDICT
# ------------------------------------------------------------

prediction = model.predict(sample)[0]

probabilities = model.predict_proba(sample)[0]

classes = model.classes_


# ------------------------------------------------------------
# 7. DISPLAY PREDICTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              HERCULES ANALYSIS")
print("=" * 60)

print(
    f"\nPrediction: {prediction}"
)

print("\nClass probabilities:")

for class_name, probability in zip(
    classes,
    probabilities,
):

    print(
        f"{class_name:20} "
        f"{probability * 100:6.2f}%"
    )


# ------------------------------------------------------------
# 8. CONFIDENCE
# ------------------------------------------------------------

confidence = probabilities.max() * 100

print(
    f"\nModel confidence: {confidence:.2f}%"
)

print(
    f"NASA label:        {actual_label}"
)


# ============================================================
# 9. PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

print(
    "This may take a little while..."
)


# Use a representative sample of the dataset.
# This keeps the explanation reasonably fast.

explanation_data = data[
    FEATURES + [TARGET]
].dropna(
    subset=[TARGET]
)


# Limit the number of rows used for explanation.

explanation_data = explanation_data.sample(
    n=min(2000, len(explanation_data)),
    random_state=42,
)


X_explain = explanation_data[FEATURES]

y_explain = explanation_data[TARGET]


# Calculate permutation importance.

permutation = permutation_importance(
    model,
    X_explain,
    y_explain,
    n_repeats=5,
    random_state=42,
    scoring="f1_macro",
    n_jobs=-1,
)


# ------------------------------------------------------------
# 10. BUILD IMPORTANCE TABLE
# ------------------------------------------------------------

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": permutation.importances_mean,
    }
)


importance_df = importance_df.sort_values(
    by="importance",
    ascending=False,
)


# ------------------------------------------------------------
# 11. DISPLAY TOP SIGNALS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TOP MODEL SIGNALS")
print("=" * 60)

for _, row in importance_df.head(10).iterrows():

    feature = row["feature"]

    importance = row["importance"]

    print(
        f"{feature:25} "
        f"{importance:.6f}"
    )


# ------------------------------------------------------------
# 12. RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              FINAL RESULT")
print("=" * 60)

if prediction == actual_label:

    print(
        "\n✅ HERCULES prediction matches NASA label."
    )

else:

    print(
        "\n❌ HERCULES prediction differs from NASA label."
    )


# ------------------------------------------------------------
# 13. SCIENTIFIC DISCLAIMER
# ------------------------------------------------------------

print("\nNOTE:")

print(
    "Model confidence represents the machine-learning "
    "model's predicted class probability."
)

print(
    "It is not a replacement for scientific confirmation."
)


# ------------------------------------------------------------
# 14. COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("          EXPLANATION COMPLETE")
print("=" * 60)

