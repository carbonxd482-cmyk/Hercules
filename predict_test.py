from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# HERCULES V1.1 — PROBABILITY TEST
# ============================================================

print("=" * 60)
print("          HERCULES V1.1 PROBABILITY ENGINE")
print("=" * 60)

# 1. Paths


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hercules_v1.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

# 2. LOAD MODEL + DATA

print("\nLoading HERCULES model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")


data = pd.read_csv(DATA_PATH)

print(f"Loaded {len(data)} observations.")

# 3. FEATURES

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

# 4. TAKE ONE REAL NASA OBSERVATION

sample = data[FEATURES].iloc[[0]]

# 5. PREDICT

prediction = model.predict(sample)[0]

probabilities = model.predict_proba(sample)[0]

classes = model.classes_

# 6. DISPLAY RESULT

print("\n" + "=" * 60)
print("              HERCULES ANALYSIS")
print("=" * 60)

print(f"\nPrediction: {prediction}")

print("\nProbabilities:")

for class_name, probability in zip(classes, probabilities):

    print(
        f"{class_name:20} "
        f"{probability * 100:6.2f}%"
    )

# 7. MODEL CONFIDENCE

confidence = probabilities.max() * 100

print(f"\nModel confidence: {confidence:.2f}%")


# 8. ACTUAL NASA LABEL

actual = data["koi_disposition"].iloc[0]

print(f"NASA label:        {actual}")

# 9. CORRECT / INCORRECT

if prediction == actual:

    print("\nResult: ✅ Correct prediction")

else:

    print("\nResult: ❌ Incorrect prediction")


print("\n" + "=" * 60)
print("       PROBABILITY ENGINE TEST COMPLETE")
print("=" * 60)

