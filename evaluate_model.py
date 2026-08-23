from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

# HERCULES V1.1 — FULL MODEL EVALUATION

print("=" * 60)
print("           HERCULES V1.1 EVALUATION")
print("=" * 60)

# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hercules_v1.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. FEATURES

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


# 3. LOAD MODEL + DATA

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")


print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH)

X = data[FEATURES]
y = data[TARGET]

# 4. RECREATE THE SAME TEST SET

print("\nRecreating test set...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Test observations: {len(X_test)}")

# 5. PREDICTIONS

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# 6. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

# 7. CLASSIFICATION REPORT


print("\nClassification report:\n")

report = classification_report(
    y_test,
    y_pred,
    digits=4,
)

print(report)

# 8. CONFUSION MATRIX

labels = model.classes_

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)


print("Confusion matrix:\n")

print(matrix)


# 9. CREATE VISUALIZATION

print("\nCreating confusion matrix visualization...")


fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels,
)

display.plot(
    ax=ax,
    values_format="d",
)

ax.set_title(
    "HERCULES V1.1 — Confusion Matrix"
)

plt.tight_layout()


# 10. SAVE IMAGE

output_path = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


print("\nConfusion matrix saved to:")

print(output_path)

# 11. FINISHED


print("\n" + "=" * 60)
print("       HERCULES EVALUATION COMPLETE")
print("=" * 60)