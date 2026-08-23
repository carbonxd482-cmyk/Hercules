from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# HERCULES V1 — MODEL TRAINING

print("=" * 60)
print("              HERCULES AI V1")
print("              MODEL TRAINING")
print("=" * 60)

# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv" 
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. LOAD DATA

print("\nLoading NASA/Kepler dataset ...")

data = pd.read_csv(DATA_PATH)

print(f"Loaded {len(data)} observations.")

# 3. SELECT FEATURES

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

TARGET = "koi_disposition"

X = data[FEATURES]
y = data[TARGET]

print(f"\nFeatures used: {len(FEATURES)}")
print(f"Target: {TARGET}")

# 4. TRAIN / TEST SPLIT

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Testing observations:  {len(X_test)}")

# 5. BUILD ML PIPELINE

print("\nBuilding HERCULES pipeline...")

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]
)

# 6. TRAIN
# ============================================================

print("\nTraining HERCULES...")

pipeline.fit(X_train, y_train)

print("Training complete!")

# 7. TEST
# ============================================================

print("\nEvaluating HERCULES on unseen test data...")

y_pred = pipeline.predict(X_test)

# 8. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")

# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
    )
)

# 10. CONFUSION MATRIX
# ============================================================

print("\nConfusion matrix:")

labels = pipeline.classes_

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

print("\nLabels:")
print(labels)

print("\nMatrix:")
print(matrix)

# 11. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "hercules_v1.pkl"

joblib.dump(
    pipeline,
    model_path,
)

print("\nModel saved to:")
print(model_path)


# 12. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("          HERCULES V1 TRAINING COMPLETE")
print("=" * 60)
