from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# HERCULES V2
# HISTOGRAM GRADIENT BOOSTING
# ============================================================

print("=" * 60)
print("              HERCULES AI V2")
print("         HISTOGRAM GRADIENT BOOSTING")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = MODEL_DIR / "hercules_v2.pkl"


# ============================================================
# 2. FEATURES
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

TARGET = "koi_disposition"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading NASA/Kepler dataset...")

data = pd.read_csv(DATA_PATH)

print(f"Loaded {len(data)} observations.")


X = data[FEATURES]
y = data[TARGET]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\nCreating fixed train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Testing observations:  {len(X_test)}")


# ============================================================
# 5. BUILD V2 PIPELINE
# ============================================================

print("\nBuilding HERCULES V2 pipeline...")


pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# 6. TRAIN
# ============================================================

print("\nTraining HERCULES V2...")

pipeline.fit(
    X_train,
    y_train,
)

print("Training complete!")


# ============================================================
# 7. EVALUATE
# ============================================================

print("\nEvaluating HERCULES V2...")

predictions = pipeline.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions,
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
)


print("\n" + "=" * 60)
print("HERCULES V2 PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy:     {accuracy * 100:.2f}%"
)

print(
    f"Macro F1:     {macro_f1 * 100:.2f}%"
)

print(
    f"Weighted F1:  {weighted_f1 * 100:.2f}%"
)


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("Confusion matrix:\n")

labels = pipeline.classes_

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

print(matrix)

print("\nLabels:")

print(labels)


# ============================================================
# 10. SAVE MODEL
# ============================================================

print("\nSaving HERCULES V2...")

joblib.dump(
    pipeline,
    MODEL_PATH,
)

print("\nModel saved to:")

print(MODEL_PATH)


# ============================================================
# 11. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("       HERCULES V2 TRAINING COMPLETE")
print("=" * 60)
