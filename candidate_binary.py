from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


print("=" * 60)
print("       HERCULES CANDIDATE DETECTION EXPERIMENT")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"


# ============================================================
# FEATURES
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
# LOAD DATA
# ============================================================

print("\nLoading NASA/Kepler dataset...")

data = pd.read_csv(DATA_PATH)

print(f"Observations: {len(data)}")


X = data[FEATURES]

y = (
    data[TARGET] == "CANDIDATE"
).astype(int)


print("\nBinary target distribution:")

print(
    y.value_counts()
)

# ============================================================
# SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\nTraining:", len(X_train))
print("Testing: ", len(X_test))

# ============================================================
# MODEL
# ============================================================

print("\nTraining candidate detector...")


model = Pipeline(
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


model.fit(
    X_train,
    y_train,
)

# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("       CANDIDATE DETECTOR PERFORMANCE")
print("=" * 60)


print(
    "\nClassification report:\n"
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "NOT CANDIDATE",
            "CANDIDATE",
        ],
        digits=4,
    )
)


print("\nConfusion matrix:\n")

print(
    confusion_matrix(
        y_test,
        predictions,
    )
)


# ============================================================
# ROC AUC
# ============================================================

auc = roc_auc_score(
    y_test,
    probabilities,
)


print(
    f"\nROC-AUC: {auc:.4f}"
)

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("       CANDIDATE EXPERIMENT COMPLETE")
print("=" * 60)

