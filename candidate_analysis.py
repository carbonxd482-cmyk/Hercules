from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# HERCULES V2 — CANDIDATE ERROR ANALYSIS


print("=" * 60)
print("       HERCULES V2 CANDIDATE ERROR ANALYSIS")
print("=" * 60)

# PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hercules_v2.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

# FEATURES


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


# LOAD

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")


print("\nLoading NASA/Kepler dataset...")

data = pd.read_csv(DATA_PATH)

print(f"Observations: {len(data)}")

# RECREATE SAME TEST SET


X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# PREDICT

print("\nGenerating predictions...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

classes = model.classes_

# FIND CANDIDATE OBJECTS


results = X_test.copy()

results["actual"] = y_test.values

results["predicted"] = predictions


for index, class_name in enumerate(classes):

    results[
        f"prob_{class_name.replace(' ', '_').lower()}"
    ] = probabilities[:, index]


# CANDIDATE SUBSET

candidate_results = results[
    results["actual"] == "CANDIDATE"
].copy()


print(
    f"\nActual CANDIDATE objects: "
    f"{len(candidate_results)}"
)


# CANDIDATE PREDICTION DISTRIBUTION

print("\nWhat does HERCULES predict for candidates?")

print(
    candidate_results[
        "predicted"
    ].value_counts()
)

# CANDIDATE PROBABILITY STATISTICS

print("\nCandidate probability statistics:")

print(
    candidate_results[
        "prob_candidate"
    ].describe()
)

# MISCLASSIFIED CANDIDATES

missed = candidate_results[
    candidate_results["predicted"] != "CANDIDATE"
].copy()


print(
    f"\nMissed candidates: {len(missed)}"
)

print(
    f"Candidate recall: "
    f"{(1 - len(missed) / len(candidate_results)) * 100:.2f}%"
)

# MOST CONFIDENTLY MISSED CANDIDATES

print(
    "\nMost confidently misclassified candidates:"
)

columns_to_show = [
    "actual",
    "predicted",
    "prob_candidate",
    "prob_confirmed",
    "prob_false_positive",
    "koi_model_snr",
    "koi_period",
    "koi_prad",
    "koi_depth",
    "koi_ror",
]


print(
    missed[
        columns_to_show
    ]
    .sort_values(
        "prob_candidate",
        ascending=True,
    )
    .head(10)
    .to_string(
        index=False
    )
)

# COMPLETE

print("\n" + "=" * 60)
print("     CANDIDATE ERROR ANALYSIS COMPLETE")
print("=" * 60)

