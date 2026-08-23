from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# HERCULES V2 — MODEL TOURNAMENT

print("=" * 60)
print("             HERCULES V2 MODEL TOURNAMENT")
print("=" * 60)

# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

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

# 3. LOAD DATA


print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH)

X = data[FEATURES]
y = data[TARGET]

print(f"Observations: {len(data)}")
print(f"Features: {len(FEATURES)}")

# 4. SAME TRAIN / TEST SPLIT

print("\nCreating fixed train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print(f"Training: {len(X_train)}")
print(f"Testing:  {len(X_test)}")

# 5. MODELS


models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),
}

# 6. TRAIN + EVALUATE

results = []


for name, classifier in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {name}")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

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

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
        }
    )

    print(
        f"Accuracy:     {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1:     {macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1:  {weighted_f1 * 100:.2f}%"
    )


# 7. RESULTS TABLE

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False,
)


print("\n")
print("=" * 60)
print("              MODEL LEADERBOARD")
print("=" * 60)

print()

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro F1": "{:.4f}".format,
            "Weighted F1": "{:.4f}".format,
        },
    )
)


# 8. WINNER

winner = results_df.iloc[0]

print("\n" + "=" * 60)
print("                    🏆 WINNER")
print("=" * 60)

print(f"\nModel: {winner['Model']}")
print(f"Accuracy: {winner['Accuracy'] * 100:.2f}%")
print(f"Macro F1: {winner['Macro F1'] * 100:.2f}%")


print("\n" + "=" * 60)
print("         HERCULES V2 MODEL TOURNAMENT COMPLETE")
print("=" * 60)

