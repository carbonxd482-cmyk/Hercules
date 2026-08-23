from pathlib import Path

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# HERCULES V3 FEATURE EXPERIMENT
# ============================================================

print("=" * 60)
print("          HERCULES V3 FEATURE EXPERIMENT")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"


# ============================================================
# BASE FEATURES — HERCULES V2
# ============================================================

BASE_FEATURES = [
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

df = pd.read_csv(DATA_PATH)

print(f"Observations: {len(df)}")


# ============================================================
# CREATE SCIENTIFIC SIGNAL FEATURES
# ============================================================

print("\nCreating scientific signal features...")


# IMPORTANT:
# We create all engineered columns together using pd.DataFrame
# and pd.concat instead of repeatedly inserting columns into df.
# This avoids pandas DataFrame fragmentation warnings.

scientific_features = pd.DataFrame(
    {
        # ----------------------------------------------------
        # 1. Stellar density
        # ----------------------------------------------------
        "feature_stellar_density":
            df["koi_srho"],

        # ----------------------------------------------------
        # 2. Period measurement precision
        # ----------------------------------------------------
        "feature_period_precision":
            df["koi_period"]
            / (
                df["koi_period_err1"].abs()
                + 1e-9
            ),

        # ----------------------------------------------------
        # 3. Transit depth precision
        # ----------------------------------------------------
        "feature_depth_precision":
            df["koi_depth"]
            / (
                df["koi_depth_err1"].abs()
                + 1e-9
            ),

        # ----------------------------------------------------
        # 4. Planet radius precision
        # ----------------------------------------------------
        "feature_radius_precision":
            df["koi_prad"]
            / (
                df["koi_prad_err1"].abs()
                + 1e-9
            ),

        # ----------------------------------------------------
        # 5. Transit radius / planet radius relationship
        # ----------------------------------------------------
        "feature_radius_ratio":
            df["koi_ror"]
            / (
                df["koi_prad"].abs()
                + 1e-9
            ),

        # ----------------------------------------------------
        # 6. Transit duration / orbital period
        # ----------------------------------------------------
        "feature_duration_period_ratio":
            df["koi_duration"]
            / (
                df["koi_period"].abs()
                + 1e-9
            ),

        # ----------------------------------------------------
        # 7. Transit depth relative to Kepler magnitude
        # ----------------------------------------------------
        "feature_depth_mag_signal":
            df["koi_depth"]
            / (
                df["koi_kepmag"].abs()
                + 1e-9
            ),
    },
    index=df.index,
)


# Add all engineered features at once
df = pd.concat(
    [df, scientific_features],
    axis=1,
)


# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SET_V2 = BASE_FEATURES.copy()


# ------------------------------------------------------------
# V3-A
# V2 + stellar density
# ------------------------------------------------------------

FEATURE_SET_V3A = (
    BASE_FEATURES
    + [
        "feature_stellar_density",
    ]
)


# ------------------------------------------------------------
# V3-B
# V2 + measurement uncertainty information
# ------------------------------------------------------------

FEATURE_SET_V3B = (
    BASE_FEATURES
    + [
        "feature_stellar_density",

        "koi_period_err1",
        "koi_period_err2",

        "koi_depth_err1",
        "koi_depth_err2",

        "koi_ror_err1",
        "koi_ror_err2",

        "koi_prad_err1",
        "koi_prad_err2",

        "koi_dor_err1",
        "koi_dor_err2",
    ]
)


# ------------------------------------------------------------
# V3-C
# V2 + scientifically engineered signals
# ------------------------------------------------------------

FEATURE_SET_V3C = (
    BASE_FEATURES
    + [
        "feature_stellar_density",

        "feature_period_precision",
        "feature_depth_precision",
        "feature_radius_precision",
        "feature_radius_ratio",
        "feature_duration_period_ratio",
        "feature_depth_mag_signal",
    ]
)


# ============================================================
# FEATURE SET COLLECTION
# ============================================================

all_feature_sets = {
    "V2 BASELINE": FEATURE_SET_V2,

    "V3-A STELLAR DENSITY": FEATURE_SET_V3A,

    "V3-B MEASUREMENT UNCERTAINTY": FEATURE_SET_V3B,

    "V3-C SCIENTIFIC SIGNALS": FEATURE_SET_V3C,
}


# ============================================================
# VERIFY FEATURES
# ============================================================

print("\nChecking feature sets...")

for name, features in all_feature_sets.items():

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        print(f"\n❌ {name}")

        for feature in missing:
            print(
                f"   Missing: {feature}"
            )

    else:

        print(
            f"✓ {name}: "
            f"{len(features)} features"
        )


# ============================================================
# FIXED TRAIN / TEST SPLIT
# ============================================================

print("\nCreating fixed train/test split...")


y = df[TARGET]


train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


y_train = y.loc[train_indices]

y_test = y.loc[test_indices]


print(
    f"Training observations: "
    f"{len(train_indices)}"
)

print(
    f"Testing observations:  "
    f"{len(test_indices)}"
)


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model():

    return Pipeline(
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
# EXPERIMENT STORAGE
# ============================================================

results = []

models = {}


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    name,
    features,
):

    print("\n")
    print("-" * 60)
    print(f"TRAINING: {name}")
    print("-" * 60)

    print(
        f"Features: {len(features)}"
    )

    # --------------------------------------------------------
    # Prepare train/test data
    # --------------------------------------------------------

    X_train = df.loc[
        train_indices,
        features,
    ]

    X_test = df.loc[
        test_indices,
        features,
    ]

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("Training...")

    model.fit(
        X_train,
        y_train,
    )

    print("Generating predictions...")

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    candidate_recall = report[
        "CANDIDATE"
    ]["recall"]

    candidate_f1 = report[
        "CANDIDATE"
    ]["f1-score"]

    confirmed_recall = report[
        "CONFIRMED"
    ]["recall"]

    false_positive_recall = report[
        "FALSE POSITIVE"
    ]["recall"]

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        f"\nAccuracy:             "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1:             "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1:          "
        f"{weighted_f1 * 100:.2f}%"
    )

    print(
        f"Candidate Recall:     "
        f"{candidate_recall * 100:.2f}%"
    )

    print(
        f"Candidate F1:         "
        f"{candidate_f1 * 100:.2f}%"
    )

    print(
        f"Confirmed Recall:     "
        f"{confirmed_recall * 100:.2f}%"
    )

    print(
        f"False Positive Recall:"
        f" {false_positive_recall * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append(
        {
            "Model": name,
            "Features": len(features),
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Candidate Recall": candidate_recall,
            "Candidate F1": candidate_f1,
            "Confirmed Recall": confirmed_recall,
            "False Positive Recall":
                false_positive_recall,
        }
    )

    return model


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

for name, features in all_feature_sets.items():

    models[name] = run_experiment(
        name,
        features,
    )


# ============================================================
# CREATE LEADERBOARD
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("              HERCULES V3 LEADERBOARD")
print("=" * 60)


display_df = results_df.copy()


percentage_columns = [
    "Accuracy",
    "Macro F1",
    "Weighted F1",
    "Candidate Recall",
    "Candidate F1",
    "Confirmed Recall",
    "False Positive Recall",
]


for column in percentage_columns:

    display_df[column] = (
        display_df[column] * 100
    ).round(2)


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

# We prioritize Macro F1 because the dataset contains
# three classes with significant class imbalance.
#
# Candidate Recall is used as a secondary criterion.

best = results_df.sort_values(
    by=[
        "Macro F1",
        "Candidate Recall",
    ],
    ascending=False,
).iloc[0]


# ============================================================
# BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("                 🏆 BEST V3 MODEL")
print("=" * 60)

print(
    f"\nModel: "
    f"{best['Model']}"
)

print(
    f"Features: "
    f"{int(best['Features'])}"
)

print(
    f"Accuracy: "
    f"{best['Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{best['Macro F1'] * 100:.2f}%"
)

print(
    f"Candidate Recall: "
    f"{best['Candidate Recall'] * 100:.2f}%"
)

print(
    f"Candidate F1: "
    f"{best['Candidate F1'] * 100:.2f}%"
)

print(
    f"Confirmed Recall: "
    f"{best['Confirmed Recall'] * 100:.2f}%"
)

print(
    f"False Positive Recall: "
    f"{best['False Positive Recall'] * 100:.2f}%"
)


# ============================================================
# V3 INTERPRETATION
# ============================================================

print("\n")
print("=" * 60)
print("              HERCULES V3 INTERPRETATION")
print("=" * 60)

print(
    "\nV3 experiments test whether additional "
    "scientific information improves HERCULES."
)

print(
    "\nV2 provides the baseline."
)

print(
    "V3-A adds stellar density."
)

print(
    "V3-B adds measurement uncertainty."
)

print(
    "V3-C adds engineered scientific signals."
)

print(
    "\nThe winning model is selected using "
    "Macro F1 followed by Candidate Recall."
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("       HERCULES V3 FEATURE EXPERIMENT COMPLETE")
print("=" * 60)
