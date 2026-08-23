from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ============================================================
# HERCULES V4 — CANDIDATE HUNTER
# ============================================================

print("=" * 60)
print("           HERCULES V4 CANDIDATE HUNTER")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "koi_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FEATURES
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

SCIENTIFIC_FEATURES = [
    "feature_stellar_density",
    "feature_period_precision",
    "feature_depth_precision",
    "feature_radius_precision",
    "feature_radius_ratio",
    "feature_duration_period_ratio",
    "feature_depth_mag_signal",
]


FEATURES = (
    BASE_FEATURES
    + SCIENTIFIC_FEATURES
)


TARGET = "koi_disposition"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading NASA/Kepler dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Observations: {len(df)}"
)

# ============================================================
# CREATE SCIENTIFIC FEATURES
# ============================================================

print("\nCreating scientific features...")


scientific_features = pd.DataFrame(
    {
        "feature_stellar_density":
            df["koi_srho"],

        "feature_period_precision":
            df["koi_period"]
            / (
                df["koi_period_err1"].abs()
                + 1e-9
            ),

        "feature_depth_precision":
            df["koi_depth"]
            / (
                df["koi_depth_err1"].abs()
                + 1e-9
            ),

        "feature_radius_precision":
            df["koi_prad"]
            / (
                df["koi_prad_err1"].abs()
                + 1e-9
            ),

        "feature_radius_ratio":
            df["koi_ror"]
            / (
                df["koi_prad"].abs()
                + 1e-9
            ),

        "feature_duration_period_ratio":
            df["koi_duration"]
            / (
                df["koi_period"].abs()
                + 1e-9
            ),

        "feature_depth_mag_signal":
            df["koi_depth"]
            / (
                df["koi_kepmag"].abs()
                + 1e-9
            ),
    },
    index=df.index,
)

df = pd.concat(
    [df, scientific_features],
    axis=1,
)
# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    print("\nERROR: Missing features:")

    for feature in missing_features:
        print(
            f"  - {feature}"
        )

    raise SystemExit(1)


print(
    f"\nUsing {len(FEATURES)} scientific features."
)

# ============================================================
# TRAIN HERCULES V3-C
# ============================================================

print("\nTraining HERCULES V3-C...")


X = df[FEATURES]

y = df[TARGET]


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
      X,
      y,
)


print(
    "Training complete."
)


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print(
    "\nGenerating predictions for all observations..."
)


predictions = model.predict(X)

probabilities = model.predict_proba(X)

classes = model.classes_

# ============================================================
# FIND CLASS INDICES
# ============================================================

candidate_index = np.where(
    classes == "CANDIDATE"
)[0][0]

confirmed_index = np.where(
    classes == "CONFIRMED"
)[0][0]

false_positive_index = np.where(
    classes == "FALSE POSITIVE"
)[0][0]


# ============================================================
# ADD RESULTS
# ============================================================

results = df.copy()


results["hercules_prediction"] = (
    predictions
)


results["prob_candidate"] = (
    probabilities[
        :,
        candidate_index
    ]
)


results["prob_confirmed"] = (
    probabilities[
        :,
        confirmed_index
    ]
)


results["prob_false_positive"] = (
    probabilities[
        :,
        false_positive_index
    ]
)

# ============================================================
# CANDIDATE HUNTER SCORE
# ============================================================

print(
    "\nCalculating Candidate Hunter score..."
)


# Normalize useful scientific signals.

def normalize(series):

    series = series.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    median = series.median()

    series = series.fillna(
        median
    )

    minimum = series.min()

    maximum = series.max()
    # ============================================================
# CANDIDATE HUNTER SCORE
# ============================================================

print(
    "\nCalculating Candidate Hunter score..."
)


# Normalize useful scientific signals.

def normalize(series):

    series = series.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    median = series.median()

    series = series.fillna(
        median
    )

    minimum = series.min()

    maximum = series.max()
    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


signal_score = normalize(
    np.log1p(
        results["koi_model_snr"].clip(
            lower=0
        )
    )
)


transit_score = normalize(
    results["koi_num_transits"]
)
period_precision_score = normalize(
    np.log1p(
        results[
            "feature_period_precision"
        ].clip(
            lower=0
        )
    )
)


depth_precision_score = normalize(
    np.log1p(
        results[
            "feature_depth_precision"
        ].clip(
            lower=0
        )
    )
)

# ============================================================
# SCIENTIFIC PRIORITY SCORE
# ============================================================

results["signal_score"] = (
    signal_score
)


results["transit_score"] = (
    transit_score
)


results["period_precision_score"] = (
    period_precision_score
)


results["depth_precision_score"] = (
    depth_precision_score
)

# ============================================================
# FINAL HERCULES SCORE
# ============================================================

results["hercules_score"] = (
    results["prob_candidate"] * 0.45
    +
    signal_score * 0.20
    +
    transit_score * 0.15
    +
    period_precision_score * 0.10
    +
    depth_precision_score * 0.10
)


results["hercules_score"] = (
    results["hercules_score"] * 100
)

# ============================================================
# PRIORITY LEVEL
# ============================================================

def priority(score):

    if score >= 75:
        return "HIGH"

    if score >= 50:
        return "MEDIUM"

    return "LOW"


results["priority"] = (
    results["hercules_score"]
    .apply(priority)
)

# ============================================================
# TOP CANDIDATES
# ============================================================

top_candidates = (
    results[
        results["hercules_prediction"]
        != "FALSE POSITIVE"
    ]
    .sort_values(
        "hercules_score",
        ascending=False,
    )
    .head(20)
)
# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 60)
print("             TOP HERCULES CANDIDATES")
print("=" * 60)


display_columns = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "hercules_prediction",
    "prob_candidate",
    "prob_confirmed",
    "prob_false_positive",
    "hercules_score",
    "priority",
    "koi_model_snr",
    "koi_period",
    "koi_prad",
]


display = top_candidates[
    [
        column
        for column in display_columns
        if column in top_candidates.columns
    ]
].copy()

display["prob_candidate"] = (
    display["prob_candidate"] * 100
).round(2)


display["prob_confirmed"] = (
    display["prob_confirmed"] * 100
).round(2)


display["prob_false_positive"] = (
    display["prob_false_positive"] * 100
).round(2)


display["hercules_score"] = (
    display["hercules_score"]
).round(2)


print(
    display.to_string(
        index=False
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR
    / "hercules_candidate_rankings.csv"
)


results.sort_values(
    "hercules_score",
    ascending=False,
).to_csv(
    output_path,
    index=False,
)


print("\n")
print("=" * 60)
print("             CANDIDATE HUNTER COMPLETE")
print("=" * 60)

print(
    f"\nSaved full rankings to:"
)

print(
    output_path
)


# ============================================================
# SUMMARY
# ============================================================

high_priority = (
    results[
        results["priority"]
        == "HIGH"
    ]
)


print("\n")
print("HERCULES SUMMARY")
print("-" * 60)

print(
    f"Total observations: "
    f"{len(results)}"
)

print(
    f"High priority objects: "
    f"{len(high_priority)}"
)

print(
    f"Candidate predictions: "
    f"{(predictions == 'CANDIDATE').sum()}"
)

print(
    f"Confirmed predictions: "
    f"{(predictions == 'CONFIRMED').sum()}"
)

print(
    f"False positive predictions: "
    f"{(predictions == 'FALSE POSITIVE').sum()}"
)

print("\n")
print("=" * 60)
print("       HERCULES V4 CANDIDATE HUNTER READY")
print("=" * 60)
