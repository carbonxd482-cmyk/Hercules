from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# HERCULES V4.2
# SCIENTIFIC PRIORITY ENGINE
# ============================================================

print("=" * 60)
print("        HERCULES V4.2 SCIENTIFIC PRIORITY ENGINE")
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


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading NASA/Kepler dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Observations: {len(df)}"
)


# ============================================================
# CREATE ENGINEERED FEATURES
# ============================================================

print(
    "\nCreating scientific features..."
)


engineered = pd.DataFrame(
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
    [df, engineered],
    axis=1,
)


# ============================================================
# TRAIN HERCULES
# ============================================================

print(
    "\nTraining HERCULES V3-C classifier..."
)


X = df[FEATURES]

y = df["koi_disposition"]


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
    "Classifier training complete."
)


# ============================================================
# PREDICTIONS
# ============================================================

print(
    "\nGenerating scientific predictions..."
)


predictions = model.predict(X)

probabilities = model.predict_proba(X)

classes = model.classes_


candidate_index = np.where(
    classes == "CANDIDATE"
)[0][0]

confirmed_index = np.where(
    classes == "CONFIRMED"
)[0][0]

false_positive_index = np.where(
    classes == "FALSE POSITIVE"
)[0][0]


results = df.copy()


results[
    "hercules_prediction"
] = predictions


results[
    "prob_candidate"
] = probabilities[
    :, candidate_index
]


results[
    "prob_confirmed"
] = probabilities[
    :, confirmed_index
]


results[
    "prob_false_positive"
] = probabilities[
    :, false_positive_index
]


# ============================================================
# NORMALIZATION HELPER
# ============================================================

def robust_score(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    median = values.median()

    values = values.fillna(
        median
    )

    # Percentile clipping prevents
    # extreme outliers from dominating
    # the scientific score.

    lower = values.quantile(0.05)

    upper = values.quantile(0.95)

    values = values.clip(
        lower=lower,
        upper=upper,
    )

    if upper == lower:

        return pd.Series(
            0.5,
            index=values.index,
        )

    return (
        (values - lower)
        / (upper - lower)
    ).clip(
        0,
        1,
    )


# ============================================================
# 1. SIGNAL QUALITY
# ============================================================

print(
    "\nCalculating signal quality..."
)


signal_quality = robust_score(
    np.log1p(
        results[
            "koi_model_snr"
        ].clip(lower=0)
    )
)


# ============================================================
# 2. TRANSIT COVERAGE
# ============================================================

transit_coverage = robust_score(
    np.log1p(
        results[
            "koi_num_transits"
        ].clip(lower=0)
    )
)


# ============================================================
# 3. PERIOD MEASUREMENT QUALITY
# ============================================================

period_quality = robust_score(
    np.log1p(
        results[
            "feature_period_precision"
        ].clip(lower=0)
    )
)


# ============================================================
# 4. DEPTH MEASUREMENT QUALITY
# ============================================================

depth_quality = robust_score(
    np.log1p(
        results[
            "feature_depth_precision"
        ].clip(lower=0)
    )
)


# ============================================================
# 5. RADIUS MEASUREMENT QUALITY
# ============================================================

radius_quality = robust_score(
    np.log1p(
        results[
            "feature_radius_precision"
        ].clip(lower=0)
    )
)


# ============================================================
# 6. DATA COMPLETENESS
# ============================================================

important_columns = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_ror",
    "koi_prad",
    "koi_model_snr",
    "koi_num_transits",
    "koi_teq",
    "koi_steff",
    "koi_smass",
]


data_completeness = (
    results[
        important_columns
    ]
    .notna()
    .mean(axis=1)
)


# ============================================================
# STORE COMPONENT SCORES
# ============================================================

results[
    "signal_quality"
] = signal_quality * 100


results[
    "transit_coverage"
] = transit_coverage * 100


results[
    "period_quality"
] = period_quality * 100


results[
    "depth_quality"
] = depth_quality * 100


results[
    "radius_quality"
] = radius_quality * 100


results[
    "data_completeness"
] = data_completeness * 100


# ============================================================
# SCIENTIFIC PRIORITY SCORE
# ============================================================

print(
    "\nCalculating Scientific Priority Score..."
)


# The score is deliberately NOT a probability.
#
# It is a prioritization metric combining:
#
#   40% ML candidate evidence
#   20% signal quality
#   12% transit coverage
#   10% period measurement quality
#   8% depth measurement quality
#   5% radius measurement quality
#   5% data completeness

results[
    "scientific_priority_score"
] = (
    results[
        "prob_candidate"
    ] * 40

    +

    signal_quality * 20

    +

    transit_coverage * 12

    +

    period_quality * 10

    +

    depth_quality * 8

    +

    radius_quality * 5

    +

    data_completeness * 5
)


# ============================================================
# PRIORITY CLASS
# ============================================================

def assign_priority(row):

    probability = (
        row["prob_candidate"]
    )

    score = (
        row[
            "scientific_priority_score"
        ]
    )

    false_positive = (
        row[
            "prob_false_positive"
        ]
    )

    # High priority requires both
    # strong ML evidence and strong
    # scientific evidence.

    if (
        score >= 75
        and probability >= 0.70
        and false_positive < 0.20
    ):
        return "HIGH"

    if (
        score >= 55
        and probability >= 0.50
    ):
        return "MEDIUM"

    return "LOW"


results[
    "priority"
] = results.apply(
    assign_priority,
    axis=1,
)


# ============================================================
# SORT
# ============================================================

ranked = (
    results[
        results[
            "hercules_prediction"
        ] != "FALSE POSITIVE"
    ]
    .sort_values(
        "scientific_priority_score",
        ascending=False,
    )
)


# ============================================================
# TOP TARGETS
# ============================================================

top_targets = ranked.head(20)


print("\n")
print("=" * 60)
print("          HERCULES TOP SCIENTIFIC TARGETS")
print("=" * 60)


display_columns = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "hercules_prediction",
    "prob_candidate",
    "prob_false_positive",
    "scientific_priority_score",
    "priority",
    "signal_quality",
    "transit_coverage",
    "data_completeness",
    "koi_model_snr",
    "koi_period",
    "koi_prad",
]


display = top_targets[
    display_columns
].copy()


display[
    "prob_candidate"
] *= 100


display[
    "prob_false_positive"
] *= 100


numeric_columns = [
    "prob_candidate",
    "prob_false_positive",
    "scientific_priority_score",
    "signal_quality",
    "transit_coverage",
    "data_completeness",
]


display[
    numeric_columns
] = display[
    numeric_columns
].round(2)


print(
    display.to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPLETE DATASET
# ============================================================

output_path = (
    OUTPUT_DIR
    / "hercules_v4_2_rankings.csv"
)


results.sort_values(
    "scientific_priority_score",
    ascending=False,
).to_csv(
    output_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("             HERCULES V4.2 SUMMARY")
print("=" * 60)


high_count = (
    results[
        results["priority"]
        == "HIGH"
    ].shape[0]
)


medium_count = (
    results[
        results["priority"]
        == "MEDIUM"
    ].shape[0]
)


candidate_count = (
    (
        results[
            "hercules_prediction"
        ]
        == "CANDIDATE"
    )
    .sum()
)


print(
    f"\nObjects analyzed:       "
    f"{len(results)}"
)

print(
    f"Candidate predictions:  "
    f"{candidate_count}"
)

print(
    f"High priority:          "
    f"{high_count}"
)

print(
    f"Medium priority:        "
    f"{medium_count}"
)


print(
    "\nRanking saved to:"
)

print(
    output_path
)


# ============================================================
# SCIENTIFIC DISCLAIMER
# ============================================================

print("\n")
print("=" * 60)
print("             SCIENTIFIC NOTE")
print("=" * 60)

print(
    "\nThe Hercules Scientific Priority Score is"
)

print(
    "a machine-learning prioritization metric."
)

print(
    "It is NOT a probability of planetary existence"
)

print(
    "and does NOT replace astronomical validation."
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("      HERCULES V4.2 SCIENTIFIC ENGINE READY")
print("=" * 60)

