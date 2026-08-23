"""
HERCULES — data loading & shared metrics.
 
Single source of truth for:
- reading the V4.2 ranking CSV without triggering the PyArrow
  OverflowError that happens when the raw 140-column frame is passed
  straight to st.dataframe() / describe(include="all")
- computing summary counts (candidate/confirmed/false positive,
  high/medium/low) so every page agrees with every other page
- picking the current leading target
 
Nothing here is hard-coded — every number comes from whatever CSV is
currently on disk.
 
NOTE: DISPLAY_COLUMNS below is built from the column names already
referenced in the HERCULES brief (kepid, kepoi_name,
hercules_prediction, scientific_priority_score, priority,
prob_candidate, prob_false_positive, koi_model_snr, koi_period,
koi_prad). If the real CSV uses different names for any of these,
tell me and I'll fix this file — I'm not guessing past what's
already in the brief.
"""

from pathlib import Path
from typing import Optional, List, Dict
 
import pandas as pd
import streamlit as st
 
RANKINGS_PATH = Path("outputs/hercules_v4_2_rankings.csv")
 
DISPLAY_COLUMNS = [
    "kepid",
    "kepoi_name",
    "hercules_prediction",
    "scientific_priority_score",
    "priority",
    "prob_candidate",
    "prob_false_positive",
    "koi_model_snr",
    "koi_period",
    "koi_prad",
]

ID_LIKE_COLUMNS = {"kepid", "kepoi_name", "kepler_name"}
 
 
@st.cache_data
def load_rankings(path: str = str(RANKINGS_PATH)) -> pd.DataFrame:
    """Load the V4.2 ranking CSV. Cached so it isn't re-read on every rerun."""
    return pd.read_csv(path, low_memory=False)
 
 
def safe_for_streamlit(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Return a defensively-typed copy of df that Streamlit/PyArrow can
    render without the 'OverflowError: int too big to convert' crash.
 
    That crash comes from handing PyArrow a frame it can't type
    cleanly — huge raw catalog ints, NaNs mixed into an int column, or
    object columns holding a mix of strings and numbers. The fix is:
    never show the raw frame, show a narrow, explicitly-typed subset.
    """
    cols = [c for c in (columns or DISPLAY_COLUMNS) if c in df.columns]
    out = df[cols].copy()
 
    for col in out.columns:
        if col in ID_LIKE_COLUMNS or out[col].dtype == "object":
            out[col] = out[col].astype(str)
        elif pd.api.types.is_integer_dtype(out[col]) or pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype("float64")
 
    return out
 
 
def get_summary_counts(rankings: pd.DataFrame) -> Dict[str, int]:
    """
    The one place every page should pull its stat-card numbers from.
    Fixes the earlier bug where 'Candidate Signals' on Command Center
    didn't match 'candidate predictions' shown elsewhere.
    """
    pred = rankings["hercules_prediction"].astype(str).str.upper()
    prio = rankings["priority"].astype(str).str.upper()
 
    return {
        "total": len(rankings),
        "candidate": int((pred == "CANDIDATE").sum()),
        "confirmed": int((pred == "CONFIRMED").sum()),
        "false_positive": int((pred == "FALSE POSITIVE").sum()),
        "high": int((prio == "HIGH").sum()),
        "medium": int((prio == "MEDIUM").sum()),
        "low": int((prio == "LOW").sum()),
    }
 
 
def get_leading_target(rankings: pd.DataFrame) -> pd.Series:
    """Current #1 target by scientific_priority_score — not hard-coded."""
    return rankings.sort_values("scientific_priority_score", ascending=False).iloc[0]
 
 