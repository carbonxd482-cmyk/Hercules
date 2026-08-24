from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
 
# CONSTANTS

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "outputs" / "hercules_v4_2_rankings.csv"
 
CANDIDATE_LABEL = "CANDIDATE"
 
REQUIRED_COLUMNS = [
    "kepoi_name",
    "kepid",
    "hercules_prediction",
    "prob_candidate",
    "scientific_priority_score",
    "priority",
]

OPTIONAL_PROBABILITY_COLUMNS = ["prob_confirmed", "prob_false_positive"]
OPTIONAL_COMPONENT_COLUMNS = ["signal_quality", "transit_coverage", "data_completeness"]
 
SIGNAL_GROUPS = {
    "TRANSIT / SIGNAL": [
        "koi_model_snr", "koi_period", "koi_duration", "koi_depth",
        "koi_impact", "koi_ror", "koi_num_transits",
    ],
    "ORBITAL": ["koi_sma", "koi_incl", "koi_dor"],
    "PLANETARY": ["koi_prad", "koi_teq", "koi_insol"],
    "STELLAR": [
        "koi_steff", "koi_slogg", "koi_smet", "koi_srad", "koi_smass", "koi_kepmag",
    ],
}

SIGNAL_COLUMNS = [c for cols in SIGNAL_GROUPS.values() for c in cols]
 
COLORS = {
    "bg": "#05070c",
    "panel": "#0b1018",
    "panel2": "#0d141f",
    "line": "#1c2534",
    "text": "#e8edf5",
    "muted": "#7c8aa0",
    "accent": "#35e0ff",
    "amber": "#f3b84b",
    "danger": "#ff5c5c",
}

NAV_ITEMS = [
    ("01 / COMMAND CENTER", "Command Center"),
    ("02 / TARGET EXPLORER", "Target Explorer"),
    ("03 / CANDIDATE HUNTER", "Candidate Hunter"),
    ("04 / SCIENTIFIC ANALYTICS", "Scientific Analytics"),
    ("05 / ABOUT HERCULES", "About HERCULES"),
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
 
:root {
    --bg: #05070c;
    --panel: #0b1018;
    --panel-2: #0d141f;
    --line: #1c2534;
    --text: #e8edf5;
    --muted: #7c8aa0;
    --accent: #35e0ff;
    --amber: #f3b84b;
    --danger: #ff5c5c;
}

#MainMenu, footer, header { visibility: hidden; }
 
html, body, [class*="css"] { font-family: "Space Grotesk", sans-serif; }
 
.stApp {
    background: radial-gradient(circle at 85% 0%, rgba(53,224,255,.06), transparent 35%), var(--bg);
    color: var(--text);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .08;
    background-image:
        linear-gradient(rgba(255,255,255,.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.02) 1px, transparent 1px);
    background-size: 42px 42px;
}

[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.block-container { max-width: 1400px; padding-top: 2rem; }
 
.hud-k {
    font-family: "IBM Plex Mono", monospace;
    font-size: .72rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent);
}

.section-header { font-size: 1.8rem; font-weight: 700; margin: .4rem 0 1rem 0; color: var(--text); }
.sub { color: var(--muted); max-width: 720px; line-height: 1.6; }
 
.zone-header {
    font: 700 1.1rem "Space Grotesk";
    color: var(--text);
    padding-bottom: .5rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1rem;
}

/* ---------- hero ---------- */
.hero {
    padding: 2rem 0 1.6rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.8rem;
}

.hero h1 { font-size: clamp(2.1rem, 4.5vw, 3.8rem); line-height: 1.02; margin: .7rem 0; color: var(--text); }
 
.online {
    display: inline-flex;
    gap: .5rem;
    align-items: center;
    padding: .35rem .7rem;
    border: 1px solid rgba(53,224,255,.28);
    border-radius: 99px;
    color: var(--accent);
    font: 600 .68rem "IBM Plex Mono";
    letter-spacing: .12em;
    box-shadow: 0 0 14px rgba(53,224,255,.12);
}

.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: dotPulse 2s ease-in-out infinite; }
@keyframes dotPulse { 0%,100% { opacity: 1; } 50% { opacity: .45; } }
.grad-text { background: linear-gradient(90deg, var(--accent), #9df6ff); -webkit-background-clip: text; background-clip: text; color: transparent; }

/* ---------- plain cards ---------- */
.card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.05rem;
    margin-bottom: .6rem;
    transition: border-color 150ms ease, transform 150ms ease, box-shadow 150ms ease;
}

.card:hover { border-color: rgba(53,224,255,.35); transform: translateY(-1px); box-shadow: 0 8px 20px rgba(0,0,0,.3); }
.label {
    font: 600 .65rem "IBM Plex Mono"; letter-spacing: .1em; color: var(--muted); text-transform: uppercase;
}
.metric-value { font-size: 1.6rem; font-weight: 700; margin-top: .3rem; color: var(--text); }

/* ---------- telemetry strip (top-level stat rows) ---------- */
.telemetry-row { display: flex; align-items: stretch; margin: 1.3rem 0; flex-wrap: wrap; }
.telemetry-item { padding: 0 1.5rem; }
.telemetry-item:first-child { padding-left: 0; }
.telemetry-divider { width: 1px; background: var(--line); margin: .1rem 0; }
.telemetry-value { font: 700 1.9rem "Space Grotesk"; color: var(--text); line-height: 1; }
.telemetry-label {
    font: 600 .64rem "IBM Plex Mono"; letter-spacing: .1em; color: var(--muted);
    margin-top: .5rem; text-transform: uppercase;
}

/* ---------- vertical metric stack ---------- */
.metric-stack { display: flex; flex-direction: column; gap: .8rem; }
.metric-stack__value { font: 700 1.5rem "Space Grotesk"; color: var(--text); line-height: 1; }
.metric-stack__label {
    font: 600 .64rem "IBM Plex Mono"; letter-spacing: .1em; color: var(--muted);
    margin-top: .3rem; text-transform: uppercase;
}

/* ---------- telemetry list (label/value rules) ---------- */
.telemetry-list { border-top: 1px solid var(--line); }
.telemetry-row-line {
    display: flex; justify-content: space-between; align-items: center;
    padding: .5rem .1rem; border-bottom: 1px solid var(--line);
}

.tl-label { font: 600 .72rem "IBM Plex Mono"; letter-spacing: .04em; color: var(--muted); }
.tl-value { font: 600 .92rem "IBM Plex Mono"; color: var(--text); font-variant-numeric: tabular-nums; }
 
.dossier-group {
    font: 600 .66rem "IBM Plex Mono"; color: var(--muted); letter-spacing: .1em;
    text-transform: uppercase; margin: 1.2rem 0 .3rem 0;
}

/* ---------- inline readout ---------- */
.readout-inline { display: flex; align-items: baseline; gap: .8rem; margin: .3rem 0 1.1rem 0; }
.readout-inline__value { font: 700 2.1rem "Space Grotesk"; color: var(--text); }
.readout-inline__label { font: 600 .68rem "IBM Plex Mono"; letter-spacing: .1em; color: var(--muted); text-transform: uppercase; }

/* ---------- pipeline: plain cards, no glow/animation ---------- */
.pipeline-card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 1rem; flex: 1; min-width: 150px;
}
.pipeline-card--active { border-color: var(--accent); box-shadow: 0 0 20px rgba(53,224,255,.14); }
.pipeline-card__eyebrow { font: 600 .64rem "IBM Plex Mono"; letter-spacing: .1em; color: var(--muted); margin-bottom: .35rem; }
.pipeline-card--active .pipeline-card__eyebrow { color: var(--accent); }
.pipeline-card__value { font: 700 1.6rem "Space Grotesk"; color: var(--text); }
.pipeline-card--active .pipeline-card__value { color: var(--accent); }
.pipeline-card__label { font-size: .76rem; color: var(--muted); margin-top: .3rem; line-height: 1.4; }

/* ---------- score readout: plain card, no brackets/glow ---------- */
.score-card {
    position: relative;
    background: radial-gradient(circle at 25% 15%, rgba(53,224,255,.18), transparent 60%), var(--panel);
    border: 1px solid var(--accent); border-radius: 12px;
    padding: 1.5rem 1.9rem; display: inline-block;
    box-shadow: 0 0 34px rgba(53,224,255,.13);
}

.score-card__value {
    font: 700 3.6rem "IBM Plex Mono";
    background: linear-gradient(135deg, #b6f8ff, var(--accent));
    -webkit-background-clip: text; background-clip: text; color: transparent;
    line-height: 1;
    filter: drop-shadow(0 0 18px rgba(53,224,255,.4));
}

.score-card__label { font: 600 .7rem "IBM Plex Mono"; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; margin-top: .7rem; }
 
.mission-target__id { font: 700 1.5rem "Space Grotesk"; color: var(--text); }
.mission-target__kepid { font: 600 .78rem "IBM Plex Mono"; color: var(--muted); }

.why-headline { font: 700 1rem "Space Grotesk"; color: var(--accent); margin-top: .3rem; }

/* ---------- badges: plain pills, no glow ---------- */
.badge {
    display: inline-block; padding: .16rem .6rem; border-radius: 999px;
    font: 600 .66rem "IBM Plex Mono"; letter-spacing: .06em; text-transform: uppercase; margin-right: .4rem;
}
.badge--high, .badge--candidate { background: rgba(53,224,255,.15); color: var(--accent); border: 1px solid var(--accent); box-shadow: 0 0 10px rgba(53,224,255,.25); }
.badge--medium { background: rgba(243,184,75,.15); color: var(--amber); border: 1px solid var(--amber); box-shadow: 0 0 10px rgba(243,184,75,.22); }

.badge--low { background: rgba(124,138,160,.13); color: var(--muted); border: 1px solid var(--line); }
.badge--confirmed { background: rgba(232,237,245,.07); color: var(--text); border: 1px solid var(--line); }
.badge--falsepositive { background: rgba(255,92,92,.15); color: var(--danger); border: 1px solid var(--danger); box-shadow: 0 0 10px rgba(255,92,92,.22); }

/* ---------- priority component bars: static gradient, no shimmer ---------- */
.herc-bar-track { background: var(--line); border-radius: 6px; height: 6px; margin-top: .35rem; overflow: hidden; }
.herc-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, rgba(53,224,255,.35), var(--accent)); box-shadow: 0 0 8px rgba(53,224,255,.4); }
 
.safeguard {
    border-left: 2px solid var(--amber); padding: .7rem 1rem; background: rgba(243,184,75,.04);
    color: var(--muted); font-size: .8rem; border-radius: 0 8px 8px 0; margin-top: 1.6rem; line-height: 1.5;
}

/* ---------- native widgets ---------- */
div[data-baseweb="select"] > div {
    background-color: var(--panel-2) !important; border-color: var(--line) !important; color: var(--text) !important;
}
div[data-baseweb="tag"] {
    background-color: rgba(53,224,255,.13) !important; border: 1px solid var(--accent) !important; color: var(--accent) !important;
}
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
    background-color: var(--panel-2) !important; border-color: var(--line) !important; color: var(--text) !important;
}

div[data-testid="stTextInput"] input:focus { border-color: var(--accent) !important; }
.stButton > button {
    background: transparent; border: 1px solid var(--line); color: var(--text);
    font-family: "IBM Plex Mono"; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
    border-radius: 8px; transition: border-color 150ms ease, color 150ms ease;
}

.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), #1fb8d4);
    border: 1px solid var(--accent); color: #04141a; font-weight: 600;
    font-family: "IBM Plex Mono"; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
    border-radius: 8px; box-shadow: 0 0 16px rgba(53,224,255,.28);
    transition: box-shadow 150ms ease, transform 150ms ease;
}

.stDownloadButton > button:hover { box-shadow: 0 0 26px rgba(53,224,255,.42); transform: translateY(-1px); }
.stDownloadButton > button:disabled { background: var(--panel-2); color: var(--muted); box-shadow: none; border-color: var(--line); }

/* ---------- sidebar navigation rail ---------- */
[data-testid="stSidebar"] [role="radiogroup"] > label {
    display: block;
    padding: .55rem .25rem .55rem 1rem;
    border-left: 2px solid transparent;
    font: 600 .76rem "IBM Plex Mono";
    letter-spacing: .03em;
    color: var(--muted);
    transition: color 150ms ease, border-color 150ms ease;
}

[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] > label svg,
[data-testid="stSidebar"] [role="radiogroup"] > label [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover { color: var(--text); }
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    color: var(--accent) !important;
    border-left-color: var(--accent);
}

/* ---------- candidate table ---------- */
.herc-table-wrap {
    max-height: 520px; overflow: auto;
    border: 1px solid var(--line); border-radius: 8px;
}

table.herc-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
table.herc-table thead th {
    position: sticky; top: 0; background: var(--panel-2);
    color: var(--muted); font: 600 .64rem "IBM Plex Mono"; letter-spacing: .06em; text-transform: uppercase;
    text-align: left; padding: .6rem .75rem; border-bottom: 1px solid var(--line); z-index: 1;
}

table.herc-table td { padding: .5rem .75rem; border-bottom: 1px solid var(--line); color: var(--text); }
table.herc-table tbody tr:hover { background: rgba(53,224,255,.04); }
table.herc-table td.mono { font-family: "IBM Plex Mono"; color: var(--accent); }
table.herc-table td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono"; }

@media (max-width: 1000px) {
    .telemetry-row { flex-direction: column; }
    .telemetry-divider { display: none; }
    .telemetry-item { padding: .6rem 0; border-bottom: 1px solid var(--line); }
}
</style>
"""
 
 
def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

# DATA LAYER (unchanged)

@st.cache_data(show_spinner="Loading mission data...")
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
 
 
def _validate_schema(raw: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        st.error(
            "The ranking CSV is missing required columns: "
            + ", ".join(missing)
            + ". Expected the confirmed V4.2 output schema (outputs/hercules_v4_2_rankings.csv)."
        )
        st.stop()
 
 
def _to_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    finite = s.dropna()
    if len(finite) and finite.max() <= 1.0:
        s = s * 100.0
    return s.clip(lower=0, upper=100)
 
 
@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
 
    out["target"] = raw["kepoi_name"].astype(str)
    out["kepid"] = raw["kepid"].astype(str)
 
    out["classification"] = raw["hercules_prediction"].astype(str).str.upper().str.strip()
    out["is_candidate"] = out["classification"] == CANDIDATE_LABEL
 
    out["score"] = pd.to_numeric(raw["scientific_priority_score"], errors="coerce").clip(0, 100)
    out["probability"] = _to_0_100(raw["prob_candidate"])
 
    for col in OPTIONAL_PROBABILITY_COLUMNS:
        if col in raw.columns:
            out[col] = _to_0_100(raw[col])
 
    out["priority"] = raw["priority"].astype(str).str.upper().str.strip()
 
    for col in OPTIONAL_COMPONENT_COLUMNS:
        if col in raw.columns:
            out[col] = _to_0_100(raw[col])
 
    for col in SIGNAL_COLUMNS:
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce")
 
    return out.reset_index(drop=True)
 
 
def get_summary_counts(d: pd.DataFrame) -> dict:
    return {
        "total": len(d),
        "candidate": int(d["is_candidate"].sum()),
        "ranked": int(d["score"].notna().sum()),
        "high": int((d["priority"] == "HIGH").sum()),
        "medium": int((d["priority"] == "MEDIUM").sum()),
        "low": int((d["priority"] == "LOW").sum()),
    }
 
 
def get_leading_candidate(d: pd.DataFrame) -> Optional[pd.Series]:
    candidates = d[d["is_candidate"] & d["score"].notna()]
    if candidates.empty:
        return None
    return candidates.loc[candidates["score"].idxmax()]
 
 
def build_deterministic_interpretation(row: pd.Series) -> str:
    clauses = []
 
    priority = row.get("priority", "")
    probability = row.get("probability", None)
    signal_quality = row.get("signal_quality", None)
    transit_coverage = row.get("transit_coverage", None)
    data_completeness = row.get("data_completeness", None)
 
    if priority == "HIGH" and probability is not None and pd.notna(probability) and probability >= 80:
        clauses.append(
            "This observation combines a high candidate probability with high scientific "
            "priority, making it a strong candidate for follow-up investigation."
        )
    elif priority == "HIGH":
        clauses.append("This observation carries high scientific priority based on its combined signal characteristics.")
    elif priority == "MEDIUM":
        clauses.append("This observation carries medium scientific priority and may warrant secondary review.")
    else:
        clauses.append("This observation currently carries low scientific priority relative to the rest of the queue.")
 
    if signal_quality is not None and pd.notna(signal_quality):
        if signal_quality < 40:
            clauses.append("Signal quality is comparatively weak.")
        elif signal_quality >= 80:
            clauses.append("Signal quality is strong.")
 
    if transit_coverage is not None and pd.notna(transit_coverage):
        if transit_coverage < 40:
            clauses.append("Transit coverage is limited.")
        elif transit_coverage >= 80:
            clauses.append("Observational coverage of the transit is strong.")
 
    if data_completeness is not None and pd.notna(data_completeness):
        if data_completeness < 40:
            clauses.append("A meaningful portion of the expected data is missing for this target.")
        elif data_completeness >= 80:
            clauses.append("The available data for this target are relatively complete.")
 
    clauses.append("Scientific Priority Score \u2260 planetary confirmation.")
 
    return " ".join(clauses)
 
 
# ============================================================
# UI HELPERS
# ============================================================
 
def fmt_metric(value, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "\u2014"
    return f"{value:.1f}{suffix}"
 
 
def fmt_num(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "\u2014"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)
 
 
def badge_html(label: str, kind: str) -> str:
    return f'<span class="badge badge--{kind}">{label}</span>'
 
 
def classification_badge(classification: str) -> str:
    kind = {"CANDIDATE": "candidate", "CONFIRMED": "confirmed", "FALSE POSITIVE": "falsepositive"}.get(classification, "confirmed")
    return badge_html(classification, kind)
 
 
def priority_badge(priority: str) -> str:
    kind = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(priority, "low")
    return badge_html(priority, kind)
 
 
def metric_card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="metric-value">{value}</div></div>'
 
 
def component_bar(label: str, value: float) -> str:
    pct = max(0.0, min(100.0, float(value)))
    return f"""
    <div style="margin-bottom:0.6rem;">
        <div class="label">{label} &middot; {value:.1f}%</div>
        <div class="herc-bar-track"><div class="herc-bar-fill" style="width:{pct}%;"></div></div>
    </div>
    """
 
 
def render_telemetry_row(items) -> None:
    parts = ['<div class="telemetry-row">']
    for i, (value, label) in enumerate(items):
        if i > 0:
            parts.append('<div class="telemetry-divider"></div>')
        parts.append(f'<div class="telemetry-item"><div class="telemetry-value">{value}</div><div class="telemetry-label">{label}</div></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
 
 
def render_metric_stack(items) -> None:
    parts = ['<div class="metric-stack">']
    for value, label in items:
        parts.append(f'<div><div class="metric-stack__value">{value}</div><div class="metric-stack__label">{label}</div></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
 
 
def render_telemetry_list(rows) -> None:
    html = "".join(
        f'<div class="telemetry-row-line"><span class="tl-label">{l}</span><span class="tl-value">{v}</span></div>'
        for l, v in rows
    )
    st.markdown(f'<div class="telemetry-list">{html}</div>', unsafe_allow_html=True)
 
 
def render_pipeline(stages) -> None:
    """stages: (title, value, label) for a numeric stage, or (title, description)."""
    parts = ['<div style="display:flex;gap:.6rem;margin:1rem 0 1.6rem 0;flex-wrap:wrap;">']
    last = len(stages) - 1
    for i, stage in enumerate(stages):
        cls = "pipeline-card pipeline-card--active" if i == last else "pipeline-card"
        num = f"{i + 1:02d}"
        parts.append(f'<div class="{cls}"><div class="pipeline-card__eyebrow">{num} / {stage[0]}</div>')
        if len(stage) == 3:
            _, value, label = stage
            parts.append(f'<div class="pipeline-card__value">{value}</div><div class="pipeline-card__label">{label}</div>')
        else:
            _, desc = stage
            parts.append(f'<div class="pipeline-card__label" style="margin-top:.3rem;">{desc}</div>')
        parts.append("</div>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
 
 
def render_score_readout(score) -> None:
    st.markdown(
        f'<div class="score-card"><div class="score-card__value">{fmt_metric(score)}</div>'
        f'<div class="score-card__label">SCIENTIFIC PRIORITY SCORE</div></div>',
        unsafe_allow_html=True,
    )
 
 
def render_why_this_target(row: pd.Series) -> None:
    priority = row.get("priority", "")
    headline = {
        "HIGH": "HIGH SCIENTIFIC PRIORITY",
        "MEDIUM": "MEDIUM SCIENTIFIC PRIORITY",
        "LOW": "LOW SCIENTIFIC PRIORITY",
    }.get(priority, "SCIENTIFIC PRIORITY")
    st.markdown(f'<div class="why-headline">{headline}</div>', unsafe_allow_html=True)
 
    metric_rows = []
    if row.get("probability") is not None and pd.notna(row.get("probability")):
        metric_rows.append(("Candidate probability", f"{row['probability']:.1f}%"))
    for label, key in [("Signal quality", "signal_quality"), ("Transit coverage", "transit_coverage"), ("Data completeness", "data_completeness")]:
        if key in row.index and pd.notna(row[key]):
            metric_rows.append((label, f"{row[key]:.1f}%"))
    if metric_rows:
        render_telemetry_list(metric_rows)
 
    st.markdown(f'<div class="sub" style="margin-top:.8rem;">{build_deterministic_interpretation(row)}</div>', unsafe_allow_html=True)
 
 
def render_results_table(view: pd.DataFrame) -> None:
    numeric_cols = {"PRIORITY SCORE", "CANDIDATE PROBABILITY", "SIGNAL QUALITY", "TRANSIT COVERAGE", "DATA COMPLETENESS"}
    headers_html = "".join(f"<th>{c}</th>" for c in view.columns)
 
    rows_html = []
    for _, r in view.iterrows():
        cells = []
        for c in view.columns:
            val = r[c]
            if c == "PRIORITY":
                cells.append(f"<td>{priority_badge(str(val))}</td>")
            elif c == "TARGET":
                cells.append(f'<td class="mono">{val}</td>')
            elif c in numeric_cols:
                cells.append(f'<td class="num">{fmt_num(val)}</td>')
            else:
                cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
 
    table_html = (
        '<div class="herc-table-wrap"><table class="herc-table">'
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
 
 
def _dark_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor(COLORS["panel"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["line"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    return fig, ax
 
 
def bar_figure(labels, values, colors):
    fig, ax = _dark_axes((6, 3))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.bar_label(bars, padding=3, color=COLORS["text"], fontsize=8, fmt="%.0f")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(COLORS["text"])
    fig.tight_layout()
    return fig
 
 
def scatter_figure(x, y, xlabel, ylabel):
    fig, ax = _dark_axes((6, 3.4))
    ax.scatter(x, y, s=10, color=COLORS["accent"], alpha=0.55, edgecolors="none")
    ax.set_xlabel(xlabel.replace("koi_", "").upper(), color=COLORS["muted"], fontsize=8)
    ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(COLORS["text"])
    fig.tight_layout()
    return fig
 
 
# ============================================================
# PAGES
# ============================================================
 
def render_command_center(d: pd.DataFrame, counts: dict) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hud-k">SCIENTIFIC MISSION CONTROL</div>
            <div class="online" style="margin-top:.6rem;"><span class="dot"></span>CORE OPERATIONAL</div>
            <h1>Find the signals <span class="grad-text">worth looking at.</span></h1>
            <div class="sub">HERCULES classifies astronomical observations and ranks them by scientific priority so researchers can decide where to look first.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    render_telemetry_row([
        (f"{counts['total']:,}", "OBSERVATIONS"),
        (f"{counts['candidate']:,}", "CANDIDATE PREDICTIONS"),
        (f"{counts['high']:,}", "HIGH PRIORITY"),
        (f"{counts['medium']:,}", "MEDIUM PRIORITY"),
    ])
 
    st.markdown('<div class="hud-k" style="margin-top:1.4rem;">MISSION PIPELINE</div>', unsafe_allow_html=True)
    render_pipeline([
        ("OBSERVE", f"{counts['total']:,}", "observations"),
        ("CLASSIFY", f"{counts['candidate']:,}", "candidate predictions"),
        ("PRIORITIZE", f"{counts['ranked']:,}", "scientific priority ranked"),
        ("INVESTIGATE", f"{counts['high']:,}", "high-priority targets"),
    ])
 
    leading = get_leading_candidate(d)
    if leading is not None:
        st.markdown('<div class="hud-k">CURRENT TOP-RANKED CANDIDATE</div>', unsafe_allow_html=True)
 
        left, right = st.columns([1, 1.4])
        with left:
            render_score_readout(leading["score"])
            st.markdown(
                f'<div style="margin-top:.9rem;">'
                f'<div class="mission-target__id">{leading["target"]}</div>'
                f'<div class="mission-target__kepid">KEPID {leading["kepid"]}</div>'
                f'</div>'
                f'<div style="margin-top:.6rem;">{classification_badge(leading["classification"])}{priority_badge(leading["priority"])}</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(metric_card("CANDIDATE PROBABILITY", fmt_metric(leading["probability"], "%")), unsafe_allow_html=True)
            for comp_label, comp_key in [
                ("SIGNAL QUALITY", "signal_quality"),
                ("TRANSIT COVERAGE", "transit_coverage"),
                ("DATA COMPLETENESS", "data_completeness"),
            ]:
                if comp_key in leading.index and pd.notna(leading[comp_key]):
                    st.markdown(component_bar(comp_label, leading[comp_key]), unsafe_allow_html=True)
 
        st.markdown('<div class="hud-k" style="margin-top:1.5rem;">WHY THIS TARGET?</div>', unsafe_allow_html=True)
        render_why_this_target(leading)
    else:
        st.markdown('<div class="sub">No CANDIDATE-classified observation currently has a usable priority score.</div>', unsafe_allow_html=True)
 
    st.markdown(
        '<div class="safeguard"><b>SCIENTIFIC SAFEGUARD</b><br>'
        "Scientific Priority Score &ne; planetary confirmation. HERCULES prioritizes observations for "
        "investigation; it does not determine whether a planet exists.</div>",
        unsafe_allow_html=True,
    )
 
 
def render_target_explorer(d: pd.DataFrame) -> None:
    st.markdown('<div class="hud-k">TARGET EXPLORER</div><div class="section-header">Inspect a signal.</div>', unsafe_allow_html=True)
 
    search = st.text_input("SEARCH TARGET", placeholder="Type a KOI name, e.g. K07259.01")
    if search:
        needle = search.strip().upper()
        limited = [t for t in d["target"].tolist() if needle in t.upper()][:200]
        if not limited:
            st.warning("No targets match that search.")
            return
    else:
        limited = d.sort_values("score", ascending=False)["target"].head(200).tolist()
        st.caption("Showing the 200 highest-priority targets. Type above to search all observations.")
 
    target = st.selectbox("SELECT TARGET", limited)
    row = d[d["target"] == target].iloc[0]
 
    id_col, assess_col = st.columns([1, 1])
    with id_col:
        st.markdown('<div class="hud-k">TARGET IDENTITY</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="mission-target__id" style="font-size:1.9rem;margin-top:.4rem;">{row["target"]}</div>'
            f'<div class="mission-target__kepid" style="margin-top:.25rem;">KEPID {row["kepid"]}</div>',
            unsafe_allow_html=True,
        )
    with assess_col:
        st.markdown('<div class="hud-k">HERCULES ASSESSMENT</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin:.5rem 0 .9rem 0;">{classification_badge(row["classification"])}{priority_badge(row["priority"])}</div>',
            unsafe_allow_html=True,
        )
        prob_items = [(fmt_metric(row.get("probability"), "%"), "CANDIDATE PROBABILITY")]
        if "prob_confirmed" in row.index and pd.notna(row["prob_confirmed"]):
            prob_items.append((fmt_metric(row["prob_confirmed"], "%"), "CONFIRMED PROBABILITY"))
        if "prob_false_positive" in row.index and pd.notna(row["prob_false_positive"]):
            prob_items.append((fmt_metric(row["prob_false_positive"], "%"), "FALSE-POSITIVE PROBABILITY"))
        render_metric_stack(prob_items)
 
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">SCIENTIFIC PRIORITY</div>', unsafe_allow_html=True)
    render_score_readout(row["score"])
 
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">PRIORITY COMPONENTS</div>', unsafe_allow_html=True)
    has_component = False
    for comp_label, comp_key in [
        ("SIGNAL QUALITY", "signal_quality"),
        ("TRANSIT COVERAGE", "transit_coverage"),
        ("DATA COMPLETENESS", "data_completeness"),
    ]:
        if comp_key in row.index and pd.notna(row[comp_key]):
            st.markdown(component_bar(comp_label, row[comp_key]), unsafe_allow_html=True)
            has_component = True
    if not has_component:
        st.caption("Priority component fields not available for this target.")
 
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">OBSERVATIONAL PROFILE</div>', unsafe_allow_html=True)
    any_measurements = False
    for group_name, cols in SIGNAL_GROUPS.items():
        available = [c for c in cols if c in row.index and pd.notna(row[c])]
        if not available:
            continue
        any_measurements = True
        st.markdown(f'<div class="dossier-group">{group_name}</div>', unsafe_allow_html=True)
        render_telemetry_list([(c.replace("koi_", "").upper(), f"{row[c]:,.4g}") for c in available])
    if not any_measurements:
        st.caption("No raw measurement columns available for this target.")
 
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">SCIENTIFIC INTERPRETATION</div>', unsafe_allow_html=True)
    render_why_this_target(row)
 
 
def render_candidate_hunter(d: pd.DataFrame) -> None:
    st.markdown('<div class="hud-k">CANDIDATE HUNTER</div><div class="section-header">Find the strongest signals.</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="hud-k" style="margin-top:.5rem;">FILTERS</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        min_score = st.slider("MINIMUM PRIORITY SCORE", 0.0, 100.0, 50.0, 1.0, key="ch_min_score")
    with f2:
        priorities = st.multiselect("PRIORITY", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"], key="ch_priorities")
    with f3:
        min_prob = st.slider("MINIMUM CANDIDATE PROBABILITY", 0.0, 100.0, 0.0, 1.0, key="ch_min_prob")
 
    optional_filters = [
        spec for spec in [
            ("signal_quality", "MINIMUM SIGNAL QUALITY", "ch_min_signal"),
            ("transit_coverage", "MINIMUM TRANSIT COVERAGE", "ch_min_transit"),
            ("data_completeness", "MINIMUM DATA COMPLETENESS", "ch_min_completeness"),
        ] if spec[0] in d.columns
    ]
    min_values = {}
    if optional_filters:
        opt_cols = st.columns(len(optional_filters))
        for col, (field, label, key) in zip(opt_cols, optional_filters):
            with col:
                min_values[field] = st.slider(label, 0.0, 100.0, 0.0, 1.0, key=key)
 
    if st.button("RESET FILTERS"):
        for k in ["ch_min_score", "ch_priorities", "ch_min_prob", "ch_min_signal", "ch_min_transit", "ch_min_completeness"]:
            st.session_state.pop(k, None)
        st.rerun()
 
    matches = d[
        d["is_candidate"]
        & (d["score"] >= min_score)
        & (d["priority"].isin(priorities))
        & (d["probability"] >= min_prob)
    ]
    for field, min_val in min_values.items():
        matches = matches[matches[field] >= min_val]
    matches = matches.sort_values("score", ascending=False)
 
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">RESULTS</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="readout-inline"><span class="readout-inline__value">{len(matches):,}</span>'
        f'<span class="readout-inline__label">MATCHING CANDIDATES</span></div>',
        unsafe_allow_html=True,
    )
 
    display_cols = ["target", "kepid", "score", "probability", "priority"]
    for c in OPTIONAL_COMPONENT_COLUMNS:
        if c in matches.columns:
            display_cols.append(c)
 
    rename_map = {
        "target": "TARGET", "kepid": "KEPID", "score": "PRIORITY SCORE",
        "probability": "CANDIDATE PROBABILITY", "priority": "PRIORITY",
        "signal_quality": "SIGNAL QUALITY", "transit_coverage": "TRANSIT COVERAGE",
        "data_completeness": "DATA COMPLETENESS",
    }
 
    if matches.empty:
        st.info("No candidates match the current filters.")
    else:
        view = matches[display_cols].head(250).rename(columns=rename_map)
        render_results_table(view)
        if len(matches) > 250:
            st.caption(f"Showing the top 250 of {len(matches):,} matching candidates by priority score. Export includes the full set.")
 
    export_view = matches[display_cols].rename(columns=rename_map)
    st.download_button(
        "EXPORT CURRENT RESULTS",
        export_view.to_csv(index=False).encode("utf-8"),
        "hercules_candidates.csv",
        "text/csv",
        disabled=matches.empty,
    )
 
 
def render_analytics(d: pd.DataFrame) -> None:
    st.markdown('<div class="hud-k">SCIENTIFIC ANALYTICS</div><div class="section-header">See the search space.</div>', unsafe_allow_html=True)
 
    render_telemetry_row([
        (fmt_metric(d['score'].mean()), "MEAN PRIORITY SCORE"),
        (fmt_metric(d['score'].max()), "TOP PRIORITY SCORE"),
        (fmt_metric(d['score'].median()), "MEDIAN PRIORITY SCORE"),
        (fmt_metric(d['probability'].median(), "%"), "MEDIAN CANDIDATE PROBABILITY"),
    ])
 
    st.markdown('<div class="zone-header" style="margin-top:1.8rem;">SCIENTIFIC OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-k">CLASSIFICATION DISTRIBUTION</div>', unsafe_allow_html=True)
    class_counts = d["classification"].value_counts()
    class_order = [c for c in ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"] if c in class_counts.index]
    class_order += [c for c in class_counts.index if c not in class_order]
    if class_order:
        class_color_map = {"CANDIDATE": COLORS["accent"], "CONFIRMED": COLORS["muted"], "FALSE POSITIVE": COLORS["danger"]}
        class_colors = [class_color_map.get(c, COLORS["muted"]) for c in class_order]
        fig0 = bar_figure(class_order, [int(class_counts[c]) for c in class_order], class_colors)
        st.pyplot(fig0)
        plt.close(fig0)
    else:
        st.caption("No classification values available.")
 
    st.markdown('<div class="zone-header" style="margin-top:2.2rem;">PRIORITY LANDSCAPE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hud-k">PRIORITY DISTRIBUTION</div>', unsafe_allow_html=True)
    priority_counts = d["priority"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    fig1 = bar_figure(priority_counts.index.tolist(), priority_counts.values.tolist(), [COLORS["accent"], COLORS["amber"], COLORS["muted"]])
    st.pyplot(fig1)
    plt.close(fig1)
 
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">CANDIDATE PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
    prob_values = d["probability"].dropna()
    if prob_values.empty:
        st.caption("No candidate probability values available.")
    else:
        bins = [0, 10, 25, 50, 75, 90, 100]
        labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
        bucketed = pd.cut(prob_values, bins=bins, labels=labels, include_lowest=True)
        bucket_counts = bucketed.value_counts().reindex(labels, fill_value=0)
        fig2 = bar_figure(labels, bucket_counts.values.tolist(), [COLORS["accent"]] * len(labels))
        st.pyplot(fig2)
        plt.close(fig2)
 
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">CANDIDATE PROBABILITY VS SCIENTIFIC PRIORITY SCORE</div>', unsafe_allow_html=True)
    pair2 = d[["probability", "score"]].dropna()
    if pair2.empty:
        st.caption("No overlapping probability/score data available.")
    else:
        if len(pair2) > 500:
            idx2 = np.linspace(0, len(pair2) - 1, 500, dtype=int)
            pair2 = pair2.iloc[idx2]
        fig4 = scatter_figure(pair2["probability"], pair2["score"], "CANDIDATE PROBABILITY (%)", "SCIENTIFIC PRIORITY SCORE")
        st.pyplot(fig4)
        plt.close(fig4)
 
    available_signals = [c for c in SIGNAL_COLUMNS if c in d.columns]
    if available_signals:
        st.markdown('<div class="zone-header" style="margin-top:2.2rem;">SIGNAL RELATIONSHIPS</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            signal = st.selectbox("SCIENTIFIC SIGNAL", available_signals)
        with sc2:
            y_axis = st.selectbox("COMPARE AGAINST", ["CANDIDATE PROBABILITY", "SCIENTIFIC PRIORITY SCORE"])
        y_col = "probability" if y_axis == "CANDIDATE PROBABILITY" else "score"
 
        pair = d[[signal, y_col]].dropna()
        if pair.empty:
            st.caption("No overlapping data for this signal and comparison.")
        else:
            if len(pair) > 500:
                idx = np.linspace(0, len(pair) - 1, 500, dtype=int)
                pair = pair.iloc[idx]
            v = d[signal].dropna()
            render_telemetry_row([
                (f"{v.min():,.4g}" if len(v) else "\u2014", "MIN"),
                (f"{v.median():,.4g}" if len(v) else "\u2014", "MEDIAN"),
                (f"{v.max():,.4g}" if len(v) else "\u2014", "MAX"),
            ])
            fig3 = scatter_figure(pair[signal], pair[y_col], signal, y_axis)
            st.pyplot(fig3)
            plt.close(fig3)
 
    st.markdown(
        '<div class="safeguard"><b>PERFORMANCE MODE</b><br>'
        "Charts use aggregated distributions or a deterministic sample of at most 500 points. "
        "The raw dataset is never rendered directly.</div>",
        unsafe_allow_html=True,
    )
 
 
def render_about(d: pd.DataFrame, counts: dict) -> None:
    st.markdown('<div class="hud-k">SYSTEM DOSSIER</div><div class="section-header">Scientific intelligence for the search space.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><div class="label">MISSION</div>'
        '<h3 style="margin:.5rem 0;color:var(--text);">HERCULES helps scientists decide where to look first.</h3>'
        '<div class="sub">HERCULES is an AI-assisted astronomical triage and prioritization system. It applies '
        "machine-learning classification to NASA/Kepler observations and combines the result with scientific "
        "signal measurements into a priority score. It does not confirm planets &mdash; a high score means an "
        "observation deserves human follow-up, not that a discovery has been made.</div></div>",
        unsafe_allow_html=True,
    )
 
    st.markdown('<div class="hud-k" style="margin-top:1.8rem;">DECISION PIPELINE</div>', unsafe_allow_html=True)
    render_pipeline([
        ("OBSERVE", "Ingest NASA / Kepler observations and transit measurements."),
        ("CLASSIFY", "HERCULES estimates whether a signal resembles a candidate observation."),
        ("PRIORITIZE", "Combine signal quality, coverage, and completeness into a scientific priority score."),
        ("INVESTIGATE", "Put the strongest observations in front of researchers for human follow-up."),
    ])
 
    render_telemetry_row([
        (f"{counts['total']:,}", "OBSERVATIONS"),
        (f"{counts['candidate']:,}", "CANDIDATES"),
        (f"{counts['high']:,}", "HIGH PRIORITY"),
    ])
 
    st.markdown(
        '<div class="safeguard"><b>SCIENTIFIC SAFEGUARD</b><br>'
        "Scientific Priority Score &ne; planetary confirmation. Model classification, scientific priority, and "
        "planetary confirmation are three distinct things &mdash; HERCULES produces the first two; the third "
        "requires human scientific follow-up.</div>",
        unsafe_allow_html=True,
    )
 
 
# ============================================================
# MAIN
# ============================================================
 
def main() -> None:
    st.set_page_config(page_title="HERCULES // Mission Control", page_icon="\u25c8", layout="wide")
    inject_css()
 
    if not CSV_PATH.exists():
        st.error(
            f"Missing dataset: expected outputs/hercules_v4_2_rankings.csv at:\n\n{CSV_PATH}\n\n"
            "Run the V4.2 pipeline to generate it, or confirm the project root is correct."
        )
        st.stop()
 
    raw = load_data(str(CSV_PATH))
 
    if raw.empty:
        st.error("outputs/hercules_v4_2_rankings.csv exists but contains no rows. Re-run the V4.2 pipeline to regenerate it.")
        st.stop()
 
    _validate_schema(raw)
    d = prepare_data(raw)
    counts = get_summary_counts(d)
 
    with st.sidebar:
        st.markdown('<div class="hud-k">AI ASTRONOMICAL TRIAGE</div>', unsafe_allow_html=True)
        st.markdown('<h2 style="letter-spacing:-.04em;margin:.3rem 0;color:var(--text);">HERCULES</h2>', unsafe_allow_html=True)
        st.markdown('<div class="online"><span class="dot"></span>SYSTEM ONLINE</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin:1.1rem 0;border-top:1px solid var(--line);"></div>', unsafe_allow_html=True)
        selected_label = st.radio("NAVIGATION", [label for label, _ in NAV_ITEMS], label_visibility="collapsed")
        page = dict(NAV_ITEMS)[selected_label]
        st.markdown('<div style="margin:1.1rem 0;border-top:1px solid var(--line);"></div>', unsafe_allow_html=True)
        st.caption(f"{counts['total']:,} observations \u2022 V4.2")
 
    if page == "Command Center":
        render_command_center(d, counts)
    elif page == "Target Explorer":
        render_target_explorer(d)
    elif page == "Candidate Hunter":
        render_candidate_hunter(d)
    elif page == "Scientific Analytics":
        render_analytics(d)
    else:
        render_about(d, counts)
 
 
if __name__ == "__main__":
    main()

