"""
HERCULES // Scientific Mission Control
AI-assisted astronomical triage and prioritization system.
 
Run from the project root:
    cd C:\\Hercules
    streamlit run src/app.py
 
Data model: outputs/hercules_v4_2_rankings.csv (confirmed 159-column
V4.2 schema). Six columns are required; everything else is optional
and its related UI hides itself gracefully if the column is missing.
"""

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
 
# Confirmed to exist. Do not remove support for these, do not treat
# any of them as optional.
REQUIRED_COLUMNS = [
    "kepoi_name",
    "kepid",
    "hercules_prediction",
    "prob_candidate",
    "scientific_priority_score",
    "priority",
]
 
# Optional — checked for existence before use anywhere they're touched.
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
    background: radial-gradient(circle at 85% 5%, rgba(53,224,255,.07), transparent 30%), var(--bg);
    color: var(--text);
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .15;
    background-image:
        linear-gradient(rgba(255,255,255,.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.02) 1px, transparent 1px);
    background-size: 42px 42px;
}
 
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.block-container { max-width: 1500px; padding-top: 2rem; }
 
.hud-k {
    font-family: "IBM Plex Mono", monospace;
    font-size: .72rem;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--accent);
}
.section-header { font-size: 1.9rem; font-weight: 700; margin: .4rem 0 1rem 0; color: var(--text); }
.sub { color: var(--muted); max-width: 700px; line-height: 1.6; }
 
.hero {
    padding: 2.1rem;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: radial-gradient(circle at 90% 15%, rgba(53,224,255,.12), transparent 30%), var(--panel);
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: clamp(2.2rem, 5vw, 4.2rem); line-height: .96; margin: .7rem 0; color: var(--text); }
 
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
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 10px var(--accent); }
 
.card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: .6rem;
}
.label {
    font: 600 .67rem "IBM Plex Mono";
    letter-spacing: .1em;
    color: var(--muted);
    text-transform: uppercase;
}
.metric-value { font-size: 1.7rem; font-weight: 700; margin-top: .3rem; color: var(--text); }
 
.funnel { display: flex; gap: .4rem; margin: 1rem 0 1.4rem 0; overflow-x: auto; }
.funnel__stage {
    flex: 1;
    min-width: 150px;
    padding: 1rem;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel);
}
.funnel__eyebrow {
    font: 600 .65rem "IBM Plex Mono";
    color: var(--accent);
    letter-spacing: .1em;
    margin-bottom: .3rem;
}
.funnel__sub { font-size: .78rem; font-weight: 600; color: var(--text); margin-bottom: .5rem; }
.funnel__value { font-size: 1.5rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.funnel__label { font-size: .76rem; color: var(--muted); margin-top: .3rem; line-height: 1.4; }
.funnel__arrow { display: flex; align-items: center; color: var(--muted); font-size: 1.1rem; padding: 0 .1rem; }
 
.target-card {
    border: 1px solid rgba(53,224,255,.3);
    border-radius: 14px;
    padding: 1.2rem;
    background: radial-gradient(circle at 90% 10%, rgba(53,224,255,.1), transparent 30%), var(--panel);
}
.target-card__id { font-family: "IBM Plex Mono"; font-size: 1.4rem; font-weight: 600; margin-top: .3rem; color: var(--text); }
.target-card__kepid { font-size: .78rem; color: var(--muted); }
.target-card__score { font: 600 2.6rem "IBM Plex Mono"; color: var(--accent); line-height: 1.1; margin-top: .2rem; }
 
.badge {
    display: inline-block;
    padding: .18rem .65rem;
    border-radius: 999px;
    font: 600 .68rem "IBM Plex Mono";
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-right: .4rem;
}
.badge--high, .badge--candidate { background: rgba(53,224,255,.15); color: var(--accent); border: 1px solid var(--accent); }
.badge--medium { background: rgba(243,184,75,.15); color: var(--amber); border: 1px solid var(--amber); }
.badge--low { background: rgba(124,138,160,.15); color: var(--muted); border: 1px solid var(--line); }
.badge--confirmed { background: rgba(232,237,245,.08); color: var(--text); border: 1px solid var(--line); }
.badge--falsepositive { background: rgba(255,92,92,.15); color: var(--danger); border: 1px solid var(--danger); }
 
.dossier-group {
    font: 600 .68rem "IBM Plex Mono";
    color: var(--muted);
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 1rem 0 .5rem 0;
    border-bottom: 1px solid var(--line);
    padding-bottom: .3rem;
}
.measure-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: .6rem; }
.measure-item { background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; padding: .7rem .8rem; }
 
.safeguard {
    border-left: 2px solid var(--amber);
    padding: .75rem 1rem;
    background: rgba(243,184,75,.05);
    color: var(--muted);
    font-size: .8rem;
    border-radius: 0 8px 8px 0;
    margin-top: 1.5rem;
    line-height: 1.5;
}
 
@media (max-width: 900px) {
    .funnel { flex-direction: column; }
}
</style>
"""
 
 
def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
 
 
# ============================================================
# DATA LAYER
# ============================================================
 
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
    """Convert a 0-1 or already-0-100 numeric series to 0-100, without double-scaling."""
    s = pd.to_numeric(series, errors="coerce")
    finite = s.dropna()
    if len(finite) and finite.max() <= 1.0:
        s = s * 100.0
    return s.clip(lower=0, upper=100)
 
 
@st.cache_data(show_spinner=False)
def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build the small, UI-facing dataframe from confirmed real V4.2 columns only.
    The six REQUIRED_COLUMNS are assumed present (validated before this runs).
    Everything else is copied over only if the column actually exists —
    nothing here invents a column or fabricates a value. The underlying
    HERCULES model / ranking calculations are never touched here — this
    function only reshapes already-computed columns for display.
    """
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
    """Highest scientific_priority_score among rows HERCULES actually classified as CANDIDATE."""
    candidates = d[d["is_candidate"] & d["score"].notna()]
    if candidates.empty:
        return None
    return candidates.loc[candidates["score"].idxmax()]
 
 
def build_deterministic_interpretation(row: pd.Series) -> str:
    """
    Rule-based interpretation text built only from real values on this row.
    No LLM, no hallucination — every clause is gated on an actual number.
    Always closes with the exact scientific safeguard line.
    """
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
 
    if probability is not None and pd.notna(probability):
        clauses.append(f"Candidate probability: {probability:.1f}%.")
 
    if signal_quality is not None and pd.notna(signal_quality):
        clauses.append(f"Signal quality: {signal_quality:.1f}%.")
        if signal_quality < 40:
            clauses.append("Signal quality is comparatively weak.")
        elif signal_quality >= 80:
            clauses.append("Signal quality is strong.")
 
    if transit_coverage is not None and pd.notna(transit_coverage):
        clauses.append(f"Transit coverage: {transit_coverage:.1f}%.")
        if transit_coverage < 40:
            clauses.append("Transit coverage is limited.")
        elif transit_coverage >= 80:
            clauses.append("Observational coverage of the transit is strong.")
 
    if data_completeness is not None and pd.notna(data_completeness):
        clauses.append(f"Data completeness: {data_completeness:.1f}%.")
        if data_completeness < 40:
            clauses.append("A meaningful portion of the expected data is missing for this target.")
        elif data_completeness >= 80:
            clauses.append("The available data for this target are relatively complete.")
 
    clauses.append("Scientific Priority Score \u2260 planetary confirmation.")
 
    return " ".join(clauses)

# UI HELPERS

def metric_card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="metric-value">{value}</div></div>'
 
 
def fmt_metric(value, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "\u2014"
    return f"{value:.1f}{suffix}"
 
 
def badge_html(label: str, kind: str) -> str:
    return f'<span class="badge badge--{kind}">{label}</span>'
 
 
def classification_badge(classification: str) -> str:
    kind = {
        "CANDIDATE": "candidate",
        "CONFIRMED": "confirmed",
        "FALSE POSITIVE": "falsepositive",
    }.get(classification, "confirmed")
    return badge_html(classification, kind)
 
 
def priority_badge(priority: str) -> str:
    kind = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(priority, "low")
    return badge_html(priority, kind)
 
 
def component_bar(label: str, value: float) -> str:
    pct = max(0.0, min(100.0, float(value)))
    return f"""
    <div style="margin-bottom:0.6rem;">
        <div class="label">{label} &middot; {value:.1f}%</div>
        <div style="background:{COLORS['line']};border-radius:6px;height:8px;margin-top:0.3rem;overflow:hidden;">
            <div style="background:{COLORS['accent']};width:{pct}%;height:100%;"></div>
        </div>
    </div>
    """
 
 
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
    ax.bar(labels, values, color=colors, width=0.6)
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

# PAGES

def render_command_center(d: pd.DataFrame, counts: dict) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hud-k">SCIENTIFIC MISSION CONTROL</div>
            <div class="online" style="margin-top:.6rem;"><span class="dot"></span>CORE OPERATIONAL</div>
            <h1>Find the signals<br>worth looking at.</h1>
            <div class="sub">HERCULES classifies astronomical observations and ranks them by scientific priority so researchers can decide where to look first.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("OBSERVATIONS", f"{counts['total']:,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("CANDIDATE PREDICTIONS", f"{counts['candidate']:,}"), unsafe_allow_html=True)
    c3.markdown(metric_card("HIGH PRIORITY", f"{counts['high']:,}"), unsafe_allow_html=True)
    c4.markdown(metric_card("MEDIUM PRIORITY", f"{counts['medium']:,}"), unsafe_allow_html=True)
 
    # OBSERVE -> CLASSIFY -> PRIORITIZE -> INVESTIGATE.
    # PRIORITIZE reflects observations that actually received a valid
    # scientific priority score, not MEDIUM priority specifically —
    # MEDIUM is not "the ranking stage", it's one priority bucket.
    st.markdown('<div class="hud-k" style="margin-top:2rem;">MISSION PIPELINE</div>', unsafe_allow_html=True)
    stages = [
        ("01", "OBSERVE", "NASA / KEPLER", f"{counts['total']:,}", "observations"),
        ("02", "CLASSIFY", "HERCULES ML TRIAGE", f"{counts['candidate']:,}", "candidate predictions"),
        ("03", "PRIORITIZE", "SCIENTIFIC PRIORITY ENGINE", f"{counts['ranked']:,}", "scientific priority ranked"),
        ("04", "INVESTIGATE", "HUMAN FOLLOW-UP", f"{counts['high']:,}", "high-priority targets"),
    ]
    parts = ['<div class="funnel">']
    for i, (num, title, sub, value, label) in enumerate(stages):
        parts.append(
            f'<div class="funnel__stage">'
            f'<div class="funnel__eyebrow">{num} / {title}</div>'
            f'<div class="funnel__sub">{sub}</div>'
            f'<div class="funnel__value">{value}</div>'
            f'<div class="funnel__label">{label}</div>'
            f'</div>'
        )
        if i < len(stages) - 1:
            parts.append('<div class="funnel__arrow">&rarr;</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
 
    leading = get_leading_candidate(d)
    if leading is not None:
        st.markdown('<div class="hud-k" style="margin-top:.5rem;">CURRENT TOP-RANKED CANDIDATE</div>', unsafe_allow_html=True)
        left, right = st.columns([1.2, 1.4])
        with left:
            st.markdown(
                f"""<div class="target-card">
                    <div class="label">TARGET NAME</div>
                    <div class="target-card__id">{leading['target']}</div>
                    <div class="target-card__kepid">KEPID: {leading['kepid']}</div>
                    <div class="label" style="margin-top:1rem;">SCIENTIFIC PRIORITY SCORE</div>
                    <div class="target-card__score">{fmt_metric(leading['score'])}</div>
                    <div style="margin-top:.8rem;">{classification_badge(leading['classification'])}{priority_badge(leading['priority'])}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with right:
            m1, m2 = st.columns(2)
            m1.markdown(metric_card("CANDIDATE PROBABILITY", fmt_metric(leading['probability'], "%")), unsafe_allow_html=True)
            m2.markdown(metric_card("PRIORITY", leading["priority"]), unsafe_allow_html=True)
            for comp_label, comp_key in [
                ("SIGNAL QUALITY", "signal_quality"),
                ("TRANSIT COVERAGE", "transit_coverage"),
                ("DATA COMPLETENESS", "data_completeness"),
            ]:
                if comp_key in leading.index and pd.notna(leading[comp_key]):
                    st.markdown(component_bar(comp_label, leading[comp_key]), unsafe_allow_html=True)
 
        st.markdown('<div class="hud-k" style="margin-top:1.25rem;">WHY THIS TARGET?</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{build_deterministic_interpretation(leading)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card">No CANDIDATE-classified observation currently has a usable priority score.</div>',
            unsafe_allow_html=True,
        )
 
    st.markdown(
        '<div class="safeguard"><b>SCIENTIFIC SAFEGUARD</b><br>'
        "Scientific Priority Score &ne; planetary confirmation. HERCULES prioritizes observations for "
        "investigation; it does not determine whether a planet exists.</div>",
        unsafe_allow_html=True,
    )
 
 
def render_target_explorer(d: pd.DataFrame) -> None:
    st.markdown(
        '<div class="hud-k">TARGET EXPLORER</div>'
        '<div class="section-header">Inspect a signal.</div>',
        unsafe_allow_html=True,
    )
 
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
    # .iloc[0] deliberately handles duplicate target names without crashing —
    # if kepoi_name is ever non-unique, the first match is shown; KEPID is
    # displayed alongside it so the specific row is still identifiable.
    row = d[d["target"] == target].iloc[0]
 
    # TARGET IDENTITY
    st.markdown('<div class="hud-k" style="margin-top:1rem;">TARGET IDENTITY</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="target-card">
            <div class="label">TARGET NAME</div>
            <div class="target-card__id">{row['target']}</div>
            <div class="target-card__kepid">KEPID: {row['kepid']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
 
    # HERCULES ASSESSMENT
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">HERCULES ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="margin-bottom:.6rem;">{classification_badge(row["classification"])}{priority_badge(row["priority"])}</div>',
        unsafe_allow_html=True,
    )
    prob_cards = [("CANDIDATE PROBABILITY", row.get("probability"))]
    if "prob_confirmed" in row.index and pd.notna(row["prob_confirmed"]):
        prob_cards.append(("CONFIRMED PROBABILITY", row["prob_confirmed"]))
    if "prob_false_positive" in row.index and pd.notna(row["prob_false_positive"]):
        prob_cards.append(("FALSE-POSITIVE PROBABILITY", row["prob_false_positive"]))
    prob_cols = st.columns(len(prob_cards))
    for col, (label, value) in zip(prob_cols, prob_cards):
        col.markdown(metric_card(label, fmt_metric(value, "%")), unsafe_allow_html=True)
 
    # SCIENTIFIC PRIORITY — made visually dominant
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">SCIENTIFIC PRIORITY</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card"><div class="label">SCIENTIFIC PRIORITY SCORE</div>'
        f'<div class="target-card__score">{fmt_metric(row["score"])}</div></div>',
        unsafe_allow_html=True,
    )
 
    # PRIORITY COMPONENTS
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
 
    # OBSERVATIONAL PROFILE
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">OBSERVATIONAL PROFILE</div>', unsafe_allow_html=True)
    any_measurements = False
    for group_name, cols in SIGNAL_GROUPS.items():
        available = [c for c in cols if c in row.index and pd.notna(row[c])]
        if not available:
            continue
        any_measurements = True
        st.markdown(f'<div class="dossier-group">{group_name}</div>', unsafe_allow_html=True)
        grid = "".join(
            f'<div class="measure-item"><div class="label">{c.replace("koi_", "").upper()}</div>'
            f'<div class="metric-value" style="font-size:1.1rem;">{row[c]:,.4g}</div></div>'
            for c in available
        )
        st.markdown(f'<div class="measure-grid">{grid}</div>', unsafe_allow_html=True)
    if not any_measurements:
        st.caption("No raw measurement columns available for this target.")
 
    # SCIENTIFIC INTERPRETATION
    st.markdown('<div class="hud-k" style="margin-top:1.5rem;">SCIENTIFIC INTERPRETATION</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{build_deterministic_interpretation(row)}</div>', unsafe_allow_html=True)
 
 
def render_candidate_hunter(d: pd.DataFrame) -> None:
    st.markdown(
        '<div class="hud-k">CANDIDATE HUNTER</div>'
        '<div class="section-header">Find the strongest signals.</div>',
        unsafe_allow_html=True,
    )
 
    st.markdown('<div class="hud-k" style="margin-top:.5rem;">FILTERS</div>', unsafe_allow_html=True)
 
    f1, f2, f3 = st.columns(3)
    with f1:
        min_score = st.slider("MINIMUM PRIORITY SCORE", 0.0, 100.0, 50.0, 1.0, key="ch_min_score")
    with f2:
        priorities = st.multiselect(
            "PRIORITY", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"], key="ch_priorities"
        )
    with f3:
        min_prob = st.slider("MINIMUM CANDIDATE PROBABILITY", 0.0, 100.0, 0.0, 1.0, key="ch_min_prob")
 
    optional_filters = [
        spec
        for spec in [
            ("signal_quality", "MINIMUM SIGNAL QUALITY", "ch_min_signal"),
            ("transit_coverage", "MINIMUM TRANSIT COVERAGE", "ch_min_transit"),
            ("data_completeness", "MINIMUM DATA COMPLETENESS", "ch_min_completeness"),
        ]
        if spec[0] in d.columns
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
 
    st.markdown(metric_card("MATCHING CANDIDATES", f"{len(matches):,}"), unsafe_allow_html=True)
 
    display_cols = ["target", "kepid", "score", "probability", "priority"]
    for c in OPTIONAL_COMPONENT_COLUMNS:
        if c in matches.columns:
            display_cols.append(c)
 
    rename_map = {
        "target": "TARGET",
        "kepid": "KEPID",
        "score": "PRIORITY SCORE",
        "probability": "CANDIDATE PROBABILITY",
        "priority": "PRIORITY",
        "signal_quality": "SIGNAL QUALITY",
        "transit_coverage": "TRANSIT COVERAGE",
        "data_completeness": "DATA COMPLETENESS",
    }
 
    if matches.empty:
        st.info("No candidates match the current filters.")
    else:
        view = matches[display_cols].head(250).rename(columns=rename_map)
        st.dataframe(view, use_container_width=True, hide_index=True, height=520)
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
    st.markdown(
        '<div class="hud-k">SCIENTIFIC ANALYTICS</div>'
        '<div class="section-header">See the search space.</div>',
        unsafe_allow_html=True,
    )
 
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("MEAN PRIORITY SCORE", fmt_metric(d['score'].mean())), unsafe_allow_html=True)
    m2.markdown(metric_card("TOP PRIORITY SCORE", fmt_metric(d['score'].max())), unsafe_allow_html=True)
    m3.markdown(metric_card("MEDIAN PRIORITY SCORE", fmt_metric(d['score'].median())), unsafe_allow_html=True)
    m4.markdown(metric_card("MEDIAN CANDIDATE PROBABILITY", fmt_metric(d['probability'].median(), "%")), unsafe_allow_html=True)
 
    st.markdown('<div class="hud-k" style="margin-top:2rem;">01 / CLASSIFICATION DISTRIBUTION</div>', unsafe_allow_html=True)
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
 
    st.markdown('<div class="hud-k" style="margin-top:2rem;">02 / PRIORITY LANDSCAPE</div>', unsafe_allow_html=True)
    priority_counts = d["priority"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    fig1 = bar_figure(
        priority_counts.index.tolist(),
        priority_counts.values.tolist(),
        [COLORS["accent"], COLORS["amber"], COLORS["muted"]],
    )
    st.pyplot(fig1)
    plt.close(fig1)
 
    st.markdown('<div class="hud-k" style="margin-top:2rem;">03 / CANDIDATE PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
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
 
    st.markdown('<div class="hud-k" style="margin-top:2rem;">04 / CANDIDATE PROBABILITY VS SCIENTIFIC PRIORITY SCORE</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="hud-k" style="margin-top:2rem;">05 / SCIENTIFIC SIGNAL INSPECTOR</div>', unsafe_allow_html=True)
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
            s1, s2, s3 = st.columns(3)
            s1.markdown(metric_card("MIN", f"{v.min():,.4g}" if len(v) else "\u2014"), unsafe_allow_html=True)
            s2.markdown(metric_card("MEDIAN", f"{v.median():,.4g}" if len(v) else "\u2014"), unsafe_allow_html=True)
            s3.markdown(metric_card("MAX", f"{v.max():,.4g}" if len(v) else "\u2014"), unsafe_allow_html=True)
 
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
    st.markdown(
        '<div class="hud-k">SYSTEM DOSSIER</div>'
        '<div class="section-header">Scientific intelligence for the search space.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><div class="label">MISSION</div>'
        '<h3 style="margin:.5rem 0;color:var(--text);">HERCULES helps scientists decide where to look first.</h3>'
        '<div class="sub">HERCULES is an AI-assisted astronomical triage and prioritization system. It applies '
        "machine-learning classification to NASA/Kepler observations and combines the result with scientific "
        "signal measurements into a priority score. It does not confirm planets &mdash; a high score means an "
        "observation deserves human follow-up, not that a discovery has been made.</div></div>",
        unsafe_allow_html=True,
    )
 
    stages = [
        ("OBSERVE", "Ingest NASA / Kepler observations and transit measurements."),
        ("CLASSIFY", "HERCULES estimates whether a signal resembles a candidate observation."),
        ("RANK", "Combine signal quality, coverage, and completeness into a scientific priority score."),
        ("INVESTIGATE", "Put the strongest observations in front of researchers for human follow-up."),
    ]
    parts = ['<div class="funnel" style="margin-top:1.5rem;">']
    for i, (title, desc) in enumerate(stages):
        parts.append(
            f'<div class="funnel__stage">'
            f'<div class="funnel__eyebrow">{title}</div>'
            f'<div class="funnel__label">{desc}</div>'
            f'</div>'
        )
        if i < len(stages) - 1:
            parts.append('<div class="funnel__arrow">&rarr;</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
 
    a, b, c = st.columns(3)
    a.markdown(metric_card("OBSERVATIONS", f"{counts['total']:,}"), unsafe_allow_html=True)
    b.markdown(metric_card("CANDIDATES", f"{counts['candidate']:,}"), unsafe_allow_html=True)
    c.markdown(metric_card("HIGH PRIORITY", f"{counts['high']:,}"), unsafe_allow_html=True)
 
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
        st.error(
            "outputs/hercules_v4_2_rankings.csv exists but contains no rows. "
            "Re-run the V4.2 pipeline to regenerate it."
        )
        st.stop()
 
    _validate_schema(raw)
    d = prepare_data(raw)
    counts = get_summary_counts(d)
 
    with st.sidebar:
        st.markdown('<div class="hud-k">AI SCIENTIFIC DISCOVERY SYSTEM</div>', unsafe_allow_html=True)
        st.markdown('<h2 style="letter-spacing:-.04em;margin:.3rem 0;color:var(--text);">HERCULES</h2>', unsafe_allow_html=True)
        st.markdown('<div class="online"><span class="dot"></span>SYSTEM ONLINE</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin:1rem 0;border-top:1px solid var(--line);"></div>', unsafe_allow_html=True)
        page = st.radio(
            "NAVIGATION",
            ["Command Center", "Target Explorer", "Candidate Hunter", "Scientific Analytics", "About HERCULES"],
            label_visibility="collapsed",
        )
        st.markdown('<div style="margin:1rem 0;border-top:1px solid var(--line);"></div>', unsafe_allow_html=True)
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

