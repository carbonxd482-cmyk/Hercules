"""
HERCULES — theme injection.
 
Central place for the design language from the brief:
- near-black navy base, not pure black
- ONE accent for high-priority / candidate (cyan below) — don't also
  use green for this anywhere else in the app
- amber for medium/general risk, stronger red reserved for
  false-positive
- monospaced uppercase eyebrow labels, bold hero headings
- restyled native Streamlit widgets (select, multiselect, slider,
  number input) so they stop looking like a bootstrap admin form
 
Streamlit's internal DOM/class names can shift between versions — if
a selector below stops matching after an upgrade, right-click →
Inspect on the widget and update the selector; the CSS variables at
the top won't need to change.
"""

import streamlit as st
 
CSS = """
<style>
:root {
    --herc-bg: #05070c;
    --herc-bg-elevated: #0b1018;
    --herc-border: #1c2534;
    --herc-text: #e8edf5;
    --herc-text-muted: #7c8aa0;
    --herc-accent: #35e0ff;      /* HIGH priority / CANDIDATE — the one accent */
    --herc-risk: #ff9d47;        /* MEDIUM / general caution */
    --herc-risk-strong: #ff5c5c; /* FALSE POSITIVE */
}
 
html, body, [class*="css"] {
    background-color: var(--herc-bg) !important;
    color: var(--herc-text) !important;
}
 
.herc-eyebrow {
    font-family: "JetBrains Mono", "Courier New", monospace;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.75rem;
    color: var(--herc-text-muted);
}
 
/* selectbox / multiselect */
div[data-baseweb="select"] > div {
    background-color: var(--herc-bg-elevated) !important;
    border-color: var(--herc-border) !important;
    color: var(--herc-text) !important;
}
div[data-baseweb="tag"] {
    background-color: rgba(53, 224, 255, 0.15) !important;
    border: 1px solid var(--herc-accent) !important;
    color: var(--herc-accent) !important;
}
 
/* slider */
div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--herc-accent) !important;
    border-color: var(--herc-accent) !important;
}
 
/* number input */
div[data-testid="stNumberInput"] input {
    background-color: var(--herc-bg-elevated) !important;
    border-color: var(--herc-border) !important;
    color: var(--herc-text) !important;
}
 
/* metric cards */
div[data-testid="stMetric"] {
    background-color: var(--herc-bg-elevated);
    border: 1px solid var(--herc-border);
    border-radius: 10px;
    padding: 1rem;
}
 
.herc-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.herc-badge--high        { background: rgba(53,224,255,0.15);  color: var(--herc-accent);      border: 1px solid var(--herc-accent); }
.herc-badge--medium      { background: rgba(255,157,71,0.15);  color: var(--herc-risk);         border: 1px solid var(--herc-risk); }
.herc-badge--low         { background: rgba(124,138,160,0.15); color: var(--herc-text-muted);   border: 1px solid var(--herc-border); }
.herc-badge--candidate   { background: rgba(53,224,255,0.15);  color: var(--herc-accent);       border: 1px solid var(--herc-accent); }
.herc-badge--confirmed   { background: rgba(232,237,245,0.10); color: var(--herc-text);         border: 1px solid var(--herc-border); }
.herc-badge--falsepositive { background: rgba(255,92,92,0.15); color: var(--herc-risk-strong);  border: 1px solid var(--herc-risk-strong); }
</style>
"""
 
 
def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
 
 
def badge_html(label: str, kind: str) -> str:
    """kind: 'high' | 'medium' | 'low' | 'candidate' | 'confirmed' | 'falsepositive'"""
    return f'<span class="herc-badge herc-badge--{kind}">{label}</span>'
 