import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# HERCULES — COMMAND CENTER V1
# UI rebuild around the existing V4.2 rankings
# ============================================================

st.set_page_config(
    page_title="HERCULES — Scientific Priority Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

RANKING_FILES = [
    OUTPUTS / "hercules_v4_2_rankings.csv",
    OUTPUTS / "hercules_candidate_rankings.csv",
]


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #050912;
    --bg2: #08101d;
    --panel: #0a1220;
    --panel2: #0d1726;
    --border: rgba(145, 170, 200, 0.13);
    --text: #edf5ff;
    --muted: #718198;
    --cyan: #53e6d1;
    --cyan-soft: rgba(83, 230, 209, 0.12);
    --amber: #f5b94c;
    --red: #ff6b6b;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 78% 8%,
            rgba(50, 180, 190, 0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 12% 35%,
            rgba(45, 95, 160, 0.06),
            transparent 30%
        ),
        #050912;
    color: var(--text);
}

/* subtle scientific grid */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.18;
    background-image:
        linear-gradient(
            rgba(100, 140, 170, 0.035) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(100, 140, 170, 0.035) 1px,
            transparent 1px
        );
    background-size: 56px 56px;
    mask-image: linear-gradient(
        to bottom,
        black,
        transparent 85%
    );
}

/* remove Streamlit chrome */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* sidebar */

section[data-testid="stSidebar"] {
    background: #060c16;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

.sidebar-brand {
    padding: 0 0.4rem 1.5rem 0.4rem;
}

.sidebar-logo {
    font-family: 'DM Mono', monospace;
    font-size: 1.15rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    color: var(--text);
}

.sidebar-sub {
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.13em;
    color: var(--muted);
}

.nav-label {
    margin: 1.2rem 0 0.5rem 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    color: #536276;
}

/* main width */

.block-container {
    max-width: 1450px;
    padding-top: 2.2rem;
    padding-bottom: 5rem;
}

/* eyebrow */

.eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.67rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.9rem;
}

/* hero */

.hero {
    position: relative;
    overflow: hidden;
    min-height: 330px;
    padding: 3.1rem 3.2rem;
    border: 1px solid var(--border);
    border-radius: 24px;
    background:
        radial-gradient(
            circle at 86% 42%,
            rgba(83, 230, 209, 0.10),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            rgba(13, 23, 38, 0.96),
            rgba(5, 9, 18, 0.92)
        );
    box-shadow:
        0 30px 90px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.025);
}

.hero::after {
    content: "";
    position: absolute;
    width: 390px;
    height: 390px;
    right: -90px;
    top: -130px;
    border: 1px solid rgba(83, 230, 209, 0.12);
    border-radius: 50%;
    box-shadow:
        0 0 0 45px rgba(83,230,209,0.018),
        0 0 0 95px rgba(83,230,209,0.012);
}

.hero-title {
    position: relative;
    z-index: 2;
    max-width: 800px;
    font-size: clamp(2.7rem, 5vw, 5.2rem);
    line-height: 0.98;
    font-weight: 800;
    letter-spacing: -0.055em;
    color: var(--text);
}

.hero-title span {
    color: var(--cyan);
}

.hero-copy {
    position: relative;
    z-index: 2;
    max-width: 700px;
    margin-top: 1.4rem;
    color: #8795aa;
    line-height: 1.8;
    font-size: 0.98rem;
}

.status-row {
    position: relative;
    z-index: 3;
    display: flex;
    gap: 0.55rem;
    margin-top: 1.7rem;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.42rem 0.72rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.025);
    color: #9aacbf;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 12px var(--cyan);
}

/* section */

.section {
    margin-top: 2.5rem;
}

.section-head {
    display: flex;
    justify-content: space-between;
    align-items: end;
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text);
}

.section-desc {
    margin-top: 0.25rem;
    color: var(--muted);
    font-size: 0.78rem;
}

/* stats */

.stat-card {
    min-height: 145px;
    padding: 1.25rem 1.35rem;
    border-radius: 17px;
    border: 1px solid var(--border);
    background: linear-gradient(
        145deg,
        rgba(12, 22, 37, 0.95),
        rgba(7, 13, 23, 0.95)
    );
}

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.59rem;
    letter-spacing: 0.15em;
    color: #63738a;
    text-transform: uppercase;
}

.stat-value {
    margin-top: 1rem;
    font-size: 2.25rem;
    line-height: 1;
    font-weight: 700;
    letter-spacing: -0.045em;
    color: var(--text);
}

.stat-note {
    margin-top: 0.65rem;
    color: #627187;
    font-size: 0.72rem;
}

.stat-card.primary {
    border-color: rgba(83,230,209,0.28);
    background:
        radial-gradient(
            circle at 90% 0%,
            rgba(83,230,209,0.11),
            transparent 42%
        ),
        linear-gradient(
            145deg,
            rgba(11, 31, 35, 0.95),
            rgba(7, 16, 24, 0.95)
        );
}

.stat-card.primary .stat-value {
    color: var(--cyan);
}

/* pipeline */

.pipeline {
    display: flex;
    align-items: stretch;
    gap: 0.65rem;
    margin-top: 1.4rem;
}

.pipeline-node {
    position: relative;
    flex: 1;
    min-width: 0;
    padding: 1.25rem 1.1rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(9,17,29,0.88);
}

.pipeline-node:last-child {
    border-color: rgba(83,230,209,0.28);
    background:
        radial-gradient(
            circle at 100% 0%,
            rgba(83,230,209,0.13),
            transparent 45%
        ),
        rgba(9,24,28,0.92);
}

.pipeline-kicker {
    font-family: 'DM Mono', monospace;
    color: #65758a;
    font-size: 0.55rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.pipeline-number {
    margin-top: 0.55rem;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.04em;
}

.pipeline-name {
    margin-top: 0.35rem;
    font-size: 0.72rem;
    color: #7f8da0;
}

.pipeline-arrow {
    display: flex;
    align-items: center;
    color: #405166;
    font-size: 1rem;
}

/* target */

.target-card {
    padding: 1.6rem;
    border: 1px solid rgba(83,230,209,0.20);
    border-radius: 19px;
    background:
        radial-gradient(
            circle at 100% 0%,
            rgba(83,230,209,0.08),
            transparent 40%
        ),
        #09131f;
}

.target-kicker {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.17em;
    color: var(--cyan);
}

.target-name {
    margin-top: 0.55rem;
    font-size: 1.8rem;
    font-weight: 750;
    letter-spacing: -0.04em;
}

.target-id {
    margin-top: 0.25rem;
    color: #65758a;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
}

.target-score {
    margin-top: 1.4rem;
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -0.06em;
    color: var(--cyan);
}

.target-score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    color: #63738a;
}

/* disclaimer */

.disclaimer {
    margin-top: 2rem;
    padding: 1rem 1.15rem;
    border-left: 2px solid rgba(245,185,76,0.55);
    background: rgba(245,185,76,0.035);
    color: #8290a3;
    font-size: 0.75rem;
    line-height: 1.65;
}

/* buttons */

.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #0b1523;
    color: #b8c5d5;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: rgba(83,230,209,0.35);
    color: var(--cyan);
}

/* selectbox / inputs */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #0a1422;
    border-color: var(--border);
    border-radius: 10px;
}

label {
    color: #75859a !important;
}

/* responsive */

@media (max-width: 900px) {
    .hero {
        padding: 2rem;
    }

    .pipeline {
        flex-direction: column;
    }

    .pipeline-arrow {
        justify-content: center;
        transform: rotate(90deg);
    }
}

</style>
""",
    unsafe_allow_html=True,
)
# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_rankings():
    for file in RANKING_FILES:
        if file.exists():
            return pd.read_csv(file), file.name

    return None, None


rankings, ranking_filename = load_rankings()

# ============================================================
# SAFE COLUMN HELPERS
# ============================================================

def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def numeric_sum(df, col, value):
    if not col:
        return 0

    return int(
        pd.to_numeric(df[col], errors="coerce")
        .eq(value)
        .sum()
    )

# ============================================================
# CALCULATE SUMMARY
# ============================================================

if rankings is not None and not rankings.empty:

    total_objects = len(rankings)

    candidate_col = find_col(
        rankings,
        [
            "hercules_prediction",
            "prediction",
            "koi_disposition",
        ],
    )

    priority_col = find_col(
        rankings,
        [
            "priority",
            "scientific_priority",
        ],
    )

    score_col = find_col(
        rankings,
        [
            "scientific_priority_score",
            "hercules_score",
        ],
    )

    kepoi_col = find_col(
        rankings,
        [
            "kepoi_name",
            "kepoi",
            "kepler_name",
        ],
    )

    kepid_col = find_col(
        rankings,
        [
            "kepid",
            "kepid_name",
        ],
    )

    # candidate predictions
    candidate_predictions = 0

    if candidate_col:
        candidate_predictions = int(
            rankings[candidate_col]
            .astype(str)
            .str.upper()
            .eq("CANDIDATE")
            .sum()
        )

    # priorities
    high_priority = 0
    medium_priority = 0

    if priority_col:
        priority_values = (
            rankings[priority_col]
            .astype(str)
            .str.upper()
        )

        high_priority = int(
            priority_values.eq("HIGH").sum()
        )

        medium_priority = int(
            priority_values.eq("MEDIUM").sum()
        )

    # top target
    top_target = "—"
    top_score = np.nan
    top_kepid = "—"

    if score_col:
        scores = pd.to_numeric(
            rankings[score_col],
            errors="coerce",
        )

        valid = rankings.loc[scores.notna()].copy()

        if not valid.empty:
            valid["_score"] = pd.to_numeric(
                valid[score_col],
                errors="coerce",
            )

            target_row = valid.sort_values(
                "_score",
                ascending=False,
            ).iloc[0]

            top_score = target_row["_score"]

            if kepoi_col:
                top_target = str(
                    target_row[kepoi_col]
                )

            if kepid_col:
                top_kepid = str(
                    target_row[kepid_col]
                )

else:
    total_objects = 9564
    candidate_predictions = 1957
    high_priority = 50
    medium_priority = 921
    top_target = "K07259.01"
    top_score = 95.17
    top_kepid = "9944201"

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">HERCULES</div>
            <div class="sidebar-sub">
                SCIENTIFIC PRIORITY ENGINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-label">NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Target Explorer",
            "Candidate Hunter",
            "Scientific Analytics",
            "About HERCULES",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            font-family:'DM Mono';
            font-size:0.58rem;
            letter-spacing:0.12em;
            color:#536276;
        ">
        SYSTEM
        </div>

        <div style="
            margin-top:0.5rem;
            color:#8795aa;
            font-size:0.72rem;
        ">
        V4.2 SCIENTIFIC PRIORITY ENGINE
        </div>

        <div style="
            margin-top:0.45rem;
            color:#59697e;
            font-size:0.68rem;
        ">
        NASA / KEPLER
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                ◈ HERCULES / COMMAND CENTER
            </div>

            <div class="hero-title">
                From raw signal<br>
                to <span>scientific priority.</span>
            </div>

            <div class="hero-copy">
                HERCULES evaluates astronomical observations,
                applies machine-learning classification, and
                ranks observations according to their potential
                value for scientific follow-up.
            </div>

            <div class="status-row">
                <div class="status">
                    <span class="status-dot"></span>
                    SYSTEM ONLINE
                </div>

                <div class="status">
                    V4.2 ENGINE
                </div>

                <div class="status">
                    NASA / KEPLER
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # STAT CARDS
    # ========================================================

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    Observations analyzed
                </div>
                <div class="stat-value">
                    {total_objects:,}
                </div>
                <div class="stat-note">
                    NASA / Kepler observations
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    Candidate predictions
                </div>
                <div class="stat-value">
                    {candidate_predictions:,}
                </div>
                <div class="stat-note">
                    Classified by HERCULES
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">
                    Medium priority
                </div>
                <div class="stat-value">
                    {medium_priority:,}
                </div>
                <div class="stat-note">
                    Investigation queue
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="stat-card primary">
                <div class="stat-label">
                    High priority
                </div>
                <div class="stat-value">
                    {high}
                </div>
                <div class="stat-note">
                    Highest scientific priority
                </div>
            </div>
            """.format(high=high_priority),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

   # ========================================================
    # PIPELINE
    # ========================================================

    st.markdown(
        """
        <div class="section">

            <div class="section-head">
                <div>
                    <div class="eyebrow">
                        DECISION PIPELINE
                    </div>

                    <div class="section-title">
                        From observation to investigation
                    </div>

                    <div class="section-desc">
                        HERCULES progressively reduces a large
                        observation set into a focused scientific queue.
                    </div>
                </div>
            </div>

            <div class="pipeline">

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        01 / OBSERVE
                    </div>
                    <div class="pipeline-number">
                        {observations:,}
                    </div>
                    <div class="pipeline-name">
                        astronomical observations
                    </div>
                </div>

                <div class="pipeline-arrow">→</div>

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        02 / CLASSIFY
                    </div>
                    <div class="pipeline-number">
                        {candidates:,}
                    </div>
                    <div class="pipeline-name">
                        candidate predictions
                    </div>
                </div>

                <div class="pipeline-arrow">→</div>

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        03 / RANK
                    </div>
                    <div class="pipeline-number">
                        {medium:,}
                    </div>
                    <div class="pipeline-name">
                        medium priority
                    </div>
                </div>

                <div class="pipeline-arrow">→</div>

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        04 / PRIORITIZE
                    </div>
                    <div class="pipeline-number">
                        {high:,}
                    </div>
                    <div class="pipeline-name">
                        high priority
                    </div>
                </div>

                <div class="pipeline-arrow">→</div>

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        05 / INVESTIGATE
                    </div>
                    <div class="pipeline-number">
                        {target}
                    </div>
                    <div class="pipeline-name">
                        top scientific target
                    </div>
                </div>

            </div>

        </div>
        """.format(
            observations=total_objects,
            candidates=candidate_predictions,
            medium=medium_priority,
            high=high_priority,
            target=top_target,
        ),
        unsafe_allow_html=True,
    )

  # ========================================================
    # TOP TARGET
    # ========================================================

    st.markdown(
        '<div class="section">',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1])

    with left:

        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="eyebrow">
                        CURRENT LEADING TARGET
                    </div>

                    <div class="section-title">
                        Highest scientific priority
                    </div>

                    <div class="section-desc">
                        The observation currently ranked highest
                        by the Scientific Priority Engine.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_text = (
            f"{top_score:.2f}"
            if pd.notna(top_score)
            else "—"
        )

        st.markdown(
            f"""
            <div class="target-card">

                <div class="target-kicker">
                    HERCULES TARGET
                </div>

                <div class="target-name">
                    {top_target}
                </div>

                <div class="target-id">
                    KEPID: {top_kepid}
                </div>

                <div class="target-score-label">
                    SCIENTIFIC PRIORITY SCORE
                </div>

                <div class="target-score">
                    {score_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class="section-head">
                <div>
                    <div class="eyebrow">
                        SYSTEM INTERPRETATION
                    </div>

                    <div class="section-title">
                        What HERCULES does
                    </div>
                </div>
            </div>

            <div style="
                padding:1.5rem;
                border:1px solid rgba(145,170,200,0.13);
                border-radius:19px;
                background:#09131f;
                color:#8795aa;
                line-height:1.8;
                font-size:0.82rem;
            ">
                HERCULES combines machine-learning classification
                with measurable properties of the astronomical
                observation to create a structured investigation
                queue.

                <br><br>

                The system helps researchers decide <b style="
                    color:#edf5ff;
                ">where to look first</b>.

                <br><br>

                It does not determine planetary existence.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="disclaimer">
            <b style="color:#f5b94c;">
                SCIENTIFIC NOTE
            </b>
            <br>
            Scientific Priority Score is a machine-learning
            prioritization metric. It is not a probability of
            planetary existence and does not replace astronomical
            validation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OTHER PAGES — TEMPORARY CLEAN PLACEHOLDERS
# ============================================================

elif page == "Target Explorer":

    st.markdown(
        """
        <div class="eyebrow">TARGET EXPLORER</div>

        <div class="hero-title" style="font-size:3rem;">
            Inspect a single <span>observation.</span>
        </div>

        <div class="hero-copy">
            Target Explorer is the next interface we will rebuild
            around the V4.2 scientific ranking data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rankings is not None:

        search_col = find_col(
            rankings,
            ["kepoi_name", "kepid"],
        )

        if search_col:

            target = st.selectbox(
                "Select target",
                rankings[search_col]
                .dropna()
                .astype(str)
                .tolist(),
            )

            row = rankings[
                rankings[search_col].astype(str) == target
            ].iloc[0]

            st.json(
                row.to_dict(),
                expanded=False,
            )


elif page == "Candidate Hunter":

    st.markdown(
        """
        <div class="eyebrow">CANDIDATE HUNTER</div>

        <div class="hero-title" style="font-size:3rem;">
            Find the observations worth <span>investigating.</span>
        </div>

        <div class="hero-copy">
            The Candidate Hunter interface will be rebuilt next
            with a scientific ranking table, semantic priority
            badges and investigation controls.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rankings is not None:
        st.dataframe(
            rankings.head(100),
            width="stretch",
            hide_index=True,
        )


elif page == "Scientific Analytics":

    st.markdown(
        """
        <div class="eyebrow">SCIENTIFIC ANALYTICS</div>

        <div class="hero-title" style="font-size:3rem;">
            Understand the <span>search space.</span>
        </div>

        <div class="hero-copy">
            Scientific Analytics will contain the dark themed
            analytical visualizations in the next build stage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rankings is not None:
        st.dataframe(
            rankings.describe(include="all"),
            width="stretch",
        )


elif page == "About HERCULES":

    st.markdown(
        """
        <div class="eyebrow">ABOUT HERCULES</div>

        <div class="hero-title" style="font-size:3rem;">
            An intelligence layer for <span>astronomical triage.</span>
        </div>

        <div class="hero-copy">
            HERCULES evaluates astronomical observations using
            machine-learning classification and scientific signal
            information to prioritize observations for potential
            human follow-up.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section">

            <div class="pipeline">

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        01
                    </div>
                    <div class="pipeline-number">
                        OBSERVE
                    </div>
                    <div class="pipeline-name">
                        ingest astronomical observations
                    </div>
                </div>

                <div class="pipeline-arrow">→</div>

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        02
                    </div>
                    <div class="pipeline-number">
                        CLASSIFY
                    </div>
                    <div class="pipeline-name">
                        evaluate candidate likelihood
                    </div>
                </div>

                <div class="pipeline-arrow">→</div>

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        03
                    </div>
                    <div class="pipeline-number">
                        RANK
                    </div>
                    <div class="pipeline-name">
                        calculate scientific priority
                    </div>
                </div>

                <div class="pipeline-arrow">→</div>

                <div class="pipeline-node">
                    <div class="pipeline-kicker">
                        04
                    </div>
                    <div class="pipeline-number">
                        INVESTIGATE
                    </div>
                    <div class="pipeline-name">
                        focus human attention
                    </div>
                </div>

            </div>

        </div>

        <div class="disclaimer">
            <b style="color:#f5b94c;">
                IMPORTANT
            </b>
            <br>
            HERCULES produces prioritization metrics. It does not
            claim that an observation is a confirmed exoplanet.
            Astronomical validation remains necessary.
        </div>
        """,
        unsafe_allow_html=True,   
   )

