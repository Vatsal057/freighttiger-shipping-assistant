"""Interactive Streamlit Dashboard for FreightTiger Smart Shipping Cost Assistant.

Designed for Round 1 Reviewers and Round 2 Live Walkthrough:
1. Executive KPI Summary & Live Reproducibility Hash.
2. Direct Match Verification against FreightTiger's Sample Output Rows.
3. Corridor Cost Trend Visualizer (Weekly Cost per Tonne-KM vs Trailing History & Peer Averages).
4. Full Anomaly Audit Table (Justified vs Unexplained Spikes).
5. Planted Trap & Context Note Inspector (Explaining Disqualifications).
6. Interactive Plain-English Assistant (Bonus Stretch Goal).

Styling follows Google's Material Design language: Roboto type, Google blue
primary, rounded surfaces, generous spacing, large touch targets, and text
labels that never rely on colour alone.
"""

from __future__ import annotations

import csv
import html
import re
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core_analysis import (
    load_shipments,
    aggregate_weekly_route_costs,
    compute_baselines,
)
from src.justification import load_context_notes, evaluate_route_week_justification
from src.pipeline import run_pipeline, write_submission_csv, answer_natural_language_query
from src.cli import compute_file_sha256

# Streamlit Page Config
st.set_page_config(
    page_title="FreightTiger | Smart Shipping Cost Assistant",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Material Design theme (Google UI language) ---
MATERIAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

html, body, [class*="css"] {
    font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
}
.stApp {
    background-color: #f8f9fa;
    color: #202124;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
p, li, .stMarkdown {
    line-height: 1.65;
    font-size: 1rem;
    color: #202124;
}
h1 {
    font-weight: 400 !important;
    font-size: 2rem !important;
    letter-spacing: -0.01em;
    color: #202124 !important;
    padding-bottom: 0.25rem;
}
h2, h3 {
    font-weight: 500 !important;
    color: #202124 !important;
}
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #5f6368 !important;
    font-size: 0.95rem !important;
}
/* KPI cards: white Material surfaces with outline and soft elevation */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(60,64,67,.08);
}
[data-testid="stMetricLabel"] p {
    color: #5f6368 !important;
    font-size: 0.8rem !important;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #202124 !important;
    font-size: 1.75rem !important;
    font-weight: 500;
}
/* Buttons: Material pill shape, 48px touch target */
.stButton > button, .stDownloadButton > button {
    border-radius: 999px !important;
    min-height: 48px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    font-size: 0.95rem;
    border: 1px solid #dadce0 !important;
    background-color: #ffffff !important;
    color: #1a73e8 !important;
    transition: box-shadow .15s ease, background-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #f0f6fe !important;
    box-shadow: 0 1px 3px rgba(60,64,67,.25);
}
.stButton > button[kind="primary"] {
    background-color: #1a73e8 !important;
    color: #ffffff !important;
    border-color: #1a73e8 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1765cc !important;
}
button:focus-visible, a:focus-visible, input:focus-visible,
[tabindex]:focus-visible {
    outline: 3px solid #a8c7fa !important;
    outline-offset: 2px;
}
/* Tabs: pill indicator like Google product UIs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid #dadce0;
    padding-bottom: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 10px 18px;
    min-height: 44px;
    background-color: #ffffff;
    border: 1px solid #dadce0;
    color: #3c4043;
    font-weight: 500;
    font-size: 0.95rem;
}
.stTabs [aria-selected="true"] {
    background-color: #e8f0fe !important;
    color: #1a73e8 !important;
    border-color: #a8c7fa !important;
}
/* Inputs: rounded, tall, clear focus ring */
.stTextInput input, .stSelectbox [data-baseweb="select"] > div {
    border-radius: 12px !important;
    min-height: 48px;
    border-color: #dadce0 !important;
    font-size: 1rem;
}
.stTextInput input:focus, .stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 1px #1a73e8;
}
.stTextInput label p, .stSelectbox label p, .stRadio label p {
    font-size: 1rem !important;
    font-weight: 500;
    color: #202124 !important;
}
/* Data tables: card surface with rounded corners */
[data-testid="stDataFrame"] {
    border: 1px solid #dadce0;
    border-radius: 12px;
    overflow: hidden;
    background-color: #ffffff;
}
/* Alerts: rounded Material containers */
[data-testid="stAlert"] {
    border-radius: 12px;
    font-size: 1rem;
}
hr {
    border-color: #e0e0e0;
}
footer { visibility: hidden; }
/* Google Material Symbols icon font */
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded';
    font-weight: normal;
    font-style: normal;
    font-size: 1.25em;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
    vertical-align: -0.16em;
    margin-right: 8px;
    color: #1a73e8;
}
/* Verdict badges: words first, colour as support */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 500;
    font-size: 0.95rem;
}
.badge-justified { background-color: #e6f4ea; color: #137333; border: 1px solid #a3d9b1; }
.badge-review { background-color: #fef7e0; color: #b06000; border: 1px solid #f5d67b; }
/* Assistant answer: one card per finding */
.qa-head { font-size: 1.05rem; font-weight: 500; color: #202124; margin: 4px 0 14px 0; }
.qa-card {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-left: 5px solid #1a73e8;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(60,64,67,.08);
}
.qa-card.justified { border-left-color: #188038; }
.qa-card.review { border-left-color: #e37400; }
.qa-week { font-weight: 700; font-size: 1.02rem; color: #202124; }
.qa-cost { font-weight: 500; color: #202124; }
.qa-meta { color: #5f6368; font-size: 0.92rem; margin-top: 2px; }
.qa-reason { color: #3c4043; margin-top: 8px; line-height: 1.6; }
.qa-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
.qa-notes {
    background-color: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 12px 18px;
    margin-top: 4px;
}
.qa-notes-title { font-weight: 500; color: #202124; margin-bottom: 6px; }
.qa-note { color: #3c4043; line-height: 1.6; margin-bottom: 6px; }
.qa-note b { color: #202124; }
.qa-prose {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 12px;
    padding: 16px 20px;
    line-height: 1.7;
    color: #202124;
    box-shadow: 0 1px 2px rgba(60,64,67,.08);
}
/* Calm theme overrides: muted blues, sage, and sand replace the vivid defaults */
.stApp { background-color: #f6f7f5; color: #2e3336; }
p, li, .stMarkdown { color: #2e3336; }
h1, h2, h3 { color: #2e3336 !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #6e757c !important; }
[data-testid="stMetric"] { border-color: #e0e3e1; box-shadow: 0 1px 2px rgba(60,64,67,.05); }
[data-testid="stMetricLabel"] p { color: #6e757c !important; }
[data-testid="stMetricValue"] { color: #2e3336 !important; }
.stButton > button, .stDownloadButton > button {
    border-color: #dde1df !important;
    color: #5e7f9d !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #eef3f6 !important;
    box-shadow: 0 1px 3px rgba(60,64,67,.15);
}
.stButton > button[kind="primary"] {
    background-color: #5e7f9d !important;
    border-color: #5e7f9d !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover { background-color: #4f6d87 !important; }
button:focus-visible, a:focus-visible, input:focus-visible, [tabindex]:focus-visible {
    outline-color: #b7c9d9 !important;
}
.stTextInput input:focus, .stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: #5e7f9d !important;
    box-shadow: 0 0 0 1px #5e7f9d;
}
.stTextInput label p, .stSelectbox label p, .stRadio label p { color: #2e3336 !important; }
.material-symbols-rounded { color: #5e7f9d; }
.badge-justified { background-color: #e9f1ea; color: #4c6b54; border-color: #bfd4c3; }
.badge-review { background-color: #f8f1de; color: #85703f; border-color: #e6d7ab; }
.qa-head { color: #2e3336; }
.qa-card { border-color: #e0e3e1; border-left-color: #5e7f9d; box-shadow: 0 1px 2px rgba(60,64,67,.05); }
.qa-card.justified { border-left-color: #7fa189; }
.qa-card.review { border-left-color: #c2a968; }
.qa-week, .qa-cost { color: #2e3336; }
.qa-meta { color: #6e757c; }
.qa-reason { color: #454b4f; }
.qa-notes { background-color: #f6f7f5; border-color: #e0e3e1; }
.qa-notes-title, .qa-note b { color: #2e3336; }
.qa-note { color: #454b4f; }
.qa-prose { border-color: #e0e3e1; color: #2e3336; box-shadow: 0 1px 2px rgba(60,64,67,.05); }
</style>
"""
st.markdown(MATERIAL_CSS, unsafe_allow_html=True)

DATA_PATH = Path("AI Intern Case Study/shipment_records.csv")
NOTES_PATH = Path("AI Intern Case Study/context_notes.csv")
SAMPLE_PATH = Path("AI Intern Case Study/sample_output_format_v2.csv")
OUTPUT_PATH = Path("submission_output.csv")

@st.cache_data
def get_pipeline_data():
    shipments = load_shipments(DATA_PATH)
    weekly = aggregate_weekly_route_costs(shipments)
    baselines = compute_baselines(weekly)
    out_df = run_pipeline(DATA_PATH, NOTES_PATH)
    notes = load_context_notes(NOTES_PATH)
    file_hash = compute_file_sha256(OUTPUT_PATH) if OUTPUT_PATH.exists() else "N/A"
    return shipments, weekly, baselines, out_df, notes, file_hash


shipments_df, weekly_df, baselines_df, out_df, notes, file_hash = get_pipeline_data()


def load_sample_rows() -> pd.DataFrame:
    """Read the sample output file tolerantly.

    The sample file's reason text can contain commas without quoting, which
    breaks a strict CSV parse. The first seven columns never contain commas,
    so any extra fields are merged back into the final reason column.
    """
    with open(SAMPLE_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], [r for r in rows[1:] if r]
    fixed = []
    for row in data:
        if len(row) > len(header):
            row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]
        fixed.append(row)
    return pd.DataFrame(fixed, columns=header)

_FINDING_RE = re.compile(
    r"Week (?P<week>\d{4}-\d{2}-\d{2}): cost was ₹(?P<cost>[\d.]+)/tonne-km "
    r"\((?P<hist>[^;]+); (?P<peer>[^)]+)\)\. Verdict: (?P<verdict>[^.]+)\. Reason: (?P<reason>.*)",
    re.S,
)


def render_answer(answer: str) -> None:
    """Render the assistant's answer as readable Material cards.

    The deterministic answer lists one finding per bullet. Each finding
    becomes its own card with the week, cost, comparisons, a verdict badge,
    and the reason as a separate paragraph. Free-form prose answers are
    shown in a single reading card.
    """
    esc = html.escape
    notes_split = answer.split("\n\nApplicable Context Notes:")
    main_text = notes_split[0].strip()
    notes_text = notes_split[1].strip() if len(notes_split) > 1 else ""

    if "\u2022" not in main_text:
        paras = "".join(f"<p>{esc(p)}</p>" for p in main_text.split("\n\n") if p.strip())
        st.markdown(f"<div class='qa-prose'>{paras}</div>", unsafe_allow_html=True)
        return

    blocks = [b.strip() for b in main_text.split("\u2022") if b.strip()]
    head, items = blocks[0], blocks[1:]
    out = [f"<div class='qa-head'>{esc(head)}</div>"]
    for item in items:
        m = _FINDING_RE.match(item)
        if not m:
            item_html = esc(item).replace("\n", "<br>")
            out.append(f"<div class='qa-card'><div class='qa-reason'>{item_html}</div></div>")
            continue
        justified = m.group("verdict").strip() == "No (justified)"
        card_cls = "justified" if justified else "review"
        badge_cls = "badge-justified" if justified else "badge-review"
        badge_text = "No (justified)" if justified else "Yes, flagged for review"
        reason = " ".join(m.group("reason").split())
        out.append(
            f"<div class='qa-card {card_cls}'>"
            f"<div class='qa-row'><span class='qa-week'>Week of {esc(m.group('week'))}</span>"
            f"<span class='badge {badge_cls}'>{badge_text}</span></div>"
            f"<div class='qa-cost'>₹{esc(m.group('cost'))} per tonne-km</div>"
            f"<div class='qa-meta'>{esc(m.group('hist'))} &middot; {esc(m.group('peer'))}</div>"
            f"<div class='qa-reason'>{esc(reason)}</div>"
            f"</div>"
        )
    if notes_text and "No corridor-specific" not in notes_text:
        note_rows = []
        parts = re.split(r"\[(N\d+) \((\d{4}-\d{2}-\d{2})\)\]: ", notes_text)
        # parts: [prefix, id, date, text, id, date, text, ...]
        for i in range(1, len(parts) - 2, 3):
            nid, ndate, ntext = parts[i], parts[i + 1], " ".join(parts[i + 2].split())
            note_rows.append(
                f"<div class='qa-note'><b>{esc(nid)}</b> ({esc(ndate)}): {esc(ntext)}</div>"
            )
        if note_rows:
            out.append(
                "<div class='qa-notes'><div class='qa-notes-title'>Applicable context notes</div>"
                + "".join(note_rows)
                + "</div>"
            )
    st.markdown("".join(out), unsafe_allow_html=True)


# Header & Branding
st.title("FreightTiger: Smart Shipping Cost Assistant")
st.markdown(
    "This dashboard presents the case study solution for review. The pipeline groups 2,940 shipments "
    "into weekly cost per tonne-km for 7 corridors, compares each week against the route's own trailing "
    "8 weeks and against similar routes in the same week, then checks the context notes before calling "
    "any rise unexplained. **Start with Sample check** to watch the core logic reproduce your sample "
    "rows exactly."
)

# Top Metric Cards
justified_count = int((out_df["flagged"] == "No (justified)").sum())
unexplained_count = int((out_df["flagged"] == "Yes").sum())
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Shipments", f"{len(shipments_df):,}")
with col2:
    st.metric("Route-Weeks Analyzed", f"{len(baselines_df):,}")
with col3:
    st.metric("Flagged Route-Weeks", len(out_df))
with col4:
    st.metric("Justified (Note-Backed)", justified_count)
with col5:
    st.metric("Unexplained (Needs Review)", unexplained_count)

st.markdown("---")

# Navigation: Material pill tabs built from buttons (visible on every
# Streamlit version, unlike the built-in tab strip which themes wash out)
TABS = [
    "Sample check",
    "Cost trends",
    "Audit table",
    "Notes and traps",
    "Ask a question",
    "Reproducibility",
]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = TABS[0]
nav_cols = st.columns(len(TABS))
for _col, _label in zip(nav_cols, TABS):
    with _col:
        if st.button(
            _label,
            key=f"nav_{_label}",
            type="primary" if st.session_state.active_tab == _label else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_tab = _label
            st.rerun()
active_tab = st.session_state.active_tab

# Tab 1: Sample Verification
if active_tab == "Sample check":
    st.markdown("## <span class='material-symbols-rounded'>fact_check</span> Direct match against FreightTiger's sample output", unsafe_allow_html=True)
    st.write(
        "Your sample file contains 3 rows. The pipeline reproduces all three exactly: the cost, "
        "both comparison percentages, the verdict, and the cited note. Each card shows the computed "
        "value with the sample target underneath it."
    )

    sample_df = load_sample_rows()

    for _, s_row in sample_df.iterrows():
        route = str(s_row["route"])
        week = str(s_row["week_of"])
        exp_cost = f"{float(s_row['cost_per_tonne_km']):.2f}"
        exp_hist = str(s_row["vs_own_history"])
        exp_peer = str(s_row["vs_similar_routes"])

        match = out_df[(out_df["route"] == route) & (out_df["week_of"] == week)]
        with st.container():
            st.markdown(f"#### Corridor: {route} | Week of {week}")
            if len(match) == 1:
                r = match.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Cost per tonne-km (₹)", f"{r['cost_per_tonne_km']}", delta=None)
                c1.caption(f"Sample target: ₹{exp_cost}")
                c2.metric("vs own history", str(r["vs_own_history"]).split("%")[0] + "%")
                c2.caption(f"Sample target: {exp_hist}")
                c3.metric("vs similar routes", str(r["vs_similar_routes"]).split("%")[0] + "%")
                c3.caption(f"Sample target: {exp_peer}")
                if r["flagged"] == "No (justified)":
                    badge = "badge-justified"
                    verdict_text = "No (justified)"
                else:
                    badge = "badge-review"
                    verdict_text = "Yes, flagged for human review"
                note_text = r["matched_note_id"] if r["matched_note_id"] else "none"
                st.markdown(
                    f"**Verdict:** <span class='badge {badge}'>{verdict_text}</span>"
                    f" &nbsp; **Matched note:** {note_text}",
                    unsafe_allow_html=True,
                )
                st.info(f"Reason recorded in the output: {r['reason']}")
            else:
                st.error("Row missing from pipeline output.")
            st.divider()

# Tab 2: Corridor Cost Trajectory
if active_tab == "Cost trends":
    st.markdown("## <span class='material-symbols-rounded'>show_chart</span> Corridor weekly cost per tonne-km trends", unsafe_allow_html=True)
    st.write(
        "Each chart shows three lines for one corridor: the actual weekly cost, the trailing 8-week "
        "average (own-history baseline), and the same-week average of similar routes (peer baseline). "
        "A week is flagged when the actual line pulls too far away from either baseline. Try "
        "Mumbai-Pune to see slow rate creep that the peer baseline catches, or Chennai-Bangalore "
        "for the flood weeks."
    )
    selected_route = st.selectbox(
        "Select a corridor to inspect:",
        options=sorted(weekly_df["route"].unique()),
        index=0,
        help="The chart shows the corridor's actual weekly cost against its trailing 8-week history and the peer average.",
    )

    route_data = baselines_df[baselines_df["route"] == selected_route].sort_values("week_of")
    route_spikes = out_df[out_df["route"] == selected_route]

    st.write(f"Showing 104 weeks of shipping rates for **{selected_route}** ({route_data['route_type'].iloc[0]} corridor):")

    chart_df = route_data[["week_of", "cost_per_tonne_km", "hist_mean", "peer_mean"]].copy()
    chart_df.rename(
        columns={
            "cost_per_tonne_km": "Actual cost per tonne-km (₹)",
            "hist_mean": "Trailing 8-week history mean (₹)",
            "peer_mean": "Peer routes average (₹)",
        },
        inplace=True,
    )
    chart_df.set_index("week_of", inplace=True)
    st.line_chart(chart_df)

    if len(route_spikes) > 0:
        st.write(f"**Weeks where {selected_route} appears in the flagged output:**")
        st.dataframe(route_spikes[["week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id", "reason"]], use_container_width=True, hide_index=True)
    else:
        st.success(f"No cost spikes or anomalies detected for {selected_route} across the 104 weeks.")

# Tab 3: Full Anomaly Ledger
if active_tab == "Audit table":
    st.markdown("## <span class='material-symbols-rounded'>receipt_long</span> Full shipping cost anomaly audit table", unsafe_allow_html=True)
    st.write(
        "Every flagged route-week, exactly as written to `submission_output.csv`. Justified rows "
        "name the note that explains the rise; unexplained rows are the ones a human should review. "
        "The download button below gives you the file itself."
    )
    filter_choice = st.radio(
        "Filter by verdict:",
        options=[
            f"All ({len(out_df)})",
            f"Justified only ({justified_count})",
            f"Unexplained / human review ({unexplained_count})",
        ],
        horizontal=True,
    )

    if "Justified only" in filter_choice:
        display_df = out_df[out_df["flagged"] == "No (justified)"]
    elif "Unexplained" in filter_choice:
        display_df = out_df[out_df["flagged"] == "Yes"]
    else:
        display_df = out_df

    st.dataframe(display_df, use_container_width=True, height=450, hide_index=True)
    st.download_button(
        "Download submission_output.csv",
        data=out_df.to_csv(index=False),
        file_name="submission_output.csv",
        mime="text/csv",
    )

# Tab 4: Planted Trap Inspector
if active_tab == "Notes and traps":
    st.markdown("## <span class='material-symbols-rounded'>shield</span> Context notes and guardrail analysis", unsafe_allow_html=True)
    st.write(
        "All 10 context notes, classified by reading their own wording. Only 3 describe a real "
        "disruption. The other 7 say in their own words that costs were not affected, conditions "
        "were normal, or the note does not apply to this dataset, so they can never justify a flag. "
        "A valid note also counts only inside its active date window, shown in the table."
    )

    notes_records = []
    for n in notes:
        status = "Valid disruption" if n.is_valid_disruption else "Disqualified: not a disruption"
        notes_records.append(
            {
                "Note ID": n.note_id,
                "Date": n.date_str,
                "Applies to": n.applies_to,
                "Guardrail status": status,
                "Active date range": f"{n.active_start_date} to {n.active_end_date}",
                "Note text": n.note_text,
            }
        )
    st.dataframe(pd.DataFrame(notes_records), use_container_width=True, hide_index=True)

# Tab 5: Interactive Assistant (Q&A)
if active_tab == "Ask a question":
    st.markdown("## <span class='material-symbols-rounded'>chat</span> Ask the assistant in plain English (stretch goal)", unsafe_allow_html=True)
    st.write(
        "The stretch goal from the brief: ask about the findings in plain English. Answers are "
        "built only from the audit table and the context notes. A language model may rephrase an "
        "answer, but it cannot change a verdict, a number, or a cited note."
    )
    st.write("**Try one of these, or type your own below:**")
    examples = [
        "Why did Chennai-Bangalore get pricier in March 2025?",
        "Why does Mumbai-Pune keep getting flagged?",
        "Which route is the worst offender?",
    ]
    ex_cols = st.columns(len(examples))
    for _c, _q in zip(ex_cols, examples):
        with _c:
            if st.button(_q, key=f"ex_{_q}", use_container_width=True):
                st.session_state["qa_query"] = _q
                st.session_state["qa_run"] = True
                st.rerun()
    if "qa_query" not in st.session_state:
        st.session_state["qa_query"] = "Why did Chennai-Bangalore get pricier in March 2025?"
    user_query = st.text_input(
        "Your question:",
        key="qa_query",
        help="Ask about one corridor and a month, or about the whole network.",
    )
    ask_clicked = st.button("Ask assistant", type="primary")
    if ask_clicked or st.session_state.pop("qa_run", False):
        with st.spinner("Checking the audit table and context notes..."):
            answer = answer_natural_language_query(user_query, out_df, notes)
            st.markdown("### Assistant response")
            render_answer(answer)

# Tab 6: 3-Run Reproducibility Proof
if active_tab == "Reproducibility":
    st.markdown("## <span class='material-symbols-rounded'>verified</span> 3-run SHA-256 reproducibility verification", unsafe_allow_html=True)
    st.write(
        "The brief requires three untouched runs on the same input to produce identical flags "
        "and numbers. The hash of the submitted file is shown below. The button reruns the whole "
        "pipeline three times, live, and compares the hashes of what it produces."
    )
    st.markdown(f"**Canonical submission SHA-256 hash:** `{file_hash}`")

    if st.button("Run 3 consecutive live passes"):
        with st.spinner("Executing 3 untouched runs..."):
            hashes = []
            for i in range(1, 4):
                temp_df = run_pipeline(DATA_PATH, NOTES_PATH)
                run_file = Path(f"output/run_{i}_output.csv")
                write_submission_csv(temp_df, run_file)
                h = compute_file_sha256(run_file)
                hashes.append(h)
                st.write(f"Run {i}: SHA-256 = `{h}`")

            if hashes[0] == hashes[1] == hashes[2]:
                st.success("Success: all 3 runs produced byte-identical SHA-256 hashes.")
            else:
                st.error("Hashes diverged across runs.")
