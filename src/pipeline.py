"""Pipeline orchestrating core analysis, justification, and CSV reporting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List
import pandas as pd

from src.core_analysis import (
    load_shipments,
    aggregate_weekly_route_costs,
    compute_baselines,
    format_percentage_diff,
    identify_candidate_anomalies,
)
from src.justification import (
    load_context_notes,
    evaluate_route_week_justification,
    ContextNote,
)


def run_pipeline(
    shipments_path: Path | str,
    notes_path: Path | str,
    hist_threshold_pct: float = 15.0,
    peer_threshold_pct: float = 20.0,
) -> pd.DataFrame:
    """Execute end-to-end analysis and return final report DataFrame.

    Columns strictly match sample_output_format_v2.csv:
    route,week_of,cost_per_tonne_km,vs_own_history,vs_similar_routes,flagged,matched_note_id,reason
    """
    shipments = load_shipments(shipments_path)
    weekly = aggregate_weekly_route_costs(shipments)
    baselines = compute_baselines(weekly, trailing_weeks=8)
    candidates = identify_candidate_anomalies(
        baselines,
        hist_threshold_pct=hist_threshold_pct,
        peer_threshold_pct=peer_threshold_pct,
    )
    notes = load_context_notes(notes_path)

    output_rows: List[Dict[str, str]] = []
    for _, row in candidates.iterrows():
        route = str(row["route"])
        week_of = str(row["week_of"])
        cost = f"{row['cost_per_tonne_km']:.2f}"
        vs_hist = format_percentage_diff(row["vs_hist_pct"], "history")
        vs_peer = format_percentage_diff(row["vs_peer_pct"], "peer")

        flagged, matched_note_id, reason = evaluate_route_week_justification(
            route, week_of, notes
        )

        output_rows.append(
            {
                "route": route,
                "week_of": week_of,
                "cost_per_tonne_km": cost,
                "vs_own_history": vs_hist,
                "vs_similar_routes": vs_peer,
                "flagged": flagged,
                "matched_note_id": matched_note_id,
                "reason": reason,
            }
        )

    out_df = pd.DataFrame(output_rows)
    # Deterministic sorting: by week_of ascending, then route ascending
    return out_df.sort_values(["week_of", "route"]).reset_index(drop=True)


def write_submission_csv(df: pd.DataFrame, target_path: Path | str) -> None:
    """Write output dataframe to CSV with quoting ensuring byte stability."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Use standard csv writer with minimal quoting to guarantee exact formatting
    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(df.columns.tolist())
        for _, row in df.iterrows():
            writer.writerow(row.tolist())


def answer_natural_language_query(
    query: str,
    pipeline_df: pd.DataFrame,
    notes: List[ContextNote],
) -> str:
    """Stretch goal: Answer questions about shipping costs in plain English."""
    query_lower = query.lower()

    # Search by route
    matched_routes = [r for r in pipeline_df["route"].unique() if r.lower() in query_lower]
    if not matched_routes:
        # Check individual city mentions
        for r in pipeline_df["route"].unique():
            origin, dest = r.lower().split("-")
            if origin in query_lower and dest in query_lower:
                matched_routes.append(r)

    if not matched_routes:
        return (
            f"Query did not specify a known corridor (available corridors: "
            f"{', '.join(pipeline_df['route'].unique())})."
        )

    route = matched_routes[0]
    route_rows = pipeline_df[pipeline_df["route"] == route]

    # Filter by month if mentioned
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    month_match = None
    for m_name, m_num in months.items():
        if m_name in query_lower:
            month_match = f"-{m_num}-"
            break

    if month_match:
        filtered = route_rows[route_rows["week_of"].str.contains(month_match)]
    else:
        filtered = route_rows

    if len(filtered) == 0:
        return f"No cost anomalies or flags recorded for {route} during the specified period."

    answers = []
    for _, row in filtered.iterrows():
        answers.append(
            f"• Week {row['week_of']}: cost was ₹{row['cost_per_tonne_km']}/tonne-km "
            f"({row['vs_own_history']}; {row['vs_similar_routes']}). "
            f"Verdict: {row['flagged']}. Reason: {row['reason']}"
        )

    base_summary = f"Assistant findings for {route}:\n" + "\n".join(answers)

    # Use AdaptiveLLMClient to synthesize answer if external key or proxy is enabled
    from src.llm_client import AdaptiveLLMClient
    client = AdaptiveLLMClient()
    system_prompt = (
        "You are an expert shipping assistant for FreightTiger. "
        "Answer the user's question directly and concisely based solely on the provided audit findings. "
        "Do not invent facts or cite notes not present in the findings."
    )
    user_prompt = f"User Question: {query}\n\nAudit Findings:\n{base_summary}\n\nPlease provide a clear, conversational answer."
    response = client.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        deterministic_fallback_text=base_summary,
    )
    return response.text

