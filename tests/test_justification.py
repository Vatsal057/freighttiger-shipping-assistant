"""Unit and guardrail tests for the Context Note Justification and Retrieval layers."""

from pathlib import Path
import pytest
import pandas as pd

from src.justification import (
    load_context_notes,
    evaluate_route_week_justification,
    ContextNote,
    parse_context_note,
)
from src.retrieval import retrieve_relevant_notes
from src.eval_harness import run_evaluation_harness

NOTES_PATH = Path("AI Intern Case Study/context_notes.csv")


@pytest.fixture(scope="module")
def context_notes():
    return load_context_notes(NOTES_PATH)


def test_distractors_never_justify(context_notes):
    """Verify that planted distractor notes N004-N010 never justify an anomaly."""
    trap_cases = [
        ("Mumbai-Delhi", "2024-07-29"),       # N005 maintenance - costs not affected
        ("Delhi-Jaipur", "2024-05-20"),       # N007 road resurfacing
        ("Kolkata-Bhubaneswar", "2025-06-09"), # N008 freight remained normal
        ("Mumbai-Pune", "2025-09-22"),        # N006 stable demand report
        ("Mumbai-Pune", "2025-10-27"),        # N010 vehicle tracking mandate
        ("Chennai-Bangalore", "2025-03-17"),  # N009 corridor returned to normal
    ]

    for route, week_of in trap_cases:
        flagged, matched_id, reason = evaluate_route_week_justification(
            route, week_of, context_notes
        )
        assert flagged == "Yes", f"Distractor wrongly justified {route} on {week_of}"
        assert matched_id == "", f"Distractor assigned note ID {matched_id}"


def test_flood_window_boundaries(context_notes):
    """Verify strict active window gating for N001 (Feb 24 to Mar 8, 2025)."""
    # Active flood window weeks: must be justified
    flag1, id1, _ = evaluate_route_week_justification("Chennai-Bangalore", "2025-02-24", context_notes)
    assert flag1 == "No (justified)" and id1 == "N001"

    flag2, id2, _ = evaluate_route_week_justification("Chennai-Bangalore", "2025-03-03", context_notes)
    assert flag2 == "No (justified)" and id2 == "N001"

    # Post-flood week (March 10+): roads reopened March 9, must NOT be justified
    flag3, id3, _ = evaluate_route_week_justification("Chennai-Bangalore", "2025-03-10", context_notes)
    assert flag3 == "Yes" and id3 == ""

    # Scope protection: flood note must NOT justify a different route on the same date
    flag4, id4, _ = evaluate_route_week_justification("Mumbai-Pune", "2025-02-24", context_notes)
    assert flag4 == "Yes" and id4 == ""


def test_unknown_notes_and_negations():
    """Verify that unknown notes and negated phrasing do not justify cost rises."""
    # Case A: Price drops must not justify a cost rise
    row_drop = pd.Series(
        {
            "note_id": "N011",
            "date": "2025-05-05",
            "applies_to": "All Routes",
            "note": "Diesel prices fell 3% nationwide starting this week.",
        }
    )
    note_drop = parse_context_note(row_drop)
    assert not note_drop.is_valid_disruption

    # Case B: Negated disruption keywords ('no flooding reported') must not pass
    row_neg_flood = pd.Series(
        {
            "note_id": "N012",
            "date": "2025-02-24",
            "applies_to": "Chennai-Bangalore",
            "note": "No flooding reported along the Chennai-Bangalore highway corridor this week.",
        }
    )
    note_neg_flood = parse_context_note(row_neg_flood)
    assert not note_neg_flood.is_valid_disruption


def test_retrieval_sensitivity_note_removal(context_notes):
    """Verify that retrieval actively impacts the verdict (removing N002 flips verdict to unexplained)."""
    route = "Ahmedabad-Mumbai"
    week = "2025-01-20"

    # Baseline with all notes: justified by N002
    v1, id1, _ = evaluate_route_week_justification(route, week, context_notes)
    assert v1 == "No (justified)" and id1 == "N002"

    # Filtered notes without N002: must flip to unexplained 'Yes'
    notes_without_n002 = [n for n in context_notes if n.note_id != "N002"]
    v2, id2, reason2 = evaluate_route_week_justification(route, week, notes_without_n002)
    assert v2 == "Yes"
    assert id2 == ""
    assert "No matching note found" in reason2


def test_labeled_evaluation_benchmark_100_percent():
    """Verify that the evaluation harness scores 100% on the benchmark dataset."""
    res_df = run_evaluation_harness()
    assert res_df["verdict_match"].all(), "Some verdicts diverged from labeled benchmark!"
    assert res_df["note_match"].all(), "Some note attributions diverged from labeled benchmark!"


def test_adversarial_negation_suite():
    """Adversarial stress-test: 12 phrased negations must all be rejected by the polarity gate."""
    adversarial_sentences = [
        "Flooding was not reported along the corridor this week.",
        "Routine maintenance completed; freight costs were not affected.",
        "The festival surcharge was waived by local transporters.",
        "The festival surcharge was cancelled due to carrier competition.",
        "Transporters did not apply any surcharge this week.",
        "Corridor rates remained unchanged despite heavy rain.",
        "Operating costs were unaffected by new state guidelines.",
        "No major disruptions or delays reported across northern corridors.",
        "Diesel prices fell 5% following government tax cuts.",
        "Compliance costs were absorbed without rate increases.",
        "Highway authorities confirmed the route returned to normal conditions.",
        "Road freight movement remained normal throughout the week.",
    ]

    for idx, sentence in enumerate(adversarial_sentences, start=101):
        row = pd.Series(
            {
                "note_id": f"ADV_{idx}",
                "date": "2025-03-01",
                "applies_to": "All Routes",
                "note": sentence,
            }
        )
        note = parse_context_note(row)
        assert not note.is_valid_disruption, f"Adversarial negation slipped past polarity gate: '{sentence}'"


def test_date_parsing_edge_cases():
    """Verify that date range parser handles year rollover and month abbreviations."""
    from src.justification import extract_date_range

    # Year rollover: Dec 28 to Jan 3
    s1, e1 = extract_date_range("2024-12-28", "Highway repairs lasted from Dec 28 to Jan 3.")
    assert s1 == "2024-12-28" and e1 == "2025-01-03"

    # 'Sept' abbreviation
    s2, e2 = extract_date_range("2025-09-01", "Detour active from Sept 5 to Sept 12.")
    assert s2 == "2025-09-05" and e2 == "2025-09-12"

    # Full month names: 'February'
    s3, e3 = extract_date_range("2025-02-01", "Flooding detour from February 10 to February 18.")
    assert s3 == "2025-02-10" and e3 == "2025-02-18"

