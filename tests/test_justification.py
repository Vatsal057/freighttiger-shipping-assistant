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


def test_llm_justification_valid_and_unexplained(context_notes, tmp_path, monkeypatch):
    """Verify that evaluate_route_week_with_llm accurately flags valid vs unexplained cases."""
    from src.justification import evaluate_route_week_with_llm
    import src.justification as just_module

    fake_cache = tmp_path / "test_cache.json"
    monkeypatch.setattr(just_module, "CACHE_FILE", fake_cache)

    # Valid festival surcharge
    flag1, id1, reason1, usage1 = evaluate_route_week_with_llm(
        route="Ahmedabad-Mumbai",
        week_of_str="2025-01-20",
        notes=context_notes,
        vs_hist_str="+29.5% vs past average",
        vs_peer_str="+22.5% vs peers",
    )
    assert flag1 == "No (justified)"
    assert id1 == "N002"
    assert "N002" in reason1
    assert usage1["llm_calls"] == 1 or usage1["cache_hits"] == 1

    # Unexplained candidate spike: pre-gate filters out distractors so 0 LLM calls are made
    flag2, id2, reason2, usage2 = evaluate_route_week_with_llm(
        route="Delhi-Jaipur",
        week_of_str="2024-11-11",
        notes=context_notes,
        vs_hist_str="+35.5% vs past average",
        vs_peer_str="+21.0% vs peers",
    )
    assert flag2 == "Yes"
    assert id2 == ""
    assert usage2["llm_calls"] == 0


def test_llm_safety_gate_distractor_override(context_notes):
    """Verify that the safety gate overrides an LLM if it attempts to justify via a distractor note."""
    from unittest.mock import MagicMock
    from src.justification import evaluate_route_week_with_llm
    from src.llm_client import LLMResponse

    # Create a rogue/hallucinating LLM mock that attempts to justify using distractor N010
    mock_client = MagicMock()
    mock_client.generate.return_value = LLMResponse(
        text='{"verdict": "No (justified)", "matched_note_id": "N010", "evidence_quote": "compliance costs were absorbed", "reason": "Transporters added tracking cost."}',
        input_tokens=100,
        output_tokens=30,
        model_name="mock-hallucinator",
        provider="mock",
        cost_usd=0.0001,
    )

    flag, note_id, reason, usage = evaluate_route_week_with_llm(
        route="Mumbai-Pune",
        week_of_str="2025-10-27",
        notes=context_notes,
        vs_hist_str="+5.5% vs past average",
        vs_peer_str="+28.3% vs peers",
        client=mock_client,
    )

    # Safety gate MUST override the hallucinated verdict because N010 is not a valid disruption
    assert flag == "Yes", "Safety Gate failed to override hallucinated distractor justification"
    assert note_id == "", "Safety Gate allowed non-empty note_id on overridden distractor"
    assert "N010" in reason


def test_llm_invalid_evidence_quote_falls_back_to_deterministic(context_notes):
    """Verify that if evidence_quote is not in the note text, it falls back to deterministic."""
    from unittest.mock import MagicMock
    from src.justification import evaluate_route_week_with_llm, evaluate_route_week_justification
    from src.llm_client import LLMResponse

    mock_client = MagicMock()
    mock_client.generate.return_value = LLMResponse(
        text='{"verdict": "No (justified)", "matched_note_id": "N002", "evidence_quote": "THIS TEXT DOES NOT EXIST IN NOTE", "reason": "Festival surcharge."}',
        input_tokens=100,
        output_tokens=30,
        model_name="mock-bad-quote",
        provider="mock",
        cost_usd=0.0001,
    )

    det_flag, det_id, det_reason = evaluate_route_week_justification(
        "Ahmedabad-Mumbai", "2025-01-20", context_notes
    )

    flag, note_id, reason, usage = evaluate_route_week_with_llm(
        route="Ahmedabad-Mumbai",
        week_of_str="2025-01-20",
        notes=context_notes,
        vs_hist_str="+29.5% vs past average",
        vs_peer_str="+22.5% vs peers",
        client=mock_client,
    )

    # Must fall back to deterministic result
    assert flag == det_flag
    assert note_id == det_id
    assert reason == det_reason


def test_llm_response_cache_hit_makes_no_api_call(context_notes, tmp_path, monkeypatch):
    """Verify that a second identical LLM call is served from cache with 0 API calls."""
    from unittest.mock import MagicMock
    from src.justification import evaluate_route_week_with_llm
    import src.justification as just_module
    from src.llm_client import LLMResponse

    # Direct cache to tmp_path to isolate test
    fake_cache_file = tmp_path / "test_cache.json"
    monkeypatch.setattr(just_module, "CACHE_FILE", fake_cache_file)

    mock_client = MagicMock()
    mock_client.model_name = "test-cache-model"
    mock_client.generate.return_value = LLMResponse(
        text='{"verdict": "No (justified)", "matched_note_id": "N002", "evidence_quote": "temporary surcharge", "reason": "Matches note N002: festival surcharge."}',
        input_tokens=150,
        output_tokens=40,
        model_name="test-cache-model",
        provider="mock",
        cost_usd=0.0001,
    )

    # First call: cache miss, calls API
    flag1, id1, reason1, usage1 = evaluate_route_week_with_llm(
        route="Ahmedabad-Mumbai",
        week_of_str="2025-01-20",
        notes=context_notes,
        vs_hist_str="+29.5%",
        vs_peer_str="+22.5%",
        client=mock_client,
    )
    assert mock_client.generate.call_count == 1
    assert usage1["llm_calls"] == 1
    assert usage1["cache_hits"] == 0

    # Second call: identical arguments -> must be served from cache
    flag2, id2, reason2, usage2 = evaluate_route_week_with_llm(
        route="Ahmedabad-Mumbai",
        week_of_str="2025-01-20",
        notes=context_notes,
        vs_hist_str="+29.5%",
        vs_peer_str="+22.5%",
        client=mock_client,
    )
    # mock_client.generate MUST STILL be 1 (zero new calls!)
    assert mock_client.generate.call_count == 1
    assert usage2["llm_calls"] == 0
    assert usage2["cache_hits"] == 1
    assert usage2["input_tokens"] == 0
    assert usage2["output_tokens"] == 0
    assert flag1 == flag2
    assert id1 == id2


