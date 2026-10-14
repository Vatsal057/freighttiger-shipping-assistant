"""Integration tests for the complete FreightTiger shipping assistant pipeline."""

from pathlib import Path
import pytest
import pandas as pd
from src.pipeline import run_pipeline, write_submission_csv
from src.cli import compute_file_sha256

SHIPMENTS_PATH = Path("AI Intern Case Study/shipment_records.csv")
NOTES_PATH = Path("AI Intern Case Study/context_notes.csv")
EXPECTED_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


@pytest.fixture(scope="module")
def pipeline_output():
    return run_pipeline(SHIPMENTS_PATH, NOTES_PATH)


def test_output_columns_contract(pipeline_output):
    """Output columns must match sample contract exactly in name and order."""
    assert list(pipeline_output.columns) == EXPECTED_COLUMNS


def test_flagged_column_values_contract(pipeline_output):
    """Flagged column must ONLY contain 'Yes' or 'No (justified)'."""
    allowed = {"Yes", "No (justified)"}
    actual = set(pipeline_output["flagged"].unique())
    assert actual.issubset(allowed), f"Unexpected values in flagged column: {actual - allowed}"


def test_matched_note_id_consistency(pipeline_output):
    """matched_note_id must be populated if and only if flagged is 'No (justified)'."""
    for _, row in pipeline_output.iterrows():
        if row["flagged"] == "No (justified)":
            assert row["matched_note_id"] != "", f"Empty note_id for justified row: {row}"
        elif row["flagged"] == "Yes":
            assert row["matched_note_id"] == "", f"Non-empty note_id for unexplained row: {row}"


def test_date_format_contract(pipeline_output):
    """week_of must follow YYYY-MM-DD format."""
    for val in pipeline_output["week_of"]:
        assert len(val) == 10 and val[4] == "-" and val[7] == "-"


def test_reproducibility_hash(tmp_path):
    """Two independent pipeline runs must produce identical files and SHA-256."""
    file1 = tmp_path / "run1.csv"
    file2 = tmp_path / "run2.csv"

    df1 = run_pipeline(SHIPMENTS_PATH, NOTES_PATH)
    write_submission_csv(df1, file1)

    df2 = run_pipeline(SHIPMENTS_PATH, NOTES_PATH)
    write_submission_csv(df2, file2)

    hash1 = compute_file_sha256(file1)
    hash2 = compute_file_sha256(file2)

    assert hash1 == hash2, "Pipeline output is not deterministic across runs!"
