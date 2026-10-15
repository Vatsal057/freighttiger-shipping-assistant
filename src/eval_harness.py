"""Curated Regression & Benchmark Harness scoring pipeline accuracy, note attribution,
and planted distractor rejection against labeled ground-truth cases.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import run_pipeline
from src.justification import (
    load_context_notes,
    evaluate_route_week_justification,
    evaluate_route_week_with_llm,
)

DEFAULT_SHIPMENTS = Path("AI Intern Case Study/shipment_records.csv")
DEFAULT_NOTES = Path("AI Intern Case Study/context_notes.csv")
LABELED_SET = Path("tests/labeled_evaluation_set.csv")


def run_evaluation_harness(
    shipments_path: Path | str = DEFAULT_SHIPMENTS,
    notes_path: Path | str = DEFAULT_NOTES,
    labeled_path: Path | str = LABELED_SET,
    mode: str = "deterministic",
) -> pd.DataFrame:
    """Score the pipeline against the curated regression benchmark."""
    labeled_df = pd.read_csv(labeled_path)
    notes = load_context_notes(notes_path)
    pipeline_df = run_pipeline(shipments_path, notes_path, mode=mode)

    results = []
    correct_verdicts = 0
    correct_notes = 0
    total = len(labeled_df)

    for _, row in labeled_df.iterrows():
        route = str(row["route"])
        week_of = str(row["week_of"])
        exp_flag = str(row["expected_flagged"])
        exp_note = str(row["expected_note_id"]) if pd.notna(row["expected_note_id"]) else ""
        case_type = str(row["case_type"])

        match = pipeline_df[(pipeline_df["route"] == route) & (pipeline_df["week_of"] == week_of)]
        if len(match) > 0:
            actual_flag = str(match.iloc[0]["flagged"])
            raw_nid = match.iloc[0]["matched_note_id"]
            actual_note = str(raw_nid) if pd.notna(raw_nid) else ""
        else:
            if mode == "llm":
                actual_flag, actual_note, _, _ = evaluate_route_week_with_llm(route, week_of, notes)
            else:
                actual_flag, actual_note, _ = evaluate_route_week_justification(route, week_of, notes)

        verdict_match = (actual_flag == exp_flag)
        note_match = (actual_note == exp_note)

        if verdict_match:
            correct_verdicts += 1
        if note_match:
            correct_notes += 1

        results.append(
            {
                "route": route,
                "week_of": week_of,
                "case_type": case_type,
                "expected_flag": exp_flag,
                "actual_flag": actual_flag,
                "verdict_match": verdict_match,
                "expected_note": exp_note,
                "actual_note": actual_note,
                "note_match": note_match,
            }
        )

    res_df = pd.DataFrame(results)
    accuracy_verdict = (correct_verdicts / total) * 100.0
    accuracy_note = (correct_notes / total) * 100.0

    trap_cases = res_df[res_df["case_type"].str.contains("Trap|Distractor", case=False, na=False)]
    if len(trap_cases) > 0:
        traps_disqualified = int(((trap_cases["actual_flag"] == "Yes") & (trap_cases["actual_note"] == "")).sum())
        trap_rate = (traps_disqualified / len(trap_cases)) * 100.0
        trap_stat = f"{trap_rate:.1f}% ({traps_disqualified}/{len(trap_cases)} distractors rejected)"
    else:
        trap_stat = "N/A (no trap cases present)"

    scorecard_lines = [
        f"FreightTiger Evaluation Harness — Regression Benchmark Scorecard (Mode: {mode})",
        "=================================================================",
        f"Total Test Cases Evaluated: {total}",
        f"Verdict Accuracy (Yes vs No (justified)): {accuracy_verdict:.1f}% ({correct_verdicts}/{total})",
        f"Note Attribution Accuracy: {accuracy_note:.1f}% ({correct_notes}/{total})",
        f"Planted Trap Disqualification Rate: {trap_stat}",
        "",
        "Case-by-Case Breakdown:",
    ]
    for _, r in res_df.iterrows():
        v_icon = "✓" if r["verdict_match"] else "✗"
        n_icon = "✓" if r["note_match"] else "✗"
        scorecard_lines.append(f"  [{v_icon} Verdict | {n_icon} Note] {r['route']} ({r['week_of']}): {r['case_type']}")

    scorecard_text = "\n".join(scorecard_lines)
    print(scorecard_text)

    report_file = Path("output/eval_report.txt")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(scorecard_text + "\n")
    print(f"\nSaved evaluation scorecard to: {report_file}")

    return res_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FreightTiger Evaluation Harness")
    parser.add_argument(
        "--mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Evaluation mode",
    )
    args = parser.parse_args()
    run_evaluation_harness(mode=args.mode)
