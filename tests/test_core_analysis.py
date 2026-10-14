"""Unit tests for the core shipment analysis engine.

Verifies:
1. Exact match against the 3 sample output rows in sample_output_format_v2.csv.
2. Invariant: strictly no lookahead (future weeks cannot alter past baselines).
3. Early weeks with < 8 weeks of history use available history without padding.
4. Peer exclusion: own route is excluded from peer average.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.core_analysis import (
    load_shipments,
    aggregate_weekly_route_costs,
    compute_baselines,
    format_percentage_diff,
    identify_candidate_anomalies,
)

DATA_PATH = Path("AI Intern Case Study/shipment_records.csv")
SAMPLE_PATH = Path("AI Intern Case Study/sample_output_format_v2.csv")


@pytest.fixture(scope="module")
def baselines_df():
    shipments = load_shipments(DATA_PATH)
    weekly = aggregate_weekly_route_costs(shipments)
    return compute_baselines(weekly, trailing_weeks=8)


def test_sample_output_match(baselines_df):
    """Verify that the 3 rows in sample_output_format_v2.csv match exactly."""
    # Custom parse to handle unquoted comma in sample file line 4
    sample_rows = []
    with open(SAMPLE_PATH, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",", 7)
            sample_rows.append(dict(zip(header, parts)))

    for sample_row in sample_rows:
        route = sample_row["route"]
        week_of = sample_row["week_of"]
        expected_cost = float(sample_row["cost_per_tonne_km"])
        expected_hist = sample_row["vs_own_history"]
        expected_peer = sample_row["vs_similar_routes"]

        match = baselines_df[
            (baselines_df["route"] == route) & (baselines_df["week_of"] == week_of)
        ]
        assert len(match) == 1, f"Missing route-week: {route} {week_of}"
        row = match.iloc[0]

        # 1. Cost per tonne-km (rounded to 2 decimal places)
        assert round(row["cost_per_tonne_km"], 2) == expected_cost

        # 2. Formatted vs_own_history string
        hist_str = format_percentage_diff(row["vs_hist_pct"], "history")
        assert hist_str == expected_hist

        # 3. Formatted vs_similar_routes string
        peer_str = format_percentage_diff(row["vs_peer_pct"], "peer")
        assert peer_str == expected_peer


def test_strictly_no_lookahead(baselines_df):
    """Verify that earlier weeks depend only on strictly prior weeks."""
    # Check week 2 of any route: prior_weeks_available must be exactly 1
    w2 = baselines_df[baselines_df["week_of"] == "2024-01-08"]
    for _, row in w2.iterrows():
        assert row["prior_weeks_available"] == 1

    # Check week 1 of any route: prior_weeks_available must be 0 and vs_hist_pct is NaN
    w1 = baselines_df[baselines_df["week_of"] == "2024-01-01"]
    for _, row in w1.iterrows():
        assert row["prior_weeks_available"] == 0
        assert pd.isna(row["vs_hist_pct"])


def test_peer_exclusion(baselines_df):
    """Verify that a route's own cost is not included in its peer average."""
    # Short routes are Delhi-Jaipur and Mumbai-Pune.
    # Therefore, the peer average for Delhi-Jaipur must exactly equal Mumbai-Pune's cost.
    for week in ["2024-06-03", "2024-11-11", "2025-09-15"]:
        dj = baselines_df[
            (baselines_df["route"] == "Delhi-Jaipur")
            & (baselines_df["week_of"] == week)
        ].iloc[0]
        mp = baselines_df[
            (baselines_df["route"] == "Mumbai-Pune")
            & (baselines_df["week_of"] == week)
        ].iloc[0]

        assert round(dj["peer_mean"], 4) == round(mp["cost_per_tonne_km"], 4)
        assert round(mp["peer_mean"], 4) == round(dj["cost_per_tonne_km"], 4)
