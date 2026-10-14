"""Weekly shipping cost analysis and baseline calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def load_shipments(file_path: Path | str) -> pd.DataFrame:
    """Load shipment records and derive route, week, and tonne-km fields."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Shipment records file not found at {file_path}")

    df = pd.read_csv(file_path)
    required = {
        "shipment_id",
        "origin",
        "destination",
        "route_type",
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
        "shipment_date",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in shipment records: {missing}")

    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["week_of"] = df["shipment_date"].apply(lambda d: d - pd.Timedelta(days=d.weekday()))
    df["route"] = df["origin"] + "-" + df["destination"]
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def aggregate_weekly_route_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total freight cost and tonne-km by route and week."""
    grouped = (
        df.groupby(["route", "route_type", "week_of"])
        .agg(
            total_freight_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
        .reset_index()
    )

    grouped["cost_per_tonne_km"] = (
        grouped["total_freight_cost"] / grouped["total_tonne_km"]
    )
    return grouped.sort_values(["route", "week_of"]).reset_index(drop=True)


def compute_baselines(
    weekly_df: pd.DataFrame,
    trailing_weeks: int = 8,
) -> pd.DataFrame:
    """Compute rolling 8-week history and same-week peer baselines."""
    records: List[Dict] = []

    for route in weekly_df["route"].unique():
        r_df = weekly_df[weekly_df["route"] == route].sort_values("week_of").reset_index(drop=True)
        route_type = r_df["route_type"].iloc[0]

        for idx, row in r_df.iterrows():
            curr_cptk = row["cost_per_tonne_km"]
            week_of = row["week_of"]

            # Calendar-based trailing window: strictly prior weeks within 56 days (8 weeks * 7 days)
            prior_slice = r_df[
                (r_df["week_of"] < week_of)
                & (r_df["week_of"] >= week_of - pd.Timedelta(days=trailing_weeks * 7))
            ]
            if len(prior_slice) > 0:
                hist_mean = prior_slice["cost_per_tonne_km"].mean()
                vs_hist_pct = ((curr_cptk - hist_mean) / hist_mean) * 100.0
            else:
                hist_mean = None
                vs_hist_pct = None

            same_week_peers = weekly_df[
                (weekly_df["week_of"] == week_of)
                & (weekly_df["route_type"] == route_type)
                & (weekly_df["route"] != route)
            ]
            if len(same_week_peers) > 0:
                peer_mean = same_week_peers["cost_per_tonne_km"].mean()
                vs_peer_pct = ((curr_cptk - peer_mean) / peer_mean) * 100.0
            else:
                peer_mean = None
                vs_peer_pct = None

            records.append(
                {
                    "route": route,
                    "route_type": route_type,
                    "week_of": week_of.strftime("%Y-%m-%d"),
                    "cost_per_tonne_km": curr_cptk,
                    "hist_mean": hist_mean,
                    "vs_hist_pct": vs_hist_pct,
                    "peer_mean": peer_mean,
                    "vs_peer_pct": vs_peer_pct,
                    "prior_weeks_available": len(prior_slice),
                }
            )

    return pd.DataFrame(records)


def format_percentage_diff(pct: Optional[float], baseline_type: str) -> str:
    """Format baseline difference string to match sample format."""
    if pct is None:
        if baseline_type == "history":
            return "N/A (first week on this route)"
        return "N/A (no peer routes available)"

    sign = "+" if pct >= 0 else ""
    if baseline_type == "history":
        return f"{sign}{pct:.1f}% vs this route's past average"
    elif baseline_type == "peer":
        return f"{sign}{pct:.1f}% vs similar-length routes this week"
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")


def identify_candidate_anomalies(
    df_baselines: pd.DataFrame,
    hist_threshold_pct: float = 15.0,
    peer_threshold_pct: float = 20.0,
    macro_threshold_pct: float = 5.0,
    macro_min_routes: int = 4,
) -> pd.DataFrame:
    """Filter route-weeks where cost rise exceeds history or peer thresholds.

    Also identifies macro surges where a coordinated cost rise (>= macro_threshold_pct)
    impacts multiple routes (>= macro_min_routes) simultaneously, such as nationwide fuel hikes.
    """
    # 1. Standard route-specific or peer anomaly
    standard_anomaly = (
        (df_baselines["vs_hist_pct"].fillna(0) >= hist_threshold_pct)
        | (df_baselines["vs_peer_pct"].fillna(0) >= peer_threshold_pct)
    )

    # 2. Macro surge detection across the network
    macro_weeks = set()
    for week_val, grp in df_baselines.groupby("week_of"):
        rising_count = (grp["vs_hist_pct"].fillna(0) >= macro_threshold_pct).sum()
        if rising_count >= macro_min_routes:
            macro_weeks.add(week_val)

    is_macro_spike = (
        df_baselines["week_of"].isin(macro_weeks)
        & (df_baselines["vs_hist_pct"].fillna(0) >= macro_threshold_pct)
    )

    candidates = (
        df_baselines[standard_anomaly | is_macro_spike]
        .copy()
        .sort_values(["week_of", "route"])
    )
    return candidates.reset_index(drop=True)

