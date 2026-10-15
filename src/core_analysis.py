"""Weekly shipping cost analysis and baseline calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


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

    initial_count = len(df)
    valid_mask = (
        (df["quantity_tonnes"] > 0)
        & (df["distance_km"] > 0)
        & (df["freight_cost_inr"] >= 0)
    )
    dropped_count = initial_count - int(valid_mask.sum())
    if dropped_count > 0:
        logger.warning(
            "Data Quality Guard: Dropped %d invalid shipment rows (quantity<=0, distance<=0, or cost<0)",
            dropped_count
        )
        df = df[valid_mask].copy()

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
) -> pd.DataFrame:
    """Filter route-weeks where cost rise exceeds history or peer thresholds.

    Anomaly Criteria:
    1. Historical Jump: Cost jumped >= hist_threshold_pct (15%) vs its own trailing history.
    2. Peer Outlier with Genuine Rise: Cost jumped >= peer_threshold_pct (20%) vs same-week
       peers AND cost is strictly rising vs own history (vs_hist_pct > 0). This prevents
       flagging cost drops as anomalies merely because a single peer route dropped.
    3. Network-Wide Macro Surge: Coordinated systemic surge where network median cost rise
       is >= macro_threshold_pct (5%), capturing events like nationwide diesel hikes.
    """
    hist_diff = df_baselines["vs_hist_pct"].fillna(0.0)
    peer_diff = df_baselines["vs_peer_pct"].fillna(0.0)

    standard_anomaly = (hist_diff >= hist_threshold_pct) | (
        (peer_diff >= peer_threshold_pct) & (df_baselines["vs_hist_pct"] > 0.0)
    )

    macro_weeks = set()
    for week_val, grp in df_baselines.groupby("week_of"):
        valid_hist = grp["vs_hist_pct"].dropna()
        if len(valid_hist) > 0 and valid_hist.median() >= macro_threshold_pct:
            macro_weeks.add(week_val)

    is_macro_spike = (
        df_baselines["week_of"].isin(macro_weeks)
        & (df_baselines["vs_hist_pct"] >= macro_threshold_pct)
    )

    candidates = (
        df_baselines[standard_anomaly | is_macro_spike]
        .copy()
        .sort_values(["week_of", "route"])
    )
    return candidates.reset_index(drop=True)

