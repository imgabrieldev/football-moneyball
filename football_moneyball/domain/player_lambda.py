"""Player-to-team aggregation module to compute Poisson lambda.

Pure logic — receives player DataFrames (with xG/90 and weight) and
returns the team lambda (expected goals). Replaces the team-level
path where lambda = league_avg x attack_strength x opp_defense.
"""

from __future__ import annotations

import pandas as pd


def compute_xg_per_90(xg_total: float, minutes_total: float) -> float:
    """xG per 90 minutes: (xg_total / minutes_total) * 90.

    Parameters
    ----------
    xg_total : float
        Accumulated xG of the player in the period.
    minutes_total : float
        Total minutes played in the period.

    Returns
    -------
    float
        Adjusted xG/90. Returns 0.0 if minutes_total <= 0.
    """
    if minutes_total <= 0:
        return 0.0
    return (xg_total / minutes_total) * 90.0


def team_lambda_from_players(
    xi: pd.DataFrame,
    opponent_defense_factor: float = 1.0,
) -> float:
    """Compute team lambda from the aggregation of xG/90 of starters.

    lambda_team = sum(xG/90 * weight) * opponent_defense_factor

    The weight represents the probability * participation of the
    player. A player who plays every match in full has weight=1.0.

    Parameters
    ----------
    xi : pd.DataFrame
        Starters DataFrame with columns xg_per_90 and weight.
    opponent_defense_factor : float
        Defensive factor of the opponent (from calculate_team_strength).
        1.0 = league average, <1.0 = strong defense, >1.0 = weak defense.

    Returns
    -------
    float
        Expected team goal lambda. Minimum 0.15.
    """
    if xi.empty or "xg_per_90" not in xi.columns or "weight" not in xi.columns:
        return 0.15

    team_attack = float((xi["xg_per_90"] * xi["weight"]).sum())
    lam = team_attack * opponent_defense_factor
    return max(lam, 0.15)


def summarize_xi(xi: pd.DataFrame) -> list[dict]:
    """Lineup summary to display in the frontend.

    Parameters
    ----------
    xi : pd.DataFrame
        Output of probable_xi().

    Returns
    -------
    list[dict]
        List with name, xG/90 and weight of each player.
    """
    if xi.empty:
        return []
    return [
        {
            "player_id": int(row.get("player_id", 0) or 0),
            "player_name": str(row.get("player_name", "")),
            "xg_per_90": round(float(row.get("xg_per_90", 0) or 0), 3),
            "weight": round(float(row.get("weight", 0) or 0), 3),
            "minutes_total": float(row.get("minutes_total", 0) or 0),
        }
        for _, row in xi.iterrows()
    ]
