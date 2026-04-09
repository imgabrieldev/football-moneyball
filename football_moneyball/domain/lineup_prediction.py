"""Probable lineup prediction module (Probable XI).

Pure logic — zero infra deps. Given a DataFrame with per-player
aggregates for the last N matches, returns the 11 most likely
starters + a weight for each (frequency * minutes).
"""

from __future__ import annotations

import pandas as pd


def minutes_weight(
    matches_played: int,
    minutes_total: float,
    last_n: int,
) -> float:
    """Weight of being a regular starter (0.0 to 1.0).

    weight = (matches_played / last_n) * (avg_minutes / 90)

    Examples
    --------
    Player with 5/5 matches of 90 min -> 1.0 * 1.0 = 1.0
    Player with 2/5 matches of 90 min -> 0.4 * 1.0 = 0.4
    Player with 5/5 matches of 45 min -> 1.0 * 0.5 = 0.5
    Player with 1/5 matches of 45 min -> 0.2 * 0.5 = 0.1

    Parameters
    ----------
    matches_played : int
        Number of matches the player appeared in.
    minutes_total : float
        Total minutes played across matches.
    last_n : int
        Size of the reference window (5 = last 5 matches).

    Returns
    -------
    float
        Weight in the interval [0, 1].
    """
    if last_n <= 0 or matches_played <= 0:
        return 0.0
    start_rate = min(matches_played / last_n, 1.0)
    avg_minutes = minutes_total / matches_played
    minutes_rate = min(avg_minutes / 90.0, 1.0)
    return float(start_rate * minutes_rate)


def probable_xi(
    player_aggregates: pd.DataFrame,
    last_n_matches: int = 5,
) -> pd.DataFrame:
    """Return the 11 players most likely to be starters.

    Sorts by total minutes (proxy for "most used") and takes the top 11.
    Computes a weight for each in [0, 1].

    Parameters
    ----------
    player_aggregates : pd.DataFrame
        Must contain: player_id, player_name, matches_played,
        minutes_total, xg_total.
    last_n_matches : int
        Reference window used to compute the weight.

    Returns
    -------
    pd.DataFrame
        Top 11 players, sorted by minutes_total desc.
        Columns: player_id, player_name, matches_played, minutes_total,
        xg_total, xg_per_90, weight.
    """
    if player_aggregates.empty:
        return pd.DataFrame(columns=[
            "player_id", "player_name", "matches_played", "minutes_total",
            "xg_total", "xg_per_90", "weight",
        ])

    df = player_aggregates.copy()

    # Individual xG/90
    df["xg_per_90"] = df.apply(
        lambda r: (float(r["xg_total"]) / float(r["minutes_total"])) * 90.0
        if float(r["minutes_total"]) > 0
        else 0.0,
        axis=1,
    )

    # Individual weight
    df["weight"] = df.apply(
        lambda r: minutes_weight(
            int(r["matches_played"]),
            float(r["minutes_total"]),
            last_n_matches,
        ),
        axis=1,
    )

    # Top 11 by total minutes
    df = df.sort_values("minutes_total", ascending=False).head(11).reset_index(drop=True)

    cols = [
        "player_id", "player_name", "matches_played", "minutes_total",
        "xg_total", "xg_per_90", "weight",
    ]
    return df[[c for c in cols if c in df.columns]]
