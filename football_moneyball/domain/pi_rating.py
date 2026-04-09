"""Pi-Rating system (Constantinou & Fenton, 2013).

Separate HOME/AWAY ratings per team. Fixes home bias by capturing
the fact that teams perform differently home vs away.

Pure logic — zero infra deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PiRating:
    """Home/away rating of a team."""
    home: float = 0.0
    away: float = 0.0


def update_ratings(
    ratings: dict[str, PiRating],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    gamma: float = 0.04,
    goal_cap: int = 3,
) -> None:
    """Update Pi-Ratings after a match.

    Parameters
    ----------
    ratings : dict[str, PiRating]
        Current ratings (modified in place).
    home_team, away_team : str
        Team names.
    home_goals, away_goals : int
        Goals scored.
    gamma : float
        Learning rate (0.035 original EPL, 0.04 for Brasileirao).
    goal_cap : int
        Cap on the goal difference to reduce outliers (+/-3).
    """
    if home_team not in ratings:
        ratings[home_team] = PiRating()
    if away_team not in ratings:
        ratings[away_team] = PiRating()

    diff = int(home_goals) - int(away_goals)
    diff = max(-goal_cap, min(goal_cap, diff))

    expected = ratings[home_team].home - ratings[away_team].away
    error = diff - expected

    ratings[home_team].home += gamma * error
    ratings[away_team].away -= gamma * error


def compute_all_ratings(
    matches: pd.DataFrame,
    gamma: float = 0.04,
    goal_cap: int = 3,
) -> dict[str, PiRating]:
    """Compute Pi-Ratings for all teams from history.

    Parameters
    ----------
    matches : pd.DataFrame
        Columns: match_id, home_team, away_team, home_score, away_score.
        Must be sorted chronologically (by match_id or match_date).

    Returns
    -------
    dict[str, PiRating]
        Final ratings per team.
    """
    ratings: dict[str, PiRating] = {}

    # Deduplicate matches (each match_id appears once)
    if "home_team" in matches.columns:
        df = matches.drop_duplicates(subset=["match_id"]).sort_values("match_id")
    else:
        # Alternative format: team, is_home, goals
        # Pivot to home_team/away_team
        df = _pivot_match_data(matches)

    for _, row in df.iterrows():
        ht = row.get("home_team", "")
        at = row.get("away_team", "")
        hg = row.get("home_score", row.get("home_goals", 0))
        ag = row.get("away_score", row.get("away_goals", 0))

        if not ht or not at or pd.isna(hg) or pd.isna(ag):
            continue

        update_ratings(ratings, str(ht), str(at), int(hg), int(ag), gamma, goal_cap)

    return ratings


def compute_ratings_at_match(
    matches: pd.DataFrame,
    target_match_id: int,
    gamma: float = 0.04,
    goal_cap: int = 3,
) -> dict[str, PiRating]:
    """Compute ratings using only matches with match_id < target (leak-proof)."""
    prior = matches[matches["match_id"] < target_match_id]
    return compute_all_ratings(prior, gamma, goal_cap)


def rating_diff(
    ratings: dict[str, PiRating],
    home_team: str,
    away_team: str,
) -> float:
    """Pi-Rating differential: R_home[home] - R_away[away]."""
    rh = ratings.get(home_team, PiRating())
    ra = ratings.get(away_team, PiRating())
    return rh.home - ra.away


def init_promoted_teams(
    ratings: dict[str, PiRating],
    promoted: list[str],
    relegated: list[str],
) -> None:
    """Initialize promoted teams with the average of the relegated ones."""
    if not relegated:
        return
    avg_home = np.mean([ratings[t].home for t in relegated if t in ratings]) if relegated else 0.0
    avg_away = np.mean([ratings[t].away for t in relegated if t in ratings]) if relegated else 0.0
    for team in promoted:
        ratings[team] = PiRating(home=float(avg_home), away=float(avg_away))


def _pivot_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert (team, is_home, goals) format to (home_team, away_team, scores)."""
    home = df[df["is_home"] == True].copy()  # noqa: E712
    away = df[df["is_home"] == False].copy()  # noqa: E712

    home = home.rename(columns={"team": "home_team", "goals": "home_score"})
    away = away.rename(columns={"team": "away_team", "goals": "away_score"})

    merged = home[["match_id", "home_team", "home_score"]].merge(
        away[["match_id", "away_team", "away_score"]],
        on="match_id",
        how="inner",
    )
    return merged.sort_values("match_id")
