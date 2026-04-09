"""Head-to-head features — historical record between the two teams.

Pure logic: receives a list of H2H results and returns numeric features.
"""
from __future__ import annotations

from typing import Any


def compute_h2h_features(
    h2h_results: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    default_avg_goals: float = 1.3,
) -> dict[str, float]:
    """Compute H2H features from a list of results.

    Parameters
    ----------
    h2h_results : list[dict]
        List of past matches between the two teams. Each dict with:
        {home_team, away_team, home_goals, away_goals}.
    home_team, away_team : str
        Current home and away team (to orient the results).
    default_avg_goals : float
        Default value when there is no H2H history.

    Returns
    -------
    dict with 5 features:
        h2h_home_win_rate — % wins of the current home_team (in any venue)
        h2h_away_win_rate — % wins of the current away_team
        h2h_draw_rate — % draws
        h2h_home_goals_avg — goal average of the current home_team
        h2h_away_goals_avg — goal average of the current away_team
    """
    if not h2h_results:
        return {
            "h2h_home_win_rate": 0.33,
            "h2h_away_win_rate": 0.33,
            "h2h_draw_rate": 0.25,
            "h2h_home_goals_avg": default_avg_goals,
            "h2h_away_goals_avg": default_avg_goals,
            "h2h_n_matches": 0.0,
        }

    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals_total = 0
    away_goals_total = 0
    n = 0

    for match in h2h_results:
        mh = match.get("home_team", "")
        ma = match.get("away_team", "")
        hg = match.get("home_goals", 0) or 0
        ag = match.get("away_goals", 0) or 0

        # Identify who our home_team is in that match (they may have been the away side)
        if mh == home_team and ma == away_team:
            # Match with the same venue as the current one
            home_team_goals = hg
            away_team_goals = ag
        elif mh == away_team and ma == home_team:
            # Reversed venue
            home_team_goals = ag
            away_team_goals = hg
        else:
            # Match does not involve exactly the two teams — skip
            continue

        n += 1
        home_goals_total += home_team_goals
        away_goals_total += away_team_goals

        if home_team_goals > away_team_goals:
            home_wins += 1
        elif home_team_goals < away_team_goals:
            away_wins += 1
        else:
            draws += 1

    if n == 0:
        return {
            "h2h_home_win_rate": 0.33,
            "h2h_away_win_rate": 0.33,
            "h2h_draw_rate": 0.25,
            "h2h_home_goals_avg": default_avg_goals,
            "h2h_away_goals_avg": default_avg_goals,
            "h2h_n_matches": 0.0,
        }

    return {
        "h2h_home_win_rate": home_wins / n,
        "h2h_away_win_rate": away_wins / n,
        "h2h_draw_rate": draws / n,
        "h2h_home_goals_avg": home_goals_total / n,
        "h2h_away_goals_avg": away_goals_total / n,
        "h2h_n_matches": float(n),
    }
