"""Corners prediction module via Poisson.

lambda_corners_home = avg_corners_home_taken x (avg_corners_conceded_away / league_avg)
lambda_corners_away = avg_corners_away_taken x (avg_corners_conceded_home / league_avg)

Pure logic — zero infra deps.
"""

from __future__ import annotations


def predict_corners(
    home_corners_avg: float,
    away_corners_avg: float,
    home_corners_against: float,
    away_corners_against: float,
    league_corners_per_team: float = 5.0,
) -> tuple[float, float]:
    """Return (lambda_home_corners, lambda_away_corners).

    Parameters
    ----------
    home_corners_avg : float
        Average corners TAKEN by the home team in the last N matches.
    away_corners_avg : float
        Average corners TAKEN by the away team.
    home_corners_against : float
        Average corners CONCEDED by the home team.
    away_corners_against : float
        Average corners CONCEDED by the away team.
    league_corners_per_team : float
        League average corners per team per match (~5).

    Returns
    -------
    tuple[float, float]
        (lambda_home, lambda_away) with a minimum of 1.0.
    """
    if league_corners_per_team <= 0:
        league_corners_per_team = 5.0

    opp_factor_for_home = away_corners_against / league_corners_per_team
    opp_factor_for_away = home_corners_against / league_corners_per_team

    lam_home = home_corners_avg * opp_factor_for_home
    lam_away = away_corners_avg * opp_factor_for_away

    return (max(lam_home, 1.0), max(lam_away, 1.0))
