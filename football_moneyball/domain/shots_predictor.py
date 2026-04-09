"""Shots prediction module via Poisson.

Same pattern as corners: lambda_shots = avg_shots_team x opp_factor.
"""

from __future__ import annotations


def predict_shots(
    home_shots_avg: float,
    away_shots_avg: float,
    home_shots_against: float,
    away_shots_against: float,
    league_shots_per_team: float = 10.0,
) -> tuple[float, float]:
    """Return (lambda_home_shots, lambda_away_shots).

    Parameters
    ----------
    home_shots_avg, away_shots_avg : float
        Average shots taken.
    home_shots_against, away_shots_against : float
        Average shots conceded.
    league_shots_per_team : float
        League average (~10/team).

    Returns
    -------
    tuple[float, float]
        (lambda_home, lambda_away) with a minimum of 3.0.
    """
    if league_shots_per_team <= 0:
        league_shots_per_team = 10.0

    opp_factor_home = away_shots_against / league_shots_per_team
    opp_factor_away = home_shots_against / league_shots_per_team

    lam_home = home_shots_avg * opp_factor_home
    lam_away = away_shots_avg * opp_factor_away

    return (max(lam_home, 3.0), max(lam_away, 3.0))
