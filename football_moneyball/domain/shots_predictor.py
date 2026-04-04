"""Modulo de previsao de chutes via Poisson.

Mesmo padrao de corners: λ_shots = avg_shots_team × opp_factor.
"""

from __future__ import annotations


def predict_shots(
    home_shots_avg: float,
    away_shots_avg: float,
    home_shots_against: float,
    away_shots_against: float,
    league_shots_per_team: float = 10.0,
) -> tuple[float, float]:
    """Retorna (λ_home_shots, λ_away_shots).

    Parameters
    ----------
    home_shots_avg, away_shots_avg : float
        Media shots feitos.
    home_shots_against, away_shots_against : float
        Media shots sofridos.
    league_shots_per_team : float
        Media da liga (~10/time).

    Returns
    -------
    tuple[float, float]
        (λ_home, λ_away) com minimo de 3.0.
    """
    if league_shots_per_team <= 0:
        league_shots_per_team = 10.0

    opp_factor_home = away_shots_against / league_shots_per_team
    opp_factor_away = home_shots_against / league_shots_per_team

    lam_home = home_shots_avg * opp_factor_home
    lam_away = away_shots_avg * opp_factor_away

    return (max(lam_home, 3.0), max(lam_away, 3.0))
