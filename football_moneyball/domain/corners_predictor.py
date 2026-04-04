"""Modulo de previsao de escanteios via Poisson.

λ_corners_home = avg_corners_home_feitos × (avg_corners_sofridos_away / league_avg)
λ_corners_away = avg_corners_away_feitos × (avg_corners_sofridos_home / league_avg)

Logica pura — zero deps de infra.
"""

from __future__ import annotations


def predict_corners(
    home_corners_avg: float,
    away_corners_avg: float,
    home_corners_against: float,
    away_corners_against: float,
    league_corners_per_team: float = 5.0,
) -> tuple[float, float]:
    """Retorna (λ_home_corners, λ_away_corners).

    Parameters
    ----------
    home_corners_avg : float
        Media de corners que o home FEZ nos ultimos N jogos.
    away_corners_avg : float
        Media de corners que o away FEZ.
    home_corners_against : float
        Media de corners que o home SOFREU.
    away_corners_against : float
        Media de corners que o away SOFREU.
    league_corners_per_team : float
        Media de corners por time por jogo na liga (~5).

    Returns
    -------
    tuple[float, float]
        (λ_home, λ_away) com minimo de 1.0.
    """
    if league_corners_per_team <= 0:
        league_corners_per_team = 5.0

    opp_factor_for_home = away_corners_against / league_corners_per_team
    opp_factor_for_away = home_corners_against / league_corners_per_team

    lam_home = home_corners_avg * opp_factor_for_home
    lam_away = away_corners_avg * opp_factor_for_away

    return (max(lam_home, 1.0), max(lam_away, 1.0))
