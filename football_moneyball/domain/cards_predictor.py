"""Cards prediction module via Poisson + referee factor.

lambda_cards = base_cards x referee_factor

Simplification of ZIP: we use pure Poisson, we can later validate whether
it is worth migrating.

Pure logic — zero infra deps.
"""

from __future__ import annotations


def predict_cards(
    home_cards_avg: float,
    away_cards_avg: float,
    home_fouls_avg: float,
    away_fouls_avg: float,
    referee_factor: float = 1.0,
    derby_factor: float = 1.0,
) -> tuple[float, float]:
    """Return (lambda_home_cards, lambda_away_cards).

    Combines historical cards received + fouls committed by the opponent
    (proxy for "violent team against you"), adjusted by the referee.

    Parameters
    ----------
    home_cards_avg : float
        Average cards received by the home team in the last N.
    away_cards_avg : float
        Average cards received by the away team.
    home_fouls_avg : float
        Average fouls committed by the home team.
    away_fouls_avg : float
        Average fouls committed by the away team.
    referee_factor : float
        Referee factor (1.0 = average, 1.5 = strict).
    derby_factor : float
        1.2 in a derby, 1.0 otherwise.

    Returns
    -------
    tuple[float, float]
        (lambda_home_cards, lambda_away_cards) with a minimum of 0.5.
    """
    # Base: cards_received + contribution from opponent fouls
    # (assuming ~1 card per 7-8 dangerous fouls)
    fouls_to_cards = 0.15

    base_home = home_cards_avg + (away_fouls_avg * fouls_to_cards * 0.3)
    base_away = away_cards_avg + (home_fouls_avg * fouls_to_cards * 0.3)

    lam_home = base_home * referee_factor * derby_factor
    lam_away = base_away * referee_factor * derby_factor

    return (max(lam_home, 0.5), max(lam_away, 0.5))
