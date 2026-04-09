"""Referee modeling module (referee strictness).

Sofascore exposes career totals: yellowCards, redCards, yellowRedCards,
games. We compute `cards_per_game` directly (we do not need empirical Bayes
shrinkage when we have plenty of samples).

Pure logic — zero infra deps.
"""

from __future__ import annotations


def referee_strictness_factor(
    referee_cards_per_game: float,
    league_avg_cards_per_game: float,
) -> float:
    """Multiplicative factor for the cards lambda.

    1.0 = referee at the league average
    1.5 = referee 50% stricter
    0.7 = referee 30% more lenient

    Parameters
    ----------
    referee_cards_per_game : float
        Average cards per match for this referee.
    league_avg_cards_per_game : float
        Average cards per match in the league.

    Returns
    -------
    float
        Multiplicative factor clamped to [0.5, 2.0].
    """
    if league_avg_cards_per_game <= 0:
        return 1.0
    if referee_cards_per_game <= 0:
        return 1.0
    factor = referee_cards_per_game / league_avg_cards_per_game
    # Clamp to avoid extremes
    return max(0.5, min(factor, 2.0))


def cards_per_game_from_totals(
    yellow_total: int,
    yellowred_total: int,
    red_total: int,
    matches: int,
) -> float:
    """Compute cards_per_game from career totals."""
    if matches <= 0:
        return 0.0
    total_cards = yellow_total + yellowred_total + red_total
    return total_cards / matches
