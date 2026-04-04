"""Modulo de modelagem de arbitros (referee strictness).

Sofascore expoe totais de carreira: yellowCards, redCards, yellowRedCards,
games. Calculamos `cards_per_game` direto (nao precisamos empirical Bayes
shrinkage quando temos muita amostra).

Logica pura — zero deps de infra.
"""

from __future__ import annotations


def referee_strictness_factor(
    referee_cards_per_game: float,
    league_avg_cards_per_game: float,
) -> float:
    """Fator multiplicativo pra λ de cartoes.

    1.0 = juiz na media da liga
    1.5 = juiz 50% mais rigoroso
    0.7 = juiz 30% mais leniente

    Parameters
    ----------
    referee_cards_per_game : float
        Media de cartoes por jogo desse arbitro.
    league_avg_cards_per_game : float
        Media de cartoes por jogo na liga.

    Returns
    -------
    float
        Fator multiplicativo clamped em [0.5, 2.0].
    """
    if league_avg_cards_per_game <= 0:
        return 1.0
    if referee_cards_per_game <= 0:
        return 1.0
    factor = referee_cards_per_game / league_avg_cards_per_game
    # Clamp pra evitar extremos
    return max(0.5, min(factor, 2.0))


def cards_per_game_from_totals(
    yellow_total: int,
    yellowred_total: int,
    red_total: int,
    matches: int,
) -> float:
    """Calcula cards_per_game a partir dos totais de carreira."""
    if matches <= 0:
        return 0.0
    total_cards = yellow_total + yellowred_total + red_total
    return total_cards / matches
