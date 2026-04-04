"""Modulo de previsao de cartoes via Poisson + referee factor.

λ_cards = base_cards × referee_factor

Simplificacao do ZIP: usamos Poisson puro, validamos depois se vale migrar.

Logica pura — zero deps de infra.
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
    """Retorna (λ_home_cards, λ_away_cards).

    Combina cartoes recebidos historicos + faltas cometidas pelo adversario
    (proxy de "time violento contra vc"), ajustado pelo juiz.

    Parameters
    ----------
    home_cards_avg : float
        Media cartoes recebidos pelo home nos ultimos N.
    away_cards_avg : float
        Media cartoes recebidos pelo away.
    home_fouls_avg : float
        Media faltas cometidas pelo home.
    away_fouls_avg : float
        Media faltas cometidas pelo away.
    referee_factor : float
        Fator do arbitro (1.0 = media, 1.5 = rigoroso).
    derby_factor : float
        1.2 em derby, 1.0 caso contrario.

    Returns
    -------
    tuple[float, float]
        (λ_home_cards, λ_away_cards) com minimo de 0.5.
    """
    # Base: cards_recebidos + contribuicao de faltas do adversario
    # (assumindo ~1 cartao por 7-8 faltas perigosas)
    fouls_to_cards = 0.15

    base_home = home_cards_avg + (away_fouls_avg * fouls_to_cards * 0.3)
    base_away = away_cards_avg + (home_fouls_avg * fouls_to_cards * 0.3)

    lam_home = base_home * referee_factor * derby_factor
    lam_away = base_away * referee_factor * derby_factor

    return (max(lam_home, 0.5), max(lam_away, 0.5))
