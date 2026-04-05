"""Market blending — devig + ensemble com probabilidades do mercado.

Mercados de apostas (Pinnacle, Betfair) são ~80% eficientes. Blendar a previsão
do modelo com a do mercado reduz erro significativamente.

Lógica pura: zero deps de infra.
"""
from __future__ import annotations


def devig_odds(odds_home: float, odds_draw: float, odds_away: float) -> dict[str, float]:
    """Remove vig das odds 1x2. Retorna probabilidades implícitas normalizadas.

    Método: normalização proporcional (mais simples, shrinkage power removido).

    Parameters
    ----------
    odds_home, odds_draw, odds_away : float
        Odds decimais (>1.0). Ex: 1.85, 3.40, 4.20.

    Returns
    -------
    dict com p_home, p_draw, p_away normalizados (soma=1.0).
    """
    if odds_home <= 1.0 or odds_draw <= 1.0 or odds_away <= 1.0:
        # Odds inválidas — retorna uniforme
        return {"p_home": 1 / 3, "p_draw": 1 / 3, "p_away": 1 / 3}

    # Implied probs brutas (com vig)
    p_h = 1.0 / odds_home
    p_d = 1.0 / odds_draw
    p_a = 1.0 / odds_away

    total = p_h + p_d + p_a
    if total <= 0:
        return {"p_home": 1 / 3, "p_draw": 1 / 3, "p_away": 1 / 3}

    return {
        "p_home": p_h / total,
        "p_draw": p_d / total,
        "p_away": p_a / total,
    }


def blend_with_market(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    alpha: float = 0.6,
) -> dict[str, float]:
    """Combina probs do modelo com probs devigged do mercado.

    p_final = alpha * p_modelo + (1 - alpha) * p_market

    Parameters
    ----------
    model_probs : dict com home_win_prob, draw_prob, away_win_prob.
    market_probs : dict com p_home, p_draw, p_away (devigged).
    alpha : float
        Peso do modelo [0, 1]. 0.6 = 60% modelo, 40% mercado.

    Returns
    -------
    dict com home_win_prob, draw_prob, away_win_prob blendado.
    """
    a = max(0.0, min(1.0, alpha))

    p_h = a * model_probs.get("home_win_prob", 0.33) + (1 - a) * market_probs.get("p_home", 0.33)
    p_d = a * model_probs.get("draw_prob", 0.33) + (1 - a) * market_probs.get("p_draw", 0.33)
    p_a = a * model_probs.get("away_win_prob", 0.33) + (1 - a) * market_probs.get("p_away", 0.33)

    total = p_h + p_d + p_a
    if total > 0:
        p_h /= total
        p_d /= total
        p_a /= total

    return {
        "home_win_prob": p_h,
        "draw_prob": p_d,
        "away_win_prob": p_a,
    }


def consensus_devig(
    bookmaker_odds: list[dict[str, float]],
) -> dict[str, float] | None:
    """Calcula consenso devigged de múltiplas casas.

    Parameters
    ----------
    bookmaker_odds : list[dict]
        Lista de dicts com odds_home, odds_draw, odds_away de cada casa.

    Returns
    -------
    dict com p_home, p_draw, p_away (média das devigged). None se vazio.
    """
    if not bookmaker_odds:
        return None

    totals = {"p_home": 0.0, "p_draw": 0.0, "p_away": 0.0}
    n = 0
    for bo in bookmaker_odds:
        try:
            d = devig_odds(
                float(bo["odds_home"]),
                float(bo["odds_draw"]),
                float(bo["odds_away"]),
            )
            totals["p_home"] += d["p_home"]
            totals["p_draw"] += d["p_draw"]
            totals["p_away"] += d["p_away"]
            n += 1
        except (KeyError, ValueError, TypeError):
            continue

    if n == 0:
        return None

    return {k: v / n for k, v in totals.items()}
