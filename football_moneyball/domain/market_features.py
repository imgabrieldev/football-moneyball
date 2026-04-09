"""Market blending — devig + ensemble with market probabilities.

Betting markets (Pinnacle, Betfair) are ~80% efficient. Blending the model
prediction with the market significantly reduces error.

Pure logic: zero infra deps.
"""
from __future__ import annotations


def devig_odds(odds_home: float, odds_draw: float, odds_away: float) -> dict[str, float]:
    """Remove vig from 1x2 odds. Returns normalized implied probabilities.

    Method: proportional normalization (simpler, power shrinkage removed).

    Parameters
    ----------
    odds_home, odds_draw, odds_away : float
        Decimal odds (>1.0). Ex: 1.85, 3.40, 4.20.

    Returns
    -------
    dict with p_home, p_draw, p_away normalized (sum=1.0).
    """
    if odds_home <= 1.0 or odds_draw <= 1.0 or odds_away <= 1.0:
        # Invalid odds — return uniform
        return {"p_home": 1 / 3, "p_draw": 1 / 3, "p_away": 1 / 3}

    # Raw implied probs (with vig)
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
    """Combine model probs with devigged market probs.

    p_final = alpha * p_model + (1 - alpha) * p_market

    Parameters
    ----------
    model_probs : dict with home_win_prob, draw_prob, away_win_prob.
    market_probs : dict with p_home, p_draw, p_away (devigged).
    alpha : float
        Model weight [0, 1]. 0.6 = 60% model, 40% market.

    Returns
    -------
    dict with blended home_win_prob, draw_prob, away_win_prob.
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
    """Compute the devigged consensus from multiple bookmakers.

    Parameters
    ----------
    bookmaker_odds : list[dict]
        List of dicts with odds_home, odds_draw, odds_away of each bookmaker.

    Returns
    -------
    dict with p_home, p_draw, p_away (mean of devigged). None if empty.
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
