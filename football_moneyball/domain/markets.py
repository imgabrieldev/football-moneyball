"""Modulo de derivacao de mercados de apostas.

A partir do output do Monte Carlo (probabilidades + score_matrix),
deriva todos os mercados de apostas que a Betfair oferece sem
precisar de modelo novo.
"""

from __future__ import annotations


def _poisson_prob(k: int, lam: float) -> float:
    """P(X=k) pra Poisson."""
    if lam <= 0 or k < 0:
        return 0.0
    from math import exp, factorial
    return exp(-lam) * (lam ** k) / factorial(k)


def _generate_score_matrix_from_xg(home_xg: float, away_xg: float) -> dict:
    """Gera score_matrix via Poisson analitico quando nao temos Monte Carlo."""
    if not home_xg or not away_xg:
        return {}
    scores = {}
    for h in range(6):
        for a in range(6):
            prob = _poisson_prob(h, home_xg) * _poisson_prob(a, away_xg)
            if prob > 0.005:  # > 0.5%
                scores[f"{h}x{a}"] = round(prob, 4)
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:15])


def derive_all_markets(prediction: dict) -> dict:
    """Deriva todos os mercados de apostas de uma previsao Monte Carlo.

    Recebe o dict de predict_match/simulate_match e retorna mercados
    completos pra Betfair.

    Parameters
    ----------
    prediction : dict
        Output do Monte Carlo com home_win_prob, draw_prob, away_win_prob,
        over_05..over_35, btts_prob, score_matrix.

    Returns
    -------
    dict
        Mercados: match_odds, over_under, btts, correct_score, asian_handicap.
    """
    home = prediction.get("home_team", "?")
    away = prediction.get("away_team", "?")
    score_matrix = prediction.get("score_matrix", {})

    # Se nao tem score_matrix (predictions pre-computadas), gerar via Poisson
    if not score_matrix:
        home_xg = prediction.get("home_xg") or prediction.get("home_xg_expected")
        away_xg = prediction.get("away_xg") or prediction.get("away_xg_expected")
        if home_xg and away_xg:
            score_matrix = _generate_score_matrix_from_xg(float(home_xg), float(away_xg))

    return {
        "match_odds": _derive_match_odds(prediction, home, away),
        "over_under": _derive_over_under(prediction),
        "btts": _derive_btts(prediction),
        "correct_score": _derive_correct_score(score_matrix),
        "asian_handicap": _derive_asian_handicap(score_matrix, home, away),
    }


def _derive_match_odds(pred: dict, home: str, away: str) -> list[dict]:
    """Mercado 1X2."""
    return [
        {"outcome": f"Vitória {home}", "prob": pred.get("home_win_prob", 0), "fair_odds": _prob_to_odds(pred.get("home_win_prob", 0))},
        {"outcome": "Empate", "prob": pred.get("draw_prob", 0), "fair_odds": _prob_to_odds(pred.get("draw_prob", 0))},
        {"outcome": f"Vitória {away}", "prob": pred.get("away_win_prob", 0), "fair_odds": _prob_to_odds(pred.get("away_win_prob", 0))},
    ]


def _derive_over_under(pred: dict) -> list[dict]:
    """Over/Under 0.5, 1.5, 2.5, 3.5."""
    lines = [
        ("0.5", pred.get("over_05", 0)),
        ("1.5", pred.get("over_15", 0)),
        ("2.5", pred.get("over_25", 0)),
        ("3.5", pred.get("over_35", 0)),
    ]
    result = []
    for line, over_prob in lines:
        under_prob = 1.0 - over_prob
        result.append({
            "line": line,
            "over_prob": round(over_prob, 4),
            "under_prob": round(under_prob, 4),
            "over_odds": _prob_to_odds(over_prob),
            "under_odds": _prob_to_odds(under_prob),
        })
    return result


def _derive_btts(pred: dict) -> dict:
    """Both Teams To Score."""
    btts = pred.get("btts_prob", 0)
    return {
        "yes_prob": round(btts, 4),
        "no_prob": round(1.0 - btts, 4),
        "yes_odds": _prob_to_odds(btts),
        "no_odds": _prob_to_odds(1.0 - btts),
    }


def _derive_correct_score(score_matrix: dict) -> list[dict]:
    """Top placares mais provaveis com odds justos."""
    if not score_matrix:
        return []

    return [
        {
            "score": score,
            "prob": round(prob, 4),
            "fair_odds": _prob_to_odds(prob),
        }
        for score, prob in sorted(score_matrix.items(), key=lambda x: -x[1])[:10]
    ]


def _derive_asian_handicap(score_matrix: dict, home: str, away: str) -> list[dict]:
    """Asian Handicap derivado do score matrix.

    Calcula probabilidade de cada handicap somando os placares relevantes.
    """
    if not score_matrix:
        return []

    # Parse scores
    parsed = []
    for score_str, prob in score_matrix.items():
        try:
            parts = score_str.split("x")
            h, a = int(parts[0]), int(parts[1])
            parsed.append((h, a, prob))
        except (ValueError, IndexError):
            continue

    # Extend: pegar TODOS os placares simulados, nao so top 10
    # O score_matrix tem top 10, mas pra handicap precisamos de todos
    # Vamos usar os que temos como aproximacao

    handicaps = []
    for line in [-0.5, -1.5, -2.5, 0.5, 1.5, 2.5]:
        # Home handicap: home_goals + line > away_goals?
        home_prob = sum(prob for h, a, prob in parsed if (h + line) > a)
        away_prob = sum(prob for h, a, prob in parsed if (h + line) < a)
        # Normalizar (score_matrix nao soma 100%)
        total = home_prob + away_prob
        if total > 0:
            home_prob /= total
            away_prob /= total

        if line < 0:
            label = f"{home} {line}"
        else:
            label = f"{home} +{line}"

        handicaps.append({
            "line": line,
            "label": label,
            "home_prob": round(home_prob, 4),
            "away_prob": round(away_prob, 4),
            "home_odds": _prob_to_odds(home_prob),
            "away_odds": _prob_to_odds(away_prob),
        })

    return handicaps


def _prob_to_odds(prob: float) -> float:
    """Converte probabilidade em odds decimais justos."""
    if prob <= 0:
        return 99.0
    if prob >= 1:
        return 1.01
    return round(1.0 / prob, 2)
