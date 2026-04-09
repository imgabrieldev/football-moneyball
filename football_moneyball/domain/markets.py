"""Betting markets derivation module.

Given the output of the Monte Carlo (probabilities + score_matrix),
it derives all the betting markets that Betfair offers without
needing a new model.
"""

from __future__ import annotations


def _poisson_prob(k: int, lam: float) -> float:
    """P(X=k) for Poisson."""
    if lam <= 0 or k < 0:
        return 0.0
    from math import exp, factorial
    return exp(-lam) * (lam ** k) / factorial(k)


def _generate_score_matrix_from_xg(home_xg: float, away_xg: float) -> dict:
    """Generate score_matrix via analytic Poisson when we have no Monte Carlo."""
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
    """Derive all betting markets from a Monte Carlo prediction.

    Receives the predict_match/simulate_match dict and returns complete
    markets for Betfair.

    Parameters
    ----------
    prediction : dict
        Monte Carlo output with home_win_prob, draw_prob, away_win_prob,
        over_05..over_35, btts_prob, score_matrix.

    Returns
    -------
    dict
        Markets: match_odds, over_under, btts, correct_score, asian_handicap.
    """
    home = prediction.get("home_team", "?")
    away = prediction.get("away_team", "?")
    score_matrix = prediction.get("score_matrix", {})

    # If there is no score_matrix (pre-computed predictions), generate via Poisson
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
    """1X2 market."""
    return [
        {"outcome": f"{home} Win", "prob": pred.get("home_win_prob", 0), "fair_odds": _prob_to_odds(pred.get("home_win_prob", 0))},
        {"outcome": "Draw", "prob": pred.get("draw_prob", 0), "fair_odds": _prob_to_odds(pred.get("draw_prob", 0))},
        {"outcome": f"{away} Win", "prob": pred.get("away_win_prob", 0), "fair_odds": _prob_to_odds(pred.get("away_win_prob", 0))},
    ]


def _derive_over_under(pred: dict) -> list[dict]:
    """Over/Under 0.5, 1.5, 2.5, 3.5.

    Uses stored fields if available, otherwise derives analytically from xG
    via Poisson (P(total > line)).
    """
    # Try using stored fields
    stored = {
        "0.5": pred.get("over_05"),
        "1.5": pred.get("over_15"),
        "2.5": pred.get("over_25"),
        "3.5": pred.get("over_35"),
    }

    # If any line is missing, derive analytically
    home_xg = pred.get("home_xg") or pred.get("home_xg_expected")
    away_xg = pred.get("away_xg") or pred.get("away_xg_expected")

    def _over_prob_poisson(line: float, h_xg: float, a_xg: float) -> float:
        """P(total_goals > line) via independent bivariate Poisson."""
        if not h_xg or not a_xg:
            return 0.0
        # P(H=i) * P(A=j) where i+j > line
        prob = 0.0
        for i in range(10):
            for j in range(10):
                if i + j > line:
                    prob += _poisson_prob(i, h_xg) * _poisson_prob(j, a_xg)
        return prob

    lines = [("0.5", 0.5), ("1.5", 1.5), ("2.5", 2.5), ("3.5", 3.5)]
    result = []
    for label, line_val in lines:
        over_prob = stored.get(label)
        if over_prob is None or over_prob == 0:
            # Derive from xG
            if home_xg and away_xg:
                over_prob = _over_prob_poisson(line_val, float(home_xg), float(away_xg))
            else:
                over_prob = 0.0
        under_prob = 1.0 - over_prob
        result.append({
            "line": label,
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
    """Top most likely scores with fair odds."""
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
    """Asian Handicap derived from the score matrix.

    Computes the probability of each handicap by summing the relevant scores.
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

    # Extend: take ALL simulated scores, not just top 10
    # The score_matrix has the top 10, but for the handicap we need all
    # We will use what we have as an approximation

    handicaps = []
    for line in [-0.5, -1.5, -2.5, 0.5, 1.5, 2.5]:
        # Home handicap: home_goals + line > away_goals?
        home_prob = sum(prob for h, a, prob in parsed if (h + line) > a)
        away_prob = sum(prob for h, a, prob in parsed if (h + line) < a)
        # Normalize (score_matrix does not sum to 100%)
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
    """Convert probability into fair decimal odds."""
    if prob <= 0:
        return 99.0
    if prob >= 1:
        return 1.01
    return round(1.0 / prob, 2)
