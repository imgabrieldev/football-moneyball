"""Value bet detection module.

Compares model probabilities with bookmaker odds
to identify bets with positive expectation.
"""

from __future__ import annotations


def odds_to_implied_prob(odds: float) -> float:
    """Convert decimal odds into implied probability.

    Parameters
    ----------
    odds : float
        Decimal odds (e.g. 1.80).

    Returns
    -------
    float
        Implied probability (e.g. 0.556).
    """
    if odds <= 0:
        return 0.0
    return round(1.0 / odds, 4)


def remove_vig(probs: list[float]) -> list[float]:
    """Remove the margin (vig/juice) from implied probabilities.

    Normalizes the probabilities to sum to 1.0.

    Parameters
    ----------
    probs : list[float]
        Implied probabilities (sum > 1.0 with vig).

    Returns
    -------
    list[float]
        Normalized real probabilities.
    """
    total = sum(probs)
    if total <= 0:
        return probs
    return [round(p / total, 4) for p in probs]


def calculate_edge(model_prob: float, implied_prob: float) -> float:
    """Compute the edge (advantage) of the model over the odds.

    Parameters
    ----------
    model_prob : float
        Probability estimated by the model.
    implied_prob : float
        Implied probability from the odds.

    Returns
    -------
    float
        Edge (positive = value bet).
    """
    return round(model_prob - implied_prob, 4)


def expected_value(model_prob: float, odds: float) -> float:
    """Compute the expected value of a bet.

    Parameters
    ----------
    model_prob : float
        Estimated probability.
    odds : float
        Decimal odds.

    Returns
    -------
    float
        EV per unit staked (positive = profitable).
    """
    return round(model_prob * odds - 1.0, 4)


# Mapping from prediction keys to odds market/outcome
_PREDICTION_TO_MARKET = {
    "home_win_prob": ("h2h", "Home"),
    "draw_prob": ("h2h", "Draw"),
    "away_win_prob": ("h2h", "Away"),
    "over_25": ("totals", "Over"),
    "btts_prob": ("btts", "Yes"),
}


def find_value_bets(
    predictions: dict,
    odds_data: list[dict],
    min_edge: float = 0.05,
    markets: list[str] | None = None,
) -> list[dict]:
    """Identify value bets by comparing the model with odds.

    Parameters
    ----------
    predictions : dict
        Output of simulate_match() with model probabilities.
    odds_data : list[dict]
        List of bookmaker odds in the format:
        [{"name": "bet365", "markets": [{"market": "h2h", "outcome": "Home", "odds": 1.80, ...}]}]
    min_edge : float
        Minimum edge to consider a value bet (default 3%).
    markets : list[str], optional
        Markets to consider. None = all.

    Returns
    -------
    list[dict]
        Value bets found with: market, outcome, model_prob,
        best_odds, bookmaker, implied_prob, edge, ev.
    """
    if markets is None:
        markets = ["h2h", "totals", "btts"]

    value_bets = []

    # Translation: "Home"/"Away" -> team names (odds API uses team names)
    home_team = predictions.get("home_team", "")
    away_team = predictions.get("away_team", "")
    outcome_aliases = {
        "Home": {home_team, "Home"},
        "Away": {away_team, "Away"},
        "Draw": {"Draw"},
        "Over": {"Over"},
        "Yes": {"Yes"},
    }

    for pred_key, (market, outcome) in _PREDICTION_TO_MARKET.items():
        if market not in markets:
            continue

        model_prob = predictions.get(pred_key, 0)
        if model_prob <= 0:
            continue

        matching_outcomes = outcome_aliases.get(outcome, {outcome})

        # Find best odds across bookmakers for this market/outcome
        best_odds = 0.0
        best_bookmaker = ""

        for bm in odds_data:
            bm_name = bm.get("name", "")
            for m in bm.get("markets", []):
                if m.get("market") == market and m.get("outcome") in matching_outcomes:
                    if m.get("odds", 0) > best_odds:
                        best_odds = m["odds"]
                        best_bookmaker = bm_name

        if best_odds <= 1.0:
            continue

        implied = odds_to_implied_prob(best_odds)
        edge = calculate_edge(model_prob, implied)
        ev = expected_value(model_prob, best_odds)

        if edge >= min_edge:
            value_bets.append({
                "market": market,
                "outcome": outcome,
                "model_prob": model_prob,
                "best_odds": best_odds,
                "bookmaker": best_bookmaker,
                "implied_prob": implied,
                "edge": edge,
                "ev": ev,
            })

    # Also check Under for totals
    under_prob = 1.0 - predictions.get("over_25", 0.5)
    for bm in odds_data:
        for m in bm.get("markets", []):
            if m.get("market") == "totals" and m.get("outcome") == "Under":
                implied = odds_to_implied_prob(m.get("odds", 0))
                edge = calculate_edge(under_prob, implied)
                if edge >= min_edge:
                    value_bets.append({
                        "market": "totals",
                        "outcome": "Under",
                        "model_prob": under_prob,
                        "best_odds": m["odds"],
                        "bookmaker": bm.get("name", ""),
                        "implied_prob": implied,
                        "edge": edge,
                        "ev": expected_value(under_prob, m["odds"]),
                    })
                break

    # BTTS No
    btts_no_prob = 1.0 - predictions.get("btts_prob", 0.5)
    for bm in odds_data:
        for m in bm.get("markets", []):
            if m.get("market") == "btts" and m.get("outcome") == "No":
                implied = odds_to_implied_prob(m.get("odds", 0))
                edge = calculate_edge(btts_no_prob, implied)
                if edge >= min_edge:
                    value_bets.append({
                        "market": "btts",
                        "outcome": "No",
                        "model_prob": btts_no_prob,
                        "best_odds": m["odds"],
                        "bookmaker": bm.get("name", ""),
                        "implied_prob": implied,
                        "edge": edge,
                        "ev": expected_value(btts_no_prob, m["odds"]),
                    })
                break

    return sorted(value_bets, key=lambda x: -x["edge"])
