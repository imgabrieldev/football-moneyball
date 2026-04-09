"""Multi-dimensional Monte Carlo — simulates goals + corners + cards + shots + HT.

Each simulation = 1 complete match with all metrics sampled from independent
Poisson. From each simulation we derive ~50+ different markets.

Pure logic — zero infra deps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_full_match(
    lambdas: dict,
    n_simulations: int = 10_000,
    seed: int | None = None,
    dixon_coles_rho: float | None = -0.10,
) -> pd.DataFrame:
    """Simulate N full matches (goals with Dixon-Coles correction).

    Parameters
    ----------
    lambdas : dict
        Expected keys:
        home_goals, away_goals,
        home_corners, away_corners,
        home_cards, away_cards,
        home_shots, away_shots,
        home_ht_goals, away_ht_goals.
    n_simulations : int
        Number of simulations.
    seed : int, optional
    dixon_coles_rho : float | None
        Dixon-Coles correction applied only to goals (negative = more draws).
        None disables it (reverts to independent Poisson).

    Returns
    -------
    pd.DataFrame
        One row per simulation, columns: home_goals, away_goals,
        home_corners, away_corners, total_corners, home_cards, away_cards,
        total_cards, home_shots, away_shots, ht_home, ht_away.
    """
    rng = np.random.default_rng(seed)

    lam_h = max(lambdas.get("home_goals", 1.3), 0.1)
    lam_a = max(lambdas.get("away_goals", 1.1), 0.1)
    if dixon_coles_rho is not None:
        from football_moneyball.domain.calibration import sample_scores_dixon_coles
        home_goals, away_goals = sample_scores_dixon_coles(
            lam_h, lam_a, dixon_coles_rho, n_simulations, seed=seed,
        )
    else:
        home_goals = rng.poisson(lam_h, n_simulations)
        away_goals = rng.poisson(lam_a, n_simulations)
    home_corners = rng.poisson(max(lambdas.get("home_corners", 5.0), 0.5), n_simulations)
    away_corners = rng.poisson(max(lambdas.get("away_corners", 4.5), 0.5), n_simulations)
    home_cards = rng.poisson(max(lambdas.get("home_cards", 2.0), 0.1), n_simulations)
    away_cards = rng.poisson(max(lambdas.get("away_cards", 2.0), 0.1), n_simulations)
    home_shots = rng.poisson(max(lambdas.get("home_shots", 12.0), 1.0), n_simulations)
    away_shots = rng.poisson(max(lambdas.get("away_shots", 10.0), 1.0), n_simulations)
    ht_home = rng.poisson(max(lambdas.get("home_ht_goals", 0.6), 0.05), n_simulations)
    ht_away = rng.poisson(max(lambdas.get("away_ht_goals", 0.5), 0.05), n_simulations)

    return pd.DataFrame({
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "total_corners": home_corners + away_corners,
        "home_cards": home_cards,
        "away_cards": away_cards,
        "total_cards": home_cards + away_cards,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "ht_home": ht_home,
        "ht_away": ht_away,
    })


def derive_markets_from_sims(sim_df: pd.DataFrame) -> dict:
    """Derive multi-dimensional markets from the simulation DataFrame.

    Returns
    -------
    dict
        corners, cards, ht_result, ht_ft, margin, shots_home, shots_away.
    """
    n = len(sim_df)
    if n == 0:
        return {}

    def _ou(col: str, lines: list[float]) -> list[dict]:
        values = sim_df[col]
        return [
            {
                "line": line,
                "over_prob": round(float((values > line).sum() / n), 4),
                "under_prob": round(float((values <= line).sum() / n), 4),
            }
            for line in lines
        ]

    # HT result
    ht_home_wins = float(((sim_df.ht_home > sim_df.ht_away).sum()) / n)
    ht_draws = float(((sim_df.ht_home == sim_df.ht_away).sum()) / n)
    ht_away_wins = float(((sim_df.ht_home < sim_df.ht_away).sum()) / n)

    # HT/FT combos
    home_wins = sim_df.home_goals > sim_df.away_goals
    draws = sim_df.home_goals == sim_df.away_goals
    away_wins = sim_df.home_goals < sim_df.away_goals
    ht_home_w = sim_df.ht_home > sim_df.ht_away
    ht_draw = sim_df.ht_home == sim_df.ht_away
    ht_away_w = sim_df.ht_home < sim_df.ht_away

    ht_ft = {
        "home_home": round(float((ht_home_w & home_wins).sum() / n), 4),
        "home_draw": round(float((ht_home_w & draws).sum() / n), 4),
        "home_away": round(float((ht_home_w & away_wins).sum() / n), 4),
        "draw_home": round(float((ht_draw & home_wins).sum() / n), 4),
        "draw_draw": round(float((ht_draw & draws).sum() / n), 4),
        "draw_away": round(float((ht_draw & away_wins).sum() / n), 4),
        "away_home": round(float((ht_away_w & home_wins).sum() / n), 4),
        "away_draw": round(float((ht_away_w & draws).sum() / n), 4),
        "away_away": round(float((ht_away_w & away_wins).sum() / n), 4),
    }

    # Margin of victory
    margin_diff = sim_df.home_goals - sim_df.away_goals
    margins = {}
    for d in [-3, -2, -1, 0, 1, 2, 3]:
        margins[str(d)] = round(float((margin_diff == d).sum() / n), 4)
    margins["home_by_4_plus"] = round(float((margin_diff >= 4).sum() / n), 4)
    margins["away_by_4_plus"] = round(float((margin_diff <= -4).sum() / n), 4)

    return {
        "corners": _ou("total_corners", [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]),
        "corners_home": _ou("home_corners", [3.5, 4.5, 5.5, 6.5]),
        "corners_away": _ou("away_corners", [3.5, 4.5, 5.5, 6.5]),
        "cards": _ou("total_cards", [2.5, 3.5, 4.5, 5.5, 6.5]),
        "cards_home": _ou("home_cards", [1.5, 2.5, 3.5]),
        "cards_away": _ou("away_cards", [1.5, 2.5, 3.5]),
        "shots_home": _ou("home_shots", [9.5, 11.5, 13.5, 15.5]),
        "shots_away": _ou("away_shots", [9.5, 11.5, 13.5, 15.5]),
        "ht_result": {
            "home_prob": round(ht_home_wins, 4),
            "draw_prob": round(ht_draws, 4),
            "away_prob": round(ht_away_wins, 4),
        },
        "ht_ft": ht_ft,
        "margin_of_victory": margins,
    }
