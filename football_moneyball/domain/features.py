"""Centralized feature engineering for CatBoost v1.15.0.

Pure functions — zero infra dependencies (DB, API).
Receive already-extracted data and return numeric features.
"""

from __future__ import annotations


def compute_xg_form_ema(
    xg_history: list[float],
    alpha: float = 0.15,
    default: float = 1.2,
) -> float:
    """EMA of xG produced in the last N matches.

    Unlike xg_avg (simple mean of the last 5), this EMA
    weights recent matches more heavily and uses a larger window.
    """
    if not xg_history:
        return default
    ema = default
    for xg in xg_history:
        ema = alpha * xg + (1 - alpha) * ema
    return ema


def compute_xg_diff_ema(
    xg_for: list[float],
    xg_against: list[float],
    alpha: float = 0.15,
    default: float = 0.0,
) -> float:
    """EMA of (xG For - xG Against). Measures offensive vs defensive dominance."""
    n = min(len(xg_for), len(xg_against))
    if n == 0:
        return default
    ema = default
    for i in range(n):
        diff = xg_for[i] - xg_against[i]
        ema = alpha * diff + (1 - alpha) * ema
    return ema


def compute_coach_features(coach_info: dict | None) -> dict:
    """Extract coach features from the dict returned by the repo.

    Expected: {coach_name, games_since_change, coach_change_recent, coach_win_rate}
    Returns a dict with 3 features (for home OR away — caller prefixes).
    """
    if not coach_info:
        return {
            "tenure_days": 180.0,
            "win_rate": 0.40,
            "changed_30d": 0.0,
        }
    games = coach_info.get("games_since_change", 10)
    # Proxy: ~3.5 days per match in the Brasileirao
    tenure_approx = games * 3.5
    return {
        "tenure_days": min(float(tenure_approx), 365.0),
        "win_rate": float(coach_info.get("coach_win_rate", 0.40)),
        "changed_30d": 1.0 if coach_info.get("coach_change_recent", False) else 0.0,
    }


def compute_standings_features(
    standings_info: dict | None,
) -> dict:
    """Extract standings features from the dict returned by the repo.

    Expected: {home_position, away_position, position_gap, points_gap, both_in_relegation}
    """
    if not standings_info:
        return {
            "home_position": 10.0,
            "away_position": 10.0,
            "position_gap": 0.0,
            "home_points_last_5": 7.0,
            "away_points_last_5": 7.0,
        }
    return {
        "home_position": float(standings_info.get("home_position", 10)),
        "away_position": float(standings_info.get("away_position", 10)),
        "position_gap": float(abs(standings_info.get("position_gap", 0))),
        "home_points_last_5": float(standings_info.get("home_points_last_5", 7)),
        "away_points_last_5": float(standings_info.get("away_points_last_5", 7)),
    }


def compute_points_last_n(
    results: list[float],
    n: int = 5,
) -> float:
    """Compute points in the last N matches (3=win, 1=draw, 0=loss).

    results: list of floats where 1.0=win, 0.5=draw, 0.0=loss.
    """
    if not results:
        return 7.0  # ~1.4 pts/match * 5 matches
    last_n = results[-n:]
    pts = 0.0
    for r in last_n:
        if r >= 0.9:
            pts += 3.0
        elif r >= 0.4:
            pts += 1.0
    return pts
