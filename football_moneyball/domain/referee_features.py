"""Referee features — referee tendencies.

Pure logic: receives a dict with referee stats and returns numeric features.
"""
from __future__ import annotations

from typing import Any

# Historical Brasileirao average — used as fallback when the ref is unknown
LEAGUE_AVG_CARDS_PER_GAME = 4.2


def compute_referee_features(
    referee_stats: dict[str, Any] | None,
    league_avg_cards: float = LEAGUE_AVG_CARDS_PER_GAME,
    min_matches: int = 5,
) -> dict[str, float]:
    """Compute referee features.

    Parameters
    ----------
    referee_stats : dict | None
        {referee_id, name, matches, yellow_total, red_total, cards_per_game}.
        None = unknown/unassigned referee -> uses league averages.
    league_avg_cards : float
        League average cards/match (default 4.2 for Brasileirao).
    min_matches : int
        Minimum matches to use the ref's stats (avoids outliers).

    Returns
    -------
    dict with 3 features:
        ref_cards_per_game — historical average
        ref_strictness — normalized deviation vs league [-1, 1]
        ref_experience — proxy based on #matches (clipped to [0, 1])
    """
    if not referee_stats or referee_stats.get("matches", 0) < min_matches:
        return {
            "ref_cards_per_game": league_avg_cards,
            "ref_strictness": 0.0,
            "ref_experience": 0.0,
        }

    cpg = float(referee_stats.get("cards_per_game") or league_avg_cards)
    matches = int(referee_stats.get("matches") or 0)

    # Strictness: normalized deviation [-1, 1]. ~2 stdev from mean (clip).
    strictness = (cpg - league_avg_cards) / max(league_avg_cards, 0.1)
    strictness = max(-1.0, min(1.0, strictness))

    # Experience: sigmoid-like, 30 matches = saturation
    experience = min(1.0, matches / 30.0)

    return {
        "ref_cards_per_game": cpg,
        "ref_strictness": strictness,
        "ref_experience": experience,
    }
