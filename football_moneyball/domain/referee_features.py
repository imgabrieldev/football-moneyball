"""Referee features — tendências do árbitro.

Lógica pura: recebe dict com stats do árbitro e retorna features numéricas.
"""
from __future__ import annotations

from typing import Any

# Média histórica Brasileirão — usada como fallback quando ref é desconhecido
LEAGUE_AVG_CARDS_PER_GAME = 4.2


def compute_referee_features(
    referee_stats: dict[str, Any] | None,
    league_avg_cards: float = LEAGUE_AVG_CARDS_PER_GAME,
    min_matches: int = 5,
) -> dict[str, float]:
    """Calcula features de árbitro.

    Parameters
    ----------
    referee_stats : dict | None
        {referee_id, name, matches, yellow_total, red_total, cards_per_game}.
        None = árbitro desconhecido/não designado → usa médias da liga.
    league_avg_cards : float
        Média de cartões/jogo da liga (default 4.2 pro Brasileirão).
    min_matches : int
        Mínimo de jogos pra usar stats do ref (evita outliers).

    Returns
    -------
    dict com 3 features:
        ref_cards_per_game — média histórica
        ref_strictness — desvio normalizado vs liga [-1, 1]
        ref_experience — proxy baseado em #jogos (clip a [0, 1])
    """
    if not referee_stats or referee_stats.get("matches", 0) < min_matches:
        return {
            "ref_cards_per_game": league_avg_cards,
            "ref_strictness": 0.0,
            "ref_experience": 0.0,
        }

    cpg = float(referee_stats.get("cards_per_game") or league_avg_cards)
    matches = int(referee_stats.get("matches") or 0)

    # Strictness: desvio normalizado [-1, 1]. ~2 stdev da média (clip).
    strictness = (cpg - league_avg_cards) / max(league_avg_cards, 0.1)
    strictness = max(-1.0, min(1.0, strictness))

    # Experience: sigmoid-like, 30 jogos = saturação
    experience = min(1.0, matches / 30.0)

    return {
        "ref_cards_per_game": cpg,
        "ref_strictness": strictness,
        "ref_experience": experience,
    }
