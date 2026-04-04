"""Modulo de predicao de mercados individuais por jogador.

Usa Poisson com lambda = metric_per_90 × minutes_expected / 90.

Logica pura — zero deps de infra.
"""

from __future__ import annotations

from math import exp, factorial

import pandas as pd


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X ≤ k) para Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    cdf = 0.0
    for i in range(k + 1):
        cdf += exp(-lam) * (lam ** i) / factorial(i)
    return cdf


def predict_player_goal(
    xg_per_90: float,
    minutes_expected: float,
) -> float:
    """P(jogador ≥ 1 gol) via Poisson.

    λ_player = xg_per_90 × minutes_expected / 90
    P(≥ 1) = 1 - e^(-λ)

    Parameters
    ----------
    xg_per_90 : float
        xG por 90 min do jogador.
    minutes_expected : float
        Minutos esperados (0-90).

    Returns
    -------
    float
        Probabilidade em [0, 1].
    """
    if xg_per_90 <= 0 or minutes_expected <= 0:
        return 0.0
    lam = xg_per_90 * minutes_expected / 90.0
    return round(1.0 - exp(-lam), 4)


def predict_player_multiple_goals(
    xg_per_90: float,
    minutes_expected: float,
    n: int = 2,
) -> float:
    """P(jogador marca ≥ n gols) via Poisson.

    P(X ≥ n) = 1 - P(X ≤ n-1)
    """
    if xg_per_90 <= 0 or minutes_expected <= 0:
        return 0.0
    lam = xg_per_90 * minutes_expected / 90.0
    return round(1.0 - _poisson_cdf(n - 1, lam), 4)


def predict_player_assist(
    xa_per_90: float,
    minutes_expected: float,
) -> float:
    """P(jogador ≥ 1 assistencia) via Poisson.

    Mesma formula do gol, mas com xA/90.
    """
    if xa_per_90 <= 0 or minutes_expected <= 0:
        return 0.0
    lam = xa_per_90 * minutes_expected / 90.0
    return round(1.0 - exp(-lam), 4)


def predict_player_shots(
    shots_per_90: float,
    minutes_expected: float,
    lines: list[float] | None = None,
) -> list[dict]:
    """P(shots do jogador ≥ line) pra cada linha.

    Returns
    -------
    list[dict]
        [{"line": 0.5, "over_prob": 0.82}, ...]
    """
    if lines is None:
        lines = [0.5, 1.5, 2.5]
    if shots_per_90 <= 0 or minutes_expected <= 0:
        return [{"line": l, "over_prob": 0.0} for l in lines]

    lam = shots_per_90 * minutes_expected / 90.0
    result = []
    for line in lines:
        # P(shots > line) = P(shots ≥ ceil(line)) = 1 - cdf(floor(line))
        k = int(line)
        prob = 1.0 - _poisson_cdf(k, lam)
        result.append({"line": line, "over_prob": round(prob, 4)})
    return result


def predict_player_scores_or_assists(
    xg_per_90: float,
    xa_per_90: float,
    minutes_expected: float,
) -> float:
    """P(jogador marca OU assiste) via independencia.

    P(A ou B) = 1 - P(nao A) × P(nao B)
    """
    p_goal = predict_player_goal(xg_per_90, minutes_expected)
    p_assist = predict_player_assist(xa_per_90, minutes_expected)
    return round(1.0 - (1.0 - p_goal) * (1.0 - p_assist), 4)


def compute_team_player_props(
    player_aggregates: pd.DataFrame,
    last_n: int = 5,
    min_matches: int = 3,
    top_n: int = 5,
) -> list[dict]:
    """Calcula props pra top N jogadores mais regulares.

    Parameters
    ----------
    player_aggregates : pd.DataFrame
        Deve conter: player_id, player_name, matches_played, minutes_total,
        xg_total, xa_total (opcional), shots_total, assists_total (opcional).
    last_n : int
        Janela historica usada pro aggregates.
    min_matches : int
        Minimo de partidas pra incluir jogador.
    top_n : int
        Quantos jogadores retornar por time.

    Returns
    -------
    list[dict]
        Lista ordenada por minutes_total. Cada dict tem probs por mercado.
    """
    if player_aggregates.empty:
        return []

    df = player_aggregates.copy()

    # Filtrar por min_matches
    df = df[df["matches_played"] >= min_matches]
    if df.empty:
        return []

    # Ordenar por minutes_total descendente, top N
    df = df.sort_values("minutes_total", ascending=False).head(top_n)

    results = []
    for _, row in df.iterrows():
        minutes_total = float(row.get("minutes_total", 0) or 0)
        matches = int(row.get("matches_played", 0) or 0)
        if matches == 0:
            continue

        # Minutos esperados = media de minutos por jogo
        minutes_expected = min(minutes_total / matches, 90.0)

        xg_per_90 = (float(row.get("xg_total", 0) or 0) / minutes_total) * 90.0 if minutes_total > 0 else 0
        xa_per_90 = (float(row.get("xa_total", 0) or 0) / minutes_total) * 90.0 if minutes_total > 0 else 0
        shots_per_90 = (float(row.get("shots_total", 0) or 0) / minutes_total) * 90.0 if minutes_total > 0 else 0

        goal_prob = predict_player_goal(xg_per_90, minutes_expected)
        goal_2plus_prob = predict_player_multiple_goals(xg_per_90, minutes_expected, n=2)
        assist_prob = predict_player_assist(xa_per_90, minutes_expected)
        scores_or_assists_prob = predict_player_scores_or_assists(
            xg_per_90, xa_per_90, minutes_expected
        )
        shots_markets = predict_player_shots(shots_per_90, minutes_expected)

        results.append({
            "player_id": int(row.get("player_id", 0) or 0),
            "player_name": str(row.get("player_name", "")),
            "minutes_expected": round(minutes_expected, 1),
            "xg_per_90": round(xg_per_90, 3),
            "xa_per_90": round(xa_per_90, 3),
            "shots_per_90": round(shots_per_90, 3),
            "goal_prob": goal_prob,
            "goal_2plus_prob": goal_2plus_prob,
            "assist_prob": assist_prob,
            "scores_or_assists_prob": scores_or_assists_prob,
            "shots": shots_markets,
        })

    return results
