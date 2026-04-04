"""Modulo de agregacao jogador->time pra calcular lambda (λ) de Poisson.

Logica pura — recebe DataFrames de jogadores (com xG/90 e peso) e
retorna o lambda (esperado de gols) do time. Substitui o path
team-level onde λ = league_avg × attack_strength × opp_defense.
"""

from __future__ import annotations

import pandas as pd


def compute_xg_per_90(xg_total: float, minutes_total: float) -> float:
    """xG por 90 minutos: (xg_total / minutes_total) × 90.

    Parameters
    ----------
    xg_total : float
        xG acumulado do jogador no periodo.
    minutes_total : float
        Minutos totais jogados no periodo.

    Returns
    -------
    float
        xG/90 ajustado. Retorna 0.0 se minutes_total <= 0.
    """
    if minutes_total <= 0:
        return 0.0
    return (xg_total / minutes_total) * 90.0


def team_lambda_from_players(
    xi: pd.DataFrame,
    opponent_defense_factor: float = 1.0,
) -> float:
    """Calcula λ do time a partir de agregacao de xG/90 dos titulares.

    λ_time = Σ(xG/90 × weight) × opponent_defense_factor

    O peso (weight) representa a probabilidade × participacao do
    jogador. Jogador que joga todos os jogos completos tem weight=1.0.

    Parameters
    ----------
    xi : pd.DataFrame
        DataFrame dos titulares com colunas xg_per_90 e weight.
    opponent_defense_factor : float
        Fator defensivo do adversario (de calculate_team_strength).
        1.0 = media da liga, <1.0 = defesa boa, >1.0 = defesa ruim.

    Returns
    -------
    float
        λ esperado de gols do time. Minimo 0.15.
    """
    if xi.empty or "xg_per_90" not in xi.columns or "weight" not in xi.columns:
        return 0.15

    team_attack = float((xi["xg_per_90"] * xi["weight"]).sum())
    lam = team_attack * opponent_defense_factor
    return max(lam, 0.15)


def summarize_xi(xi: pd.DataFrame) -> list[dict]:
    """Resumo da escalacao pra exibir no frontend.

    Parameters
    ----------
    xi : pd.DataFrame
        Output de probable_xi().

    Returns
    -------
    list[dict]
        Lista com nome, xG/90 e weight de cada jogador.
    """
    if xi.empty:
        return []
    return [
        {
            "player_id": int(row.get("player_id", 0) or 0),
            "player_name": str(row.get("player_name", "")),
            "xg_per_90": round(float(row.get("xg_per_90", 0) or 0), 3),
            "weight": round(float(row.get("weight", 0) or 0), 3),
            "minutes_total": float(row.get("minutes_total", 0) or 0),
        }
        for _, row in xi.iterrows()
    ]
