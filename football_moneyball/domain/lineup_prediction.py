"""Modulo de predicao de escalacao provavel (Probable XI).

Logica pura — zero deps de infra. Dado um DataFrame com agregados
por jogador nos ultimos N jogos, retorna os 11 titulares mais
provaveis + peso de cada um (frequencia × minutos).
"""

from __future__ import annotations

import pandas as pd


def minutes_weight(
    matches_played: int,
    minutes_total: float,
    last_n: int,
) -> float:
    """Peso de ser titular regular (0.0 a 1.0).

    weight = (matches_played / last_n) × (avg_minutes / 90)

    Exemplos
    --------
    Player com 5/5 jogos de 90 min → 1.0 × 1.0 = 1.0
    Player com 2/5 jogos de 90 min → 0.4 × 1.0 = 0.4
    Player com 5/5 jogos de 45 min → 1.0 × 0.5 = 0.5
    Player com 1/5 jogos de 45 min → 0.2 × 0.5 = 0.1

    Parameters
    ----------
    matches_played : int
        Numero de partidas que o jogador entrou.
    minutes_total : float
        Total de minutos jogados nas partidas.
    last_n : int
        Tamanho da janela de referencia (5 = ultimos 5 jogos).

    Returns
    -------
    float
        Peso no intervalo [0, 1].
    """
    if last_n <= 0 or matches_played <= 0:
        return 0.0
    start_rate = min(matches_played / last_n, 1.0)
    avg_minutes = minutes_total / matches_played
    minutes_rate = min(avg_minutes / 90.0, 1.0)
    return float(start_rate * minutes_rate)


def probable_xi(
    player_aggregates: pd.DataFrame,
    last_n_matches: int = 5,
) -> pd.DataFrame:
    """Retorna os 11 jogadores mais provaveis de serem titulares.

    Ordena por minutos totais (proxy de "mais usado") e pega os 11
    primeiros. Calcula peso de cada um em [0, 1].

    Parameters
    ----------
    player_aggregates : pd.DataFrame
        Deve conter: player_id, player_name, matches_played,
        minutes_total, xg_total.
    last_n_matches : int
        Janela de referencia usada pra calcular weight.

    Returns
    -------
    pd.DataFrame
        Top 11 jogadores, ordenados por minutes_total desc.
        Colunas: player_id, player_name, matches_played, minutes_total,
        xg_total, xg_per_90, weight.
    """
    if player_aggregates.empty:
        return pd.DataFrame(columns=[
            "player_id", "player_name", "matches_played", "minutes_total",
            "xg_total", "xg_per_90", "weight",
        ])

    df = player_aggregates.copy()

    # xG/90 individual
    df["xg_per_90"] = df.apply(
        lambda r: (float(r["xg_total"]) / float(r["minutes_total"])) * 90.0
        if float(r["minutes_total"]) > 0
        else 0.0,
        axis=1,
    )

    # Weight individual
    df["weight"] = df.apply(
        lambda r: minutes_weight(
            int(r["matches_played"]),
            float(r["minutes_total"]),
            last_n_matches,
        ),
        axis=1,
    )

    # Top 11 por minutos totais
    df = df.sort_values("minutes_total", ascending=False).head(11).reset_index(drop=True)

    cols = [
        "player_id", "player_name", "matches_played", "minutes_total",
        "xg_total", "xg_per_90", "weight",
    ]
    return df[[c for c in cols if c in df.columns]]
