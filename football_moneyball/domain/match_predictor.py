"""Modulo de previsao de partidas via Monte Carlo + Poisson.

Estima probabilidades de resultados de partidas de futebol usando
distribuicao de Poisson sobre Expected Goals (xG) e simulacao
Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# xG Estimation
# ---------------------------------------------------------------------------

def estimate_team_xg(
    team_history: pd.DataFrame,
    opponent_history: pd.DataFrame,
    is_home: bool,
    n_games: int = 6,
    decay: float = 0.85,
    home_advantage: float = 0.25,
) -> float:
    """Estima o xG esperado de um time para uma partida futura.

    Usa media ponderada exponencial dos ultimos N jogos, ajustada
    pela forca do adversario (xG concedido) e fator casa.

    Parameters
    ----------
    team_history : pd.DataFrame
        Historico do time com coluna 'xg' (ultimas partidas, mais
        recente primeiro).
    opponent_history : pd.DataFrame
        Historico do adversario com coluna 'xg' (gols sofridos).
    is_home : bool
        Se o time joga em casa.
    n_games : int
        Numero de jogos para considerar.
    decay : float
        Fator de decaimento exponencial (0-1). Jogos mais recentes
        pesam mais.
    home_advantage : float
        Bonus de xG para time da casa.

    Returns
    -------
    float
        xG esperado (sempre >= 0.1).
    """
    # Get recent xG values
    recent_xg = team_history["xg"].head(n_games).values
    if len(recent_xg) == 0:
        return 1.0  # default

    # Exponential weighted average
    weights = np.array([decay ** i for i in range(len(recent_xg))])
    weights /= weights.sum()
    avg_xg = float(np.dot(recent_xg, weights))

    # Opponent strength adjustment
    # If opponent concedes more than average, boost our xG
    if not opponent_history.empty and "xg_against" in opponent_history.columns:
        opp_xga = opponent_history["xg_against"].head(n_games).mean()
        league_avg_xg = 1.25  # ~1.25 xG per team per game in Brasileirao
        strength_factor = opp_xga / league_avg_xg if league_avg_xg > 0 else 1.0
        avg_xg *= strength_factor

    # Home advantage
    if is_home:
        avg_xg += home_advantage

    return max(avg_xg, 0.1)


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

def simulate_match(
    home_xg: float,
    away_xg: float,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Simula uma partida N vezes via Monte Carlo + Poisson.

    Sorteia gols de cada time a partir de distribuicao Poisson com
    parametro lambda = xG esperado. Calcula probabilidades de todos
    os mercados de apostas.

    Parameters
    ----------
    home_xg : float
        xG esperado do time da casa.
    away_xg : float
        xG esperado do time visitante.
    n_simulations : int
        Numero de simulacoes Monte Carlo.
    seed : int, optional
        Seed para reprodutibilidade.

    Returns
    -------
    dict
        Dicionario com probabilidades:
        - home_win_prob, draw_prob, away_win_prob
        - over_05, over_15, over_25, over_35 (probabilidades)
        - btts_prob (ambas marcam)
        - most_likely_score (str "HxA")
        - score_matrix (dict de score -> probabilidade)
        - home_xg, away_xg (inputs)
    """
    rng = np.random.default_rng(seed)

    home_goals = rng.poisson(home_xg, n_simulations)
    away_goals = rng.poisson(away_xg, n_simulations)
    total_goals = home_goals + away_goals

    # Match outcomes
    home_wins = (home_goals > away_goals).sum()
    draws = (home_goals == away_goals).sum()
    away_wins = (home_goals < away_goals).sum()

    # Over/Under
    over_05 = (total_goals > 0.5).sum()
    over_15 = (total_goals > 1.5).sum()
    over_25 = (total_goals > 2.5).sum()
    over_35 = (total_goals > 3.5).sum()

    # BTTS
    btts = ((home_goals > 0) & (away_goals > 0)).sum()

    # Score matrix
    score_counts: dict[str, int] = {}
    for h, a in zip(home_goals, away_goals):
        key = f"{h}x{a}"
        score_counts[key] = score_counts.get(key, 0) + 1

    # Most likely score
    most_likely = max(score_counts, key=score_counts.get) if score_counts else "0x0"

    # Top 10 scores by probability
    score_matrix = {
        k: round(v / n_simulations, 4)
        for k, v in sorted(score_counts.items(), key=lambda x: -x[1])[:10]
    }

    n = n_simulations
    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win_prob": round(home_wins / n, 4),
        "draw_prob": round(draws / n, 4),
        "away_win_prob": round(away_wins / n, 4),
        "over_05": round(over_05 / n, 4),
        "over_15": round(over_15 / n, 4),
        "over_25": round(over_25 / n, 4),
        "over_35": round(over_35 / n, 4),
        "btts_prob": round(btts / n, 4),
        "most_likely_score": most_likely,
        "score_matrix": score_matrix,
        "simulations": n_simulations,
    }


def poisson_pmf(k: int, lam: float) -> float:
    """Calcula P(X=k) para distribuicao Poisson com parametro lambda.

    Parameters
    ----------
    k : int
        Numero de ocorrencias.
    lam : float
        Parametro lambda (taxa media).

    Returns
    -------
    float
        Probabilidade P(X=k).
    """
    if lam <= 0 or k < 0:
        return 0.0
    from math import factorial
    return float(np.exp(-lam) * (lam ** k) / factorial(k))
