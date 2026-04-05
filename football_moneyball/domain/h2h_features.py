"""Head-to-head features — históricos entre os dois times.

Lógica pura: recebe lista de resultados H2H e retorna features numéricas.
"""
from __future__ import annotations

from typing import Any


def compute_h2h_features(
    h2h_results: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    default_avg_goals: float = 1.3,
) -> dict[str, float]:
    """Calcula features H2H a partir de lista de resultados.

    Parameters
    ----------
    h2h_results : list[dict]
        Lista de partidas passadas entre os dois times. Cada dict com:
        {home_team, away_team, home_goals, away_goals}.
    home_team, away_team : str
        Time mandante e visitante atuais (pra orientar os resultados).
    default_avg_goals : float
        Valor default quando não há histórico H2H.

    Returns
    -------
    dict com 5 features:
        h2h_home_win_rate — % vitórias do home_team atual (em qualquer mando)
        h2h_away_win_rate — % vitórias do away_team atual
        h2h_draw_rate — % empates
        h2h_home_goals_avg — média gols do home_team atual
        h2h_away_goals_avg — média gols do away_team atual
    """
    if not h2h_results:
        return {
            "h2h_home_win_rate": 0.33,
            "h2h_away_win_rate": 0.33,
            "h2h_draw_rate": 0.25,
            "h2h_home_goals_avg": default_avg_goals,
            "h2h_away_goals_avg": default_avg_goals,
            "h2h_n_matches": 0.0,
        }

    home_wins = 0
    away_wins = 0
    draws = 0
    home_goals_total = 0
    away_goals_total = 0
    n = 0

    for match in h2h_results:
        mh = match.get("home_team", "")
        ma = match.get("away_team", "")
        hg = match.get("home_goals", 0) or 0
        ag = match.get("away_goals", 0) or 0

        # Identificar quem é nosso home_team naquele jogo (ele pode ter sido visitante)
        if mh == home_team and ma == away_team:
            # Jogo no mesmo mando que o atual
            home_team_goals = hg
            away_team_goals = ag
        elif mh == away_team and ma == home_team:
            # Jogo invertido
            home_team_goals = ag
            away_team_goals = hg
        else:
            # Partida não envolve exatos os dois times — pular
            continue

        n += 1
        home_goals_total += home_team_goals
        away_goals_total += away_team_goals

        if home_team_goals > away_team_goals:
            home_wins += 1
        elif home_team_goals < away_team_goals:
            away_wins += 1
        else:
            draws += 1

    if n == 0:
        return {
            "h2h_home_win_rate": 0.33,
            "h2h_away_win_rate": 0.33,
            "h2h_draw_rate": 0.25,
            "h2h_home_goals_avg": default_avg_goals,
            "h2h_away_goals_avg": default_avg_goals,
            "h2h_n_matches": 0.0,
        }

    return {
        "h2h_home_win_rate": home_wins / n,
        "h2h_away_win_rate": away_wins / n,
        "h2h_draw_rate": draws / n,
        "h2h_home_goals_avg": home_goals_total / n,
        "h2h_away_goals_avg": away_goals_total / n,
        "h2h_n_matches": float(n),
    }
