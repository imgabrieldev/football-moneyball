"""Features contextuais: tecnico, lesoes, fadiga, posicao.

Logica pura — recebe dicts de contexto e retorna features numericas.
Sem deps de infra.

Research-backed (v1.6.0):
- "New manager bounce" modesto (~10 jogos) — regressao a media predomina
- Interim coaches NAO sao piores → usar win_rate real
- Fixture congestion afeta injury risk > performance direta
- Player Impact Metric: top-N ausentes reduz outcome
"""

from __future__ import annotations


def coach_features(coach_info: dict | None) -> dict:
    """Converte dict de coach em features ML-ready.

    Parameters
    ----------
    coach_info : dict | None
        {games_since_change, coach_change_recent, coach_win_rate}
        None = sem info, usa defaults neutros.

    Returns
    -------
    dict
        Chaves: coach_win_rate (0-1), games_since_change (0-100),
        coach_change_recent (0 ou 1).
    """
    if not coach_info:
        return {
            "coach_win_rate": 0.5,
            "games_since_change": 10,
            "coach_change_recent": 0,
        }

    # Clamp win_rate em [0.0, 1.0]
    win_rate = float(coach_info.get("coach_win_rate", 0.5) or 0.5)
    win_rate = max(0.0, min(win_rate, 1.0))

    # Clamp games_since em [0, 100]
    games_since = int(coach_info.get("games_since_change", 10) or 10)
    games_since = max(0, min(games_since, 100))

    recent = 1 if coach_info.get("coach_change_recent") else 0

    return {
        "coach_win_rate": win_rate,
        "games_since_change": games_since,
        "coach_change_recent": recent,
    }


def injury_features(injury_info: dict | None) -> dict:
    """Converte dict de desfalques em features.

    Parameters
    ----------
    injury_info : dict | None
        {key_players_out, xg_contribution_missing}.

    Returns
    -------
    dict
        key_players_out (0-N), xg_contribution_missing (0.0-1.0).
    """
    if not injury_info:
        return {"key_players_out": 0, "xg_contribution_missing": 0.0}

    key_out = int(injury_info.get("key_players_out", 0) or 0)
    key_out = max(0, min(key_out, 5))  # clamp

    xg_missing = float(injury_info.get("xg_contribution_missing", 0.0) or 0.0)
    xg_missing = max(0.0, min(xg_missing, 1.0))

    return {
        "key_players_out": key_out,
        "xg_contribution_missing": xg_missing,
    }


def fixture_features(
    games_last_7d: int = 0,
    games_next_7d: int = 0,
) -> dict:
    """Features de fadiga/calendario apertado.

    Returns
    -------
    dict
        games_last_7d (0-3), games_next_7d (0-3).
    """
    return {
        "games_last_7d": max(0, min(int(games_last_7d or 0), 5)),
        "games_next_7d": max(0, min(int(games_next_7d or 0), 5)),
    }


def position_features(gap_info: dict | None) -> dict:
    """Features de contexto de tabela.

    Parameters
    ----------
    gap_info : dict | None
        {home_position, away_position, position_gap, points_gap,
         both_in_relegation}

    Returns
    -------
    dict
        home_position (1-20), away_position (1-20), position_gap (-19 a 19),
        both_in_relegation (0/1).
    """
    if not gap_info:
        return {
            "home_position": 10, "away_position": 10,
            "position_gap": 0, "both_in_relegation": 0,
        }

    return {
        "home_position": int(gap_info.get("home_position", 10) or 10),
        "away_position": int(gap_info.get("away_position", 10) or 10),
        "position_gap": int(gap_info.get("position_gap", 0) or 0),
        "both_in_relegation": 1 if gap_info.get("both_in_relegation") else 0,
    }
