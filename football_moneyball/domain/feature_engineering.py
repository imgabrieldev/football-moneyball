"""Feature engineering para predicao de λ via ML.

Funcoes puras que constroem vetores de features a partir de dicts/DataFrames
de estatisticas de times. Zero deps de infra.

Feature layout (12 dimensoes):
  0: team_goals_for_avg
  1: team_goals_against_avg
  2: team_xg_for_avg
  3: team_xg_against_avg
  4: team_corners_for
  5: team_cards_for
  6: opp_goals_for_avg
  7: opp_goals_against_avg
  8: opp_xg_for_avg
  9: opp_xg_against_avg
  10: league_goals_per_team
  11: is_home (0 ou 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_DIM = 12


def build_team_features(
    team_stats: dict,
    opponent_stats: dict,
    league_avg: dict,
    is_home: bool,
) -> np.ndarray:
    """Constroi vetor de 12 features pra predizer λ do time.

    Parameters
    ----------
    team_stats : dict
        Medias do time (atacante). Chaves esperadas:
        goals_for, goals_against, xg_for, xg_against, corners_for, cards_for.
    opponent_stats : dict
        Medias do adversario. Mesmas chaves.
    league_avg : dict
        Medias da liga. Chaves: goals_per_team, corners_per_team.
    is_home : bool
        True se o time (atacante) joga em casa.

    Returns
    -------
    np.ndarray
        Array de 12 floats.
    """
    league_goals = league_avg.get("goals_per_team", 1.3)

    features = np.array([
        float(team_stats.get("goals_for", league_goals)),
        float(team_stats.get("goals_against", league_goals)),
        float(team_stats.get("xg_for", league_goals)),
        float(team_stats.get("xg_against", league_goals)),
        float(team_stats.get("corners_for", 5.0)),
        float(team_stats.get("cards_for", 2.0)),
        float(opponent_stats.get("goals_for", league_goals)),
        float(opponent_stats.get("goals_against", league_goals)),
        float(opponent_stats.get("xg_for", league_goals)),
        float(opponent_stats.get("xg_against", league_goals)),
        float(league_goals),
        1.0 if is_home else 0.0,
    ], dtype=np.float64)

    return features


def _team_rolling_stats(
    past_matches: pd.DataFrame,
    team: str,
    last_n: int = 5,
) -> dict:
    """Calcula medias dos ultimos N jogos do time nos dados fornecidos.

    past_matches: DataFrame com colunas match_id, home_team, away_team,
    home_goals, away_goals, home_xg, away_xg, home_corners, away_corners,
    home_cards, away_cards.
    """
    if past_matches.empty or "home_team" not in past_matches.columns:
        return {
            "goals_for": 1.3, "goals_against": 1.3,
            "xg_for": 1.3, "xg_against": 1.3,
            "corners_for": 5.0, "cards_for": 2.0,
        }

    mask = (past_matches["home_team"] == team) | (past_matches["away_team"] == team)
    team_matches = past_matches[mask].tail(last_n)

    if team_matches.empty:
        return {
            "goals_for": 1.3, "goals_against": 1.3,
            "xg_for": 1.3, "xg_against": 1.3,
            "corners_for": 5.0, "cards_for": 2.0,
        }

    goals_for, goals_against = [], []
    xg_for, xg_against = [], []
    corners_for, cards_for = [], []

    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        if is_home:
            goals_for.append(row.get("home_goals", 0) or 0)
            goals_against.append(row.get("away_goals", 0) or 0)
            xg_for.append(row.get("home_xg", 0) or 0)
            xg_against.append(row.get("away_xg", 0) or 0)
            corners_for.append(row.get("home_corners", 0) or 0)
            cards_for.append(row.get("home_cards", 0) or 0)
        else:
            goals_for.append(row.get("away_goals", 0) or 0)
            goals_against.append(row.get("home_goals", 0) or 0)
            xg_for.append(row.get("away_xg", 0) or 0)
            xg_against.append(row.get("home_xg", 0) or 0)
            corners_for.append(row.get("away_corners", 0) or 0)
            cards_for.append(row.get("away_cards", 0) or 0)

    return {
        "goals_for": float(np.mean(goals_for)),
        "goals_against": float(np.mean(goals_against)),
        "xg_for": float(np.mean(xg_for)),
        "xg_against": float(np.mean(xg_against)),
        "corners_for": float(np.mean(corners_for)),
        "cards_for": float(np.mean(cards_for)),
    }


def build_training_dataset(
    matches_with_stats: pd.DataFrame,
    target: str = "goals",
    last_n: int = 5,
    min_prior: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Constroi (X, y) para treino usando APENAS dados anteriores a cada partida.

    Para cada partida (home, away) gera 2 samples:
    - Sample 1: features(home vs away, is_home=1), y = home_{target}
    - Sample 2: features(away vs home, is_home=0), y = away_{target}

    Parameters
    ----------
    matches_with_stats : pd.DataFrame
        Colunas obrigatorias: match_id, match_date, home_team, away_team,
        home_goals, away_goals, home_xg, away_xg, home_corners, away_corners,
        home_cards, away_cards.
    target : str
        'goals' | 'corners' | 'cards'.
    last_n : int
        Janela de features (ultimos N jogos).
    min_prior : int
        Minimo de jogos anteriores pra incluir na amostra.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X com shape (2n, 12), y com shape (2n,).
    """
    target_col_home = f"home_{target}"
    target_col_away = f"away_{target}"

    df = matches_with_stats.sort_values("match_date").reset_index(drop=True)

    # League averages globais (aproximacao — nao time-varying pra simplicidade)
    league_avg = {
        "goals_per_team": float(
            (df["home_goals"].fillna(0).mean() + df["away_goals"].fillna(0).mean()) / 2
        ) if not df.empty else 1.3,
        "corners_per_team": float(
            (df["home_corners"].fillna(5).mean() + df["away_corners"].fillna(5).mean()) / 2
        ) if not df.empty else 5.0,
    }

    X_rows = []
    y_rows = []

    for idx, row in df.iterrows():
        past = df.iloc[:idx]
        home = row["home_team"]
        away = row["away_team"]

        home_hist = _team_rolling_stats(past, home, last_n)
        away_hist = _team_rolling_stats(past, away, last_n)

        # Contar quantos jogos historicos ja teve pra cada time
        n_home_prior = ((past["home_team"] == home) | (past["away_team"] == home)).sum()
        n_away_prior = ((past["home_team"] == away) | (past["away_team"] == away)).sum()

        if n_home_prior < min_prior or n_away_prior < min_prior:
            continue

        # Sample 1: home ataca
        X_rows.append(build_team_features(home_hist, away_hist, league_avg, is_home=True))
        y_rows.append(float(row.get(target_col_home, 0) or 0))

        # Sample 2: away ataca
        X_rows.append(build_team_features(away_hist, home_hist, league_avg, is_home=False))
        y_rows.append(float(row.get(target_col_away, 0) or 0))

    if not X_rows:
        return np.zeros((0, FEATURE_DIM)), np.zeros(0)

    return np.vstack(X_rows), np.array(y_rows)
