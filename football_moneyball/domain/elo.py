"""Elo rating dinamico para times de futebol.

Implementacao FiveThirtyEight-style: home advantage + margin of victory
multiplier. Ratings comecam em 1500 e sao atualizados apos cada jogo.

Logica pura — zero deps de infra.

Referencias:
- 538 blog: https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
- Wikipedia Elo: https://en.wikipedia.org/wiki/Elo_rating_system
"""

from __future__ import annotations

from math import log
from typing import Iterable

import pandas as pd


class EloRating:
    """Elo rating dinamico com home advantage e margin-of-victory.

    Parameters
    ----------
    initial : float
        Rating inicial pra times novos (default 1500).
    k : float
        K-factor — magnitude de update por jogo (default 20).
        Maior K = mudanca mais rapida. 538 usa 20 pra NFL.
    home_advantage : float
        Bonus em pontos pro mandante na formula de expected score (default 50).
    """

    def __init__(
        self,
        initial: float = 1500.0,
        k: float = 20.0,
        home_advantage: float = 50.0,
    ) -> None:
        self.ratings: dict[str, float] = {}
        self.initial = initial
        self.k = k
        self.home_advantage = home_advantage

    def get(self, team: str) -> float:
        """Retorna rating atual do time (1500 se novo)."""
        return self.ratings.get(team, self.initial)

    def set(self, team: str, rating: float) -> None:
        self.ratings[team] = rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """P(A vence) baseado em diferenca de ratings.

        Formula Elo classica: 1 / (1 + 10^((rating_b - rating_a) / 400))
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _mov_multiplier(self, goal_diff: int, elo_diff: float) -> float:
        """Margin of Victory multiplier (538-style).

        Ajusta magnitude do update pelo placar — vitoria por 3+ vale mais
        que 1x0. Tambem corrige 'autocorrelation bias' (time forte
        ganhar por muito nao deveria subir proporcionalmente).

        Formula 538: ln(|goal_diff| + 1) × 2.2 / (elo_diff_favored × 0.001 + 2.2)
        """
        abs_diff = abs(goal_diff)
        if abs_diff == 0:
            return 1.0
        return log(abs_diff + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
    ) -> tuple[float, float]:
        """Atualiza ratings apos um jogo.

        Returns
        -------
        tuple[float, float]
            (delta_home, delta_away) — mudancas nos ratings.
        """
        elo_home = self.get(home)
        elo_away = self.get(away)

        # Effective rating do mandante inclui home advantage
        expected_home = self.expected_score(
            elo_home + self.home_advantage, elo_away
        )

        # Actual score: 1 win, 0.5 draw, 0 loss
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals < away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5

        # Margin of victory multiplier
        goal_diff = home_goals - away_goals
        # elo_diff_favored: diff do favorito perspective (quem ganhou)
        if actual_home == 1.0:
            elo_diff_fav = (elo_home + self.home_advantage) - elo_away
        elif actual_home == 0.0:
            elo_diff_fav = elo_away - (elo_home + self.home_advantage)
        else:
            elo_diff_fav = 0.0

        mov = self._mov_multiplier(goal_diff, elo_diff_fav)

        # Update
        delta_home = self.k * mov * (actual_home - expected_home)
        delta_away = -delta_home  # zero-sum

        self.ratings[home] = elo_home + delta_home
        self.ratings[away] = elo_away + delta_away

        return (delta_home, delta_away)


def compute_elo_timeline(
    matches_df: pd.DataFrame,
    k: float = 20.0,
    initial: float = 1500.0,
    home_advantage: float = 50.0,
) -> dict[tuple[int, str], float]:
    """Replay cronologico de todos os matches, retorna Elo PRE-match de cada time.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Colunas obrigatorias: match_id, match_date, home_team, away_team,
        home_goals, away_goals.
    k : float
        K-factor do Elo.

    Returns
    -------
    dict[tuple[int, str], float]
        Mapa {(match_id, team): elo_antes_do_jogo}.
        Usado pra evitar data leak em features de treino.
    """
    elo = EloRating(initial=initial, k=k, home_advantage=home_advantage)
    timeline: dict[tuple[int, str], float] = {}

    if matches_df.empty:
        return timeline

    # Sort cronologicamente
    df = matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    for _, row in df.iterrows():
        mid = int(row["match_id"])
        home = str(row["home_team"])
        away = str(row["away_team"])

        # Gravar Elo ANTES de atualizar
        timeline[(mid, home)] = elo.get(home)
        timeline[(mid, away)] = elo.get(away)

        # Skip matches sem resultado
        hg = row.get("home_goals")
        ag = row.get("away_goals")
        if hg is None or ag is None or pd.isna(hg) or pd.isna(ag):
            continue

        elo.update(home, away, int(hg), int(ag))

    return timeline


def final_elo_ratings(
    matches_df: pd.DataFrame,
    k: float = 20.0,
    initial: float = 1500.0,
    home_advantage: float = 50.0,
) -> dict[str, float]:
    """Ratings finais apos processar todos os matches."""
    elo = EloRating(initial=initial, k=k, home_advantage=home_advantage)
    if matches_df.empty:
        return {}
    df = matches_df.sort_values(["match_date", "match_id"])
    for _, row in df.iterrows():
        hg = row.get("home_goals")
        ag = row.get("away_goals")
        if hg is None or ag is None or pd.isna(hg) or pd.isna(ag):
            continue
        elo.update(
            str(row["home_team"]),
            str(row["away_team"]),
            int(hg),
            int(ag),
        )
    return dict(elo.ratings)
