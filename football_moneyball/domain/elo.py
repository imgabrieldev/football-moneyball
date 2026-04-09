"""Dynamic Elo rating for football teams.

FiveThirtyEight-style implementation: home advantage + margin of victory
multiplier. Ratings start at 1500 and are updated after each match.

Pure logic — zero infra deps.

References:
- 538 blog: https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
- Wikipedia Elo: https://en.wikipedia.org/wiki/Elo_rating_system
"""

from __future__ import annotations

from math import log
from typing import Iterable

import pandas as pd


class EloRating:
    """Dynamic Elo rating with home advantage and margin-of-victory.

    Parameters
    ----------
    initial : float
        Initial rating for new teams (default 1500).
    k : float
        K-factor — magnitude of update per match (default 20).
        Larger K = faster change. 538 uses 20 for NFL.
    home_advantage : float
        Point bonus for the home team in the expected score formula (default 50).
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
        """Return current rating of the team (1500 if new)."""
        return self.ratings.get(team, self.initial)

    def set(self, team: str, rating: float) -> None:
        self.ratings[team] = rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """P(A wins) based on rating difference.

        Classic Elo formula: 1 / (1 + 10^((rating_b - rating_a) / 400))
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _mov_multiplier(self, goal_diff: int, elo_diff: float) -> float:
        """Margin of Victory multiplier (538-style).

        Adjusts update magnitude by scoreline — a 3+ win is worth more
        than 1x0. Also corrects for 'autocorrelation bias' (a strong
        team winning by a lot should not rise proportionally).

        538 formula: ln(|goal_diff| + 1) * 2.2 / (elo_diff_favored * 0.001 + 2.2)
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
        """Update ratings after a match.

        Returns
        -------
        tuple[float, float]
            (delta_home, delta_away) — changes in the ratings.
        """
        elo_home = self.get(home)
        elo_away = self.get(away)

        # Effective rating of the home team includes home advantage
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
        # elo_diff_favored: diff from the favorite's perspective (who won)
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
    """Chronological replay of all matches, returns each team's PRE-match Elo.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Required columns: match_id, match_date, home_team, away_team,
        home_goals, away_goals.
    k : float
        Elo K-factor.

    Returns
    -------
    dict[tuple[int, str], float]
        Mapping {(match_id, team): elo_before_match}.
        Used to avoid data leakage in training features.
    """
    elo = EloRating(initial=initial, k=k, home_advantage=home_advantage)
    timeline: dict[tuple[int, str], float] = {}

    if matches_df.empty:
        return timeline

    # Sort chronologically
    df = matches_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    for _, row in df.iterrows():
        mid = int(row["match_id"])
        home = str(row["home_team"])
        away = str(row["away_team"])

        # Record Elo BEFORE updating
        timeline[(mid, home)] = elo.get(home)
        timeline[(mid, away)] = elo.get(away)

        # Skip matches without a result
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
    """Final ratings after processing all matches."""
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
