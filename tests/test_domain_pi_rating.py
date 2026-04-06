"""Testes para football_moneyball.domain.pi_rating."""

import numpy as np
import pandas as pd

from football_moneyball.domain.pi_rating import (
    PiRating,
    compute_all_ratings,
    init_promoted_teams,
    rating_diff,
    update_ratings,
)


class TestUpdateRatings:
    def test_home_win_increases_home_rating(self):
        r = {"A": PiRating(0.0, 0.0), "B": PiRating(0.0, 0.0)}
        update_ratings(r, "A", "B", 3, 0)
        assert r["A"].home > 0
        assert r["B"].away < 0

    def test_draw_no_change_when_expected(self):
        r = {"A": PiRating(1.0, 0.0), "B": PiRating(0.0, 1.0)}
        # expected diff = 1.0 - 1.0 = 0, actual diff = 0 → no error
        update_ratings(r, "A", "B", 1, 1)
        assert abs(r["A"].home - 1.0) < 0.01
        assert abs(r["B"].away - 1.0) < 0.01

    def test_goal_cap(self):
        r = {"A": PiRating(), "B": PiRating()}
        update_ratings(r, "A", "B", 7, 0, goal_cap=3)
        r2 = {"A": PiRating(), "B": PiRating()}
        update_ratings(r2, "A", "B", 3, 0, goal_cap=3)
        assert abs(r["A"].home - r2["A"].home) < 1e-9

    def test_new_team_initialized(self):
        r = {}
        update_ratings(r, "NewTeam", "OldTeam", 1, 0)
        assert "NewTeam" in r
        assert "OldTeam" in r


class TestComputeAllRatings:
    def test_from_match_data(self):
        data = pd.DataFrame({
            "match_id": [1, 1, 2, 2],
            "team": ["A", "B", "B", "A"],
            "goals": [2, 1, 3, 0],
            "xg": [1.5, 1.0, 2.0, 0.5],
            "is_home": [True, False, True, False],
        })
        ratings = compute_all_ratings(data)
        assert "A" in ratings
        assert "B" in ratings
        # A won at home, lost away → home rating > away rating
        assert ratings["A"].home > ratings["A"].away

    def test_convergence(self):
        # 100 matches, A always wins 2-1 at home
        rows = []
        for i in range(100):
            rows.extend([
                {"match_id": i, "team": "A", "goals": 2, "xg": 1.5, "is_home": True},
                {"match_id": i, "team": "B", "goals": 1, "xg": 1.0, "is_home": False},
            ])
        data = pd.DataFrame(rows)
        ratings = compute_all_ratings(data)
        # A's home rating should converge near 0.5 (goal diff 1 × gamma damping)
        assert 0.3 < ratings["A"].home < 1.5


class TestRatingDiff:
    def test_basic(self):
        r = {"A": PiRating(1.0, -0.5), "B": PiRating(0.5, 0.3)}
        assert abs(rating_diff(r, "A", "B") - 0.7) < 1e-9

    def test_unknown_team_zero(self):
        r = {"A": PiRating(1.0, 0.0)}
        assert abs(rating_diff(r, "A", "Unknown") - 1.0) < 1e-9


class TestInitPromotedTeams:
    def test_average_of_relegated(self):
        r = {
            "Rel1": PiRating(-0.5, -0.3),
            "Rel2": PiRating(-0.7, -0.5),
        }
        init_promoted_teams(r, ["NewTeam"], ["Rel1", "Rel2"])
        assert abs(r["NewTeam"].home - (-0.6)) < 1e-9
        assert abs(r["NewTeam"].away - (-0.4)) < 1e-9
