"""Testes para football_moneyball.domain.elo."""

import pandas as pd

from football_moneyball.domain.elo import (
    EloRating,
    compute_elo_timeline,
    final_elo_ratings,
)


class TestEloRatingInit:
    def test_default_rating(self):
        elo = EloRating()
        assert elo.get("NewTeam") == 1500.0

    def test_custom_initial(self):
        elo = EloRating(initial=1000)
        assert elo.get("NewTeam") == 1000.0

    def test_set_rating(self):
        elo = EloRating()
        elo.set("TeamA", 1700)
        assert elo.get("TeamA") == 1700.0


class TestExpectedScore:
    def test_equal_ratings(self):
        elo = EloRating()
        assert abs(elo.expected_score(1500, 1500) - 0.5) < 1e-9

    def test_higher_rating_favored(self):
        elo = EloRating()
        assert elo.expected_score(1700, 1500) > 0.5
        assert elo.expected_score(1500, 1700) < 0.5

    def test_large_gap(self):
        # 400 Elo gap → 10:1 odds → ~0.909 expected
        elo = EloRating()
        p = elo.expected_score(1900, 1500)
        assert 0.9 < p < 0.92


class TestUpdate:
    def test_draw_between_equals(self):
        elo = EloRating(k=20, home_advantage=0)
        elo.set("A", 1500); elo.set("B", 1500)
        elo.update("A", "B", 1, 1)
        # Draw com expected 0.5 → delta = 0
        assert abs(elo.get("A") - 1500) < 1e-6
        assert abs(elo.get("B") - 1500) < 1e-6

    def test_home_advantage_matters(self):
        # With home advantage, draw favors away slightly
        elo = EloRating(k=20, home_advantage=50)
        elo.set("A", 1500); elo.set("B", 1500)
        elo.update("A", "B", 1, 1)
        # Home expected > 0.5, actual = 0.5 → home loses rating
        assert elo.get("A") < 1500
        assert elo.get("B") > 1500

    def test_zero_sum(self):
        elo = EloRating(k=20, home_advantage=0)
        elo.set("A", 1500); elo.set("B", 1500)
        elo.update("A", "B", 2, 0)
        total_before = 3000
        total_after = elo.get("A") + elo.get("B")
        assert abs(total_after - total_before) < 1e-6

    def test_underdog_beats_favorite(self):
        elo = EloRating(k=20, home_advantage=0)
        elo.set("Strong", 1700); elo.set("Weak", 1300)
        elo.update("Weak", "Strong", 1, 0)
        # MoV dampens delta for big elo_diff, mas weak ganha ~10pts
        assert elo.get("Weak") > 1305
        assert elo.get("Strong") < 1695

    def test_favorite_beats_underdog_small_delta(self):
        # Favorite beating underdog should gain little (already expected)
        elo = EloRating(k=20, home_advantage=0)
        elo.set("Strong", 1700); elo.set("Weak", 1300)
        elo.update("Strong", "Weak", 1, 0)
        # Expected outcome, small delta
        delta = elo.get("Strong") - 1700
        assert 0 < delta < 5

    def test_mov_multiplier_scales_wins(self):
        # Win by 3 gives bigger delta than win by 1
        elo1 = EloRating(k=20, home_advantage=0)
        elo1.set("A", 1500); elo1.set("B", 1500)
        elo1.update("A", "B", 1, 0)
        delta_1 = elo1.get("A") - 1500

        elo3 = EloRating(k=20, home_advantage=0)
        elo3.set("A", 1500); elo3.set("B", 1500)
        elo3.update("A", "B", 3, 0)
        delta_3 = elo3.get("A") - 1500

        assert delta_3 > delta_1

    def test_k_factor_scaling(self):
        elo20 = EloRating(k=20, home_advantage=0)
        elo20.set("A", 1500); elo20.set("B", 1500)
        elo20.update("A", "B", 1, 0)
        delta_20 = elo20.get("A") - 1500

        elo40 = EloRating(k=40, home_advantage=0)
        elo40.set("A", 1500); elo40.set("B", 1500)
        elo40.update("A", "B", 1, 0)
        delta_40 = elo40.get("A") - 1500

        # Roughly 2x bigger
        assert abs(delta_40 / delta_20 - 2.0) < 0.1


class TestComputeEloTimeline:
    def _make_matches(self):
        """3 matches sinteticos."""
        return pd.DataFrame([
            {"match_id": 1, "match_date": "2026-01-01",
             "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0},
            {"match_id": 2, "match_date": "2026-01-08",
             "home_team": "B", "away_team": "C", "home_goals": 1, "away_goals": 1},
            {"match_id": 3, "match_date": "2026-01-15",
             "home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 1},
        ])

    def test_pre_match_elo_first_match(self):
        df = self._make_matches()
        timeline = compute_elo_timeline(df)
        # Match 1: A and B both start at 1500
        assert timeline[(1, "A")] == 1500.0
        assert timeline[(1, "B")] == 1500.0

    def test_elo_propagates(self):
        df = self._make_matches()
        timeline = compute_elo_timeline(df)
        # Match 2: B played match 1 (lost to A), so B's elo should be < 1500
        assert timeline[(2, "B")] < 1500.0
        # C played nothing yet → 1500
        assert timeline[(2, "C")] == 1500.0

    def test_final_ratings(self):
        df = self._make_matches()
        ratings = final_elo_ratings(df)
        # A won 2, B lost 1 + drew 1, C drew 1 + lost 1
        # Expected: A highest, C lowest
        assert ratings["A"] > ratings["B"]
        assert ratings["A"] > ratings["C"]

    def test_empty_df(self):
        assert compute_elo_timeline(pd.DataFrame()) == {}
        assert final_elo_ratings(pd.DataFrame()) == {}

    def test_skip_unresolved(self):
        df = pd.DataFrame([
            {"match_id": 1, "match_date": "2026-01-01",
             "home_team": "A", "away_team": "B", "home_goals": None, "away_goals": None},
        ])
        ratings = final_elo_ratings(df)
        # No matches resolved → empty
        assert ratings == {}
