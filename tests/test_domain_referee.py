"""Tests for football_moneyball.domain.referee."""

from football_moneyball.domain.referee import (
    cards_per_game_from_totals,
    referee_strictness_factor,
)


class TestRefereeStrictness:
    def test_average_referee(self):
        assert referee_strictness_factor(4.5, 4.5) == 1.0

    def test_strict_referee(self):
        # 6 cards/game vs 4 avg = 1.5 strictness
        assert referee_strictness_factor(6.0, 4.0) == 1.5

    def test_lenient_referee(self):
        # 2 cards/game vs 4 avg = 0.5 strictness
        assert referee_strictness_factor(2.0, 4.0) == 0.5

    def test_clamp_upper(self):
        # 12 vs 4 = 3.0 -> clamped to 2.0
        assert referee_strictness_factor(12.0, 4.0) == 2.0

    def test_clamp_lower(self):
        # 0.5 vs 4 = 0.125 -> clamped to 0.5
        assert referee_strictness_factor(0.5, 4.0) == 0.5

    def test_zero_league(self):
        assert referee_strictness_factor(5.0, 0) == 1.0

    def test_zero_referee(self):
        assert referee_strictness_factor(0, 4.0) == 1.0


class TestCardsPerGameFromTotals:
    def test_klein_example(self):
        # Rafael Klein: 607 yellow + 12 yellowred + 30 red / 106 games
        result = cards_per_game_from_totals(607, 12, 30, 106)
        assert abs(result - 6.122) < 0.01

    def test_zero_matches(self):
        assert cards_per_game_from_totals(100, 0, 5, 0) == 0.0

    def test_all_yellow(self):
        assert cards_per_game_from_totals(40, 0, 0, 10) == 4.0
