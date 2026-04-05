"""Testes para football_moneyball.domain.referee_features."""

from football_moneyball.domain.referee_features import (
    LEAGUE_AVG_CARDS_PER_GAME,
    compute_referee_features,
)


class TestComputeRefereeFeatures:
    def test_none_returns_defaults(self):
        f = compute_referee_features(None)
        assert f["ref_cards_per_game"] == LEAGUE_AVG_CARDS_PER_GAME
        assert f["ref_strictness"] == 0.0
        assert f["ref_experience"] == 0.0

    def test_strict_referee_positive_strictness(self):
        stats = {"referee_id": 1, "matches": 20, "cards_per_game": 5.5}
        f = compute_referee_features(stats, league_avg_cards=4.2)
        assert f["ref_strictness"] > 0.3
        assert f["ref_cards_per_game"] == 5.5

    def test_lenient_referee_negative_strictness(self):
        stats = {"referee_id": 1, "matches": 20, "cards_per_game": 2.8}
        f = compute_referee_features(stats, league_avg_cards=4.2)
        assert f["ref_strictness"] < -0.2

    def test_experience_caps_at_one(self):
        stats = {"referee_id": 1, "matches": 100, "cards_per_game": 4.0}
        f = compute_referee_features(stats)
        assert f["ref_experience"] == 1.0

    def test_low_sample_returns_defaults(self):
        stats = {"referee_id": 1, "matches": 3, "cards_per_game": 8.0}
        f = compute_referee_features(stats, min_matches=5)
        assert f["ref_strictness"] == 0.0
        assert f["ref_cards_per_game"] == LEAGUE_AVG_CARDS_PER_GAME
