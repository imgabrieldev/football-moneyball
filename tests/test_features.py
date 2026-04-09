"""Unit tests for domain/features.py (v1.15.0)."""

import pytest
from football_moneyball.domain.features import (
    compute_xg_form_ema,
    compute_xg_diff_ema,
    compute_coach_features,
    compute_standings_features,
    compute_points_last_n,
)


class TestXgFormEma:
    def test_empty_history_returns_default(self):
        assert compute_xg_form_ema([]) == 1.2
        assert compute_xg_form_ema([], default=2.0) == 2.0

    def test_single_value(self):
        result = compute_xg_form_ema([2.0], alpha=0.15, default=1.2)
        expected = 0.15 * 2.0 + 0.85 * 1.2
        assert abs(result - expected) < 1e-6

    def test_weights_recent_more(self):
        # Constant xG should converge to that value
        result = compute_xg_form_ema([3.0] * 50, alpha=0.15)
        assert abs(result - 3.0) < 0.01

    def test_reacts_to_change(self):
        # Low xG then sudden spike
        low = compute_xg_form_ema([0.5] * 10, alpha=0.15)
        high = compute_xg_form_ema([0.5] * 10 + [3.0], alpha=0.15)
        assert high > low


class TestXgDiffEma:
    def test_empty_returns_default(self):
        assert compute_xg_diff_ema([], []) == 0.0

    def test_mismatched_lengths_uses_min(self):
        result = compute_xg_diff_ema([2.0, 1.5], [1.0], alpha=0.15)
        expected = 0.15 * (2.0 - 1.0) + 0.85 * 0.0
        assert abs(result - expected) < 1e-6

    def test_positive_dominance(self):
        # Team that consistently outperforms
        result = compute_xg_diff_ema([2.0] * 10, [1.0] * 10)
        assert result > 0.5


class TestCoachFeatures:
    def test_none_returns_defaults(self):
        result = compute_coach_features(None)
        assert result["tenure_days"] == 180.0
        assert result["win_rate"] == 0.40
        assert result["changed_30d"] == 0.0

    def test_new_coach(self):
        result = compute_coach_features({
            "games_since_change": 2,
            "coach_change_recent": True,
            "coach_win_rate": 0.50,
        })
        assert result["tenure_days"] == 7.0  # 2 * 3.5
        assert result["win_rate"] == 0.50
        assert result["changed_30d"] == 1.0

    def test_veteran_coach(self):
        result = compute_coach_features({
            "games_since_change": 50,
            "coach_change_recent": False,
            "coach_win_rate": 0.65,
        })
        assert result["tenure_days"] == 175.0  # 50 * 3.5
        assert result["win_rate"] == 0.65
        assert result["changed_30d"] == 0.0

    def test_tenure_capped_at_365(self):
        result = compute_coach_features({
            "games_since_change": 200,
            "coach_change_recent": False,
            "coach_win_rate": 0.5,
        })
        assert result["tenure_days"] == 365.0


class TestStandingsFeatures:
    def test_none_returns_defaults(self):
        result = compute_standings_features(None)
        assert result["home_position"] == 10.0
        assert result["away_position"] == 10.0
        assert result["position_gap"] == 0.0

    def test_with_data(self):
        result = compute_standings_features({
            "home_position": 3,
            "away_position": 15,
            "position_gap": -12,
        })
        assert result["home_position"] == 3.0
        assert result["away_position"] == 15.0
        assert result["position_gap"] == 12.0  # abs value


class TestPointsLastN:
    def test_empty_returns_default(self):
        assert compute_points_last_n([]) == 7.0

    def test_all_wins(self):
        assert compute_points_last_n([1.0] * 5) == 15.0

    def test_all_draws(self):
        assert compute_points_last_n([0.5] * 5) == 5.0

    def test_all_losses(self):
        assert compute_points_last_n([0.0] * 5) == 0.0

    def test_mixed(self):
        # 2 wins + 1 draw + 2 losses = 7 pts
        results = [1.0, 1.0, 0.5, 0.0, 0.0]
        assert compute_points_last_n(results) == 7.0

    def test_only_last_n(self):
        # 10 results, only last 5 count
        results = [0.0] * 5 + [1.0] * 5
        assert compute_points_last_n(results, n=5) == 15.0


class TestBuildMatchFeatures:
    def test_returns_43_features(self):
        from football_moneyball.domain.catboost_predictor import (
            build_match_features, N_FEATURES,
        )
        features = build_match_features(
            pi_ratings={},
            home_team="TeamA",
            away_team="TeamB",
            home_form=[],
            away_form=[],
            home_gd=[],
            away_gd=[],
        )
        assert len(features) == N_FEATURES
        assert N_FEATURES == 43
