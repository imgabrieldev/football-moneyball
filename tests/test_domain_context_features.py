"""Tests for football_moneyball.domain.context_features."""

from football_moneyball.domain.context_features import (
    coach_features,
    fixture_features,
    injury_features,
    position_features,
)


class TestCoachFeatures:
    def test_defaults_when_none(self):
        result = coach_features(None)
        assert result["coach_win_rate"] == 0.5
        assert result["games_since_change"] == 10
        assert result["coach_change_recent"] == 0

    def test_win_rate_clamped(self):
        result = coach_features({"coach_win_rate": 1.5})
        assert result["coach_win_rate"] == 1.0
        result = coach_features({"coach_win_rate": -0.3})
        assert result["coach_win_rate"] == 0.0

    def test_recent_flag(self):
        assert coach_features({"coach_change_recent": True})["coach_change_recent"] == 1
        assert coach_features({"coach_change_recent": False})["coach_change_recent"] == 0

    def test_games_clamp(self):
        assert coach_features({"games_since_change": 500})["games_since_change"] == 100
        assert coach_features({"games_since_change": -5})["games_since_change"] == 0

    def test_typical_values(self):
        info = {
            "coach_win_rate": 0.73,
            "games_since_change": 5,
            "coach_change_recent": True,
        }
        result = coach_features(info)
        assert abs(result["coach_win_rate"] - 0.73) < 1e-9
        assert result["games_since_change"] == 5
        assert result["coach_change_recent"] == 1


class TestInjuryFeatures:
    def test_defaults(self):
        result = injury_features(None)
        assert result["key_players_out"] == 0
        assert result["xg_contribution_missing"] == 0.0

    def test_clamp_key_out(self):
        assert injury_features({"key_players_out": 100})["key_players_out"] == 5
        assert injury_features({"key_players_out": -1})["key_players_out"] == 0

    def test_xg_missing_clamp(self):
        assert injury_features({"xg_contribution_missing": 2.0})["xg_contribution_missing"] == 1.0
        assert injury_features({"xg_contribution_missing": -0.1})["xg_contribution_missing"] == 0.0

    def test_typical(self):
        info = {"key_players_out": 2, "xg_contribution_missing": 0.45}
        result = injury_features(info)
        assert result["key_players_out"] == 2
        assert abs(result["xg_contribution_missing"] - 0.45) < 1e-9


class TestFixtureFeatures:
    def test_defaults(self):
        result = fixture_features()
        assert result["games_last_7d"] == 0
        assert result["games_next_7d"] == 0

    def test_clamp(self):
        assert fixture_features(games_last_7d=10)["games_last_7d"] == 5

    def test_typical(self):
        result = fixture_features(games_last_7d=2, games_next_7d=3)
        assert result["games_last_7d"] == 2
        assert result["games_next_7d"] == 3


class TestPositionFeatures:
    def test_defaults(self):
        result = position_features(None)
        assert result["home_position"] == 10
        assert result["away_position"] == 10
        assert result["position_gap"] == 0
        assert result["both_in_relegation"] == 0

    def test_typical_values(self):
        gap = {
            "home_position": 8, "away_position": 15,
            "position_gap": -7, "both_in_relegation": False,
        }
        result = position_features(gap)
        assert result["home_position"] == 8
        assert result["away_position"] == 15
        assert result["position_gap"] == -7
        assert result["both_in_relegation"] == 0

    def test_both_in_relegation(self):
        result = position_features({
            "home_position": 18, "away_position": 19,
            "position_gap": -1, "both_in_relegation": True,
        })
        assert result["both_in_relegation"] == 1
