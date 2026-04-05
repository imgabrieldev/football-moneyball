"""Testes para football_moneyball.domain.h2h_features."""

from football_moneyball.domain.h2h_features import compute_h2h_features


class TestComputeH2HFeatures:
    def test_empty_history_returns_defaults(self):
        features = compute_h2h_features([], "Flamengo", "Vasco")
        assert features["h2h_home_win_rate"] == 0.33
        assert features["h2h_draw_rate"] == 0.25
        assert features["h2h_n_matches"] == 0.0

    def test_dominance_home_team(self):
        # Flamengo ganhou todos os 3 jogos
        history = [
            {"home_team": "Flamengo", "away_team": "Vasco", "home_goals": 3, "away_goals": 0},
            {"home_team": "Vasco", "away_team": "Flamengo", "home_goals": 0, "away_goals": 2},
            {"home_team": "Flamengo", "away_team": "Vasco", "home_goals": 2, "away_goals": 1},
        ]
        f = compute_h2h_features(history, "Flamengo", "Vasco")
        assert f["h2h_home_win_rate"] == 1.0
        assert f["h2h_away_win_rate"] == 0.0
        assert f["h2h_draw_rate"] == 0.0
        assert f["h2h_home_goals_avg"] == 7 / 3  # 3+2+2=7
        assert f["h2h_away_goals_avg"] == 1 / 3  # 0+0+1=1
        assert f["h2h_n_matches"] == 3.0

    def test_handles_reverse_fixture(self):
        # Team A foi visitante em alguns jogos
        history = [
            {"home_team": "B", "away_team": "A", "home_goals": 1, "away_goals": 2},
        ]
        f = compute_h2h_features(history, "A", "B")
        # A (home atual) ganhou o jogo sendo visitante
        assert f["h2h_home_win_rate"] == 1.0
        assert f["h2h_home_goals_avg"] == 2.0
        assert f["h2h_away_goals_avg"] == 1.0

    def test_all_draws(self):
        history = [
            {"home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 1},
            {"home_team": "B", "away_team": "A", "home_goals": 2, "away_goals": 2},
        ]
        f = compute_h2h_features(history, "A", "B")
        assert f["h2h_draw_rate"] == 1.0
        assert f["h2h_home_win_rate"] == 0.0
        assert f["h2h_away_win_rate"] == 0.0

    def test_ignores_irrelevant_matches(self):
        history = [
            {"home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0},
            {"home_team": "C", "away_team": "D", "home_goals": 1, "away_goals": 1},  # irrelevante
        ]
        f = compute_h2h_features(history, "A", "B")
        assert f["h2h_n_matches"] == 1.0
        assert f["h2h_home_win_rate"] == 1.0
