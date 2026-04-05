"""Testes para football_moneyball.domain.feature_engineering."""

import numpy as np
import pandas as pd

from football_moneyball.domain.feature_engineering import (
    FEATURE_DIM,
    FEATURE_NAMES,
    _team_rolling_stats,
    build_rich_team_features,
    build_team_features,
    build_training_dataset,
)


class TestBuildTeamFeatures:
    def test_feature_dim(self):
        team = {"goals_for": 1.5, "goals_against": 1.0, "xg_for": 1.4,
                "xg_against": 1.1, "corners_for": 6.0, "cards_for": 2.5}
        opp = {"goals_for": 1.2, "goals_against": 1.3, "xg_for": 1.1,
               "xg_against": 1.4, "corners_for": 4.0, "cards_for": 2.0}
        league = {"goals_per_team": 1.3, "corners_per_team": 5.0}
        features = build_team_features(team, opp, league, is_home=True)
        assert features.shape == (FEATURE_DIM,)
        assert features.dtype == np.float64

    def test_is_home_flag(self):
        team = {"goals_for": 1.5}
        opp = {"goals_for": 1.2}
        league = {"goals_per_team": 1.3}
        f_home = build_team_features(team, opp, league, is_home=True)
        f_away = build_team_features(team, opp, league, is_home=False)
        # is_home sits at index 11
        assert f_home[11] == 1.0
        assert f_away[11] == 0.0

    def test_feature_names_match_dim(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_59_features(self):
        assert FEATURE_DIM == 59

    def test_team_vs_opp_ordering(self):
        team = {"goals_for": 2.0, "goals_against": 0.5, "xg_for": 1.8,
                "xg_against": 0.6, "corners_for": 7.0, "cards_for": 3.0}
        opp = {"goals_for": 0.5, "goals_against": 2.0, "xg_for": 0.6,
               "xg_against": 1.8, "corners_for": 3.0, "cards_for": 1.5}
        league = {"goals_per_team": 1.3, "corners_per_team": 5.0}
        features = build_team_features(team, opp, league, is_home=True)
        # Primeiros 6 sao do team, proximos 4 do opponent
        assert features[0] == 2.0  # team_goals_for
        assert features[4] == 7.0  # team_corners_for
        assert features[6] == 0.5  # opp_goals_for

    def test_rich_features_elo_diff(self):
        team = {"goals_for": 1.5}
        opp = {"goals_for": 1.2}
        league = {"goals_per_team": 1.3}
        features = build_rich_team_features(
            team, opp, league, is_home=True,
            team_elo=1700, opp_elo=1500,
            team_rest_days=5, opp_rest_days=7,
        )
        # elo_diff sits at index 12
        assert features[12] == 200.0

    def test_rich_features_rest_days(self):
        team = {"goals_for": 1.5}
        opp = {"goals_for": 1.2}
        league = {"goals_per_team": 1.3}
        features = build_rich_team_features(
            team, opp, league, is_home=True,
            team_elo=1500, opp_elo=1500,
            team_rest_days=3, opp_rest_days=10,
        )
        # rest_days: team at 19, opp at 23
        assert features[19] == 3.0
        assert features[23] == 10.0

    def test_fallback_defaults(self):
        team = {}
        opp = {}
        league = {}
        features = build_team_features(team, opp, league, is_home=False)
        assert features.shape == (FEATURE_DIM,)

    def test_h2h_features_slot(self):
        from football_moneyball.domain.feature_engineering import (
            build_context_aware_features,
        )
        team = {"goals_for": 1.5}
        opp = {"goals_for": 1.2}
        league = {"goals_per_team": 1.3}
        h2h = {
            "h2h_home_win_rate": 0.8, "h2h_away_win_rate": 0.0,
            "h2h_draw_rate": 0.2, "h2h_home_goals_avg": 2.4, "h2h_away_goals_avg": 0.6,
        }
        features = build_context_aware_features(
            team, opp, league, is_home=True, h2h_features=h2h,
        )
        # H2H wins at index 48
        assert features[48] == 0.8
        assert features[49] == 0.0
        assert features[50] == 0.2

    def test_referee_features_slot(self):
        from football_moneyball.domain.feature_engineering import (
            build_context_aware_features,
        )
        team = {"goals_for": 1.5}
        opp = {"goals_for": 1.2}
        league = {"goals_per_team": 1.3}
        ref = {"ref_cards_per_game": 5.1, "ref_strictness": 0.25, "ref_experience": 0.9}
        features = build_context_aware_features(
            team, opp, league, is_home=True, referee_features=ref,
        )
        # Referee at indices 53-55
        assert features[53] == 5.1
        assert features[54] == 0.25
        assert features[55] == 0.9


class TestTeamRollingStats:
    def _make_history(self):
        """Histórico: Team A joga 3 jogos."""
        return pd.DataFrame([
            {"match_id": 1, "home_team": "A", "away_team": "B",
             "home_goals": 2, "away_goals": 1, "home_xg": 1.8, "away_xg": 0.9,
             "home_corners": 6, "away_corners": 3, "home_cards": 2, "away_cards": 3},
            {"match_id": 2, "home_team": "C", "away_team": "A",
             "home_goals": 0, "away_goals": 1, "home_xg": 0.5, "away_xg": 1.2,
             "home_corners": 4, "away_corners": 5, "home_cards": 3, "away_cards": 2},
            {"match_id": 3, "home_team": "A", "away_team": "D",
             "home_goals": 3, "away_goals": 0, "home_xg": 2.5, "away_xg": 0.3,
             "home_corners": 8, "away_corners": 2, "home_cards": 1, "away_cards": 4},
        ])

    def test_team_A_goals_for_avg(self):
        hist = self._make_history()
        stats = _team_rolling_stats(hist, "A", last_n=5)
        # A: 2 (home vs B), 1 (away @ C), 3 (home vs D) → 2.0 avg
        assert abs(stats["goals_for"] - 2.0) < 1e-9

    def test_team_A_goals_against(self):
        hist = self._make_history()
        stats = _team_rolling_stats(hist, "A", last_n=5)
        # A sofreu: 1, 0, 0 → 0.33
        assert abs(stats["goals_against"] - 1/3) < 1e-9

    def test_empty_history(self):
        stats = _team_rolling_stats(pd.DataFrame(), "A", last_n=5)
        assert stats["goals_for"] == 1.3  # fallback default


class TestBuildTrainingDataset:
    def _make_matches(self, n=10):
        """n matches alternando times A/B vs C/D."""
        teams = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"),
                 ("B", "C"), ("C", "A"), ("D", "B"), ("D", "A"), ("C", "B")]
        rows = []
        for i in range(n):
            h, a = teams[i % len(teams)]
            rows.append({
                "match_id": i + 1,
                "match_date": f"2026-04-{i+1:02d}",
                "home_team": h, "away_team": a,
                "home_goals": (i % 4), "away_goals": ((i+1) % 3),
                "home_xg": 1.0 + (i % 3) * 0.3, "away_xg": 0.8 + (i % 2) * 0.3,
                "home_corners": 5 + (i % 4), "away_corners": 4 + (i % 3),
                "home_cards": 2 + (i % 3), "away_cards": 2 + ((i+1) % 3),
            })
        return pd.DataFrame(rows)

    def test_returns_X_y(self):
        matches = self._make_matches(n=10)
        X, y = build_training_dataset(matches, target="goals", min_prior=2)
        assert X.shape[1] == FEATURE_DIM
        assert len(X) == len(y)
        assert len(X) > 0

    def test_two_samples_per_match(self):
        # Com min_prior=0, cada match gera 2 samples
        matches = self._make_matches(n=10)
        X, y = build_training_dataset(matches, target="goals", min_prior=0)
        assert len(X) == 20  # 10 partidas × 2

    def test_min_prior_filters(self):
        matches = self._make_matches(n=5)
        # Com min_prior=10, nenhum jogo tem histórico suficiente
        X, y = build_training_dataset(matches, target="goals", min_prior=10)
        assert len(X) == 0

    def test_corners_target(self):
        matches = self._make_matches(n=10)
        X, y = build_training_dataset(matches, target="corners", min_prior=2)
        # y deve ter valores de corners (5-8)
        assert y.min() >= 0
        assert y.max() < 20

    def test_cards_target(self):
        matches = self._make_matches(n=10)
        X, y = build_training_dataset(matches, target="cards", min_prior=2)
        assert y.min() >= 0
        assert y.max() < 10
