"""Testes para predict_match_player_aware — pipeline end-to-end com dados sinteticos."""

import pandas as pd

from football_moneyball.domain.match_predictor import (
    predict_match, predict_match_player_aware,
)


def _make_match_data():
    """Constroi DataFrame team-level sintetico (10 jogos, 2 times)."""
    rows = []
    for mid in range(1, 11):
        # Home = Team A (time forte), Away = Team B (time medio)
        rows.append({
            "match_id": mid, "team": "Team A", "goals": 2, "xg": 1.8, "is_home": True,
        })
        rows.append({
            "match_id": mid, "team": "Team B", "goals": 1, "xg": 1.0, "is_home": False,
        })
    return pd.DataFrame(rows)


def _make_player_aggregates(n_players: int, xg_per_player: float = 0.15) -> pd.DataFrame:
    """Constroi agregados sintéticos: N jogadores, todos com mesmo xG/90."""
    rows = []
    for i in range(n_players):
        rows.append({
            "player_id": i + 1,
            "player_name": f"Player {i+1}",
            "matches_played": 5,
            "minutes_total": 450,
            "xg_total": xg_per_player * 5,  # 5 jogos completos com xG/90 fixo
        })
    return pd.DataFrame(rows)


class TestPredictMatchPlayerAware:
    def test_returns_valid_probabilities(self):
        match_data = _make_match_data()
        home_aggs = _make_player_aggregates(15, xg_per_player=0.15)
        away_aggs = _make_player_aggregates(15, xg_per_player=0.10)

        result = predict_match_player_aware(
            home_team="Team A",
            away_team="Team B",
            all_match_data=match_data,
            home_player_aggregates=home_aggs,
            away_player_aggregates=away_aggs,
            seed=42,
        )

        probs = (
            result["home_win_prob"]
            + result["draw_prob"]
            + result["away_win_prob"]
        )
        assert abs(probs - 1.0) < 0.01

    def test_metadata_fields(self):
        match_data = _make_match_data()
        home_aggs = _make_player_aggregates(11)
        away_aggs = _make_player_aggregates(11)

        result = predict_match_player_aware(
            home_team="Team A",
            away_team="Team B",
            all_match_data=match_data,
            home_player_aggregates=home_aggs,
            away_player_aggregates=away_aggs,
            seed=42,
        )

        assert result["lineup_type"] == "probable-xi"
        assert result["model_version"] == "v1.10.0"
        assert len(result["home_xi"]) == 11
        assert len(result["away_xi"]) == 11
        assert "home_team_attack" in result["pipeline"]

    def test_home_team_favored_with_better_xi(self):
        match_data = _make_match_data()
        # Home tem jogadores de melhor xG/90
        home_aggs = _make_player_aggregates(11, xg_per_player=0.25)
        away_aggs = _make_player_aggregates(11, xg_per_player=0.05)

        result = predict_match_player_aware(
            home_team="Team A",
            away_team="Team B",
            all_match_data=match_data,
            home_player_aggregates=home_aggs,
            away_player_aggregates=away_aggs,
            seed=42,
        )

        assert result["home_win_prob"] > result["away_win_prob"]

    def test_consistent_with_seed(self):
        match_data = _make_match_data()
        home_aggs = _make_player_aggregates(11)
        away_aggs = _make_player_aggregates(11)

        r1 = predict_match_player_aware(
            home_team="Team A", away_team="Team B",
            all_match_data=match_data,
            home_player_aggregates=home_aggs,
            away_player_aggregates=away_aggs,
            seed=42,
        )
        r2 = predict_match_player_aware(
            home_team="Team A", away_team="Team B",
            all_match_data=match_data,
            home_player_aggregates=home_aggs,
            away_player_aggregates=away_aggs,
            seed=42,
        )
        assert r1["home_win_prob"] == r2["home_win_prob"]
        assert r1["draw_prob"] == r2["draw_prob"]

    def test_backward_compat_with_predict_match(self):
        """Path team-level (v1.0.0) continua funcionando — smoke test."""
        match_data = _make_match_data()
        result = predict_match(
            home_team="Team A",
            away_team="Team B",
            all_match_data=match_data,
            seed=42,
        )
        probs = (
            result["home_win_prob"]
            + result["draw_prob"]
            + result["away_win_prob"]
        )
        assert abs(probs - 1.0) < 0.01
