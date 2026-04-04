"""Testes para football_moneyball.domain.player_lambda."""

import pandas as pd

from football_moneyball.domain.player_lambda import (
    compute_xg_per_90,
    summarize_xi,
    team_lambda_from_players,
)


class TestComputeXgPer90:
    def test_standard(self):
        # 3 gols em 270 min → 1.0 xG/90
        assert compute_xg_per_90(3.0, 270.0) == 1.0

    def test_one_full_game(self):
        # 0.5 xG em 90 min → 0.5 xG/90
        assert compute_xg_per_90(0.5, 90.0) == 0.5

    def test_zero_minutes(self):
        assert compute_xg_per_90(1.0, 0.0) == 0.0

    def test_negative_minutes(self):
        assert compute_xg_per_90(1.0, -10.0) == 0.0


class TestTeamLambdaFromPlayers:
    def _make_xi(self, xg_per_90_list, weights):
        return pd.DataFrame({
            "xg_per_90": xg_per_90_list,
            "weight": weights,
        })

    def test_uniform_team(self):
        # 11 jogadores × 0.15 xG/90 × weight 1.0 = 1.65
        xi = self._make_xi([0.15] * 11, [1.0] * 11)
        lam = team_lambda_from_players(xi, opponent_defense_factor=1.0)
        assert abs(lam - 1.65) < 1e-9

    def test_weighted(self):
        # 11 jogadores × 0.2 xG/90 × weight 0.5 = 1.1
        xi = self._make_xi([0.2] * 11, [0.5] * 11)
        lam = team_lambda_from_players(xi, opponent_defense_factor=1.0)
        assert abs(lam - 1.1) < 1e-9

    def test_strong_defense(self):
        # opp_defense_factor = 0.5 → corta pela metade
        xi = self._make_xi([0.2] * 11, [1.0] * 11)
        lam = team_lambda_from_players(xi, opponent_defense_factor=0.5)
        assert abs(lam - 1.1) < 1e-9

    def test_weak_defense(self):
        # opp_defense_factor = 1.5 → infla
        xi = self._make_xi([0.1] * 11, [1.0] * 11)
        lam = team_lambda_from_players(xi, opponent_defense_factor=1.5)
        assert abs(lam - 1.65) < 1e-9

    def test_minimum_clamp(self):
        # 11 × 0.0 × 1.0 = 0.0 → clamp 0.15
        xi = self._make_xi([0.0] * 11, [1.0] * 11)
        lam = team_lambda_from_players(xi, opponent_defense_factor=1.0)
        assert lam == 0.15

    def test_empty_xi(self):
        assert team_lambda_from_players(pd.DataFrame()) == 0.15

    def test_missing_columns(self):
        xi = pd.DataFrame({"player_id": [1, 2, 3]})
        assert team_lambda_from_players(xi) == 0.15


class TestSummarizeXi:
    def test_format(self):
        xi = pd.DataFrame([
            {"player_id": 1, "player_name": "Neymar", "xg_per_90": 0.42, "weight": 0.9, "minutes_total": 400},
            {"player_id": 2, "player_name": "Messi", "xg_per_90": 0.55, "weight": 1.0, "minutes_total": 450},
        ])
        result = summarize_xi(xi)
        assert len(result) == 2
        assert result[0]["player_name"] == "Neymar"
        assert result[0]["xg_per_90"] == 0.42
        assert result[0]["weight"] == 0.9

    def test_empty(self):
        assert summarize_xi(pd.DataFrame()) == []
