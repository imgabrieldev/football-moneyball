"""Tests for football_moneyball.domain.player_props."""

from math import exp

import pandas as pd

from football_moneyball.domain.player_props import (
    compute_team_player_props,
    predict_player_assist,
    predict_player_goal,
    predict_player_multiple_goals,
    predict_player_scores_or_assists,
    predict_player_shots,
)


class TestPredictPlayerGoal:
    def test_basic_formula(self):
        # xg_per_90=0.5, min=90 -> lambda=0.5 -> P=1-e^(-0.5)~=0.3935
        p = predict_player_goal(0.5, 90)
        assert abs(p - (1 - exp(-0.5))) < 1e-4

    def test_zero_xg(self):
        assert predict_player_goal(0, 90) == 0.0

    def test_zero_minutes(self):
        assert predict_player_goal(0.5, 0) == 0.0

    def test_partial_minutes(self):
        # 45 min with xg_per_90=1.0 -> lambda=0.5 -> P=1-e^(-0.5)
        p = predict_player_goal(1.0, 45)
        assert abs(p - (1 - exp(-0.5))) < 1e-4

    def test_high_xg(self):
        # Exceptional striker: xg_per_90=1.0, min=90 -> P~=0.632
        p = predict_player_goal(1.0, 90)
        assert abs(p - (1 - exp(-1.0))) < 1e-4


class TestPredictMultipleGoals:
    def test_2_or_more(self):
        # lambda=1.0, P(X>=2) = 1 - P(X=0) - P(X=1) = 1 - e^-1 - e^-1
        # = 1 - 2e^-1 ~= 0.264
        p = predict_player_multiple_goals(1.0, 90, n=2)
        expected = 1 - exp(-1.0) - exp(-1.0) * 1.0  # 1 - P(0) - P(1)
        assert abs(p - expected) < 1e-4

    def test_hat_trick(self):
        # lambda=1.0, P(X>=3) should be much smaller than P(X>=2)
        p2 = predict_player_multiple_goals(1.0, 90, n=2)
        p3 = predict_player_multiple_goals(1.0, 90, n=3)
        assert p3 < p2

    def test_zero_cases(self):
        assert predict_player_multiple_goals(0, 90) == 0.0
        assert predict_player_multiple_goals(0.5, 0) == 0.0


class TestPredictPlayerAssist:
    def test_basic(self):
        p = predict_player_assist(0.3, 90)
        assert abs(p - (1 - exp(-0.3))) < 1e-4


class TestPredictPlayerShots:
    def test_returns_multiple_lines(self):
        result = predict_player_shots(3.0, 90)
        assert len(result) == 3
        assert result[0]["line"] == 0.5
        assert result[1]["line"] == 1.5
        assert result[2]["line"] == 2.5

    def test_probabilities_decrease(self):
        # P(shots>0.5) > P(shots>1.5) > P(shots>2.5)
        result = predict_player_shots(3.0, 90)
        probs = [r["over_prob"] for r in result]
        assert probs[0] > probs[1] > probs[2]

    def test_custom_lines(self):
        result = predict_player_shots(3.0, 90, lines=[0.5, 3.5])
        assert len(result) == 2
        assert result[1]["line"] == 3.5

    def test_zero_minutes(self):
        result = predict_player_shots(3.0, 0)
        assert all(r["over_prob"] == 0.0 for r in result)


class TestScoresOrAssists:
    def test_independence(self):
        # P(A or B) = 1 - (1-P(A))(1-P(B))
        p = predict_player_scores_or_assists(0.5, 0.3, 90)
        p_goal = 1 - exp(-0.5)
        p_assist = 1 - exp(-0.3)
        expected = 1 - (1 - p_goal) * (1 - p_assist)
        assert abs(p - expected) < 1e-3


class TestComputeTeamPlayerProps:
    def _make_aggregates(self, n: int = 10):
        rows = []
        for i in range(n):
            rows.append({
                "player_id": i + 1,
                "player_name": f"Player {i+1}",
                "matches_played": 5,
                "minutes_total": 450 - i * 20,
                "xg_total": 2.0 - i * 0.15,
                "xa_total": 1.5 - i * 0.1,
                "shots_total": 15.0 - i * 0.5,
            })
        return pd.DataFrame(rows)

    def test_returns_top_n(self):
        aggs = self._make_aggregates(10)
        result = compute_team_player_props(aggs, top_n=5)
        assert len(result) == 5

    def test_ordered_by_minutes(self):
        aggs = self._make_aggregates(10)
        result = compute_team_player_props(aggs, top_n=5)
        # Player 1 has 450 min (most) -> first
        assert result[0]["player_name"] == "Player 1"

    def test_fields_populated(self):
        aggs = self._make_aggregates(5)
        result = compute_team_player_props(aggs, top_n=5)
        first = result[0]
        assert "goal_prob" in first
        assert "goal_2plus_prob" in first
        assert "assist_prob" in first
        assert "scores_or_assists_prob" in first
        assert "shots" in first
        assert len(first["shots"]) == 3

    def test_filters_by_min_matches(self):
        aggs = self._make_aggregates(5)
        aggs.loc[0, "matches_played"] = 1  # too few
        result = compute_team_player_props(aggs, top_n=5, min_matches=3)
        assert all(r["player_name"] != "Player 1" for r in result)

    def test_empty_input(self):
        assert compute_team_player_props(pd.DataFrame()) == []
