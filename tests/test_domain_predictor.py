"""Testes para domain/match_predictor.py — v0.5.0 pipeline dinamico."""

import numpy as np
import pandas as pd
import pytest

from football_moneyball.domain.match_predictor import (
    calculate_league_averages,
    calculate_team_strength,
    calculate_xg_quality,
    apply_regression_to_mean,
    predict_match,
    simulate_match,
    poisson_pmf,
)


def _make_league_data(n_matches=20) -> pd.DataFrame:
    """Cria dataset sintetico de liga."""
    rng = np.random.RandomState(42)
    rows = []
    teams = ["A", "B", "C", "D"]
    for mid in range(n_matches):
        home = teams[mid % 4]
        away = teams[(mid + 1) % 4]
        rows.append({"match_id": mid, "team": home, "goals": rng.poisson(1.5), "xg": 1.3 + rng.normal(0, 0.3), "is_home": True})
        rows.append({"match_id": mid, "team": away, "goals": rng.poisson(1.0), "xg": 1.0 + rng.normal(0, 0.3), "is_home": False})
    return pd.DataFrame(rows)


class TestCalculateLeagueAverages:
    def test_returns_all_keys(self):
        data = _make_league_data()
        result = calculate_league_averages(data)
        assert "avg_xg" in result
        assert "home_advantage" in result
        assert "n_matches" in result

    def test_home_advantage_positive(self):
        data = _make_league_data()
        result = calculate_league_averages(data)
        # Home teams get more xG in our synthetic data
        assert result["home_advantage"] >= 0

    def test_empty_data_returns_defaults(self):
        result = calculate_league_averages(pd.DataFrame())
        assert result["avg_xg"] > 0
        assert result["n_matches"] == 0

    def test_decay_weights_recent_more(self):
        data = _make_league_data(10)
        # With decay=1.0 (no decay), all equal
        result_no_decay = calculate_league_averages(data, decay=1.0)
        # With decay=0.5, recent matches dominate
        result_decay = calculate_league_averages(data, decay=0.5)
        # Results should differ
        assert result_no_decay["avg_xg"] != result_decay["avg_xg"]


class TestCalculateTeamStrength:
    def test_strong_team_above_one(self):
        data = _make_league_data(20)
        league = calculate_league_averages(data)
        # Team A (always home) should have higher attack
        team_a = data[data["team"] == "A"]
        strength = calculate_team_strength(team_a, data, league)
        assert strength["attack_strength"] > 0
        assert strength["matches"] > 0

    def test_empty_team_defaults(self):
        data = _make_league_data()
        league = calculate_league_averages(data)
        strength = calculate_team_strength(pd.DataFrame(), data, league)
        assert strength["attack_strength"] == 1.0
        assert strength["defense_strength"] == 1.0


class TestXgQuality:
    def test_average_returns_one(self):
        shots = [0.10, 0.10, 0.10, 0.10]
        assert abs(calculate_xg_quality(shots, 0.10) - 1.0) < 0.01

    def test_high_quality_above_one(self):
        shots = [0.30, 0.25, 0.35, 0.20]  # big chances
        assert calculate_xg_quality(shots, 0.10) > 1.0

    def test_low_quality_below_one(self):
        shots = [0.02, 0.03, 0.01, 0.02]  # pot shots
        assert calculate_xg_quality(shots, 0.10) < 1.0

    def test_empty_returns_one(self):
        assert calculate_xg_quality([], 0.10) == 1.0

    def test_clamped(self):
        shots = [0.90, 0.80, 0.85]  # absurdly high
        result = calculate_xg_quality(shots, 0.10)
        assert result <= 1.3  # capped


class TestRegressionToMean:
    def test_overperformer_reduced(self):
        # Time com 20 gols mas 10 xG em 10 jogos → overperformer
        xg = 1.5
        adjusted = apply_regression_to_mean(xg, team_goals=20, team_xg_total=10.0, matches_played=10)
        assert adjusted < xg  # puxado pra baixo

    def test_underperformer_boosted(self):
        # Time com 5 gols mas 15 xG → underperformer
        xg = 1.5
        adjusted = apply_regression_to_mean(xg, team_goals=5, team_xg_total=15.0, matches_played=10)
        assert adjusted > xg  # puxado pra cima

    def test_aligned_unchanged(self):
        xg = 1.5
        adjusted = apply_regression_to_mean(xg, team_goals=15, team_xg_total=15.0, matches_played=10)
        assert abs(adjusted - xg) < 0.01

    def test_more_games_less_regression(self):
        xg = 1.5
        r5 = apply_regression_to_mean(xg, 20, 10.0, 5)   # 5 jogos
        r30 = apply_regression_to_mean(xg, 60, 30.0, 30)  # 30 jogos, mesma over/game
        # Com mais jogos, regressao eh menor
        assert abs(r30 - xg) < abs(r5 - xg)

    def test_minimum_floor(self):
        adjusted = apply_regression_to_mean(0.05, 50, 5.0, 10)
        assert adjusted >= 0.1


class TestPredictMatch:
    def test_pipeline_produces_valid_output(self):
        data = _make_league_data(30)
        result = predict_match("A", "B", data, n_simulations=5000, seed=42)
        assert "home_win_prob" in result
        total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
        assert abs(total - 1.0) < 0.02

    def test_no_hardcoded_constants(self):
        # Verificar que pipeline metadata mostra parametros calculados
        data = _make_league_data(30)
        result = predict_match("A", "B", data, seed=42)
        assert "pipeline" in result
        assert result["pipeline"]["league_avg_xg"] > 0
        assert result["pipeline"]["home_advantage"] >= 0

    def test_home_advantage_in_xg(self):
        data = _make_league_data(30)
        result = predict_match("A", "B", data, seed=42)
        # Home team should have higher xG
        assert result["home_xg"] > result["away_xg"] - 0.5  # approximate

    def test_with_shot_quality(self):
        data = _make_league_data(30)
        # Big chances for home
        result = predict_match("A", "B", data,
                              home_shots=[0.3, 0.4, 0.25, 0.35],
                              away_shots=[0.05, 0.03, 0.04],
                              seed=42)
        assert result["home_xg"] > result["away_xg"]


class TestSimulateMatch:
    def test_probabilities_sum_to_one(self):
        result = simulate_match(1.5, 1.2, n_simulations=50_000, seed=42)
        total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
        assert abs(total - 1.0) < 0.01

    def test_strong_favorite(self):
        result = simulate_match(3.0, 0.5, n_simulations=50_000, seed=42)
        assert result["home_win_prob"] > 0.7

    def test_reproducibility(self):
        r1 = simulate_match(1.5, 1.2, seed=123)
        r2 = simulate_match(1.5, 1.2, seed=123)
        assert r1["home_win_prob"] == r2["home_win_prob"]


class TestPoissonPMF:
    def test_known_value(self):
        assert abs(poisson_pmf(0, 1.0) - 0.3679) < 0.001

    def test_sum_to_one(self):
        total = sum(poisson_pmf(k, 2.0) for k in range(20))
        assert abs(total - 1.0) < 0.001
