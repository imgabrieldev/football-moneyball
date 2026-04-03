"""Testes para domain/match_predictor.py — previsao Monte Carlo."""

import numpy as np
import pandas as pd
import pytest

from football_moneyball.domain.match_predictor import (
    estimate_team_xg,
    simulate_match,
    poisson_pmf,
)


class TestPoissonPMF:
    def test_known_values(self):
        # P(X=0) para lambda=1 = e^-1 ≈ 0.3679
        assert abs(poisson_pmf(0, 1.0) - 0.3679) < 0.001

    def test_sum_to_one(self):
        lam = 2.0
        total = sum(poisson_pmf(k, lam) for k in range(20))
        assert abs(total - 1.0) < 0.001

    def test_zero_lambda(self):
        assert poisson_pmf(0, 0.0) == 0.0

    def test_negative_k(self):
        assert poisson_pmf(-1, 1.0) == 0.0


class TestSimulateMatch:
    def test_probabilities_sum_to_one(self):
        result = simulate_match(1.5, 1.2, n_simulations=50_000, seed=42)
        total = result["home_win_prob"] + result["draw_prob"] + result["away_win_prob"]
        assert abs(total - 1.0) < 0.01

    def test_home_advantage(self):
        # Com xG iguais, probabilidades devem ser simétricas
        result = simulate_match(1.5, 1.5, n_simulations=50_000, seed=42)
        # Home e away devem ser próximos (sem vantagem real no modelo)
        assert abs(result["home_win_prob"] - result["away_win_prob"]) < 0.05

    def test_strong_favorite(self):
        # Time muito superior deve ter alta probabilidade
        result = simulate_match(3.0, 0.5, n_simulations=50_000, seed=42)
        assert result["home_win_prob"] > 0.7

    def test_over_under_consistency(self):
        result = simulate_match(1.5, 1.5, n_simulations=50_000, seed=42)
        # over_25 + under_25 = 1.0
        under_25 = 1.0 - result["over_25"]
        assert under_25 > 0 and under_25 < 1

    def test_btts_high_xg(self):
        # Com alto xG para ambos, BTTS deve ser alta
        result = simulate_match(2.5, 2.0, n_simulations=50_000, seed=42)
        assert result["btts_prob"] > 0.6

    def test_reproducibility(self):
        r1 = simulate_match(1.5, 1.2, seed=123)
        r2 = simulate_match(1.5, 1.2, seed=123)
        assert r1["home_win_prob"] == r2["home_win_prob"]

    def test_score_matrix_exists(self):
        result = simulate_match(1.5, 1.2, seed=42)
        assert "score_matrix" in result
        assert len(result["score_matrix"]) > 0
        assert result["most_likely_score"] != ""


class TestEstimateTeamXg:
    def test_basic_estimation(self):
        history = pd.DataFrame({"xg": [1.5, 2.0, 1.0, 1.8, 1.2]})
        opponent = pd.DataFrame({"xg_against": [1.3, 1.5, 1.1]})
        xg = estimate_team_xg(history, opponent, is_home=True)
        assert xg > 0

    def test_home_advantage(self):
        history = pd.DataFrame({"xg": [1.5, 1.5, 1.5]})
        opponent = pd.DataFrame({"xg_against": [1.25]})
        home_xg = estimate_team_xg(history, opponent, is_home=True)
        away_xg = estimate_team_xg(history, opponent, is_home=False)
        assert home_xg > away_xg

    def test_minimum_xg(self):
        history = pd.DataFrame({"xg": [0.0, 0.0, 0.0]})
        opponent = pd.DataFrame({"xg_against": [0.0]})
        xg = estimate_team_xg(history, opponent, is_home=False)
        assert xg >= 0.1  # Never returns 0

    def test_empty_history_default(self):
        history = pd.DataFrame({"xg": []})
        opponent = pd.DataFrame({"xg_against": []})
        xg = estimate_team_xg(history, opponent, is_home=True)
        assert xg > 0  # Returns default
