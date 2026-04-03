"""Testes para football_moneyball.domain.rapm — RAPM via modulo hexagonal.

Estes testes ja eram puros (build_rapm_matrix, fit_rapm, cross_validate_alpha
operam sobre arrays numpy), entao a migracao e apenas troca de import paths.
"""

import numpy as np
import pandas as pd
import pytest

from football_moneyball.domain.rapm import (
    build_rapm_matrix,
    fit_rapm,
    cross_validate_alpha,
)


class TestBuildRapmMatrix:
    def test_basic_matrix_construction(self):
        stints = pd.DataFrame([
            {"home_player_ids": [1, 2], "away_player_ids": [3, 4],
             "duration_minutes": 10.0, "xg_diff": 0.5},
            {"home_player_ids": [1, 2], "away_player_ids": [3, 5],
             "duration_minutes": 5.0, "xg_diff": -0.2},
        ])
        X, y, player_ids = build_rapm_matrix(stints)
        assert X.shape[0] == 2  # 2 stints
        assert X.shape[1] == len(player_ids)
        assert len(y) == 2

    def test_home_positive_away_negative(self):
        stints = pd.DataFrame([
            {"home_player_ids": [1], "away_player_ids": [2],
             "duration_minutes": 1.0, "xg_diff": 0.0},
        ])
        X, y, player_ids = build_rapm_matrix(stints)
        idx_1 = player_ids.index(1)
        idx_2 = player_ids.index(2)
        assert X[0, idx_1] > 0  # home player positive
        assert X[0, idx_2] < 0  # away player negative


class TestFitRapm:
    def test_basic_fit(self):
        X = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
        y = np.array([0.5, -0.3, 0.4])
        player_ids = [1, 2]
        result = fit_rapm(X, y, player_ids)
        assert "rapm_value" in result.columns
        assert "offensive_rapm" in result.columns
        assert "defensive_rapm" in result.columns
        assert len(result) == 2

    def test_fit_with_spm_prior(self):
        X = np.array([[1.0, -1.0], [-1.0, 1.0]])
        y = np.array([0.5, -0.3])
        player_ids = [1, 2]
        prior = np.array([0.1, -0.1])
        result = fit_rapm(X, y, player_ids, spm_prior=prior)
        assert len(result) == 2
        # Player 1 should have higher RAPM with positive prior
        p1 = result[result["player_id"] == 1]["rapm_value"].iloc[0]
        assert p1 > 0  # Both data and prior push positive


class TestCrossValidateAlpha:
    def test_returns_float(self):
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        alpha = cross_validate_alpha(X, y)
        assert isinstance(alpha, float)
        assert alpha > 0
