"""Testes para football_moneyball.domain.catboost_predictor."""

import numpy as np
import pandas as pd

from football_moneyball.domain.catboost_predictor import (
    CATBOOST_FEATURE_NAMES,
    N_FEATURES,
    build_match_features,
    build_training_dataset,
    compute_form_ema,
    compute_gd_ema,
)
from football_moneyball.domain.pi_rating import PiRating


class TestFormEMA:
    def test_all_wins(self):
        ema = compute_form_ema([1.0] * 20)
        assert ema > 0.8

    def test_all_losses(self):
        ema = compute_form_ema([0.0] * 20)
        assert ema < 0.2

    def test_empty(self):
        assert compute_form_ema([]) == 0.5

    def test_recent_weight(self):
        # Último resultado pesa mais
        ema_recent_win = compute_form_ema([0.0, 0.0, 0.0, 1.0])
        ema_early_win = compute_form_ema([1.0, 0.0, 0.0, 0.0])
        assert ema_recent_win > ema_early_win


class TestGdEMA:
    def test_positive(self):
        ema = compute_gd_ema([2.0, 1.0, 3.0])
        assert ema > 0

    def test_empty(self):
        assert compute_gd_ema([]) == 0.0


class TestBuildMatchFeatures:
    def test_shape(self):
        ratings = {"A": PiRating(1.0, 0.5), "B": PiRating(0.3, 0.8)}
        f = build_match_features(
            ratings, "A", "B",
            home_form=[1.0, 0.5, 1.0],
            away_form=[0.0, 0.5, 0.0],
            home_gd=[2, 0, 1],
            away_gd=[-1, 0, -2],
        )
        assert f.shape == (N_FEATURES,)
        assert len(CATBOOST_FEATURE_NAMES) == N_FEATURES

    def test_pi_rating_diff_correct(self):
        ratings = {"A": PiRating(1.5, 0.0), "B": PiRating(0.0, 0.3)}
        f = build_match_features(
            ratings, "A", "B",
            home_form=[], away_form=[],
            home_gd=[], away_gd=[],
        )
        # pi_rating_diff = R_home[A] - R_away[B] = 1.5 - 0.3 = 1.2
        assert abs(f[0] - 1.2) < 1e-9


class TestBuildTrainingDataset:
    def _make_data(self, n_matches: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rows = []
        for i in range(n_matches):
            hg = int(rng.poisson(1.5))
            ag = int(rng.poisson(1.0))
            rows.extend([
                {"match_id": i, "team": f"Team{i%5}", "goals": hg,
                 "xg": 1.5 + rng.normal(0, 0.3), "is_home": True},
                {"match_id": i, "team": f"Team{(i+1)%5}", "goals": ag,
                 "xg": 1.0 + rng.normal(0, 0.3), "is_home": False},
            ])
        return pd.DataFrame(rows)

    def test_returns_arrays(self):
        data = self._make_data(80)
        X, y = build_training_dataset(data, min_history=20)
        assert X.ndim == 2
        assert X.shape[1] == N_FEATURES
        assert y.ndim == 1
        assert len(X) == len(y)
        assert len(X) > 0

    def test_labels_are_0_1_2(self):
        data = self._make_data(80)
        X, y = build_training_dataset(data, min_history=20)
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_leak_proof(self):
        # Com min_history=50 e 50 matches, deveria ter 0 samples
        data = self._make_data(50)
        X, y = build_training_dataset(data, min_history=50)
        assert len(X) == 0
