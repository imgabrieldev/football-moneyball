"""Tests for football_moneyball.domain.ml_lambda."""

import os
import tempfile

import numpy as np
import pytest

from football_moneyball.domain.ml_lambda import LambdaPredictor


def _synthetic_dataset(n: int = 100):
    """Synthetic dataset: y ~ f(X) + noise."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 12))
    # Target is a nonlinear combination of features
    y = 1.5 + 0.3 * X[:, 0] + 0.2 * X[:, 6] - 0.1 * X[:, 1] + rng.normal(0, 0.3, n)
    # clamp positive (lambda is always > 0)
    y = np.clip(y, 0.1, 5.0)
    return X, y


class TestLambdaPredictor:
    def test_train_returns_metrics(self):
        X, y = _synthetic_dataset(100)
        predictor = LambdaPredictor(target="goals")
        metrics = predictor.train(X, y)
        assert metrics["target"] == "goals"
        assert metrics["n_samples"] == 100
        assert metrics["n_features"] == 12
        assert "cv_mae_mean" in metrics
        assert metrics["cv_mae_mean"] > 0

    def test_predict_before_train_raises(self):
        predictor = LambdaPredictor()
        with pytest.raises(ValueError):
            predictor.predict(np.zeros(12))

    def test_predict_returns_clamped_value(self):
        X, y = _synthetic_dataset(100)
        predictor = LambdaPredictor()
        predictor.train(X, y)
        # Predict with any input — must return something clamped
        lam = predictor.predict(np.zeros(12))
        assert 0.1 <= lam <= 15.0

    def test_predict_reasonable(self):
        X, y = _synthetic_dataset(200)
        predictor = LambdaPredictor()
        predictor.train(X, y)
        # Predict with high values on "positive" features
        high_features = np.array([5.0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 1.0])
        lam_high = predictor.predict(high_features)
        low_features = np.array([-5.0, 0, 0, 0, 0, 0, -5.0, 0, 0, 0, 0, 0.0])
        lam_low = predictor.predict(low_features)
        # Positive features produce a larger lambda
        assert lam_high > lam_low

    def test_save_load_roundtrip(self):
        X, y = _synthetic_dataset(100)
        predictor = LambdaPredictor(target="corners")
        predictor.train(X, y)

        test_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                   1.5, 2.5, 3.5, 4.5, 1.3, 1.0])
        lam_original = predictor.predict(test_features)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            predictor.save(path)
            loaded = LambdaPredictor.load(path)
            lam_loaded = loaded.predict(test_features)
            assert abs(lam_original - lam_loaded) < 1e-9
            assert loaded.target == "corners"
            assert loaded.metadata["n_samples"] == 100
        finally:
            os.unlink(path)

    def test_too_small_dataset_raises(self):
        X = np.zeros((5, 12))
        y = np.zeros(5)
        predictor = LambdaPredictor()
        with pytest.raises(ValueError):
            predictor.train(X, y)
