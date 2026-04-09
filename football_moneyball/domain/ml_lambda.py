"""GradientBoostingRegressor wrapper to predict lambda.

Each instance predicts one target (goals, corners, cards).
Uses time-series CV in training. Persisted via joblib.

Pure logic (numpy + sklearn). Zero infra deps.
"""

from __future__ import annotations

import numpy as np


class LambdaPredictor:
    """Predicts lambda (expected value) for a metric via GBR."""

    def __init__(self, target: str = "goals") -> None:
        self.target = target
        self.model = None
        self.metadata: dict = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train GBR with Time-Series CV.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Targets (n_samples,).

        Returns
        -------
        dict
            Metadata: target, n_samples, n_features, cv_mae_mean, cv_mae_std.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import TimeSeriesSplit

        if len(X) < 10:
            raise ValueError(f"Dataset too small: {len(X)} samples")

        # Hyperparams tuned for the small-sample regime (< 200 samples)
        # Shallow trees + high regularization prevent overfitting with 24 features
        n_samples = len(X)
        if n_samples < 200:
            # Small data: less complexity
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.7,
                min_samples_leaf=8,
                min_samples_split=15,
                max_features="sqrt",
                random_state=42,
            )
        else:
            # Plenty of data: original hyperparams
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )

        # Time-series CV
        n_splits = min(5, max(2, len(X) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
            pred = self.model.predict(X[val_idx])
            scores.append(mean_absolute_error(y[val_idx], pred))

        # Final fit on all data
        self.model.fit(X, y)

        self.metadata = {
            "target": self.target,
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
            "cv_mae_mean": float(np.mean(scores)),
            "cv_mae_std": float(np.std(scores)),
        }
        return self.metadata

    def predict(self, features: np.ndarray) -> float:
        """Predict lambda for a single input. Clamped to [0.1, 15].

        Parameters
        ----------
        features : np.ndarray
            1-D array (n_features,).

        Returns
        -------
        float
            Predicted lambda, minimum 0.1.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        lam = float(self.model.predict(features.reshape(1, -1))[0])
        return max(0.1, min(lam, 15.0))

    def save(self, path: str) -> None:
        """Save model + metadata to a pickle file."""
        import joblib
        joblib.dump({"model": self.model, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str) -> "LambdaPredictor":
        """Load LambdaPredictor from pickle."""
        import joblib
        data = joblib.load(path)
        target = data.get("metadata", {}).get("target", "goals")
        instance = cls(target=target)
        instance.model = data["model"]
        instance.metadata = data.get("metadata", {})
        return instance
