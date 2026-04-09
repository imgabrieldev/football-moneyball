"""Use case: train models ML (LambdaPredictor) for goals, corners, cards."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class TrainMLModels:
    """Orquestra training of models GBR for each metric.

    Parameters
    ----------
    repo : MatchRepository
    models_dir : str
        Diretorio for salvar pickles (default: football_moneyball/models).
    """

    TARGETS = ["goals", "corners", "cards"]
    MIN_SAMPLES = 20

    def __init__(
        self, repo, models_dir: str = "football_moneyball/models",
    ) -> None:
        self.repo = repo
        self.models_dir = models_dir

    def execute(self, season: str | None = "2026") -> dict[str, Any]:
        """Treina 3 models and salva.

        Parameters
        ----------
        season : str | None
            Season for train. None = todas as temporadas disponiveis.

        Returns
        -------
        dict
            Metrics by target: {target: {cv_mae_mean, n_samples, ...}}
            Ou {"error": "..."} if insuficiente.
        """
        from football_moneyball.domain.feature_engineering import (
            build_training_dataset,
        )
        from football_moneyball.domain.ml_lambda import LambdaPredictor

        os.makedirs(self.models_dir, exist_ok=True)

        matches_df = self.repo.get_training_dataset(season)
        if matches_df.empty or len(matches_df) < 10:
            return {
                "error": (
                    f"Data insuficientes ({len(matches_df)} matches). "
                    "Minimo 10 matches with match_stats."
                ),
            }

        # v1.10.0: carregar mapping match_id -> referee_stats pros features
        match_referees = self._load_match_referees(matches_df)

        results: dict[str, Any] = {}
        for target in self.TARGETS:
            try:
                X, y = build_training_dataset(
                    matches_df, target=target, match_referees=match_referees,
                )
                if len(X) < self.MIN_SAMPLES:
                    results[target] = {
                        "error": f"Apenas {len(X)} amostras. Minimo {self.MIN_SAMPLES}.",
                    }
                    continue

                predictor = LambdaPredictor(target=target)
                metrics = predictor.train(X, y)
                path = os.path.join(self.models_dir, f"{target}_lambda.pkl")
                predictor.save(path)

                logger.info(
                    f"Trained {target}: MAE={metrics['cv_mae_mean']:.3f} "
                    f"(n={metrics['n_samples']})"
                )
                metrics["saved_to"] = path
                results[target] = metrics
            except Exception as e:
                logger.warning(f"Erro treinando {target}: {e}")
                results[target] = {"error": str(e)}

        return results

    def _load_match_referees(self, matches_df) -> dict[int, dict]:
        """Loads mapping match_id -> referee_stats pros jogos of the dataset."""
        mapping: dict[int, dict] = {}
        if matches_df.empty:
            return mapping
        for mid in matches_df["match_id"].unique():
            try:
                ref = self.repo.get_referee_for_match(int(mid))
                if ref:
                    mapping[int(mid)] = ref
            except Exception:
                continue
        return mapping
