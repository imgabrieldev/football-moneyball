"""Use case: treinar modelos ML (LambdaPredictor) pra gols, corners, cards."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class TrainMLModels:
    """Orquestra treino de modelos GBR pra cada metrica.

    Parameters
    ----------
    repo : MatchRepository
    models_dir : str
        Diretorio pra salvar pickles (default: football_moneyball/models).
    """

    TARGETS = ["goals", "corners", "cards"]
    MIN_SAMPLES = 20

    def __init__(
        self, repo, models_dir: str = "football_moneyball/models",
    ) -> None:
        self.repo = repo
        self.models_dir = models_dir

    def execute(self, season: str = "2026") -> dict[str, Any]:
        """Treina 3 modelos e salva.

        Parameters
        ----------
        season : str
            Temporada pra treinar.

        Returns
        -------
        dict
            Metricas por target: {target: {cv_mae_mean, n_samples, ...}}
            Ou {"error": "..."} se insuficiente.
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
                    f"Dados insuficientes ({len(matches_df)} partidas). "
                    "Minimo 10 partidas com match_stats."
                ),
            }

        results: dict[str, Any] = {}
        for target in self.TARGETS:
            try:
                X, y = build_training_dataset(matches_df, target=target)
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
