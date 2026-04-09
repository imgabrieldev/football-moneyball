"""Treina CatBoost 1x2 with temporal CV leak-proof.

Usa Pi-Rating + form EMA + xG as features. Save model em
{models_dir}/catboost_1x2.cbm.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from football_moneyball.domain.catboost_predictor import (
    build_training_dataset,
    train_catboost_1x2,
    CATBOOST_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


class TrainCatBoost:
    """Treina model CatBoost 1x2 from data of the repo."""

    def __init__(self, repo, models_dir: str = "football_moneyball/models") -> None:
        self.repo = repo
        self.models_dir = Path(models_dir)

    def execute(
        self,
        competition: str = "Brasileirão Série A",
        seasons: list[str] | None = None,
        draw_weight: float = 1.3,
    ) -> dict[str, Any]:
        if seasons is None:
            seasons = ["2022", "2023", "2024", "2025", "2026"]

        # Coleta data of todas as temporadas
        import pandas as pd
        all_data = pd.DataFrame()
        per_season = {}
        for season in seasons:
            data = self.repo.get_all_match_data(competition, season)
            if data.empty:
                continue
            per_season[season] = int(data["match_id"].nunique())
            all_data = pd.concat([all_data, data], ignore_index=True)
            logger.info(f"Season {season}: {per_season[season]} matches")

        if all_data.empty:
            return {"error": "Without data for training."}

        total_matches = int(all_data["match_id"].nunique())
        logger.info(f"Total: {total_matches} matches, construindo dataset...")

        # Carregar match_stats for features expandidas
        match_stats = None
        try:
            match_stats = self.repo.get_all_match_stats(competition, seasons)
            logger.info(f"Match stats carregados: {len(match_stats)} rows")
        except Exception as e:
            logger.warning(f"Match stats not available: {e}")

        # v1.15.0: Carregar coach data and standings for context features
        coach_data = None
        standings_data = None
        try:
            coach_data = self.repo.get_all_coach_data_for_training()
            logger.info(f"Coach data carregados: {len(coach_data)} entries")
        except Exception as e:
            logger.warning(f"Coach data not available: {e}")
        try:
            standings_data = self.repo.get_all_standings_for_training()
            logger.info(f"Standings data carregados: {len(standings_data)} entries")
        except Exception as e:
            logger.warning(f"Standings data not available: {e}")

        # Build training dataset (leak-proof)
        X, y = build_training_dataset(
            all_data, match_stats=match_stats,
            pi_gamma=0.04, min_history=30,
            coach_data=coach_data,
            standings_data=standings_data,
        )
        logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")

        if len(X) < 100:
            return {"error": f"Dataset muito pequeno: {len(X)} samples (min 100)."}

        # Train
        model, metrics = train_catboost_1x2(
            X, y,
            draw_weight=draw_weight,
            iterations=1000,
            depth=6,
            learning_rate=0.03,
        )

        # Save
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / "catboost_1x2.cbm"
        model.save_model(str(model_path))

        # Distribution check
        from collections import Counter
        pred_dist = Counter(model.predict(X).flatten().astype(int))
        label_names = ["home", "draw", "away"]
        pred_pct = {
            label_names[k]: round(v / len(X) * 100, 1)
            for k, v in sorted(pred_dist.items())
        }

        return {
            "n_samples": len(X),
            "total_matches": total_matches,
            "per_season": per_season,
            "saved_to": str(model_path),
            "draw_weight": draw_weight,
            "pred_distribution": pred_pct,
            **metrics,
        }
