"""Fitta calibração (Dixon-Coles ρ + Platt scaling) em dados históricos.

Gera predicoes leak-proof em todas as temporadas disponiveis e usa outcomes reais
pra fittar parametros de calibracao. Salva em {models_dir}/calibration.pkl.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from football_moneyball.domain.calibration import (
    calibrate_1x2,
    fit_dixon_coles_rho,
    fit_platt_binary,
)
from football_moneyball.domain.match_predictor import predict_match

logger = logging.getLogger(__name__)


class FitCalibration:
    """Fitta calibração do pipeline de previsão em dados leak-proof."""

    def __init__(self, repo, models_dir: str = "football_moneyball/models") -> None:
        self.repo = repo
        self.models_dir = Path(models_dir)

    def _collect_leak_proof(
        self,
        competition: str,
        season: str,
        min_prior: int = 30,
        n_sims: int = 5_000,
    ) -> list[dict[str, Any]]:
        all_data = self.repo.get_all_match_data(competition, season)
        if all_data.empty:
            return []

        match_ids = sorted(all_data["match_id"].unique())
        results = {}
        for mid in match_ids:
            md = all_data[all_data["match_id"] == mid]
            h = md[md["is_home"] == True]  # noqa: E712
            a = md[md["is_home"] == False]  # noqa: E712
            if h.empty or a.empty:
                continue
            results[mid] = {
                "home": h["team"].iloc[0], "away": a["team"].iloc[0],
                "hg": int(h["goals"].iloc[0]), "ag": int(a["goals"].iloc[0]),
            }

        records = []
        for mid in match_ids:
            if mid not in results:
                continue
            r = results[mid]
            prior = all_data[all_data["match_id"] < mid]
            if len(prior["match_id"].unique()) < min_prior:
                continue
            if prior[prior["team"] == r["home"]].empty:
                continue
            if prior[prior["team"] == r["away"]].empty:
                continue
            try:
                pred = predict_match(r["home"], r["away"], prior, n_simulations=n_sims, seed=mid)
            except Exception as exc:
                logger.debug(f"skip mid={mid}: {exc}")
                continue
            records.append({
                "home_xg": pred["home_xg"], "away_xg": pred["away_xg"],
                "p_home": pred["home_win_prob"], "p_draw": pred["draw_prob"],
                "p_away": pred["away_win_prob"],
                "hg": r["hg"], "ag": r["ag"],
            })
        return records

    def execute(
        self,
        competition: str = "Brasileirão Série A",
        seasons: list[str] | None = None,
    ) -> dict[str, Any]:
        if seasons is None:
            seasons = ["2024", "2026"]

        all_recs = []
        per_season = {}
        for season in seasons:
            recs = self._collect_leak_proof(competition, season)
            per_season[season] = len(recs)
            all_recs.extend(recs)
            logger.info(f"Temporada {season}: {len(recs)} predições leak-proof")

        if not all_recs:
            return {"error": "Sem dados para calibração."}

        # Dixon-Coles ρ (MLE analítico)
        matches = [(r["home_xg"], r["away_xg"], r["hg"], r["ag"]) for r in all_recs]
        rho = fit_dixon_coles_rho(matches)

        # Platt 3-class (one-vs-rest)
        raw = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in all_recs])
        y = np.array([
            [1 if r["hg"] > r["ag"] else 0,
             1 if r["hg"] == r["ag"] else 0,
             1 if r["hg"] < r["ag"] else 0]
            for r in all_recs
        ])
        p_home = fit_platt_binary(raw[:, 0], y[:, 0])
        p_draw = fit_platt_binary(raw[:, 1], y[:, 1])
        p_away = fit_platt_binary(raw[:, 2], y[:, 2])

        # Metrics in-sample
        cal = calibrate_1x2(raw, p_home, p_draw, p_away)
        brier_raw = float(np.mean(np.sum((raw - y) ** 2, axis=1)))
        brier_cal = float(np.mean(np.sum((cal - y) ** 2, axis=1)))
        acc_raw = float((raw.argmax(axis=1) == y.argmax(axis=1)).mean() * 100)
        acc_cal = float((cal.argmax(axis=1) == y.argmax(axis=1)).mean() * 100)

        calib = {
            "dixon_coles_rho": rho,
            "platt_home": {"a": p_home.a, "b": p_home.b},
            "platt_draw": {"a": p_draw.a, "b": p_draw.b},
            "platt_away": {"a": p_away.a, "b": p_away.b},
            "n_samples": len(all_recs),
            "per_season": per_season,
            "metrics": {
                "brier_raw": brier_raw,
                "brier_calibrated": brier_cal,
                "accuracy_raw": acc_raw,
                "accuracy_calibrated": acc_cal,
            },
        }

        self.models_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.models_dir / "calibration.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(calib, f)

        return {
            "rho": rho,
            "n_samples": len(all_recs),
            "per_season": per_season,
            "saved_to": str(out_path),
            **calib["metrics"],
        }
