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
    calibrate_1x2_isotonic,
    calibrate_1x2_temperature,
    compute_brier_3class,
    compute_ece,
    fit_dixon_coles_rho,
    fit_isotonic_binary,
    fit_lambda3,
    fit_platt_binary,
    fit_temperature,
)
from football_moneyball.domain.match_predictor import predict_match

_VALID_METHODS = ("auto", "platt", "isotonic", "temperature")

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
        method: str = "auto",
    ) -> dict[str, Any]:
        if method not in _VALID_METHODS:
            return {"error": f"method inválido: {method}. Use um de {_VALID_METHODS}"}

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

        # Dixon-Coles ρ + Bivariate λ₃ (MLE, independe do método de calibração)
        matches = [(r["home_xg"], r["away_xg"], r["hg"], r["ag"]) for r in all_recs]
        rho = fit_dixon_coles_rho(matches)
        lambda3 = fit_lambda3(matches)
        # Auto-select score method: bivariate se lambda3 significativo, senão DC
        score_method = "bivariate" if lambda3 > 0.02 else "dixon-coles"
        logger.info(
            f"Score params: rho={rho:.4f}, lambda3={lambda3:.4f}, "
            f"score_method={score_method}"
        )

        # Monta raw probs + labels one-hot
        raw = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in all_recs])
        y = np.array([
            [1 if r["hg"] > r["ag"] else 0,
             1 if r["hg"] == r["ag"] else 0,
             1 if r["hg"] < r["ag"] else 0]
            for r in all_recs
        ])

        # Time-ordered split 80/20 (leak-proof: últimos 20% são mais recentes)
        n = len(all_recs)
        cut = int(n * 0.8)
        raw_train, raw_val = raw[:cut], raw[cut:]
        y_train, y_val = y[:cut], y[cut:]

        # Fit + eval dos 3 métodos no val
        cv_results: dict[str, dict[str, float]] = {}
        fitted: dict[str, Any] = {}

        # Platt
        p_home = fit_platt_binary(raw_train[:, 0], y_train[:, 0])
        p_draw = fit_platt_binary(raw_train[:, 1], y_train[:, 1])
        p_away = fit_platt_binary(raw_train[:, 2], y_train[:, 2])
        cal_val = calibrate_1x2(raw_val, p_home, p_draw, p_away)
        cv_results["platt"] = {
            "brier_val": compute_brier_3class(cal_val, y_val),
            "ece_val": compute_ece(cal_val, y_val),
        }
        fitted["platt"] = (p_home, p_draw, p_away)

        # Isotonic
        iso_h = fit_isotonic_binary(raw_train[:, 0], y_train[:, 0])
        iso_d = fit_isotonic_binary(raw_train[:, 1], y_train[:, 1])
        iso_a = fit_isotonic_binary(raw_train[:, 2], y_train[:, 2])
        cal_val = calibrate_1x2_isotonic(raw_val, iso_h, iso_d, iso_a)
        cv_results["isotonic"] = {
            "brier_val": compute_brier_3class(cal_val, y_val),
            "ece_val": compute_ece(cal_val, y_val),
        }
        fitted["isotonic"] = (iso_h, iso_d, iso_a)

        # Temperature
        temp = fit_temperature(raw_train, y_train)
        cal_val = calibrate_1x2_temperature(raw_val, temp)
        cv_results["temperature"] = {
            "brier_val": compute_brier_3class(cal_val, y_val),
            "ece_val": compute_ece(cal_val, y_val),
        }
        fitted["temperature"] = temp

        # Método escolhido
        if method == "auto":
            chosen = min(
                cv_results.keys(),
                key=lambda k: (cv_results[k]["brier_val"], cv_results[k]["ece_val"]),
            )
        else:
            chosen = method
        logger.info(f"Método escolhido: {chosen}")

        # Re-fit no dataset completo com método vencedor (+ sempre fitta Platt como fallback)
        p_home_full = fit_platt_binary(raw[:, 0], y[:, 0])
        p_draw_full = fit_platt_binary(raw[:, 1], y[:, 1])
        p_away_full = fit_platt_binary(raw[:, 2], y[:, 2])

        calib: dict[str, Any] = {
            "method": chosen,
            "dixon_coles_rho": rho,
            "bivariate_lambda3": lambda3,
            "score_method": score_method,
            # Platt sempre salvo (backward compat + fallback)
            "platt_home": {"a": p_home_full.a, "b": p_home_full.b},
            "platt_draw": {"a": p_draw_full.a, "b": p_draw_full.b},
            "platt_away": {"a": p_away_full.a, "b": p_away_full.b},
            "n_samples": n,
            "per_season": per_season,
            "cv_results": cv_results,
        }

        # Computa cal final no dataset completo pro método vencedor + metrics
        if chosen == "platt":
            cal_full = calibrate_1x2(raw, p_home_full, p_draw_full, p_away_full)
        elif chosen == "isotonic":
            iso_h_full = fit_isotonic_binary(raw[:, 0], y[:, 0])
            iso_d_full = fit_isotonic_binary(raw[:, 1], y[:, 1])
            iso_a_full = fit_isotonic_binary(raw[:, 2], y[:, 2])
            calib["iso_home"] = {
                "x_thresholds": iso_h_full.x_thresholds,
                "y_thresholds": iso_h_full.y_thresholds,
            }
            calib["iso_draw"] = {
                "x_thresholds": iso_d_full.x_thresholds,
                "y_thresholds": iso_d_full.y_thresholds,
            }
            calib["iso_away"] = {
                "x_thresholds": iso_a_full.x_thresholds,
                "y_thresholds": iso_a_full.y_thresholds,
            }
            cal_full = calibrate_1x2_isotonic(raw, iso_h_full, iso_d_full, iso_a_full)
        elif chosen == "temperature":
            temp_full = fit_temperature(raw, y)
            calib["temperature"] = {"T": temp_full.T}
            cal_full = calibrate_1x2_temperature(raw, temp_full)
        else:
            cal_full = raw

        brier_raw = compute_brier_3class(raw, y)
        brier_cal = compute_brier_3class(cal_full, y)
        ece_raw = compute_ece(raw, y)
        ece_cal = compute_ece(cal_full, y)
        acc_raw = float((raw.argmax(axis=1) == y.argmax(axis=1)).mean() * 100)
        acc_cal = float((cal_full.argmax(axis=1) == y.argmax(axis=1)).mean() * 100)

        calib["metrics"] = {
            "brier_raw": brier_raw,
            "brier_calibrated": brier_cal,
            "ece_raw": ece_raw,
            "ece_calibrated": ece_cal,
            "accuracy_raw": acc_raw,
            "accuracy_calibrated": acc_cal,
        }

        self.models_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.models_dir / "calibration.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(calib, f)

        return {
            "method": chosen,
            "score_method": score_method,
            "rho": rho,
            "lambda3": lambda3,
            "n_samples": n,
            "n_train": cut,
            "n_val": n - cut,
            "per_season": per_season,
            "cv_results": cv_results,
            "saved_to": str(out_path),
            **calib["metrics"],
        }
