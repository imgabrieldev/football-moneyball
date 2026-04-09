"""Use case: prever todos os jogos pendentes of the matchday."""

from __future__ import annotations
import os
from typing import Any
import logging

import numpy as np

from football_moneyball.domain.match_predictor import (
    predict_match, predict_match_player_aware,
)
from football_moneyball.domain.cards_predictor import predict_cards
from football_moneyball.domain.corners_predictor import predict_corners
from football_moneyball.domain.multi_monte_carlo import (
    derive_markets_from_sims, simulate_full_match,
)
from football_moneyball.domain.referee import referee_strictness_factor
from football_moneyball.domain.shots_predictor import predict_shots

logger = logging.getLogger(__name__)

# Minimo of players by time for acionar path player-aware
MIN_PLAYERS_FOR_XI = 11

# v1.3.0 — ML targets suportados
ML_TARGETS = ("goals", "corners", "cards")


class PredictAll:
    """Roda predictor for todos os jogos that still nao aconteceram.

    Parameters
    ----------
    repo : MatchRepository
    """

    def __init__(self, repo, odds_provider=None) -> None:
        self.repo = repo
        self.odds_provider = odds_provider
        self._ml_models = self._try_load_ml_models()
        self._calibration = self._try_load_calibration()
        self._catboost_1x2 = self._try_load_catboost()
        self._elo_ratings: dict[str, float] = {}  # lazy-loaded per season
        self._pi_ratings: dict = {}  # lazy-loaded per season

    def _try_load_calibration(self) -> dict | None:
        """Loads calibracao (Dixon-Coles rho + Platt scaling) if existe."""
        import pickle
        models_dir = os.getenv("MONEYBALL_MODELS_DIR", "football_moneyball/models")
        path = os.path.join(models_dir, "calibration.pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                calib = pickle.load(f)
            logger.info(
                f"Calibracao carregada: rho={calib.get('dixon_coles_rho'):.4f}, "
                f"n_samples={calib.get('n_samples')}"
            )
            return calib
        except Exception as e:
            logger.warning(f"Erro carregando calibracao: {e}")
            return None

    def _try_load_catboost(self):
        """Loads CatBoost 1x2 if existe."""
        models_dir = os.getenv("MONEYBALL_MODELS_DIR", "football_moneyball/models")
        path = os.path.join(models_dir, "catboost_1x2.cbm")
        if not os.path.exists(path):
            return None
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(path)
            logger.info("CatBoost 1x2 carregado")
            return model
        except Exception as e:
            logger.warning(f"Erro carregando CatBoost: {e}")
            return None

    def _try_load_ml_models(self) -> dict:
        """Loads models ML if existem. Returns {} if nao trained."""
        from football_moneyball.domain.ml_lambda import LambdaPredictor

        models_dir = os.getenv("MONEYBALL_MODELS_DIR", "football_moneyball/models")
        if not os.path.isdir(models_dir):
            return {}

        models = {}
        for target in ML_TARGETS:
            path = os.path.join(models_dir, f"{target}_lambda.pkl")
            if os.path.exists(path):
                try:
                    models[target] = LambdaPredictor.load(path)
                    logger.info(f"ML model carregado: {target}")
                except Exception as e:
                    logger.warning(f"Erro carregando {target}: {e}")
        return models

    def execute(
        self,
        competition: str = "Brasileirão Série A",
        season: str = "2026",
    ) -> dict[str, Any]:
        """Preve todos os jogos with odds mas without resultado.

        Usa data historicos of the banco + pipeline v0.5.0.
        """
        # All historical data for predictions
        all_data = self.repo.get_all_match_data(competition, season)
        if all_data.empty:
            return {"error": "Without data historicos.", "predictions": []}

        # Get upcoming odds — preferir provider (tem commence_time) or cache PG
        cached_odds = None
        if self.odds_provider:
            try:
                cached_odds = self.odds_provider.get_upcoming_odds()
            except Exception:
                pass
        if not cached_odds:
            cached_odds = self.repo.get_cached_odds(max_age_hours=48)
        if not cached_odds:
            return {"error": "Without odds. Rode snapshot-odds first.", "predictions": []}

        predictions = []
        seen_matchups: set[frozenset] = set()
        for game in cached_odds:
            home = game.get("home_team", "")
            away = game.get("away_team", "")

            # If nomes vazios, extrair of the outcomes h2h
            if not home or not away:
                team_names = set()
                for bm in game.get("bookmakers", []):
                    for m in bm.get("markets", []):
                        if m.get("market") == "h2h" and m.get("outcome") != "Draw":
                            team_names.add(m["outcome"])
                team_names = sorted(team_names)
                if len(team_names) >= 2:
                    home = team_names[0]
                    away = team_names[1]

            if not home or not away:
                continue

            # v1.14.2: Dedup fixtures — evita prever same jogo with mando invertido
            matchup = frozenset([home.strip().lower(), away.strip().lower()])
            if matchup in seen_matchups:
                logger.debug(f"Skipping duplicate fixture: {home} vs {away}")
                continue
            seen_matchups.add(matchup)

            try:
                # v1.1.0: tentar path player-aware if tiver data suficientes
                home_aggs = self.repo.get_player_aggregates(home, season, last_n=5)
                away_aggs = self.repo.get_player_aggregates(away, season, last_n=5)

                # Parameters of score sampling of the calibration.pkl
                _rho = (
                    self._calibration.get("dixon_coles_rho", -0.10)
                    if self._calibration else -0.10
                )
                _lambda3 = (
                    self._calibration.get("bivariate_lambda3", 0.10)
                    if self._calibration else 0.10
                )
                _score_method = (
                    self._calibration.get("score_method", "bivariate")
                    if self._calibration else "bivariate"
                )

                if (
                    len(home_aggs) >= MIN_PLAYERS_FOR_XI
                    and len(away_aggs) >= MIN_PLAYERS_FOR_XI
                ):
                    pred = predict_match_player_aware(
                        home_team=home,
                        away_team=away,
                        all_match_data=all_data,
                        home_player_aggregates=home_aggs,
                        away_player_aggregates=away_aggs,
                        dixon_coles_rho=_rho,
                        score_method=_score_method,
                        bivariate_lambda3=_lambda3,
                    )
                else:
                    # Fallback: path team-level v1.0.0
                    home_shots = self.repo.get_team_shots(home, n_matches=6)
                    away_shots = self.repo.get_team_shots(away, n_matches=6)
                    pred = predict_match(
                        home_team=home,
                        away_team=away,
                        all_match_data=all_data,
                        home_shots=home_shots or None,
                        away_shots=away_shots or None,
                        dixon_coles_rho=_rho,
                        score_method=_score_method,
                        bivariate_lambda3=_lambda3,
                    )
                    pred["lineup_type"] = "team"
                    pred["model_version"] = "v1.0.0"

                pred["home_team"] = home
                pred["away_team"] = away
                pred["commence_time"] = game.get("commence_time", "")

                # v1.4.2: Round (estimado via repo)
                try:
                    pred["round"] = self.repo.get_round_for_date(
                        pred["commence_time"], season=season,
                    )
                except Exception:
                    pred["round"] = None

                # v1.2.0: Multi-output markets (corners, cards, shots, HT)
                try:
                    multi = self._compute_multi_markets(home, away, pred, season)
                    if multi:
                        pred["multi_markets"] = multi
                except Exception as e:
                    logger.debug(f"multi_markets failed for {home}-{away}: {e}")

                # v1.4.0: Player props (marcador, assistencia, shots individuais)
                try:
                    props = self._compute_player_props(home, away, season)
                    if props:
                        pred["player_props"] = props
                except Exception as e:
                    logger.debug(f"player_props failed for {home}-{away}: {e}")

                # v1.14.0: CatBoost 1x2 replaces Poisson probs if available
                if self._catboost_1x2:
                    self._apply_catboost_1x2(pred, home, away, all_data, season)
                else:
                    # Fallback: calibration Poisson
                    if self._calibration:
                        self._apply_calibration(pred)
                    self._apply_draw_floor(pred)

                # v1.10.0/v1.13.0: Market blending (post-calibration/CatBoost)
                self._apply_market_blending(pred, home, away)

                # v1.14.2: Re-apply floors/caps post-blending (blend can undo them)
                self._apply_draw_floor(pred)
                self._apply_confidence_cap(pred)

                predictions.append(pred)

                logger.info(
                    f"[{pred.get('lineup_type', '?')}] {home} vs {away}: "
                    f"H={pred['home_win_prob']*100:.0f}% "
                    f"D={pred['draw_prob']*100:.0f}% "
                    f"A={pred['away_win_prob']*100:.0f}%"
                )
            except Exception as e:
                logger.warning(f"Erro ao prever {home} vs {away}: {e}")

        # Persistir previsoes in the banco
        if predictions:
            self.repo.save_predictions(predictions)
            # Save to immutable history
            self.repo.save_prediction_history(predictions)
            logger.info(f"{len(predictions)} previsoes salvas in the banco")

        return {
            "predictions": predictions,
            "total": len(predictions),
        }

    def _apply_calibration(self, pred: dict) -> None:
        """Apply calibration 1x2 (platt/isotonic/temperature) baseado in the method persistido."""
        import numpy as np

        if not self._calibration:
            return

        method = self._calibration.get("method", "platt")

        raw = np.array([[
            pred.get("home_win_prob", 0.33),
            pred.get("draw_prob", 0.33),
            pred.get("away_win_prob", 0.33),
        ]])

        if method == "platt":
            from football_moneyball.domain.calibration import PlattParams, calibrate_1x2
            p_home = PlattParams(**self._calibration["platt_home"])
            p_draw = PlattParams(**self._calibration["platt_draw"])
            p_away = PlattParams(**self._calibration["platt_away"])
            cal = calibrate_1x2(raw, p_home, p_draw, p_away)
        elif method == "isotonic":
            from football_moneyball.domain.calibration import (
                IsotonicCalibrator,
                calibrate_1x2_isotonic,
            )
            iso_h = IsotonicCalibrator(**self._calibration["iso_home"])
            iso_d = IsotonicCalibrator(**self._calibration["iso_draw"])
            iso_a = IsotonicCalibrator(**self._calibration["iso_away"])
            cal = calibrate_1x2_isotonic(raw, iso_h, iso_d, iso_a)
        elif method == "temperature":
            from football_moneyball.domain.calibration import (
                TemperatureScaler,
                calibrate_1x2_temperature,
            )
            temp = TemperatureScaler(**self._calibration["temperature"])
            cal = calibrate_1x2_temperature(raw, temp)
        else:
            logger.warning(f"Method of calibration desconhecido: {method}. Skip.")
            return

        pred["home_win_prob"] = round(float(cal[0, 0]), 4)
        pred["draw_prob"] = round(float(cal[0, 1]), 4)
        pred["away_win_prob"] = round(float(cal[0, 2]), 4)
        pred["calibrated"] = True
        pred["calibration_method"] = method

    def _apply_catboost_1x2(
        self, pred: dict, home: str, away: str,
        all_data, season: str,
    ) -> None:
        """v1.14.0/v1.15.0: Substitui probs 1x2 of the Poisson by the CatBoost."""
        try:
            from football_moneyball.domain.catboost_predictor import (
                build_match_features, predict_1x2, compute_form_ema, compute_gd_ema,
                _rolling_stats,
            )
            from football_moneyball.domain.pi_rating import (
                compute_all_ratings, PiRating,
            )
            from football_moneyball.domain.features import (
                compute_xg_form_ema, compute_xg_diff_ema,
                compute_coach_features, compute_standings_features,
                compute_points_last_n,
            )

            # Pi-Ratings (cached per season)
            if not self._pi_ratings:
                self._pi_ratings = compute_all_ratings(all_data, gamma=0.04)

            # Form history
            team_results = {}
            team_gd = {}
            team_xg = {}
            team_xga = {}
            for team in [home, away]:
                tdata = all_data[all_data["team"] == team].sort_values("match_id")
                results = []
                gds = []
                xgs = []
                xgas = []
                for _, r in tdata.iterrows():
                    g = r.get("goals", 0)
                    mid = r["match_id"]
                    opp = all_data[(all_data["match_id"] == mid) & (all_data["team"] != team)]
                    og = int(opp.iloc[0]["goals"]) if not opp.empty else 0
                    oxg = float(opp.iloc[0].get("xg", 1.2)) if not opp.empty else 1.2
                    results.append(1.0 if g > og else (0.5 if g == og else 0.0))
                    gds.append(float(g - og))
                    xgs.append(float(r.get("xg", 1.2)))
                    xgas.append(oxg)
                team_results[team] = results[-20:]
                team_gd[team] = gds[-20:]
                team_xg[team] = xgs[-20:]
                team_xga[team] = xgas[-20:]

            home_xg_avg = float(np.mean(team_xg.get(home, [1.3])[-5:]))
            away_xg_avg = float(np.mean(team_xg.get(away, [1.1])[-5:]))
            home_xga_avg = float(np.mean(team_xga.get(home, [1.1])[-5:]))
            away_xga_avg = float(np.mean(team_xga.get(away, [1.3])[-5:]))

            # Market odds (if already estiverem in the pred via blending earlier)
            mkt = pred.get("market_implied", {})
            mh = mkt.get("home_win_prob", 0.40)
            md = mkt.get("draw_prob", 0.28)
            ma = mkt.get("away_win_prob", 0.32)

            # Rest days
            commence = pred.get("commence_time", "")
            hr = self.repo.get_rest_days(home, commence) if commence else 7
            ar = self.repo.get_rest_days(away, commence) if commence else 7

            # v1.15.0: Match stats rolling (FIX — was using hardcoded defaults)
            home_ms = {}
            away_ms = {}
            try:
                if not hasattr(self, "_match_stats_cache"):
                    ms = self.repo.get_all_match_stats(
                        "Brasileirão Série A", [season],
                    )
                    self._match_stats_cache = {}
                    if ms is not None and not ms.empty:
                        for _, row in ms.iterrows():
                            self._match_stats_cache[int(row["match_id"])] = row.to_dict()

                # Build per-team stats history from matches in all_data
                for team, prefix in [(home, "home"), (away, "away")]:
                    tdata = all_data[all_data["team"] == team].sort_values("match_id")
                    stats_hist = []
                    for _, r in tdata.iterrows():
                        mid = int(r["match_id"])
                        ms_row = self._match_stats_cache.get(mid, {})
                        if not ms_row:
                            continue
                        is_h = bool(r.get("is_home", True))
                        pfx = "home_" if is_h else "away_"
                        stats_hist.append({
                            "possession": ms_row.get(f"{pfx}possession", 50),
                            "shots": ms_row.get(f"{pfx}shots", 10),
                            "sot": ms_row.get(f"{pfx}sot", 4),
                            "big_chances": ms_row.get(f"{pfx}big_chances", 2),
                            "pass_accuracy": ms_row.get(f"{pfx}pass_accuracy", 78),
                            "corners": ms_row.get(f"{pfx}corners", 5),
                        })
                    if stats_hist:
                        rolled = _rolling_stats(stats_hist, n=5)
                        if prefix == "home":
                            home_ms = rolled
                        else:
                            away_ms = rolled
            except Exception as e:
                logger.debug(f"Match stats retrieval failed: {e}")

            # v1.15.0: xG Form EMA
            _hxf = compute_xg_form_ema(team_xg.get(home, []))
            _axf = compute_xg_form_ema(team_xg.get(away, []), default=1.1)
            _hxd = compute_xg_diff_ema(team_xg.get(home, []), team_xga.get(home, []))
            _axd = compute_xg_diff_ema(team_xg.get(away, []), team_xga.get(away, []))

            # v1.15.0: Coach features
            hc_info = None
            ac_info = None
            try:
                hc_info = self.repo.get_coach_change_info(home, commence)
                ac_info = self.repo.get_coach_change_info(away, commence)
            except Exception:
                pass
            hc = compute_coach_features(hc_info)
            ac = compute_coach_features(ac_info)

            # v1.15.0: Standings features
            sf = {"home_position": 10.0, "away_position": 10.0, "position_gap": 0.0,
                  "home_points_last_5": 7.0, "away_points_last_5": 7.0}
            try:
                stand = self.repo.get_standing_gap(home, away, commence)
                sf_raw = compute_standings_features(stand)
                # Override points_last_5 with computed values from form
                sf_raw["home_points_last_5"] = compute_points_last_n(team_results.get(home, []))
                sf_raw["away_points_last_5"] = compute_points_last_n(team_results.get(away, []))
                sf = sf_raw
            except Exception:
                pass

            features = build_match_features(
                pi_ratings=self._pi_ratings,
                home_team=home,
                away_team=away,
                home_form=team_results.get(home, []),
                away_form=team_results.get(away, []),
                home_gd=team_gd.get(home, []),
                away_gd=team_gd.get(away, []),
                home_xg_avg=home_xg_avg,
                away_xg_avg=away_xg_avg,
                home_xga_avg=home_xga_avg,
                away_xga_avg=away_xga_avg,
                home_rest=hr,
                away_rest=ar,
                market_home=mh,
                market_draw=md,
                market_away=ma,
                home_stats=home_ms,
                away_stats=away_ms,
                # v1.15.0 context
                home_xg_form_ema=_hxf,
                away_xg_form_ema=_axf,
                home_xg_diff_ema=_hxd,
                away_xg_diff_ema=_axd,
                home_coach_tenure=hc["tenure_days"],
                away_coach_tenure=ac["tenure_days"],
                home_coach_win_rate=hc["win_rate"],
                away_coach_win_rate=ac["win_rate"],
                home_coach_changed=hc["changed_30d"],
                away_coach_changed=ac["changed_30d"],
                home_position=sf["home_position"],
                away_position=sf["away_position"],
                position_gap=sf["position_gap"],
                home_points_5=sf["home_points_last_5"],
                away_points_5=sf["away_points_last_5"],
            )

            result = predict_1x2(self._catboost_1x2, features)
            pred["home_win_prob"] = round(result["home_win_prob"], 4)
            pred["draw_prob"] = round(result["draw_prob"], 4)
            pred["away_win_prob"] = round(result["away_win_prob"], 4)
            pred["model_version"] = "v1.15.0-catboost"
            pred["calibrated"] = True  # CatBoost MultiClass already calibrado

        except Exception as e:
            logger.warning(f"CatBoost 1x2 failed for {home}-{away}: {e}")
            # Fallback: Poisson probs (already in the pred)
            if self._calibration:
                self._apply_calibration(pred)
            self._apply_draw_floor(pred)

    def _apply_draw_floor(self, pred: dict, min_draw: float = 0.26) -> None:
        """v1.13.0/v1.14.2: Enforce minimum draw probability based on empirical rate.

        Brasileirão history: 25-26% draws. Model Poisson sistematicamente
        subestima. Floor of 26% alinhado with taxa real.
        """
        d = pred.get("draw_prob", 0.0)
        if d >= min_draw:
            return
        h = pred.get("home_win_prob", 0.33)
        a = pred.get("away_win_prob", 0.33)
        # Boost draw, tirar proporcionalmente of H and A
        deficit = min_draw - d
        total_ha = h + a
        if total_ha <= 0:
            return
        pred["draw_prob"] = round(min_draw, 4)
        pred["home_win_prob"] = round(h - deficit * (h / total_ha), 4)
        pred["away_win_prob"] = round(a - deficit * (a / total_ha), 4)

    def _apply_confidence_cap(self, pred: dict, max_prob: float = 0.75) -> None:
        """v1.14.2: Cap maximum probability to avoid overconfidence.

        Brasileirão é volátil — probs acima of 75% sao irrealistas.
        Redistribui excesso proporcionalmente between os outros outcomes.
        """
        for key in ("home_win_prob", "draw_prob", "away_win_prob"):
            p = pred.get(key, 0.0)
            if p <= max_prob:
                continue
            excess = p - max_prob
            pred[key] = round(max_prob, 4)
            # Redistribui excesso proporcionalmente
            others = [k for k in ("home_win_prob", "draw_prob", "away_win_prob") if k != key]
            other_sum = sum(pred.get(k, 0.0) for k in others)
            for k in others:
                share = pred.get(k, 0.0) / other_sum if other_sum > 0 else 0.5
                pred[k] = round(pred.get(k, 0.0) + excess * share, 4)

    def _apply_market_blending(self, pred: dict, home: str, away: str) -> None:
        """v1.10.0/v1.13.0: Blenda probs of the model with consensus devigged.

        Blend ratio alpha=0.35 (35% model, 65% mercado).
        Research: market is better calibrated than model (RPS 0.20 vs 0.24).
        """
        try:
            from football_moneyball.domain.market_features import (
                blend_with_market,
                consensus_devig,
            )

            # Fetch odds from reliable bookmakers (Pinnacle + Betfair have liquidity)
            sharp_books = ["pinnacle", "betfair_ex_uk", "matchbook", "smarkets"]
            odds_list = self.repo.get_market_odds_consensus(
                home, away, preferred_bookmakers=sharp_books,
            )
            # Fallback: qualquer home
            if not odds_list:
                odds_list = self.repo.get_market_odds_consensus(home, away)

            if not odds_list:
                return

            market = consensus_devig(odds_list)
            if not market:
                return

            blended = blend_with_market(
                {
                    "home_win_prob": pred.get("home_win_prob", 0.33),
                    "draw_prob": pred.get("draw_prob", 0.33),
                    "away_win_prob": pred.get("away_win_prob", 0.33),
                },
                market,
                alpha=0.35,
            )
            pred["home_win_prob"] = round(float(blended["home_win_prob"]), 4)
            pred["draw_prob"] = round(float(blended["draw_prob"]), 4)
            pred["away_win_prob"] = round(float(blended["away_win_prob"]), 4)
            pred["market_blended"] = True
            pred["market_implied"] = {k: round(float(v), 4) for k, v in market.items()}
        except Exception as e:
            logger.debug(f"market blending failed: {e}")

    def _compute_multi_markets(
        self, home: str, away: str, pred: dict, season: str,
    ) -> dict | None:
        """Simula corners, cards, shots, HT. Usa ML if disponivel, senao analitico."""
        # v1.5.0 — usa advanced aggregates for thelimentar rich features
        home_stats = self.repo.get_team_advanced_aggregates(home, season, last_n=5)
        away_stats = self.repo.get_team_advanced_aggregates(away, season, last_n=5)
        league = self.repo.get_league_stats_averages(season)

        if home_stats["matches"] == 0 or away_stats["matches"] == 0:
            return None

        league_corners_per_team = league["corners_per_match"] / 2
        league_shots_per_team = league["shots_per_match"] / 2

        # v1.13.0: market probs as features pro ML
        _market_probs = pred.get("market_implied")
        if _market_probs:
            _market_probs = {
                "market_home_prob": _market_probs.get("home_win_prob", 0.40),
                "market_draw_prob": _market_probs.get("draw_prob", 0.28),
                "market_away_prob": _market_probs.get("away_win_prob", 0.32),
            }

        ml_used = False

        # v1.3.0 — ML corners if model carregado
        if "corners" in self._ml_models:
            lam_home_corners, lam_away_corners = self._ml_predict_pair(
                home_stats, away_stats, league, target="corners",
                home_team=home, away_team=away,
                commence_time=pred.get("commence_time", ""), season=season,
            )
            ml_used = True
        else:
            lam_home_corners, lam_away_corners = predict_corners(
                home_corners_avg=home_stats["corners_for"],
                away_corners_avg=away_stats["corners_for"],
                home_corners_against=home_stats["corners_against"],
                away_corners_against=away_stats["corners_against"],
                league_corners_per_team=league_corners_per_team,
            )

        # v1.3.0 — ML cards if model carregado
        if "cards" in self._ml_models:
            lam_home_cards, lam_away_cards = self._ml_predict_pair(
                home_stats, away_stats, league, target="cards",
                home_team=home, away_team=away,
                commence_time=pred.get("commence_time", ""), season=season,
            )
            ml_used = True
        else:
            lam_home_cards, lam_away_cards = predict_cards(
                home_cards_avg=home_stats["cards_for"],
                away_cards_avg=away_stats["cards_for"],
                home_fouls_avg=home_stats["fouls_committed"],
                away_fouls_avg=away_stats["fouls_committed"],
                referee_factor=1.0,
                derby_factor=1.0,
            )

        # Shots: sempre analitico (mercado menos critico)
        lam_home_shots, lam_away_shots = predict_shots(
            home_shots_avg=home_stats["shots_for"],
            away_shots_avg=away_stats["shots_for"],
            home_shots_against=home_stats["shots_against"],
            away_shots_against=away_stats["shots_against"],
            league_shots_per_team=league_shots_per_team,
        )

        # v1.3.0/v1.5.0 — ML goals ENSEMBLE with analytical (70% analytical + 30% ML)
        # Research: com < 200 samples, analytical Dixon-Coles bate ML puro.
        # Ensemble weighted evita regression enquanto permite ML adicionar sinal.
        lam_home_goals_analytical = pred.get("home_xg", 1.3)
        lam_away_goals_analytical = pred.get("away_xg", 1.1)
        if "goals" in self._ml_models:
            lam_home_ml, lam_away_ml = self._ml_predict_pair(
                home_stats, away_stats, league, target="goals",
                home_team=home, away_team=away,
                commence_time=pred.get("commence_time", ""), season=season,
                market_probs=_market_probs,
            )
            # Ensemble: 60% analytical + 40% ML (v1.8.0 with 810 samples)
            # Research: with 10+ samples/feature, ML bate analytical in accuracy
            lam_home_goals = 0.6 * lam_home_goals_analytical + 0.4 * lam_home_ml
            lam_away_goals = 0.6 * lam_away_goals_analytical + 0.4 * lam_away_ml
            ml_used = True
        else:
            lam_home_goals = lam_home_goals_analytical
            lam_away_goals = lam_away_goals_analytical

        pred["ml_used"] = ml_used

        # Monte Carlo multi-dim
        lambdas = {
            "home_goals": lam_home_goals,
            "away_goals": lam_away_goals,
            "home_corners": lam_home_corners,
            "away_corners": lam_away_corners,
            "home_cards": lam_home_cards,
            "away_cards": lam_away_cards,
            "home_shots": lam_home_shots,
            "away_shots": lam_away_shots,
            "home_ht_goals": lam_home_goals * 0.45,
            "away_ht_goals": lam_away_goals * 0.45,
        }
        sim_df = simulate_full_match(lambdas, n_simulations=10_000, seed=42)
        return derive_markets_from_sims(sim_df)

    def _compute_player_props(
        self, home: str, away: str, season: str,
    ) -> dict | None:
        """v1.4.0: calcula marcador/assistencia/shots for top players."""
        from football_moneyball.domain.player_props import compute_team_player_props

        home_aggs = self.repo.get_player_aggregates(home, season, last_n=5)
        away_aggs = self.repo.get_player_aggregates(away, season, last_n=5)

        if home_aggs.empty or away_aggs.empty:
            return None

        return {
            "home": compute_team_player_props(home_aggs, top_n=5),
            "away": compute_team_player_props(away_aggs, top_n=5),
        }

    def _ensure_elo_loaded(self, season: str) -> None:
        """Loads Elo ratings of the season (lazy, 1x by instancia)."""
        if self._elo_ratings:
            return
        from football_moneyball.domain.elo import final_elo_ratings
        try:
            df = self.repo.get_training_dataset(season)
            self._elo_ratings = final_elo_ratings(df)
        except Exception:
            self._elo_ratings = {}

    def _ml_predict_pair(
        self,
        home_stats: dict,
        away_stats: dict,
        league: dict,
        target: str,
        home_team: str = "",
        away_team: str = "",
        commence_time: str = "",
        season: str = "2026",
    ) -> tuple[float, float]:
        """Usa LambdaPredictor for prever (λ_home, λ_away) with rich + context features."""
        from football_moneyball.domain.feature_engineering import (
            build_context_aware_features, build_rich_team_features, FEATURE_DIM,
        )

        # Elo ratings (current — used as aprox pre-match nessa call)
        self._ensure_elo_loaded(season)
        home_elo = self._elo_ratings.get(home_team, 1500.0)
        away_elo = self._elo_ratings.get(away_team, 1500.0)

        # Rest days
        home_rest = self.repo.get_rest_days(home_team, commence_time) if commence_time else 7
        away_rest = self.repo.get_rest_days(away_team, commence_time) if commence_time else 7

        # v1.6.0: Context features
        home_context = None
        away_context = None
        try:
            home_context = self._build_team_context(home_team, away_team, commence_time, is_home=True)
            away_context = self._build_team_context(away_team, home_team, commence_time, is_home=False)
        except Exception:
            pass

        # v1.8.0: Playing style features
        home_style = None
        away_style = None
        try:
            home_style = self.repo.get_team_style_aggregates(home_team, season, last_n=5)
            away_style = self.repo.get_team_style_aggregates(away_team, season, last_n=5)
        except Exception:
            pass

        # v1.10.0: H2H + Referee features
        h2h_home_feats = None
        h2h_away_feats = None
        ref_feats = None
        try:
            from football_moneyball.domain.h2h_features import compute_h2h_features
            from football_moneyball.domain.referee_features import (
                compute_referee_features,
            )
            h2h_history = self.repo.get_h2h_history(
                home_team, away_team, ref_date=commence_time, last_n=5,
            )
            h2h_home_feats = compute_h2h_features(h2h_history, home_team, away_team)
            h2h_away_feats = compute_h2h_features(h2h_history, away_team, home_team)
            # Referee: without match_id of futuro, usa None (defaults)
            ref_feats = compute_referee_features(None)
        except Exception:
            pass

        league_avg = {
            "goals_per_team": 1.3,
            "corners_per_team": league.get("corners_per_match", 10.0) / 2,
        }

        # v1.13.0: Market-implied probs as features
        market_probs = kwargs.get("market_probs")

        X_home = build_context_aware_features(
            home_stats, away_stats, league_avg, is_home=True,
            team_elo=home_elo, opp_elo=away_elo,
            team_rest_days=home_rest, opp_rest_days=away_rest,
            team_context=home_context, opp_context=away_context,
            team_style=home_style, opp_style=away_style,
            h2h_features=h2h_home_feats, referee_features=ref_feats,
            market_probs=market_probs,
        )
        X_away = build_context_aware_features(
            away_stats, home_stats, league_avg, is_home=False,
            team_elo=away_elo, opp_elo=home_elo,
            team_rest_days=away_rest, opp_rest_days=home_rest,
            team_context=away_context, opp_context=home_context,
            team_style=away_style, opp_style=home_style,
            h2h_features=h2h_away_feats, referee_features=ref_feats,
            market_probs=market_probs,
        )

        # Backward compat: if model was trained with menos features, truncar
        model = self._ml_models[target]
        if model.model is not None and hasattr(model.model, "n_features_in_"):
            if model.model.n_features_in_ != FEATURE_DIM:
                X_home = X_home[:model.model.n_features_in_]
                X_away = X_away[:model.model.n_features_in_]

        lam_home = model.predict(X_home)
        lam_away = model.predict(X_away)
        return (lam_home, lam_away)

    def _build_team_context(
        self, team: str, opponent: str, commence_time: str, is_home: bool,
    ) -> dict:
        """Monta dict of contexto pro time (coach + injuries + fixtures + position)."""
        try:
            coach = self.repo.get_coach_change_info(team, commence_time)
        except Exception:
            coach = None
        try:
            injuries = self.repo.get_key_players_out(team, ref_date=commence_time)
        except Exception:
            injuries = None
        try:
            glast = self.repo.get_games_in_window(team, -7, 0, commence_time)
            gnext = self.repo.get_games_in_window(team, 0, 7, commence_time)
        except Exception:
            glast, gnext = 0, 0
        try:
            if is_home:
                gap = self.repo.get_standing_gap(team, opponent, commence_time)
            else:
                gap_swap = self.repo.get_standing_gap(opponent, team, commence_time)
                # Inverter: home/away ficam trocados
                gap = {
                    "home_position": gap_swap.get("home_position", 10),
                    "away_position": gap_swap.get("away_position", 10),
                    "position_gap": gap_swap.get("position_gap", 0),
                    "both_in_relegation": gap_swap.get("both_in_relegation", False),
                }
        except Exception:
            gap = None

        return {
            "coach": coach,
            "injuries": injuries,
            "fixtures": {"games_last_7d": glast, "games_next_7d": gnext},
            "position": gap,
        }
