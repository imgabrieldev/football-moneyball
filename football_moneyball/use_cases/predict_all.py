"""Use case: prever todos os jogos pendentes da rodada."""

from __future__ import annotations
import os
from typing import Any
import logging

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

# Minimo de jogadores por time pra acionar path player-aware
MIN_PLAYERS_FOR_XI = 11

# v1.3.0 — ML targets suportados
ML_TARGETS = ("goals", "corners", "cards")


class PredictAll:
    """Roda predictor pra todos os jogos que ainda nao aconteceram.

    Parameters
    ----------
    repo : MatchRepository
    """

    def __init__(self, repo, odds_provider=None) -> None:
        self.repo = repo
        self.odds_provider = odds_provider
        self._ml_models = self._try_load_ml_models()
        self._calibration = self._try_load_calibration()
        self._elo_ratings: dict[str, float] = {}  # lazy-loaded per season

    def _try_load_calibration(self) -> dict | None:
        """Carrega calibracao (Dixon-Coles rho + Platt scaling) se existe."""
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

    def _try_load_ml_models(self) -> dict:
        """Carrega modelos ML se existem. Retorna {} se nao treinados."""
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
        """Preve todos os jogos com odds mas sem resultado.

        Usa dados historicos do banco + pipeline v0.5.0.
        """
        # All historical data for predictions
        all_data = self.repo.get_all_match_data(competition, season)
        if all_data.empty:
            return {"error": "Sem dados historicos.", "predictions": []}

        # Get upcoming odds — preferir provider (tem commence_time) ou cache PG
        cached_odds = None
        if self.odds_provider:
            try:
                cached_odds = self.odds_provider.get_upcoming_odds()
            except Exception:
                pass
        if not cached_odds:
            cached_odds = self.repo.get_cached_odds(max_age_hours=48)
        if not cached_odds:
            return {"error": "Sem odds. Rode snapshot-odds primeiro.", "predictions": []}

        predictions = []
        for game in cached_odds:
            home = game.get("home_team", "")
            away = game.get("away_team", "")

            # Se nomes vazios, extrair dos outcomes h2h
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

            try:
                # v1.1.0: tentar path player-aware se tiver dados suficientes
                home_aggs = self.repo.get_player_aggregates(home, season, last_n=5)
                away_aggs = self.repo.get_player_aggregates(away, season, last_n=5)

                # Parâmetros de score sampling do calibration.pkl
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

                # v1.4.0: Player props (marcador, assistencia, chutes individuais)
                try:
                    props = self._compute_player_props(home, away, season)
                    if props:
                        pred["player_props"] = props
                except Exception as e:
                    logger.debug(f"player_props failed for {home}-{away}: {e}")

                # v1.9.0/v1.11.0: calibração 1x2 (platt/isotonic/temperature)
                if self._calibration:
                    self._apply_calibration(pred)

                # v1.10.0: Market blending (pós-calibração)
                self._apply_market_blending(pred, home, away)

                predictions.append(pred)

                logger.info(
                    f"[{pred.get('lineup_type', '?')}] {home} vs {away}: "
                    f"H={pred['home_win_prob']*100:.0f}% "
                    f"D={pred['draw_prob']*100:.0f}% "
                    f"A={pred['away_win_prob']*100:.0f}%"
                )
            except Exception as e:
                logger.warning(f"Erro ao prever {home} vs {away}: {e}")

        # Persistir previsoes no banco
        if predictions:
            self.repo.save_predictions(predictions)
            # Save to immutable history
            self.repo.save_prediction_history(predictions)
            logger.info(f"{len(predictions)} previsoes salvas no banco")

        return {
            "predictions": predictions,
            "total": len(predictions),
        }

    def _apply_calibration(self, pred: dict) -> None:
        """Aplica calibração 1x2 (platt/isotonic/temperature) baseado no método persistido."""
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
            logger.warning(f"Método de calibração desconhecido: {method}. Skip.")
            return

        pred["home_win_prob"] = round(float(cal[0, 0]), 4)
        pred["draw_prob"] = round(float(cal[0, 1]), 4)
        pred["away_win_prob"] = round(float(cal[0, 2]), 4)
        pred["calibrated"] = True
        pred["calibration_method"] = method

    def _apply_market_blending(self, pred: dict, home: str, away: str) -> None:
        """v1.10.0: Blenda probs do modelo com consensus devigged do mercado.

        Blend ratio alpha=0.65 (65% modelo, 35% mercado) — ajustável.
        """
        try:
            from football_moneyball.domain.market_features import (
                blend_with_market,
                consensus_devig,
            )

            # Busca odds de casas confiáveis (Pinnacle + Betfair têm liquidez)
            sharp_books = ["pinnacle", "betfair_ex_uk", "matchbook", "smarkets"]
            odds_list = self.repo.get_market_odds_consensus(
                home, away, preferred_bookmakers=sharp_books,
            )
            # Fallback: qualquer casa
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
                alpha=0.65,
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
        """Simula corners, cards, shots, HT. Usa ML se disponivel, senao analitico."""
        # v1.5.0 — usa advanced aggregates para alimentar rich features
        home_stats = self.repo.get_team_advanced_aggregates(home, season, last_n=5)
        away_stats = self.repo.get_team_advanced_aggregates(away, season, last_n=5)
        league = self.repo.get_league_stats_averages(season)

        if home_stats["matches"] == 0 or away_stats["matches"] == 0:
            return None

        league_corners_per_team = league["corners_per_match"] / 2
        league_shots_per_team = league["shots_per_match"] / 2

        ml_used = False

        # v1.3.0 — ML corners se modelo carregado
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

        # v1.3.0 — ML cards se modelo carregado
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

        # v1.3.0/v1.5.0 — ML goals ENSEMBLE com analytical (70% analytical + 30% ML)
        # Research: com < 200 samples, analytical Dixon-Coles bate ML puro.
        # Ensemble weighted evita regressao enquanto permite ML adicionar sinal.
        lam_home_goals_analytical = pred.get("home_xg", 1.3)
        lam_away_goals_analytical = pred.get("away_xg", 1.1)
        if "goals" in self._ml_models:
            lam_home_ml, lam_away_ml = self._ml_predict_pair(
                home_stats, away_stats, league, target="goals",
                home_team=home, away_team=away,
                commence_time=pred.get("commence_time", ""), season=season,
            )
            # Ensemble: 60% analytical + 40% ML (v1.8.0 com 810 samples)
            # Research: com 10+ samples/feature, ML bate analytical em accuracy
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
        """v1.4.0: calcula marcador/assistencia/chutes pra top jogadores."""
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
        """Carrega Elo ratings da temporada (lazy, 1x por instancia)."""
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
        """Usa LambdaPredictor pra prever (λ_home, λ_away) com rich + context features."""
        from football_moneyball.domain.feature_engineering import (
            build_context_aware_features, build_rich_team_features, FEATURE_DIM,
        )

        # Elo ratings (current — usado como aprox pre-match nessa chamada)
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
            # Referee: sem match_id de futuro, usa None (defaults)
            ref_feats = compute_referee_features(None)
        except Exception:
            pass

        league_avg = {
            "goals_per_team": 1.3,
            "corners_per_team": league.get("corners_per_match", 10.0) / 2,
        }

        X_home = build_context_aware_features(
            home_stats, away_stats, league_avg, is_home=True,
            team_elo=home_elo, opp_elo=away_elo,
            team_rest_days=home_rest, opp_rest_days=away_rest,
            team_context=home_context, opp_context=away_context,
            team_style=home_style, opp_style=away_style,
            h2h_features=h2h_home_feats, referee_features=ref_feats,
        )
        X_away = build_context_aware_features(
            away_stats, home_stats, league_avg, is_home=False,
            team_elo=away_elo, opp_elo=home_elo,
            team_rest_days=away_rest, opp_rest_days=home_rest,
            team_context=away_context, opp_context=home_context,
            team_style=away_style, opp_style=home_style,
            h2h_features=h2h_away_feats, referee_features=ref_feats,
        )

        # Backward compat: se modelo foi treinado com menos features, truncar
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
        """Monta dict de contexto pro time (coach + injuries + fixtures + position)."""
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
