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
                    )
                    pred["lineup_type"] = "team"
                    pred["model_version"] = "v1.0.0"

                pred["home_team"] = home
                pred["away_team"] = away
                pred["commence_time"] = game.get("commence_time", "")

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

    def _compute_multi_markets(
        self, home: str, away: str, pred: dict, season: str,
    ) -> dict | None:
        """Simula corners, cards, shots, HT. Usa ML se disponivel, senao analitico."""
        home_stats = self.repo.get_team_stats_aggregates(home, season, last_n=5)
        away_stats = self.repo.get_team_stats_aggregates(away, season, last_n=5)
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

        # v1.3.0 — ML goals sobrescreve home_xg/away_xg se modelo carregado
        lam_home_goals = pred.get("home_xg", 1.3)
        lam_away_goals = pred.get("away_xg", 1.1)
        if "goals" in self._ml_models:
            lam_home_goals, lam_away_goals = self._ml_predict_pair(
                home_stats, away_stats, league, target="goals",
            )
            ml_used = True

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

    def _ml_predict_pair(
        self, home_stats: dict, away_stats: dict, league: dict, target: str,
    ) -> tuple[float, float]:
        """Usa LambdaPredictor pra prever (λ_home, λ_away) de um target."""
        from football_moneyball.domain.feature_engineering import (
            build_team_features,
        )

        # league_avg adaptado pro target
        league_avg = {
            "goals_per_team": 1.3,
            "corners_per_team": league.get("corners_per_match", 10.0) / 2,
        }

        X_home = build_team_features(home_stats, away_stats, league_avg, is_home=True)
        X_away = build_team_features(away_stats, home_stats, league_avg, is_home=False)

        lam_home = self._ml_models[target].predict(X_home)
        lam_away = self._ml_models[target].predict(X_away)
        return (lam_home, lam_away)
