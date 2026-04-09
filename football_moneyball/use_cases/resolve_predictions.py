"""Use case: resolver previsoes pendentes contra resultados reais."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResolvePredictions:
    """Resolve previsoes pendentes comparando with resultados in the banco.

    Busca previsoes with status='pending', tenta encontrar o resultado
    real in the table matches (via fuzzy match of nomes), and updates
    o history with metrics of acuracia.

    Parameters
    ----------
    repo : PostgresRepository
        Repositorio with acesso ao banco.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(self) -> dict[str, Any]:
        """Resolve previsoes pendentes contra resultados reais.

        Returns
        -------
        dict
            Chaves: 'resolved', 'still_pending', 'errors'.
        """
        from football_moneyball.domain.track_record import (
            resolve_prediction,
            resolve_value_bet,
        )
        from football_moneyball.domain.match_predictor import _fuzzy_match_team

        pending = self.repo.get_pending_predictions()
        if not pending:
            return {"resolved": 0, "still_pending": 0, "errors": 0}

        # Load all matches with results
        from sqlalchemy import text
        result = self.repo._session.execute(text(
            "SELECT home_team, away_team, home_score, away_score "
            "FROM matches WHERE home_score IS NOT NULL"
        ))
        matches_with_results = [
            {
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_score": r.home_score,
                "away_score": r.away_score,
            }
            for r in result
        ]

        if not matches_with_results:
            return {
                "resolved": 0,
                "still_pending": len(pending),
                "errors": 0,
            }

        # Build lookup: normalized team pairs -> result
        known_teams = list(set(
            m["home_team"] for m in matches_with_results
        ) | set(
            m["away_team"] for m in matches_with_results
        ))

        resolved_count = 0
        error_count = 0

        for pred in pending:
            try:
                home = pred.get("home_team", "")
                away = pred.get("away_team", "")

                # Fuzzy match against known teams
                matched_home = _fuzzy_match_team(home, known_teams)
                matched_away = _fuzzy_match_team(away, known_teams)

                # Find the match result (tentando direto and invertido — old preds tinham bug)
                match_result = None
                inverted = False
                for m in matches_with_results:
                    if m["home_team"] == matched_home and m["away_team"] == matched_away:
                        match_result = m
                        break
                    if m["home_team"] == matched_away and m["away_team"] == matched_home:
                        match_result = m
                        inverted = True
                        break

                if match_result is None:
                    continue

                # If prediction is inverted vs match, swap scores to resolve correctly
                if inverted:
                    home_goals = int(match_result["away_score"])
                    away_goals = int(match_result["home_score"])
                else:
                    home_goals = int(match_result["home_score"])
                    away_goals = int(match_result["away_score"])

                # Resolve prediction
                resolution = resolve_prediction(pred, home_goals, away_goals)
                self.repo.resolve_prediction_in_db(pred["id"], resolution)
                resolved_count += 1

                # Resolve associated value bets
                actual_outcome = resolution["actual_outcome"]
                total_goals = home_goals + away_goals
                self._resolve_value_bets_for_match(
                    pred["match_key"], actual_outcome, total_goals
                )

                logger.info(
                    f"Resolvido: {home} vs {away} "
                    f"({home_goals}-{away_goals}) "
                    f"1X2={'OK' if resolution['correct_1x2'] else 'MISS'}"
                )

            except Exception as e:
                logger.warning(f"Erro ao resolver {pred.get('home_team')} vs {pred.get('away_team')}: {e}")
                error_count += 1

        still_pending = len(pending) - resolved_count
        return {
            "resolved": resolved_count,
            "still_pending": still_pending,
            "errors": error_count,
        }

    def _resolve_value_bets_for_match(
        self,
        match_key: int,
        actual_outcome: str,
        total_goals: int,
    ) -> None:
        """Resolve value bets associadas a a match."""
        from football_moneyball.domain.track_record import resolve_value_bet
        from football_moneyball.adapters.orm import ValueBetHistory

        bets = (
            self.repo._session.query(ValueBetHistory)
            .filter(
                ValueBetHistory.match_key == match_key,
                ValueBetHistory.won.is_(None),
            )
            .all()
        )

        for bet in bets:
            bet_dict = {
                "market": bet.market,
                "outcome": bet.outcome,
                "best_odds": bet.best_odds,
                "kelly_stake": bet.kelly_stake,
                "home_team": bet.home_team,
            }
            resolution = resolve_value_bet(bet_dict, actual_outcome, total_goals)
            self.repo.resolve_value_bet_in_db(bet.id, resolution)
