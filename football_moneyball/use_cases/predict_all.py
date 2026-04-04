"""Use case: prever todos os jogos pendentes da rodada."""

from __future__ import annotations
from typing import Any
import logging

from football_moneyball.domain.match_predictor import predict_match

logger = logging.getLogger(__name__)


class PredictAll:
    """Roda predictor pra todos os jogos que ainda nao aconteceram.

    Parameters
    ----------
    repo : MatchRepository
    """

    def __init__(self, repo) -> None:
        self.repo = repo

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

        # Get upcoming odds (games that haven't happened yet)
        cached_odds = self.repo.get_cached_odds(max_age_hours=48)
        if not cached_odds:
            return {"error": "Sem odds no banco. Rode snapshot-odds primeiro.", "predictions": []}

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
                # Get shot quality
                home_shots = self.repo.get_team_shots(home, n_matches=6)
                away_shots = self.repo.get_team_shots(away, n_matches=6)

                pred = predict_match(
                    home_team=home,
                    away_team=away,
                    all_match_data=all_data,
                    home_shots=home_shots or None,
                    away_shots=away_shots or None,
                )

                pred["home_team"] = home
                pred["away_team"] = away
                predictions.append(pred)

                logger.info(
                    f"{home} vs {away}: "
                    f"H={pred['home_win_prob']*100:.0f}% "
                    f"D={pred['draw_prob']*100:.0f}% "
                    f"A={pred['away_win_prob']*100:.0f}%"
                )
            except Exception as e:
                logger.warning(f"Erro ao prever {home} vs {away}: {e}")

        return {
            "predictions": predictions,
            "total": len(predictions),
        }
