"""Use case: previsao of resultado of a match.

Pipeline v0.5.0 — parameters dinamicos calculados of todas as matches
of the season, zero constantes hardcoded.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_moneyball.domain.match_predictor import predict_match


class PredictMatch:
    """Preve o resultado of a match via pipeline avancado.

    Busca todos os data of the season in the banco, calcula parameters
    dinamicos (league averages, team strengths, regression), and roda
    Monte Carlo.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio for buscar history.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        match_id: int,
        home_team: str,
        away_team: str,
        n_simulations: int = 10_000,
        competition: str | None = "Brasileirão Série A",
        season: str | None = "2026",
    ) -> dict[str, Any]:
        """Runs previsao with pipeline complete.

        Parameters
        ----------
        match_id : int
            ID of the match.
        home_team, away_team : str
            Nomes of the times.
        n_simulations : int
            Simulacoes Monte Carlo.

        Returns
        -------
        dict
            Probabilidades + metadados of the pipeline.
        """
        # Buscar TODOS os data of the season
        all_match_data = self.repo.get_all_match_data(competition, season)

        if all_match_data.empty:
            return {"error": "Without data historicos in the banco.", "home_team": home_team, "away_team": away_team}

        # Shot quality (xG by shot of the last jogos)
        home_shots = self.repo.get_team_shots(home_team, n_matches=6)
        away_shots = self.repo.get_team_shots(away_team, n_matches=6)

        # Pipeline complete — tudo calculado dinamicamente
        prediction = predict_match(
            home_team=home_team,
            away_team=away_team,
            all_match_data=all_match_data,
            home_shots=home_shots or None,
            away_shots=away_shots or None,
            n_simulations=n_simulations,
        )

        prediction["match_id"] = match_id
        prediction["home_team"] = home_team
        prediction["away_team"] = away_team

        return prediction
