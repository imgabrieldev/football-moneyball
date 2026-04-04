"""Use case: previsao de resultado de uma partida.

Pipeline v0.5.0 — parametros dinamicos calculados de todas as partidas
da temporada, zero constantes hardcoded.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_moneyball.domain.match_predictor import predict_match


class PredictMatch:
    """Preve o resultado de uma partida via pipeline avancado.

    Busca todos os dados da temporada no banco, calcula parametros
    dinamicos (league averages, team strengths, regression), e roda
    Monte Carlo.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio para buscar historico.
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
        """Executa previsao com pipeline completo.

        Parameters
        ----------
        match_id : int
            ID da partida.
        home_team, away_team : str
            Nomes dos times.
        n_simulations : int
            Simulacoes Monte Carlo.

        Returns
        -------
        dict
            Probabilidades + metadados do pipeline.
        """
        # Buscar TODOS os dados da temporada
        all_match_data = self.repo.get_all_match_data(competition, season)

        if all_match_data.empty:
            return {"error": "Sem dados historicos no banco.", "home_team": home_team, "away_team": away_team}

        # Shot quality (xG por chute dos ultimos jogos)
        home_shots = self.repo.get_team_shots(home_team, n_matches=6)
        away_shots = self.repo.get_team_shots(away_team, n_matches=6)

        # Pipeline completo — tudo calculado dinamicamente
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
