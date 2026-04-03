"""Use case: previsao de resultado de uma partida."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_moneyball.domain.match_predictor import estimate_team_xg, simulate_match


class PredictMatch:
    """Preve o resultado de uma partida via Monte Carlo.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio para buscar historico de xG.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        match_id: int,
        home_team: str,
        away_team: str,
        n_simulations: int = 10_000,
    ) -> dict[str, Any]:
        """Executa a previsao de uma partida.

        Busca historico de xG de ambos os times, estima xG esperado
        e roda simulacao Monte Carlo.

        Parameters
        ----------
        match_id : int
            ID da partida.
        home_team, away_team : str
            Nomes dos times.
        n_simulations : int
            Numero de simulacoes.

        Returns
        -------
        dict
            Previsao com probabilidades de todos os mercados.
        """
        # Buscar historico de xG dos times
        home_history = self._get_team_history(home_team)
        away_history = self._get_team_history(away_team)

        # Historico defensivo (xG sofrido)
        home_defensive = self._get_defensive_history(home_team)
        away_defensive = self._get_defensive_history(away_team)

        # Estimar xG esperado
        home_xg = estimate_team_xg(
            team_history=home_history,
            opponent_history=away_defensive,
            is_home=True,
        )
        away_xg = estimate_team_xg(
            team_history=away_history,
            opponent_history=home_defensive,
            is_home=False,
        )

        # Simular
        prediction = simulate_match(home_xg, away_xg, n_simulations)
        prediction["match_id"] = match_id
        prediction["home_team"] = home_team
        prediction["away_team"] = away_team

        return prediction

    def _get_team_history(self, team: str) -> pd.DataFrame:
        """Busca historico de xG ofensivo do time."""
        try:
            all_metrics = self.repo.get_all_metrics(None, None)
            if all_metrics.empty:
                return pd.DataFrame({"xg": [1.2]})

            team_metrics = all_metrics[all_metrics["team"] == team]
            if team_metrics.empty:
                return pd.DataFrame({"xg": [1.2]})

            # Agregar xG por partida
            per_match = (
                team_metrics.groupby("match_id")
                .agg(xg=("xg", "sum"))
                .reset_index()
                .sort_values("match_id", ascending=False)
            )
            return per_match
        except Exception:
            return pd.DataFrame({"xg": [1.2]})

    def _get_defensive_history(self, team: str) -> pd.DataFrame:
        """Busca historico de xG sofrido pelo time (xG dos adversarios)."""
        try:
            all_metrics = self.repo.get_all_metrics(None, None)
            if all_metrics.empty:
                return pd.DataFrame({"xg_against": [1.2]})

            # Para cada partida do time, pegar xG do adversario
            team_matches = all_metrics[all_metrics["team"] == team]["match_id"].unique()
            opponent_xg = []
            for mid in team_matches:
                match_data = all_metrics[all_metrics["match_id"] == mid]
                opp_data = match_data[match_data["team"] != team]
                if not opp_data.empty:
                    opponent_xg.append({"match_id": mid, "xg_against": opp_data["xg"].sum()})

            if not opponent_xg:
                return pd.DataFrame({"xg_against": [1.2]})

            return pd.DataFrame(opponent_xg).sort_values("match_id", ascending=False)
        except Exception:
            return pd.DataFrame({"xg_against": [1.2]})
