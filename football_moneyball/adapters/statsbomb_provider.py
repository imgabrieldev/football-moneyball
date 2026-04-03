"""Adapter StatsBomb — provedor de dados via statsbombpy.

Encapsula todas as chamadas a API do StatsBomb (dados abertos), expondo
uma interface padronizada para obtencao de eventos, lineups, competicoes
e metadados de partidas.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsbombpy import sb


class StatsBombProvider:
    """Provedor de dados usando StatsBomb open data."""

    def get_match_events(self, match_id: int) -> pd.DataFrame:
        """Retorna todos os eventos de uma partida."""
        return sb.events(match_id=match_id)

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        """Retorna os lineups de uma partida, indexados por time."""
        return sb.lineups(match_id=match_id)

    def get_competitions(self) -> pd.DataFrame:
        """Retorna as competicoes disponiveis nos dados abertos.

        Filtra apenas competicoes com partidas disponiveis quando a coluna
        'match_available' existe.
        """
        comps = sb.competitions()
        if "match_available" in comps.columns:
            return comps[comps["match_available"].notna()].reset_index(drop=True)
        return comps

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Retorna todas as partidas de uma competicao/temporada."""
        return sb.matches(competition_id=competition_id, season_id=season_id)

    def get_match_info(self, match_id: int) -> dict[str, Any]:
        """Retorna metadados de uma partida especifica.

        Busca as informacoes da partida (competicao, temporada, times, placar)
        a partir dos dados abertos do StatsBomb. Itera sobre todas as
        competicoes ate encontrar a partida correspondente.

        Parameters
        ----------
        match_id : int
            Identificador da partida no StatsBomb.

        Returns
        -------
        dict
            Dicionario com chaves: ``match_id``, ``competition``, ``season``,
            ``match_date``, ``home_team``, ``away_team``, ``home_score``,
            ``away_score``.
        """
        comps = sb.competitions()
        for _, comp in comps.iterrows():
            try:
                matches = sb.matches(
                    competition_id=comp["competition_id"],
                    season_id=comp["season_id"],
                )
            except Exception:
                continue

            match_row = matches[matches["match_id"] == match_id]
            if not match_row.empty:
                m = match_row.iloc[0]
                return {
                    "match_id": int(m["match_id"]),
                    "competition": m.get(
                        "competition", comp.get("competition_name", "")
                    ),
                    "season": m.get("season", comp.get("season_name", "")),
                    "match_date": str(m.get("match_date", "")),
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "home_score": int(m.get("home_score", 0)),
                    "away_score": int(m.get("away_score", 0)),
                }

        warnings.warn(
            f"Partida {match_id} nao encontrada nos dados abertos do StatsBomb."
        )
        return {"match_id": match_id}
