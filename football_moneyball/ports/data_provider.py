"""Port para provedores de dados de futebol."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    """Interface para fontes de dados de futebol (StatsBomb, Sofascore, etc).

    Define o contrato que qualquer provedor de dados deve implementar para
    ser usado pelo Football Moneyball. A implementacao padrao utiliza
    statsbombpy, mas o sistema pode ser estendido para outros provedores.
    """

    def get_match_events(self, match_id: int) -> pd.DataFrame:
        """Retorna eventos de uma partida como DataFrame.

        O DataFrame deve conter, no minimo, as colunas: type, player,
        player_id, team, location, period, minute, second, timestamp.
        Colunas adicionais dependem do tipo de evento (ex: pass_end_location,
        shot_outcome, shot_statsbomb_xg).

        Parameters
        ----------
        match_id : int
            Identificador da partida.

        Returns
        -------
        pd.DataFrame
            DataFrame com todos os eventos da partida.
        """
        ...

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        """Retorna lineups por time com dados de posicao.

        O dicionario mapeia nome do time para um DataFrame com colunas:
        player_id, player_name, jersey_number, positions (lista de dicts
        com position_id e position).

        Parameters
        ----------
        match_id : int
            Identificador da partida.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapeamento nome_do_time -> DataFrame de jogadores.
        """
        ...

    def get_competitions(self) -> pd.DataFrame:
        """Lista competicoes disponiveis.

        Retorna DataFrame com colunas: competition_id, competition_name,
        season_id, season_name, e opcionalmente match_available.

        Returns
        -------
        pd.DataFrame
            DataFrame com as competicoes disponiveis.
        """
        ...

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Lista partidas de uma competicao/temporada.

        Retorna DataFrame com colunas: match_id, match_date, home_team,
        away_team, home_score, away_score, competition, season.

        Parameters
        ----------
        competition_id : int
            Identificador da competicao.
        season_id : int
            Identificador da temporada.

        Returns
        -------
        pd.DataFrame
            DataFrame com informacoes de todas as partidas.
        """
        ...

    def get_match_info(self, match_id: int) -> dict:
        """Retorna metadados de uma partida (competicao, placar, times).

        O dicionario retornado deve conter as chaves: match_id, competition,
        season, match_date, home_team, away_team, home_score, away_score.

        Parameters
        ----------
        match_id : int
            Identificador da partida.

        Returns
        -------
        dict
            Dicionario com metadados da partida.
        """
        ...
