"""Port para persistencia de dados."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class MatchRepository(Protocol):
    """Interface para operacoes de persistencia.

    Define o contrato que qualquer implementacao de repositorio deve seguir.
    A implementacao padrao utiliza PostgreSQL + pgvector via SQLAlchemy,
    mas o sistema pode ser estendido para outros backends (SQLite, DuckDB, etc).
    """

    # ------------------------------------------------------------------
    # Match operations
    # ------------------------------------------------------------------

    def match_exists(self, match_id: int) -> bool:
        """Verifica se uma partida ja esta cadastrada no banco."""
        ...

    def save_match(self, match_data: dict) -> None:
        """Insere ou atualiza os dados de uma partida.

        O dicionario deve conter: match_id, competition, season,
        match_date, home_team, away_team, home_score, away_score.
        """
        ...

    def get_match_data(self, match_id: int) -> pd.DataFrame:
        """Retorna os dados de uma partida como DataFrame."""
        ...

    def get_season_matches(self, competition: str, season: str) -> list:
        """Retorna lista de match_ids de uma competicao/temporada."""
        ...

    # ------------------------------------------------------------------
    # Player metrics
    # ------------------------------------------------------------------

    def save_player_metrics(self, metrics_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza metricas de jogadores para uma partida.

        O DataFrame deve conter player_id e colunas compativeis com
        PlayerMatchMetrics.
        """
        ...

    def get_player_metrics(
        self, player_name: str, season: str | None = None
    ) -> pd.DataFrame:
        """Retorna metricas de um jogador, opcionalmente filtradas por temporada."""
        ...

    def get_all_metrics(self, competition: str, season: str) -> pd.DataFrame:
        """Retorna metricas de todos os jogadores de uma competicao/temporada."""
        ...

    # ------------------------------------------------------------------
    # Pass network
    # ------------------------------------------------------------------

    def save_pass_network(self, edges_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza arestas da rede de passes de uma partida.

        O DataFrame deve conter: passer_id, receiver_id, weight,
        e opcionalmente passer_name, receiver_name, features.
        """
        ...

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def save_embeddings(self, embeddings_df: pd.DataFrame) -> None:
        """Insere ou atualiza embeddings vetoriais de jogadores.

        O DataFrame deve conter: player_id, season, embedding (lista de floats),
        e opcionalmente player_name, cluster_label, archetype, position_group.
        """
        ...

    def get_embedding(
        self, player_name: str, season: str | None = None
    ) -> Any:
        """Retorna o embedding de um jogador para uma temporada."""
        ...

    # ------------------------------------------------------------------
    # Similarity / Complementarity
    # ------------------------------------------------------------------

    def find_similar_players(
        self, player_name: str, season: str, limit: int = 10
    ) -> pd.DataFrame:
        """Busca jogadores com embeddings similares via distancia cosseno.

        Retorna DataFrame com: player_name, team, archetype,
        position_group, distance, similarity.
        """
        ...

    def find_complementary_players(
        self, player_name: str, season: str, limit: int = 10
    ) -> pd.DataFrame:
        """Busca jogadores com perfil complementar (mais dissimilar).

        Retorna DataFrame com: player_name, team, archetype,
        position_group, distance, similarity.
        """
        ...

    # ------------------------------------------------------------------
    # Stints (RAPM)
    # ------------------------------------------------------------------

    def save_stints(self, stints_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza stints (periodos de jogo) de uma partida.

        O DataFrame deve conter: stint_number, home_player_ids,
        away_player_ids, duration_minutes, home_xg, away_xg, xg_diff,
        boundary_type.
        """
        ...

    def get_cached_stints(self, match_id: int) -> pd.DataFrame:
        """Retorna stints previamente persistidos para uma partida."""
        ...

    # ------------------------------------------------------------------
    # Action values (xT / VAEP)
    # ------------------------------------------------------------------

    def save_action_values(self, values_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza valores de acao (xT/VAEP) de uma partida.

        O DataFrame deve conter: event_index, player_id, player_name,
        team, action_type, start_x, start_y, end_x, end_y, xt_value,
        vaep_value, vaep_offensive, vaep_defensive.
        """
        ...

    # ------------------------------------------------------------------
    # Pressing
    # ------------------------------------------------------------------

    def save_pressing_metrics(
        self, metrics_df: pd.DataFrame, match_id: int
    ) -> None:
        """Insere ou atualiza metricas de pressing de uma partida.

        O DataFrame deve conter: team, ppda, pressing_success_rate,
        counter_pressing_fraction, high_turnovers,
        shot_ending_high_turnovers, pressing_zone_1..6.
        """
        ...

    def get_pressing_metrics(
        self, team: str, season: str | None = None
    ) -> list:
        """Retorna metricas de pressing de um time, opcionalmente por temporada."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Fecha a conexao/sessao do repositorio."""
        ...
