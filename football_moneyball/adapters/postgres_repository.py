"""Repositorio PostgreSQL para o Football Moneyball.

Encapsula todas as operacoes de persistencia e consulta ao banco de dados,
incluindo upserts de metricas, busca por similaridade via pgvector e
queries de suporte a relatorios.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from football_moneyball.adapters.orm import (
    ActionValue,
    Match,
    PassNetwork,
    PlayerEmbedding,
    PlayerMatchMetrics,
    PressingMetrics,
    Stint,
)


class PostgresRepository:
    """Repositorio de acesso a dados via PostgreSQL + pgvector.

    Todos os metodos de persistencia e consulta sao expostos como metodos
    de instancia, recebendo a sessao SQLAlchemy no construtor.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        """Retorna a sessao SQLAlchemy subjacente."""
        return self._session

    # =====================================================================
    # Verificacao de existencia
    # =====================================================================

    def match_exists(self, match_id: int) -> bool:
        """Verifica se uma partida ja esta cadastrada no banco."""
        return (
            self._session.query(Match)
            .filter_by(match_id=match_id)
            .first()
            is not None
        )

    # =====================================================================
    # Upserts (save)
    # =====================================================================

    def save_match(self, match_data: dict) -> None:
        """Insere ou atualiza os dados de uma partida.

        Recebe um dicionario com as chaves correspondentes as colunas da
        tabela matches.
        """
        existing = self._session.get(Match, match_data["match_id"])
        if existing:
            for key, value in match_data.items():
                setattr(existing, key, value)
        else:
            self._session.add(Match(**match_data))
        self._session.commit()

    def save_player_metrics(self, metrics_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza metricas de jogadores para uma partida.

        O DataFrame deve conter uma coluna 'player_id' e colunas compativeis
        com os campos de PlayerMatchMetrics.
        """
        metrics_df = metrics_df.copy()
        metrics_df["match_id"] = match_id

        columns = {c.key for c in PlayerMatchMetrics.__table__.columns}

        for _, row in metrics_df.iterrows():
            data = {k: v for k, v in row.to_dict().items() if k in columns}
            existing = self._session.get(
                PlayerMatchMetrics, (data["match_id"], data["player_id"])
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                self._session.add(PlayerMatchMetrics(**data))

        self._session.commit()

    def save_pass_network(self, edges_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza as arestas da rede de passes de uma partida.

        O DataFrame deve conter as colunas 'passer_id', 'receiver_id',
        'weight' e opcionalmente 'features'.
        """
        edges_df = edges_df.copy()
        edges_df["match_id"] = match_id

        columns = {c.key for c in PassNetwork.__table__.columns}

        for _, row in edges_df.iterrows():
            data = {k: v for k, v in row.to_dict().items() if k in columns}
            pk = (data["match_id"], data["passer_id"], data["receiver_id"])
            existing = self._session.get(PassNetwork, pk)
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                self._session.add(PassNetwork(**data))

        self._session.commit()

    def save_embeddings(self, embeddings_df: pd.DataFrame) -> None:
        """Insere ou atualiza embeddings vetoriais de jogadores.

        O DataFrame deve conter 'player_id', 'season', 'embedding'
        (lista de floats), e opcionalmente 'player_name', 'cluster_label'
        e 'archetype'.
        """
        columns = {c.key for c in PlayerEmbedding.__table__.columns}

        for _, row in embeddings_df.iterrows():
            data = {k: v for k, v in row.to_dict().items() if k in columns}
            existing = self._session.get(
                PlayerEmbedding, (data["player_id"], data["season"])
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                self._session.add(PlayerEmbedding(**data))

        self._session.commit()

    def save_stints(self, stints_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza os stints (periodos de jogo) de uma partida.

        O DataFrame deve conter 'stint_number' e colunas compativeis com
        o modelo Stint.
        """
        stints_df = stints_df.copy()
        stints_df["match_id"] = match_id

        columns = {c.key for c in Stint.__table__.columns}

        for _, row in stints_df.iterrows():
            data = {k: v for k, v in row.to_dict().items() if k in columns}
            existing = self._session.get(
                Stint, (data["match_id"], data["stint_number"])
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                self._session.add(Stint(**data))

        self._session.commit()

    def save_action_values(self, values_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza valores de acao (xT/VAEP) de uma partida."""
        values_df = values_df.copy()
        values_df["match_id"] = match_id
        columns = {c.key for c in ActionValue.__table__.columns}
        for _, row in values_df.iterrows():
            data = {k: v for k, v in row.to_dict().items() if k in columns}
            existing = self._session.get(
                ActionValue, (data["match_id"], data["event_index"])
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                self._session.add(ActionValue(**data))
        self._session.commit()

    def save_pressing_metrics(self, metrics_df: pd.DataFrame, match_id: int) -> None:
        """Insere ou atualiza metricas de pressing de uma partida."""
        metrics_df = metrics_df.copy()
        metrics_df["match_id"] = match_id
        columns = {c.key for c in PressingMetrics.__table__.columns}
        for _, row in metrics_df.iterrows():
            data = {k: v for k, v in row.to_dict().items() if k in columns}
            existing = self._session.get(
                PressingMetrics, (data["match_id"], data["team"])
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                self._session.add(PressingMetrics(**data))
        self._session.commit()

    # =====================================================================
    # Consultas basicas
    # =====================================================================

    def get_player_metrics(
        self,
        player_name: str,
        season: str | None = None,
    ) -> pd.DataFrame:
        """Retorna as metricas de um jogador, opcionalmente filtradas por temporada.

        Faz join com a tabela matches para obter a temporada quando o filtro
        e aplicado.
        """
        query = self._session.query(PlayerMatchMetrics).filter(
            PlayerMatchMetrics.player_name == player_name
        )

        if season is not None:
            query = (
                query.join(Match, Match.match_id == PlayerMatchMetrics.match_id)
                .filter(Match.season == season)
            )

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        columns = [c.key for c in PlayerMatchMetrics.__table__.columns]
        data = [{col: getattr(r, col) for col in columns} for r in rows]
        return pd.DataFrame(data)

    def get_match_data(self, match_id: int) -> pd.DataFrame:
        """Retorna metricas dos jogadores de uma partida como DataFrame."""
        rows = (
            self._session.query(PlayerMatchMetrics)
            .filter_by(match_id=match_id)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        columns = [c.key for c in PlayerMatchMetrics.__table__.columns]
        data = [{col: getattr(r, col) for col in columns} for r in rows]
        return pd.DataFrame(data)

    def get_season_matches(
        self,
        competition: str,
        season: str,
    ) -> list[Match]:
        """Retorna todas as partidas de uma competicao/temporada."""
        return (
            self._session.query(Match)
            .filter(Match.competition == competition, Match.season == season)
            .all()
        )

    def get_cached_stints(self, match_id: int) -> list[Stint]:
        """Retorna os stints ja persistidos para uma partida."""
        return (
            self._session.query(Stint)
            .filter(Stint.match_id == match_id)
            .all()
        )

    def get_all_metrics(
        self,
        competition: str | None = None,
        season: str | None = None,
    ) -> pd.DataFrame:
        """Retorna todas as metricas de jogadores, filtradas por competicao/temporada.

        Faz join com matches para aplicar os filtros.
        """
        query = self._session.query(PlayerMatchMetrics)

        if competition or season:
            query = query.join(Match, Match.match_id == PlayerMatchMetrics.match_id)
            if competition:
                query = query.filter(Match.competition == competition)
            if season:
                query = query.filter(Match.season == season)

        rows = query.all()
        if not rows:
            return pd.DataFrame()

        columns = [c.key for c in PlayerMatchMetrics.__table__.columns]
        data = [{col: getattr(r, col) for col in columns} for r in rows]
        return pd.DataFrame(data)

    def get_pressing_metrics(
        self,
        team: str,
        season: str | None = None,
    ) -> list[PressingMetrics]:
        """Retorna metricas de pressing de um time, opcionalmente por temporada."""
        query = self._session.query(PressingMetrics).filter(
            PressingMetrics.team == team
        )
        if season:
            query = query.join(
                Match, Match.match_id == PressingMetrics.match_id
            ).filter(Match.season == season)
        return query.all()

    def get_embedding(
        self,
        player_name: str,
        season: str | None = None,
    ) -> Optional[PlayerEmbedding]:
        """Retorna o embedding de um jogador por temporada."""
        query = self._session.query(PlayerEmbedding).filter(
            PlayerEmbedding.player_name == player_name
        )
        if season:
            query = query.filter(PlayerEmbedding.season == season)
        return query.first()

    # =====================================================================
    # Busca por similaridade (pgvector)
    # =====================================================================

    def find_similar_players(
        self,
        player_name: str,
        season: str,
        limit: int = 10,
        cross_position: bool = False,
    ) -> pd.DataFrame:
        """Encontra jogadores com estilo de jogo mais parecido via pgvector.

        Utiliza distancia cosseno (operador ``<=>``) na tabela
        ``player_embeddings``. Por padrao, filtra apenas jogadores do mesmo
        grupo posicional.
        """
        position_filter = ""
        if not cross_position:
            position_filter = (
                " AND pe2.position_group = ("
                "   SELECT pe1.position_group"
                "   FROM player_embeddings pe1"
                "   WHERE pe1.player_name = :name AND pe1.season = :season"
                "   LIMIT 1"
                " )"
            )

        query = text(f"""
            SELECT pe2.player_id, pe2.player_name, pe2.team,
                   pe2.archetype, pe2.position_group, pe2.season,
                   pe2.cluster_label,
                   pe2.embedding <=> (
                       SELECT pe1.embedding
                       FROM player_embeddings pe1
                       WHERE pe1.player_name = :name AND pe1.season = :season
                   ) AS distance
            FROM player_embeddings pe2
            WHERE pe2.season = :season AND pe2.player_name != :name
            {position_filter}
            ORDER BY distance
            LIMIT :limit
        """)

        result = self._session.execute(
            query,
            {"name": player_name, "season": season, "limit": limit},
        )
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if not df.empty:
            df["similarity"] = 1.0 - df["distance"]

        return df

    # =====================================================================
    # Busca por complementaridade (pgvector)
    # =====================================================================

    def find_complementary_players(
        self,
        player_name: str,
        season: str,
        limit: int = 10,
        cross_position: bool = False,
    ) -> pd.DataFrame:
        """Encontra jogadores com perfil complementar (mais dissimilar).

        Ordena pela maior distancia cosseno, retornando jogadores que cobrem
        caracteristicas opostas ao jogador de referencia.
        """
        position_filter = ""
        if not cross_position:
            position_filter = (
                " AND pe2.position_group = ("
                "   SELECT pe1.position_group"
                "   FROM player_embeddings pe1"
                "   WHERE pe1.player_name = :name AND pe1.season = :season"
                "   LIMIT 1"
                " )"
            )

        query = text(f"""
            SELECT pe2.player_name, pe2.team, pe2.archetype,
                   pe2.position_group,
                   pe2.embedding <=> (
                       SELECT pe1.embedding
                       FROM player_embeddings pe1
                       WHERE pe1.player_name = :name AND pe1.season = :season
                   ) AS distance
            FROM player_embeddings pe2
            WHERE pe2.season = :season AND pe2.player_name != :name
            {position_filter}
            ORDER BY distance DESC
            LIMIT :limit
        """)

        result = self._session.execute(
            query,
            {"name": player_name, "season": season, "limit": limit},
        )
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if not df.empty:
            df["similarity"] = 1.0 - df["distance"]

        return df

    # =====================================================================
    # Recomendacao por perfil sintetico (pgvector)
    # =====================================================================

    def recommend_by_profile(
        self,
        embedding: list[float],
        season: str,
        limit: int = 10,
        position_group: str | None = None,
    ) -> pd.DataFrame:
        """Recomenda jogadores mais proximos de um embedding sintetico.

        Recebe um embedding ja projetado no espaco PCA e consulta os vizinhos
        mais proximos via pgvector.

        Parameters
        ----------
        embedding:
            Vetor de embedding ja transformado via PCA/scaler.
        season:
            Temporada para filtrar.
        limit:
            Numero maximo de resultados.
        position_group:
            Grupo posicional para filtrar resultados. Se ``None``, busca
            entre todos os jogadores.
        """
        embedding_literal = "[" + ",".join(str(v) for v in embedding) + "]"

        position_filter = ""
        params: dict[str, Any] = {
            "embedding": embedding_literal,
            "season": season,
            "limit": limit,
        }
        if position_group is not None:
            position_filter = " AND position_group = :pos_group"
            params["pos_group"] = position_group

        query = text(f"""
            SELECT player_name, team, archetype, position_group,
                   embedding <=> :embedding::vector AS distance
            FROM player_embeddings
            WHERE season = :season
            {position_filter}
            ORDER BY distance
            LIMIT :limit
        """)

        result = self._session.execute(query, params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if not df.empty:
            df["similarity"] = 1.0 - df["distance"]

        return df

    # =====================================================================
    # Compatibilidade com elenco-alvo
    # =====================================================================

    def compute_compatibility(
        self,
        player_name: str,
        season: str,
        target_team: str,
    ) -> list[dict]:
        """Calcula compatibilidade entre o jogador e o elenco do time-alvo.

        Usa distancia cosseno no espaco de embeddings: menor distancia
        significa estilo mais similar, maior distancia significa mais
        complementar.
        """
        query = text("""
            SELECT
                pe2.player_name,
                pe2.archetype,
                pe2.embedding <=> pe1.embedding AS distance
            FROM player_embeddings pe1
            JOIN player_embeddings pe2
                ON pe2.player_id != pe1.player_id
                AND pe2.season = pe1.season
            WHERE pe1.player_name = :player_name
              AND pe1.season = :season
              AND pe2.player_name IN (
                  SELECT DISTINCT pmm.player_name
                  FROM player_match_metrics pmm
                  JOIN matches m ON m.match_id = pmm.match_id
                  WHERE pmm.team = :target_team AND m.season = :season
              )
            ORDER BY distance
        """)

        result = self._session.execute(query, {
            "player_name": player_name,
            "season": season,
            "target_team": target_team,
        })

        rows = result.fetchall()
        compatibility = []
        for row in rows:
            name, archetype, distance = row
            compatibility.append({
                "player_name": name,
                "archetype": archetype,
                "distance": round(float(distance), 4),
                "similarity": round(1.0 - float(distance), 4),
            })

        return compatibility

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def close(self) -> None:
        """Fecha a sessao SQLAlchemy."""
        self._session.close()
