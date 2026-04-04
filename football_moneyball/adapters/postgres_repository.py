"""Repositorio PostgreSQL para o Football Moneyball.

Encapsula todas as operacoes de persistencia e consulta ao banco de dados,
incluindo upserts de metricas, busca por similaridade via pgvector e
queries de suporte a relatorios.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


def _fuzzy_team_match(db_session, team: str) -> str:
    """Encontra nome de time no DB ignorando acentos/case.

    Se 'Sao Paulo' vem da odds API mas DB tem 'São Paulo', resolve.
    Retorna o nome DO DB se achou, senao devolve original.
    """
    if not team:
        return team
    nfkd = unicodedata.normalize("NFKD", team.strip())
    norm_input = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    query = text("""
        SELECT DISTINCT team FROM player_match_metrics
        WHERE team IS NOT NULL AND team != ''
    """)
    try:
        rows = db_session.execute(query).fetchall()
    except Exception:
        return team

    for row in rows:
        db_team = row[0]
        if not db_team:
            continue
        nfkd_db = unicodedata.normalize("NFKD", db_team.strip())
        norm_db = "".join(c for c in nfkd_db if not unicodedata.combining(c)).lower()
        if norm_db == norm_input:
            return db_team
    # Substring fallback
    for row in rows:
        db_team = row[0]
        if not db_team:
            continue
        nfkd_db = unicodedata.normalize("NFKD", db_team.strip())
        norm_db = "".join(c for c in nfkd_db if not unicodedata.combining(c)).lower()
        if norm_input in norm_db or norm_db in norm_input:
            return db_team
    return team


def _stable_match_key(home: str, away: str) -> int:
    """Gera match_key estavel (deterministico entre processos) de home+away.

    Normaliza acentos antes pra ter 'Grêmio' e 'Gremio' como mesma chave.
    """
    def _norm(s: str) -> str:
        nfkd = unicodedata.normalize("NFKD", s.strip())
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    key_str = f"{_norm(home)}-{_norm(away)}"
    digest = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (10**9)

from football_moneyball.adapters.orm import (
    ActionValue,
    LeagueStanding,
    Match,
    MatchLineup,
    MatchOdds,
    MatchStats,
    PassNetwork,
    PlayerEmbedding,
    PlayerInjury,
    PlayerMatchMetrics,
    PredictionHistory,
    PressingMetrics,
    RefereeStats,
    Stint,
    TeamCoach,
    ValueBetHistory,
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
    # v0.5.0 — Prediction queries
    # =====================================================================

    def get_all_match_data(
        self, competition: str | None = None, season: str | None = None
    ) -> pd.DataFrame:
        """Retorna todos os jogos: match_id, team, goals, xg, is_home.

        Uma linha por time por partida. Usado pelo predictor pra
        calcular parametros dinamicos.
        """
        query = text("""
            SELECT pmm.match_id, pmm.team,
                   SUM(pmm.goals) as goals,
                   SUM(pmm.xg) as xg,
                   CASE WHEN pmm.team = m.home_team THEN true ELSE false END as is_home
            FROM player_match_metrics pmm
            JOIN matches m ON m.match_id = pmm.match_id
            WHERE (:comp IS NULL OR m.competition = :comp)
              AND (:season IS NULL OR m.season = :season)
            GROUP BY pmm.match_id, pmm.team, m.home_team
            ORDER BY pmm.match_id
        """)
        return pd.read_sql(query, self._session.bind, params={
            "comp": competition, "season": season,
        })

    def get_player_aggregates(
        self,
        team: str,
        season: str | None = None,
        last_n: int = 5,
    ) -> pd.DataFrame:
        # Fuzzy match: resolver acentos (ex: 'Sao Paulo' → 'São Paulo')
        team = _fuzzy_team_match(self._session, team)
        """Retorna agregacao por jogador nos ultimos N jogos do time.

        Usado pelo pipeline player-aware pra construir probable XI e
        calcular xG/90 individual.

        Parameters
        ----------
        team : str
            Nome do time.
        season : str, optional
            Temporada. Se None, todas.
        last_n : int
            Quantos jogos recentes considerar.

        Returns
        -------
        pd.DataFrame
            Colunas: player_id, player_name, matches_played, minutes_total,
            xg_total, xa_total, shots_total, shots_on_target_total,
            goals_total, assists_total, fouls_total, crosses_total, tackles_total.
        """
        query = text("""
            WITH recent_matches AS (
                SELECT DISTINCT pmm.match_id, m.match_date
                FROM player_match_metrics pmm
                JOIN matches m ON m.match_id = pmm.match_id
                WHERE pmm.team = :team
                  AND (:season IS NULL OR m.season = :season)
                ORDER BY m.match_date DESC, pmm.match_id DESC
                LIMIT :last_n
            )
            SELECT pmm.player_id,
                   MAX(pmm.player_name) AS player_name,
                   COUNT(DISTINCT pmm.match_id) AS matches_played,
                   COALESCE(SUM(pmm.minutes_played), 0) AS minutes_total,
                   COALESCE(SUM(pmm.xg), 0) AS xg_total,
                   COALESCE(SUM(pmm.xa), 0) AS xa_total,
                   COALESCE(SUM(pmm.shots), 0) AS shots_total,
                   COALESCE(SUM(pmm.shots_on_target), 0) AS shots_on_target_total,
                   COALESCE(SUM(pmm.goals), 0) AS goals_total,
                   COALESCE(SUM(pmm.assists), 0) AS assists_total,
                   COALESCE(SUM(pmm.fouls_committed), 0) AS fouls_total,
                   COALESCE(SUM(pmm.crosses), 0) AS crosses_total,
                   COALESCE(SUM(pmm.tackles), 0) AS tackles_total
            FROM player_match_metrics pmm
            JOIN recent_matches rm ON rm.match_id = pmm.match_id
            WHERE pmm.team = :team
            GROUP BY pmm.player_id
            ORDER BY minutes_total DESC
        """)
        return pd.read_sql(query, self._session.bind, params={
            "team": team, "season": season, "last_n": last_n,
        })

    def save_match_lineups(self, lineups: list[dict]) -> None:
        """Persiste lineups (provaveis ou confirmadas) de partidas.

        Parameters
        ----------
        lineups : list[dict]
            Cada dict: match_key, player_id, team, side, player_name,
            position, is_starter, jersey_number, source, fetched_at.
        """
        from datetime import datetime
        now = datetime.now().isoformat()

        for row in lineups:
            data = {
                "match_key": int(row["match_key"]),
                "player_id": int(row["player_id"]),
                "team": str(row.get("team", "")),
                "side": str(row.get("side", "")),
                "player_name": str(row.get("player_name", "")),
                "position": str(row.get("position", "")),
                "is_starter": int(bool(row.get("is_starter", True))),
                "jersey_number": int(row.get("jersey_number", 0) or 0),
                "source": str(row.get("source", "probable")),
                "fetched_at": str(row.get("fetched_at", now)),
            }
            existing = self._session.get(
                MatchLineup, (data["match_key"], data["player_id"])
            )
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
            else:
                self._session.add(MatchLineup(**data))
        self._session.commit()

    def get_match_lineup(self, match_key: int) -> dict[str, list[dict]]:
        """Retorna lineup de uma partida indexada por lado.

        Returns
        -------
        dict
            {"home": [...], "away": [...]} com dicts de jogadores.
            Vazio se nao houver lineup.
        """
        rows = (
            self._session.query(MatchLineup)
            .filter(MatchLineup.match_key == match_key)
            .all()
        )
        result: dict[str, list[dict]] = {"home": [], "away": []}
        for r in rows:
            side = r.side or "home"
            result.setdefault(side, []).append({
                "player_id": r.player_id,
                "player_name": r.player_name,
                "team": r.team,
                "position": r.position,
                "is_starter": bool(r.is_starter),
                "jersey_number": r.jersey_number,
                "source": r.source,
            })
        return result

    def get_team_shots(self, team: str, n_matches: int = 6) -> list[float]:
        """Lista de xG por chute do time nos ultimos N jogos (via action_values)."""
        query = text("""
            SELECT av.vaep_offensive as shot_xg
            FROM action_values av
            JOIN matches m ON m.match_id = av.match_id
            WHERE av.team = :team
              AND av.action_type = 'Shot'
              AND av.vaep_offensive IS NOT NULL
              AND av.match_id IN (
                  SELECT DISTINCT match_id FROM player_match_metrics
                  WHERE team = :team
                  ORDER BY match_id DESC LIMIT :n
              )
            ORDER BY av.match_id DESC
        """)
        result = self._session.execute(query, {
            "team": team, "n": n_matches,
        })
        return [float(row.shot_xg) for row in result if row.shot_xg]

    def get_latest_match_date(self, competition: str | None = None) -> str | None:
        """Retorna data da partida mais recente."""
        query = text("""
            SELECT MAX(match_date)::text as latest
            FROM matches
            WHERE (:comp IS NULL OR competition = :comp)
        """)
        result = self._session.execute(query, {"comp": competition}).scalar()
        return result

    # =====================================================================
    # v0.7.0 — Predictions (pre-computed)
    # =====================================================================

    def save_predictions(self, predictions: list[dict]) -> None:
        """Persiste previsoes pre-computadas."""
        from datetime import datetime
        from football_moneyball.adapters.orm import MatchPrediction
        now = datetime.now().isoformat()

        def _float(v):
            """Converte numpy float pra Python float nativo."""
            if v is None:
                return None
            return float(v)

        for pred in predictions:
            match_key = _stable_match_key(pred.get('home_team',''), pred.get('away_team',''))
            data = {
                "match_id": match_key,
                "home_team": str(pred.get("home_team", "")),
                "away_team": str(pred.get("away_team", "")),
                "home_xg_expected": _float(pred.get("home_xg")),
                "away_xg_expected": _float(pred.get("away_xg")),
                "home_win_prob": _float(pred.get("home_win_prob")),
                "draw_prob": _float(pred.get("draw_prob")),
                "away_win_prob": _float(pred.get("away_win_prob")),
                "over_25_prob": _float(pred.get("over_25")),
                "btts_prob": _float(pred.get("btts_prob")),
                "most_likely_score": str(pred.get("most_likely_score", "")),
                "simulations": int(pred.get("simulations", 10000)),
                "predicted_at": now,
                "commence_time": str(pred.get("commence_time", "")),
                "lineup_type": str(pred.get("lineup_type", "team")),
                "model_version": str(pred.get("model_version", "v1.0.0")),
                "multi_markets": pred.get("multi_markets"),
                "player_props": pred.get("player_props"),
                "ml_used": int(bool(pred.get("ml_used", False))),
                "round": int(pred["round"]) if pred.get("round") else None,
            }
            existing = self._session.get(MatchPrediction, match_key)
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
            else:
                self._session.add(MatchPrediction(**data))
        self._session.commit()

    def get_predictions(self) -> list[dict]:
        """Retorna previsoes pre-computadas ordenadas por commence_time ASC (proximas primeiro)."""
        from football_moneyball.adapters.orm import MatchPrediction
        rows = (
            self._session.query(MatchPrediction)
            .order_by(MatchPrediction.commence_time.asc())
            .all()
        )
        return [
            {
                "home_team": r.home_team, "away_team": r.away_team,
                "home_xg": r.home_xg_expected, "away_xg": r.away_xg_expected,
                "home_win_prob": r.home_win_prob, "draw_prob": r.draw_prob,
                "away_win_prob": r.away_win_prob, "over_25": r.over_25_prob,
                "btts_prob": r.btts_prob, "most_likely_score": r.most_likely_score,
                "simulations": r.simulations, "predicted_at": r.predicted_at,
                "commence_time": r.commence_time,
                "round": r.round,
                "lineup_type": r.lineup_type, "model_version": r.model_version,
                "multi_markets": r.multi_markets,
                "player_props": r.player_props,
                "ml_used": bool(r.ml_used) if r.ml_used is not None else False,
            }
            for r in rows
        ]

    # =====================================================================
    # v0.9.0 — Track Record
    # =====================================================================

    def save_prediction_history(self, predictions: list[dict]) -> None:
        """Insere previsoes no historico imutavel.

        Nao atualiza registros existentes — cada snapshot e imutavel.
        Usa match_key + predicted_at para evitar duplicatas exatas.
        """
        from datetime import datetime

        def _float(v):
            if v is None:
                return None
            return float(v)

        now = datetime.now().isoformat()

        for pred in predictions:
            home = str(pred.get("home_team", ""))
            away = str(pred.get("away_team", ""))
            match_key = _stable_match_key(home, away)
            predicted_at = pred.get("predicted_at", now)

            # Avoid exact duplicates (same match + same prediction timestamp)
            existing = (
                self._session.query(PredictionHistory)
                .filter(
                    PredictionHistory.match_key == match_key,
                    PredictionHistory.predicted_at == predicted_at,
                )
                .first()
            )
            if existing:
                continue

            row = PredictionHistory(
                match_key=match_key,
                home_team=home,
                away_team=away,
                commence_time=str(pred.get("commence_time", "")),
                round=int(pred["round"]) if pred.get("round") else None,
                home_win_prob=_float(pred.get("home_win_prob")),
                draw_prob=_float(pred.get("draw_prob")),
                away_win_prob=_float(pred.get("away_win_prob")),
                over_25_prob=_float(pred.get("over_25")),
                btts_prob=_float(pred.get("btts_prob")),
                home_xg_expected=_float(pred.get("home_xg")),
                away_xg_expected=_float(pred.get("away_xg")),
                most_likely_score=str(pred.get("most_likely_score", "")),
                predicted_at=predicted_at,
                status="pending",
                lineup_type=str(pred.get("lineup_type", "team")),
                model_version=str(pred.get("model_version", "v1.0.0")),
            )
            self._session.add(row)

        self._session.commit()

    def save_value_bet_history(self, bets: list[dict]) -> None:
        """Insere value bets no historico imutavel."""
        for bet in bets:
            home = str(bet.get("home_team", ""))
            away = str(bet.get("away_team", ""))
            match_key = _stable_match_key(home, away)

            row = ValueBetHistory(
                prediction_id=bet.get("prediction_id"),
                match_key=match_key,
                home_team=home,
                away_team=away,
                market=bet.get("market"),
                outcome=bet.get("outcome"),
                model_prob=float(bet["model_prob"]) if bet.get("model_prob") is not None else None,
                best_odds=float(bet["best_odds"]) if bet.get("best_odds") is not None else None,
                bookmaker=bet.get("bookmaker"),
                edge=float(bet["edge"]) if bet.get("edge") is not None else None,
                kelly_stake=float(bet.get("kelly_stake") or bet.get("stake") or 0),
            )
            self._session.add(row)

        self._session.commit()

    def get_pending_predictions(self) -> list[dict]:
        """Retorna previsoes pendentes (status='pending')."""
        rows = (
            self._session.query(PredictionHistory)
            .filter(PredictionHistory.status == "pending")
            .all()
        )
        columns = [c.key for c in PredictionHistory.__table__.columns]
        return [{col: getattr(r, col) for col in columns} for r in rows]

    def resolve_prediction_in_db(self, pred_id: int, result: dict) -> None:
        """Atualiza previsao com resultado real."""
        from datetime import datetime

        row = self._session.get(PredictionHistory, pred_id)
        if not row:
            return
        row.actual_home_goals = result.get("actual_home_goals")
        row.actual_away_goals = result.get("actual_away_goals")
        row.actual_outcome = result.get("actual_outcome")
        # Colunas sao INTEGER (SQLite compat) — converter bool → int
        c1x2 = result.get("correct_1x2")
        cou = result.get("correct_over_under")
        row.correct_1x2 = int(c1x2) if c1x2 is not None else None
        row.correct_over_under = int(cou) if cou is not None else None
        row.brier_score = result.get("brier_score")
        row.status = "resolved"
        row.resolved_at = datetime.now().isoformat()
        self._session.commit()

    def resolve_value_bet_in_db(self, bet_id: int, result: dict) -> None:
        """Atualiza value bet com resultado real."""
        from datetime import datetime

        row = self._session.get(ValueBetHistory, bet_id)
        if not row:
            return
        row.won = result.get("won")
        row.profit = result.get("profit")
        row.resolved_at = datetime.now().isoformat()
        self._session.commit()

    def get_prediction_history(
        self,
        round_num: int | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Retorna historico de previsoes com filtros opcionais.

        Dedupe: prediction_history eh imutavel (1 linha por predicao), entao
        podemos ter a mesma partida N vezes. Retornamos apenas a MAIS RECENTE
        por match_key (por predicted_at desc).
        """
        query = self._session.query(PredictionHistory)
        if round_num is not None:
            query = query.filter(PredictionHistory.round == round_num)
        if status is not None:
            query = query.filter(PredictionHistory.status == status)
        query = query.order_by(
            PredictionHistory.predicted_at.desc(),
            PredictionHistory.id.desc(),
        )

        rows = query.all()
        columns = [c.key for c in PredictionHistory.__table__.columns]

        # Dedup por match_key: manter apenas a mais recente
        seen_keys = set()
        deduped = []
        for r in rows:
            mk = r.match_key
            if mk in seen_keys:
                continue
            seen_keys.add(mk)
            deduped.append({col: getattr(r, col) for col in columns})

        # Ordenar deduped por commence_time (proximas primeiro)
        deduped.sort(
            key=lambda x: (x.get("commence_time") or "", x.get("id") or 0),
            reverse=True,
        )
        return deduped

    def get_value_bet_history(self) -> list[dict]:
        """Retorna todo o historico de value bets."""
        rows = (
            self._session.query(ValueBetHistory)
            .order_by(ValueBetHistory.id.desc())
            .all()
        )
        columns = [c.key for c in ValueBetHistory.__table__.columns]
        return [{col: getattr(r, col) for col in columns} for r in rows]

    def get_track_record_summary(self) -> dict:
        """Retorna estatisticas agregadas do track record."""
        from football_moneyball.domain.track_record import calculate_track_record
        preds = self.get_prediction_history()
        return calculate_track_record(preds)

    # =====================================================================
    # v0.6.0 — Odds persistence
    # =====================================================================

    def save_odds(self, odds_data: list[dict]) -> None:
        """Persiste odds na tabela match_odds.

        Recebe lista de jogos no formato normalizado do odds_provider:
        [{"home_team": "...", "away_team": "...", "bookmakers": [{"name": "...", "markets": [...]}]}]
        """
        from datetime import datetime
        now = datetime.now().isoformat()

        for game in odds_data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            # Use hash of teams as match_id (we don't have Sofascore IDs for future matches)
            match_key = _stable_match_key(home, away)

            for bm in game.get("bookmakers", []):
                bm_name = bm.get("name", "")
                for mkt in bm.get("markets", []):
                    data = {
                        "match_id": match_key,
                        "bookmaker": bm_name,
                        "market": mkt.get("market", ""),
                        "outcome": mkt.get("outcome", ""),
                        "point": mkt.get("point", 0.0),
                        "odds": mkt.get("odds", 0.0),
                        "implied_prob": mkt.get("implied_prob", 0.0),
                        "fetched_at": now,
                        "commence_time": game.get("commence_time", ""),
                        "home_team": home,
                        "away_team": away,
                    }
                    # Upsert
                    existing = self._session.get(MatchOdds, (
                        data["match_id"], data["bookmaker"], data["market"],
                        data["outcome"], data["point"]
                    ))
                    if existing:
                        for k, v in data.items():
                            setattr(existing, k, v)
                    else:
                        self._session.add(MatchOdds(**data))
        self._session.commit()

    def get_cached_odds(self, max_age_hours: int = 24) -> list[dict] | None:
        """Busca odds do cache no PG se recentes o suficiente."""
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        rows = (
            self._session.query(MatchOdds)
            .filter(MatchOdds.fetched_at >= cutoff)
            .all()
        )

        if not rows:
            return None

        # Reconstruct normalized format
        games: dict[int, dict] = {}
        for row in rows:
            mid = row.match_id
            if mid not in games:
                games[mid] = {
                    "id": mid,
                    "home_team": getattr(row, "home_team", "") or "",
                    "away_team": getattr(row, "away_team", "") or "",
                    "commence_time": getattr(row, "commence_time", "") or "",
                    "bookmakers": {},
                }

            bm_name = row.bookmaker
            if bm_name not in games[mid]["bookmakers"]:
                games[mid]["bookmakers"][bm_name] = {"name": bm_name, "markets": []}

            games[mid]["bookmakers"][bm_name]["markets"].append({
                "market": row.market,
                "outcome": row.outcome,
                "point": row.point,
                "odds": row.odds,
                "implied_prob": row.implied_prob,
            })

        # Convert to list format
        result = []
        for game in games.values():
            game["bookmakers"] = list(game["bookmakers"].values())
            result.append(game)

        return result

    def get_odds_for_match(self, home_team: str, away_team: str) -> list[dict]:
        """Busca odds de uma partida por nomes dos times."""
        match_id = _stable_match_key(home_team, away_team)
        rows = (
            self._session.query(MatchOdds)
            .filter(MatchOdds.match_id == match_id)
            .all()
        )
        if not rows:
            return []
        bm_dict: dict[str, list] = {}
        for r in rows:
            bm_dict.setdefault(r.bookmaker, []).append({
                "market": r.market, "outcome": r.outcome,
                "point": r.point, "odds": r.odds, "implied_prob": r.implied_prob,
            })
        return [{"name": k, "markets": v} for k, v in bm_dict.items()]

    # =====================================================================
    # v1.2.0 — Match stats + referee
    # =====================================================================

    def save_match_stats(self, match_id: int, stats: dict) -> None:
        """Persiste estatisticas match-level (corners, cards, fouls, HT score)."""
        data = {
            "match_id": int(match_id),
            "home_corners": int(stats.get("home_corners", 0) or 0),
            "away_corners": int(stats.get("away_corners", 0) or 0),
            "home_yellow": int(stats.get("home_yellow", 0) or 0),
            "away_yellow": int(stats.get("away_yellow", 0) or 0),
            "home_red": int(stats.get("home_red", 0) or 0),
            "away_red": int(stats.get("away_red", 0) or 0),
            "home_fouls": int(stats.get("home_fouls", 0) or 0),
            "away_fouls": int(stats.get("away_fouls", 0) or 0),
            "home_shots": int(stats.get("home_shots", 0) or 0),
            "away_shots": int(stats.get("away_shots", 0) or 0),
            "home_sot": int(stats.get("home_sot", 0) or 0),
            "away_sot": int(stats.get("away_sot", 0) or 0),
            "home_saves": int(stats.get("home_saves", 0) or 0),
            "away_saves": int(stats.get("away_saves", 0) or 0),
            "home_possession": float(stats.get("home_possession", 0) or 0),
            "away_possession": float(stats.get("away_possession", 0) or 0),
            "ht_home_score": int(stats.get("ht_home_score", 0) or 0),
            "ht_away_score": int(stats.get("ht_away_score", 0) or 0),
            "referee_id": int(stats.get("referee_id", 0) or 0) or None,
            "referee_name": str(stats.get("referee_name", "") or ""),
        }
        existing = self._session.get(MatchStats, match_id)
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
        else:
            self._session.add(MatchStats(**data))
        self._session.commit()

    def save_referee_stats(self, referee: dict) -> None:
        """Upsert de estatisticas de arbitro."""
        from datetime import datetime
        rid = int(referee.get("referee_id", 0) or 0)
        if rid <= 0:
            return
        data = {
            "referee_id": rid,
            "name": str(referee.get("name", "")),
            "matches": int(referee.get("matches", 0) or 0),
            "yellow_total": int(referee.get("yellow_total", 0) or 0),
            "red_total": int(referee.get("red_total", 0) or 0),
            "yellowred_total": int(referee.get("yellowred_total", 0) or 0),
            "cards_per_game": float(referee.get("cards_per_game", 0) or 0),
            "last_updated": datetime.now().isoformat(),
        }
        existing = self._session.get(RefereeStats, rid)
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
        else:
            self._session.add(RefereeStats(**data))
        self._session.commit()

    def get_team_stats_aggregates(
        self, team: str, season: str | None = None, last_n: int = 5,
    ) -> dict:
        """Retorna medias de goals/xg/corners/cards/shots/fouls do time nos ultimos N jogos.

        Fuzzy match aplicado ao nome do time (resolve 'Sao Paulo' → 'São Paulo').

        Considera tambem o que o time SOFREU (pra calcular opponent factor).

        Returns
        -------
        dict
            Chaves: goals_for, goals_against, xg_for, xg_against,
            corners_for, corners_against, cards_for, shots_for,
            shots_against, fouls_committed, matches.
        """
        team = _fuzzy_team_match(self._session, team)
        query = text("""
            WITH recent_matches AS (
                SELECT m.match_id, m.home_team, m.away_team,
                       m.home_score, m.away_score,
                       CASE WHEN m.home_team = :team THEN 'home' ELSE 'away' END AS side
                FROM matches m
                WHERE (m.home_team = :team OR m.away_team = :team)
                  AND (:season IS NULL OR m.season = :season)
                ORDER BY m.match_date DESC, m.match_id DESC
                LIMIT :last_n
            ),
            xg_per_match AS (
                SELECT match_id, team, SUM(xg) AS xg
                FROM player_match_metrics
                GROUP BY match_id, team
            )
            SELECT
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN rm.home_score ELSE rm.away_score END), 0) AS goals_for,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN rm.away_score ELSE rm.home_score END), 0) AS goals_against,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN home_xg.xg ELSE away_xg.xg END), 0) AS xg_for,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN away_xg.xg ELSE home_xg.xg END), 0) AS xg_against,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN ms.home_corners ELSE ms.away_corners END), 0) AS corners_for,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN ms.away_corners ELSE ms.home_corners END), 0) AS corners_against,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN ms.home_yellow + ms.home_red ELSE ms.away_yellow + ms.away_red END), 0) AS cards_for,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN ms.home_shots ELSE ms.away_shots END), 0) AS shots_for,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN ms.away_shots ELSE ms.home_shots END), 0) AS shots_against,
                COALESCE(AVG(CASE WHEN rm.side = 'home' THEN ms.home_fouls ELSE ms.away_fouls END), 0) AS fouls_committed,
                COUNT(*) AS matches
            FROM recent_matches rm
            LEFT JOIN match_stats ms ON ms.match_id = rm.match_id
            LEFT JOIN xg_per_match home_xg
                ON home_xg.match_id = rm.match_id AND home_xg.team = rm.home_team
            LEFT JOIN xg_per_match away_xg
                ON away_xg.match_id = rm.match_id AND away_xg.team = rm.away_team
        """)
        result = self._session.execute(query, {
            "team": team, "season": season, "last_n": last_n,
        }).fetchone()
        if not result:
            return {"goals_for": 1.3, "goals_against": 1.3, "xg_for": 1.3,
                    "xg_against": 1.3, "corners_for": 5.0, "corners_against": 5.0,
                    "cards_for": 2.0, "shots_for": 10.0, "shots_against": 10.0,
                    "fouls_committed": 13.0, "matches": 0}
        return {
            "goals_for": float(result.goals_for or 0),
            "goals_against": float(result.goals_against or 0),
            "xg_for": float(result.xg_for or 0),
            "xg_against": float(result.xg_against or 0),
            "corners_for": float(result.corners_for or 0),
            "corners_against": float(result.corners_against or 0),
            "cards_for": float(result.cards_for or 0),
            "shots_for": float(result.shots_for or 0),
            "shots_against": float(result.shots_against or 0),
            "fouls_committed": float(result.fouls_committed or 0),
            "matches": int(result.matches or 0),
        }

    def get_league_stats_averages(self, season: str | None = None) -> dict:
        """Retorna medias da liga: corners/jogo, cards/jogo, shots/jogo, HT goals."""
        query = text("""
            SELECT
                COALESCE(AVG(ms.home_corners + ms.away_corners), 10.0) AS corners_per_match,
                COALESCE(AVG(ms.home_yellow + ms.away_yellow + ms.home_red + ms.away_red), 4.5) AS cards_per_match,
                COALESCE(AVG(ms.home_shots + ms.away_shots), 20.0) AS shots_per_match,
                COALESCE(AVG(ms.ht_home_score + ms.ht_away_score), 1.1) AS ht_goals_per_match,
                COUNT(*) AS matches
            FROM match_stats ms
            JOIN matches m ON m.match_id = ms.match_id
            WHERE (:season IS NULL OR m.season = :season)
        """)
        result = self._session.execute(query, {"season": season}).fetchone()
        if not result or result.matches == 0:
            return {"corners_per_match": 10.0, "cards_per_match": 4.5,
                    "shots_per_match": 20.0, "ht_goals_per_match": 1.1,
                    "matches": 0}
        return {
            "corners_per_match": float(result.corners_per_match or 10.0),
            "cards_per_match": float(result.cards_per_match or 4.5),
            "shots_per_match": float(result.shots_per_match or 20.0),
            "ht_goals_per_match": float(result.ht_goals_per_match or 1.1),
            "matches": int(result.matches or 0),
        }

    def get_rest_days(self, team: str, reference_date: str) -> int:
        """Dias desde ultimo jogo do time antes da data de referencia.

        Parameters
        ----------
        team : str
            Nome do time.
        reference_date : str
            ISO format (YYYY-MM-DD ou com timestamp).

        Returns
        -------
        int
            Dias desde ultimo jogo (minimo 1, default 7 se sem historico).
        """
        if not reference_date:
            return 7
        date_only = reference_date[:10]

        query = text("""
            SELECT MAX(match_date) AS last_date
            FROM matches
            WHERE (home_team = :team OR away_team = :team)
              AND match_date < CAST(:ref_date AS date)
              AND home_score IS NOT NULL
        """)
        try:
            result = self._session.execute(query, {
                "team": team, "ref_date": date_only,
            }).fetchone()
            if not result or not result.last_date:
                return 7
            from datetime import datetime
            ref = datetime.fromisoformat(date_only).date()
            last = result.last_date
            diff = (ref - last).days
            return max(1, min(diff, 30))  # clamp [1, 30]
        except Exception:
            return 7

    def get_team_advanced_aggregates(
        self, team: str, season: str | None = None, last_n: int = 5,
    ) -> dict:
        """Retorna agregados avancados do time para feature engineering v1.5.0.

        Estende get_team_stats_aggregates com:
        goal_diff_ema, xg_overperf, xga_overperf,
        creation_index, defensive_intensity, touches_per_match.

        Returns dict com todas as chaves que build_rich_team_features espera.
        """
        team = _fuzzy_team_match(self._session, team)
        basic = self.get_team_stats_aggregates(team, season, last_n)

        query = text("""
            WITH recent AS (
                SELECT m.match_id, m.match_date, m.home_team, m.away_team,
                       m.home_score, m.away_score,
                       CASE WHEN m.home_team = :team THEN 'home' ELSE 'away' END AS side
                FROM matches m
                WHERE (m.home_team = :team OR m.away_team = :team)
                  AND (:season IS NULL OR m.season = :season)
                  AND m.home_score IS NOT NULL
                ORDER BY m.match_date DESC, m.match_id DESC
                LIMIT :last_n
            ),
            pmm_agg AS (
                SELECT match_id, team,
                       SUM(xa) AS xa,
                       SUM(key_passes) AS key_passes,
                       SUM(tackles) AS tackles,
                       SUM(interceptions) AS interceptions,
                       SUM(pressure_regains) AS recoveries,
                       SUM(touches) AS touches
                FROM player_match_metrics
                WHERE team = :team
                GROUP BY match_id, team
            ),
            xg_agg AS (
                SELECT match_id, team, SUM(xg) AS xg
                FROM player_match_metrics
                GROUP BY match_id, team
            )
            SELECT
                -- goal diff (team perspective)
                AVG(
                    CASE WHEN r.side = 'home'
                         THEN r.home_score - r.away_score
                         ELSE r.away_score - r.home_score END
                ) AS goal_diff_avg,
                -- goals scored (for overperf calc)
                AVG(
                    CASE WHEN r.side = 'home' THEN r.home_score ELSE r.away_score END
                ) AS goals_scored_avg,
                AVG(
                    CASE WHEN r.side = 'home' THEN r.away_score ELSE r.home_score END
                ) AS goals_conceded_avg,
                -- xG for/against per match from xg_agg
                AVG(COALESCE(pmm.xa, 0)) AS xa_avg,
                AVG(COALESCE(pmm.key_passes, 0)) AS key_passes_avg,
                AVG(COALESCE(pmm.tackles, 0)) AS tackles_avg,
                AVG(COALESCE(pmm.interceptions, 0)) AS interceptions_avg,
                AVG(COALESCE(pmm.recoveries, 0)) AS recoveries_avg,
                AVG(COALESCE(pmm.touches, 0)) AS touches_avg,
                AVG(COALESCE(own_xg.xg, 0)) AS xg_for_avg,
                AVG(COALESCE(opp_xg.xg, 0)) AS xg_against_avg,
                COUNT(*) AS n
            FROM recent r
            LEFT JOIN pmm_agg pmm ON pmm.match_id = r.match_id
            LEFT JOIN xg_agg own_xg ON own_xg.match_id = r.match_id AND own_xg.team = :team
            LEFT JOIN xg_agg opp_xg ON opp_xg.match_id = r.match_id
                AND opp_xg.team != :team
        """)

        try:
            result = self._session.execute(query, {
                "team": team, "season": season, "last_n": last_n,
            }).fetchone()
        except Exception:
            result = None

        if not result or not result.n:
            # Fallback defaults
            basic.update({
                "goal_diff_ema": 0.0,
                "xg_overperf": 0.0, "xga_overperf": 0.0,
                "creation_index": 0.0, "defensive_intensity": 0.0,
                "touches_per_match": 500.0,
            })
            return basic

        goals_scored = float(result.goals_scored_avg or 0)
        goals_conceded = float(result.goals_conceded_avg or 0)
        xg_for = float(result.xg_for_avg or 0)
        xg_against = float(result.xg_against_avg or 0)
        xa = float(result.xa_avg or 0)
        kp = float(result.key_passes_avg or 0)
        tackles = float(result.tackles_avg or 0)
        interceptions = float(result.interceptions_avg or 0)
        recoveries = float(result.recoveries_avg or 0)
        touches = float(result.touches_avg or 500)

        basic.update({
            "goal_diff_ema": float(result.goal_diff_avg or 0),
            "xg_overperf": goals_scored - xg_for,
            "xga_overperf": goals_conceded - xg_against,
            "creation_index": xa + kp * 0.05,
            "defensive_intensity": tackles + interceptions + recoveries,
            "touches_per_match": touches,
        })
        return basic

    def get_training_dataset(
        self, season: str | None = None,
    ) -> pd.DataFrame:
        """Retorna todas partidas resolvidas com stats ricos pra treino ML v1.5.0.

        Join de matches + match_stats + player_match_metrics agregado.

        Returns
        -------
        pd.DataFrame
            Colunas: match_id, match_date, home_team, away_team,
            home_goals, away_goals, home_xg, away_xg,
            home_corners, away_corners, home_cards, away_cards,
            home_xa, away_xa, home_key_passes, away_key_passes,
            home_tackles, away_tackles, home_interceptions, away_interceptions,
            home_recoveries, away_recoveries, home_touches, away_touches.
        """
        query = text("""
            WITH team_agg AS (
                SELECT match_id, team,
                       SUM(xg) AS xg,
                       SUM(xa) AS xa,
                       SUM(key_passes) AS key_passes,
                       SUM(tackles) AS tackles,
                       SUM(interceptions) AS interceptions,
                       SUM(pressure_regains) AS recoveries,
                       SUM(touches) AS touches
                FROM player_match_metrics
                GROUP BY match_id, team
            )
            SELECT
                m.match_id, m.match_date, m.home_team, m.away_team,
                m.home_score AS home_goals, m.away_score AS away_goals,
                COALESCE(ha.xg, 0) AS home_xg,
                COALESCE(aa.xg, 0) AS away_xg,
                COALESCE(ms.home_corners, 0) AS home_corners,
                COALESCE(ms.away_corners, 0) AS away_corners,
                COALESCE(ms.home_yellow + ms.home_red, 0) AS home_cards,
                COALESCE(ms.away_yellow + ms.away_red, 0) AS away_cards,
                COALESCE(ha.xa, 0) AS home_xa,
                COALESCE(aa.xa, 0) AS away_xa,
                COALESCE(ha.key_passes, 0) AS home_key_passes,
                COALESCE(aa.key_passes, 0) AS away_key_passes,
                COALESCE(ha.tackles, 0) AS home_tackles,
                COALESCE(aa.tackles, 0) AS away_tackles,
                COALESCE(ha.interceptions, 0) AS home_interceptions,
                COALESCE(aa.interceptions, 0) AS away_interceptions,
                COALESCE(ha.recoveries, 0) AS home_recoveries,
                COALESCE(aa.recoveries, 0) AS away_recoveries,
                COALESCE(ha.touches, 0) AS home_touches,
                COALESCE(aa.touches, 0) AS away_touches
            FROM matches m
            LEFT JOIN match_stats ms ON ms.match_id = m.match_id
            LEFT JOIN team_agg ha ON ha.match_id = m.match_id AND ha.team = m.home_team
            LEFT JOIN team_agg aa ON aa.match_id = m.match_id AND aa.team = m.away_team
            WHERE (:season IS NULL OR m.season = :season)
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
            ORDER BY m.match_date, m.match_id
        """)
        return pd.read_sql(query, self._session.bind, params={"season": season})

    def get_round_for_date(
        self, commence_time: str, season: str | None = None,
    ) -> int | None:
        """Estima round de uma data baseado nos jogos ja ingeridos.

        Usa a rodada mais recente antes de commence_time + 1 como proxima.
        Se commence_time eh anterior ao ultimo jogo, usa o round daquela data.

        Parameters
        ----------
        commence_time : str
            ISO format (ex: '2026-04-12T21:30:00Z').
        season : str, optional

        Returns
        -------
        int | None
            Numero da rodada estimada.
        """
        if not commence_time:
            return None
        date_only = commence_time[:10]  # YYYY-MM-DD

        query = text("""
            SELECT round, match_date FROM matches
            WHERE round IS NOT NULL
              AND (:season IS NULL OR season = :season)
            ORDER BY ABS(EXTRACT(EPOCH FROM CAST(match_date AS timestamp) - CAST(:target_date AS timestamp))) ASC
            LIMIT 1
        """)
        try:
            result = self._session.execute(query, {
                "target_date": date_only, "season": season,
            }).fetchone()
            if not result:
                return None

            closest_round = int(result.round)
            closest_date = str(result.match_date)

            # Se target_date eh DEPOIS do closest match, estimar rodadas a frente
            if date_only > closest_date:
                import math
                from datetime import datetime
                d1 = datetime.fromisoformat(date_only)
                d2 = datetime.fromisoformat(closest_date)
                days_diff = (d1 - d2).days
                # Cada rodada ~7 dias no Brasileirao — usar ceil pra nao subestimar
                rounds_ahead = max(1, math.ceil(days_diff / 7.0))
                return closest_round + rounds_ahead
            return closest_round
        except Exception:
            return None

    # =====================================================================
    # v1.6.0 — Context-Aware features (coaches, injuries, standings)
    # =====================================================================

    def save_team_coach(
        self, team: str, coach_id: int, coach_name: str,
        start_match_date: str, end_match_date: str | None = None,
    ) -> None:
        """Upsert de coach-team relationship."""
        team = _fuzzy_team_match(self._session, team)
        existing = self._session.get(TeamCoach, (team, coach_id, start_match_date))
        data = {
            "team": team, "coach_id": coach_id,
            "start_match_date": start_match_date,
            "coach_name": coach_name,
            "end_match_date": end_match_date,
            "source": "sofascore",
        }
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
        else:
            self._session.add(TeamCoach(**data))
        self._session.commit()

    def save_player_injuries(
        self, match_id: int, team: str, injuries: list[dict],
    ) -> None:
        """Upsert lista de ausentes de um jogo."""
        from datetime import datetime
        now = datetime.now().isoformat()
        team = _fuzzy_team_match(self._session, team)

        for inj in injuries:
            pid = int(inj.get("player_id", 0) or 0)
            if pid <= 0:
                continue
            data = {
                "match_id": int(match_id),
                "player_id": pid,
                "player_name": str(inj.get("player_name", "") or ""),
                "team": team,
                "reason_code": int(inj.get("reason_code", 0) or 0),
                "reason_label": str(inj.get("reason_label", "") or ""),
                "fetched_at": now,
            }
            existing = self._session.get(PlayerInjury, (data["match_id"], pid))
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
            else:
                self._session.add(PlayerInjury(**data))
        self._session.commit()

    def save_league_standing(
        self, snapshot: list[dict], snapshot_date: str,
        competition: str, season: str,
    ) -> None:
        """Persiste snapshot da classificacao."""
        for row in snapshot:
            team = _fuzzy_team_match(self._session, row.get("team", ""))
            data = {
                "competition": competition,
                "season": season,
                "team": team,
                "snapshot_date": snapshot_date,
                "position": int(row.get("position", 0) or 0),
                "points": int(row.get("points", 0) or 0),
                "played": int(row.get("played", 0) or 0),
                "wins": int(row.get("wins", 0) or 0),
                "draws": int(row.get("draws", 0) or 0),
                "losses": int(row.get("losses", 0) or 0),
                "goals_for": int(row.get("goals_for", 0) or 0),
                "goals_against": int(row.get("goals_against", 0) or 0),
            }
            existing = self._session.get(
                LeagueStanding,
                (competition, season, team, snapshot_date),
            )
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
            else:
                self._session.add(LeagueStanding(**data))
        self._session.commit()

    # Context queries (pra feature engineering)

    def get_coach_change_info(self, team: str, ref_date: str) -> dict:
        """Retorna info sobre coach do time na data de referencia.

        Returns
        -------
        dict
            coach_name, games_since_change, coach_change_recent (<30d),
            coach_win_rate (pra essa relacao team-coach).
        """
        team = _fuzzy_team_match(self._session, team)
        if not ref_date:
            return {"games_since_change": 10, "coach_change_recent": False, "coach_win_rate": 0.5}
        date_only = ref_date[:10]

        query = text("""
            SELECT coach_id, coach_name, start_match_date
            FROM team_coaches
            WHERE team = :team
              AND start_match_date <= :ref
            ORDER BY start_match_date DESC
            LIMIT 1
        """)
        try:
            row = self._session.execute(query, {
                "team": team, "ref": date_only,
            }).fetchone()
        except Exception:
            row = None

        if not row:
            return {"games_since_change": 10, "coach_change_recent": False, "coach_win_rate": 0.5}

        # Count games coached + wins sob esse coach ate ref_date
        count_query = text("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN (pmm.team = :team AND home_score > away_score AND pmm.team = m.home_team)
                         OR (pmm.team = :team AND away_score > home_score AND pmm.team = m.away_team)
                         THEN 1 ELSE 0 END) AS wins
            FROM matches m
            JOIN (SELECT DISTINCT match_id, team FROM player_match_metrics) pmm
                ON pmm.match_id = m.match_id
            WHERE pmm.team = :team
              AND m.match_date >= CAST(:start AS date)
              AND m.match_date < CAST(:ref AS date)
              AND m.home_score IS NOT NULL
        """)
        try:
            stats = self._session.execute(count_query, {
                "team": team,
                "start": row.start_match_date[:10],
                "ref": date_only,
            }).fetchone()
        except Exception:
            stats = None

        total = int(stats.total or 0) if stats else 0
        wins = int(stats.wins or 0) if stats else 0
        win_rate = (wins / total) if total > 0 else 0.5

        # Dias desde inicio do coach
        from datetime import datetime
        try:
            start_d = datetime.fromisoformat(row.start_match_date[:10]).date()
            ref_d = datetime.fromisoformat(date_only).date()
            days_since = (ref_d - start_d).days
            is_recent = days_since < 30
        except Exception:
            days_since = 100
            is_recent = False

        return {
            "coach_name": row.coach_name,
            "games_since_change": total,
            "coach_change_recent": is_recent,
            "coach_win_rate": win_rate,
        }

    def get_key_players_out(
        self, team: str, match_id: int | None = None,
        ref_date: str | None = None, top_n: int = 3,
    ) -> dict:
        """Conta desfalques entre top N por xG/90 + xG contribution perdida.

        Identifica top N players do time por xG/90 historico.
        Retorna quantos estao ausentes na partida/data de referencia.
        """
        team = _fuzzy_team_match(self._session, team)

        # Top N players by xG/90
        top_query = text("""
            SELECT player_id, player_name,
                   SUM(xg) * 90.0 / NULLIF(SUM(minutes_played), 0) AS xg_per_90,
                   SUM(xg) AS total_xg, SUM(minutes_played) AS minutes
            FROM player_match_metrics
            WHERE team = :team
              AND minutes_played > 0
            GROUP BY player_id, player_name
            HAVING SUM(minutes_played) >= 90
            ORDER BY xg_per_90 DESC
            LIMIT :top_n
        """)
        try:
            top_rows = self._session.execute(top_query, {
                "team": team, "top_n": top_n,
            }).fetchall()
        except Exception:
            top_rows = []

        top_ids = {int(r.player_id) for r in top_rows}
        xg_per_90_map = {int(r.player_id): float(r.xg_per_90 or 0) for r in top_rows}
        total_top_xg = sum(xg_per_90_map.values())

        if not top_ids:
            return {"key_players_out": 0, "xg_contribution_missing": 0.0}

        # Injuries for target match
        if match_id:
            inj_query = text("""
                SELECT player_id FROM player_injuries
                WHERE match_id = :mid AND team = :team
            """)
            try:
                inj_rows = self._session.execute(inj_query, {
                    "mid": match_id, "team": team,
                }).fetchall()
            except Exception:
                inj_rows = []
        else:
            inj_rows = []

        out_ids = {int(r.player_id) for r in inj_rows}
        key_out = top_ids & out_ids
        missing_xg = sum(xg_per_90_map.get(pid, 0) for pid in key_out)
        pct_missing = (missing_xg / total_top_xg) if total_top_xg > 0 else 0.0

        return {
            "key_players_out": len(key_out),
            "xg_contribution_missing": round(pct_missing, 3),
        }

    def get_games_in_window(
        self, team: str, days_before: int, days_after: int, ref_date: str,
    ) -> int:
        """Numero de jogos do time em janela ao redor da data."""
        team = _fuzzy_team_match(self._session, team)
        if not ref_date:
            return 0
        date_only = ref_date[:10]

        query = text("""
            SELECT COUNT(*) FROM matches
            WHERE (home_team = :team OR away_team = :team)
              AND match_date >= CAST(:ref AS date) + (:days_before * INTERVAL '1 day')
              AND match_date <= CAST(:ref AS date) + (:days_after * INTERVAL '1 day')
              AND match_date != CAST(:ref AS date)
        """)
        try:
            result = self._session.execute(query, {
                "team": team, "ref": date_only,
                "days_before": days_before, "days_after": days_after,
            }).scalar()
            return int(result or 0)
        except Exception:
            return 0

    def get_standing_gap(
        self, home_team: str, away_team: str, ref_date: str,
    ) -> dict:
        """Retorna position/points gap entre os times na data de referencia."""
        home_team = _fuzzy_team_match(self._session, home_team)
        away_team = _fuzzy_team_match(self._session, away_team)
        if not ref_date:
            return {"home_position": 10, "away_position": 10, "position_gap": 0,
                    "points_gap": 0, "both_in_relegation": False}
        date_only = ref_date[:10]

        query = text("""
            SELECT team, position, points FROM league_standings
            WHERE team IN (:home, :away)
              AND snapshot_date <= :ref
            ORDER BY snapshot_date DESC
        """)
        try:
            rows = self._session.execute(query, {
                "home": home_team, "away": away_team, "ref": date_only,
            }).fetchall()
        except Exception:
            rows = []

        h_pos, a_pos, h_pts, a_pts = 10, 10, 0, 0
        seen = set()
        for r in rows:
            if r.team in seen:
                continue
            seen.add(r.team)
            if r.team == home_team:
                h_pos, h_pts = int(r.position or 10), int(r.points or 0)
            elif r.team == away_team:
                a_pos, a_pts = int(r.position or 10), int(r.points or 0)
            if len(seen) >= 2:
                break

        return {
            "home_position": h_pos,
            "away_position": a_pos,
            "position_gap": h_pos - a_pos,
            "points_gap": h_pts - a_pts,
            "both_in_relegation": h_pos >= 17 and a_pos >= 17,
        }

    def get_referee_stats_by_name(self, name: str) -> dict | None:
        """Busca arbitro por nome (fuzzy: LIKE)."""
        if not name:
            return None
        row = (
            self._session.query(RefereeStats)
            .filter(RefereeStats.name.ilike(f"%{name}%"))
            .first()
        )
        if not row:
            return None
        return {
            "referee_id": row.referee_id,
            "name": row.name,
            "matches": row.matches,
            "yellow_total": row.yellow_total,
            "red_total": row.red_total,
            "yellowred_total": row.yellowred_total,
            "cards_per_game": row.cards_per_game,
        }

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def close(self) -> None:
        """Fecha a sessao SQLAlchemy."""
        self._session.close()
