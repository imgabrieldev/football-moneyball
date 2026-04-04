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
    MatchOdds,
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
            match_key = abs(hash(f"{pred.get('home_team','')}-{pred.get('away_team','')}")) % (10**9)
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
            }
            existing = self._session.get(MatchPrediction, match_key)
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
            else:
                self._session.add(MatchPrediction(**data))
        self._session.commit()

    def get_predictions(self) -> list[dict]:
        """Retorna previsoes pre-computadas."""
        from football_moneyball.adapters.orm import MatchPrediction
        rows = self._session.query(MatchPrediction).all()
        return [
            {
                "home_team": r.home_team, "away_team": r.away_team,
                "home_xg": r.home_xg_expected, "away_xg": r.away_xg_expected,
                "home_win_prob": r.home_win_prob, "draw_prob": r.draw_prob,
                "away_win_prob": r.away_win_prob, "over_25": r.over_25_prob,
                "btts_prob": r.btts_prob, "most_likely_score": r.most_likely_score,
                "simulations": r.simulations, "predicted_at": r.predicted_at,
            }
            for r in rows
        ]

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
            match_key = abs(hash(f"{home}-{away}")) % (10**9)

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
                games[mid] = {"id": mid, "home_team": "", "away_team": "", "bookmakers": {}}

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
        match_id = abs(hash(f"{home_team}-{away_team}")) % (10**9)
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
    # Lifecycle
    # =====================================================================

    def close(self) -> None:
        """Fecha a sessao SQLAlchemy."""
        self._session.close()
