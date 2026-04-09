"""PostgreSQL repository for Football Moneyball.

Encapsulates all persistence and query operations on the database,
including metric upserts, similarity search via pgvector and
queries supporting reports.
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
    """Finds a team name in the DB ignoring accents/case.

    If 'Sao Paulo' comes from the odds API but the DB has 'Sao Paulo',
    resolves it. Returns the DB name if found, otherwise returns the
    original.
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
    """Generates a stable match_key (deterministic across processes) from home+away.

    Normalizes accents first so that 'Gremio' and 'Gremio' produce the
    same key. The key is symmetric: 'A vs B' and 'B vs A' produce the
    same key to avoid duplicates when the odds API and Sofascore
    disagree on home/away.
    """
    def _norm(s: str) -> str:
        nfkd = unicodedata.normalize("NFKD", s.strip())
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

    # Sort alphabetically so the key is independent of home/away
    pair = sorted([_norm(home), _norm(away)])
    key_str = f"{pair[0]}-{pair[1]}"
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
    """Data access repository via PostgreSQL + pgvector.

    All persistence and query methods are exposed as instance methods,
    receiving the SQLAlchemy session in the constructor.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self) -> Session:
        """Returns the underlying SQLAlchemy session."""
        return self._session

    # =====================================================================
    # Existence checks
    # =====================================================================

    def match_exists(self, match_id: int) -> bool:
        """Checks whether a match is already stored in the database."""
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
        """Inserts or updates the data of a match.

        Receives a dictionary with keys corresponding to the columns of
        the matches table.
        """
        existing = self._session.get(Match, match_data["match_id"])
        if existing:
            for key, value in match_data.items():
                setattr(existing, key, value)
        else:
            self._session.add(Match(**match_data))
        self._session.commit()

    def save_player_metrics(self, metrics_df: pd.DataFrame, match_id: int) -> None:
        """Inserts or updates player metrics for a match.

        The DataFrame must contain a 'player_id' column and columns
        compatible with the PlayerMatchMetrics fields.
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
        """Inserts or updates the edges of a match's pass network.

        The DataFrame must contain the columns 'passer_id', 'receiver_id',
        'weight' and optionally 'features'.
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
        """Inserts or updates player vector embeddings.

        The DataFrame must contain 'player_id', 'season', 'embedding'
        (list of floats), and optionally 'player_name', 'cluster_label'
        and 'archetype'.
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
        """Inserts or updates the stints (game periods) of a match.

        The DataFrame must contain 'stint_number' and columns compatible
        with the Stint model.
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
        """Inserts or updates action values (xT/VAEP) of a match."""
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
        """Inserts or updates pressing metrics of a match."""
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
    # Basic queries
    # =====================================================================

    def get_player_metrics(
        self,
        player_name: str,
        season: str | None = None,
    ) -> pd.DataFrame:
        """Returns a player's metrics, optionally filtered by season.

        Joins with the matches table to obtain the season when the filter
        is applied.
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
        """Returns a match's player metrics as a DataFrame."""
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
        """Returns all matches of a competition/season."""
        return (
            self._session.query(Match)
            .filter(Match.competition == competition, Match.season == season)
            .all()
        )

    def get_cached_stints(self, match_id: int) -> list[Stint]:
        """Returns the stints already persisted for a match."""
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
        """Returns all player metrics, filtered by competition/season.

        Joins with matches to apply the filters.
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
        """Returns a team's pressing metrics, optionally filtered by season."""
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
        """Returns a player's embedding for a given season."""
        query = self._session.query(PlayerEmbedding).filter(
            PlayerEmbedding.player_name == player_name
        )
        if season:
            query = query.filter(PlayerEmbedding.season == season)
        return query.first()

    # =====================================================================
    # Similarity search (pgvector)
    # =====================================================================

    def find_similar_players(
        self,
        player_name: str,
        season: str,
        limit: int = 10,
        cross_position: bool = False,
    ) -> pd.DataFrame:
        """Finds players with the most similar playing style via pgvector.

        Uses cosine distance (``<=>`` operator) on the
        ``player_embeddings`` table. By default, filters only players in
        the same positional group.
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
    # Complementarity search (pgvector)
    # =====================================================================

    def find_complementary_players(
        self,
        player_name: str,
        season: str,
        limit: int = 10,
        cross_position: bool = False,
    ) -> pd.DataFrame:
        """Finds players with a complementary profile (most dissimilar).

        Orders by the largest cosine distance, returning players that
        cover characteristics opposite to the reference player.
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
    # Recommendation by synthetic profile (pgvector)
    # =====================================================================

    def recommend_by_profile(
        self,
        embedding: list[float],
        season: str,
        limit: int = 10,
        position_group: str | None = None,
    ) -> pd.DataFrame:
        """Recommends players closest to a synthetic embedding.

        Receives an embedding already projected into PCA space and
        queries nearest neighbors via pgvector.

        Parameters
        ----------
        embedding:
            Embedding vector already transformed via PCA/scaler.
        season:
            Season to filter by.
        limit:
            Maximum number of results.
        position_group:
            Positional group to filter results. If ``None``, searches
            across all players.
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
    # Compatibility with a target squad
    # =====================================================================

    def compute_compatibility(
        self,
        player_name: str,
        season: str,
        target_team: str,
    ) -> list[dict]:
        """Computes compatibility between a player and a target team's squad.

        Uses cosine distance in the embeddings space: smaller distance
        means a more similar style, larger distance means more
        complementary.
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
        """Returns all games: match_id, team, goals, xg, is_home.

        One row per team per match. Used by the predictor to compute
        dynamic parameters.
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
        # Fuzzy match: resolve accents (e.g. 'Sao Paulo' -> 'Sao Paulo')
        team = _fuzzy_team_match(self._session, team)
        """Returns player aggregation over the team's last N games.

        Used by the player-aware pipeline to build the probable XI and
        compute individual xG/90.

        Parameters
        ----------
        team : str
            Team name.
        season : str, optional
            Season. If None, all seasons.
        last_n : int
            How many recent games to consider.

        Returns
        -------
        pd.DataFrame
            Columns: player_id, player_name, matches_played, minutes_total,
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
        """Persists lineups (probable or confirmed) of matches.

        Parameters
        ----------
        lineups : list[dict]
            Each dict: match_key, player_id, team, side, player_name,
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
        """Returns a match's lineup indexed by side.

        Returns
        -------
        dict
            {"home": [...], "away": [...]} with player dicts.
            Empty if no lineup exists.
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
        """List of per-shot xG for the team over the last N games (via action_values)."""
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
        """Returns the date of the most recent match."""
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
        """Persists pre-computed predictions."""
        from datetime import datetime
        from football_moneyball.adapters.orm import MatchPrediction
        now = datetime.now().isoformat()

        def _float(v):
            """Converts numpy float to native Python float."""
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
        """Returns pre-computed predictions ordered by commence_time ASC (upcoming first)."""
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
        """Inserts predictions into the immutable history.

        Does not update existing records - each snapshot is immutable.
        Uses match_key + predicted_at to avoid exact duplicates.
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
        """Inserts value bets into the immutable history."""
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
        """Returns pending predictions (status='pending')."""
        rows = (
            self._session.query(PredictionHistory)
            .filter(PredictionHistory.status == "pending")
            .all()
        )
        columns = [c.key for c in PredictionHistory.__table__.columns]
        return [{col: getattr(r, col) for col in columns} for r in rows]

    def resolve_prediction_in_db(self, pred_id: int, result: dict) -> None:
        """Updates a prediction with the actual result."""
        from datetime import datetime

        row = self._session.get(PredictionHistory, pred_id)
        if not row:
            return
        row.actual_home_goals = result.get("actual_home_goals")
        row.actual_away_goals = result.get("actual_away_goals")
        row.actual_outcome = result.get("actual_outcome")
        # Columns are INTEGER (SQLite compat) - convert bool -> int
        c1x2 = result.get("correct_1x2")
        cou = result.get("correct_over_under")
        row.correct_1x2 = int(c1x2) if c1x2 is not None else None
        row.correct_over_under = int(cou) if cou is not None else None
        row.brier_score = result.get("brier_score")
        row.status = "resolved"
        row.resolved_at = datetime.now().isoformat()
        self._session.commit()

    def resolve_value_bet_in_db(self, bet_id: int, result: dict) -> None:
        """Updates a value bet with the actual result."""
        from datetime import datetime

        row = self._session.get(ValueBetHistory, bet_id)
        if not row:
            return
        won = result.get("won")
        row.won = int(won) if won is not None else None  # bool → int (INTEGER col)
        row.profit = result.get("profit")
        row.resolved_at = datetime.now().isoformat()
        self._session.commit()

    def get_prediction_history(
        self,
        round_num: int | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Returns prediction history with optional filters.

        Dedupe: prediction_history is immutable (1 row per prediction), so
        we may have the same match N times. We return only the MOST RECENT
        per match_key (by predicted_at desc).
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

        # Dedup by team pair (symmetric): keep only the most recent
        # Exclude rehis/backtest entries (round=0 or commence_time contains "rehis")
        seen_matchups: set[frozenset] = set()
        deduped = []
        for r in rows:
            # Filter out rehis/backtest contaminating the track record
            ct = r.commence_time or ""
            if "rehis" in ct or (r.round is not None and r.round == 0):
                continue
            # Symmetric dedup: "A vs B" and "B vs A" = same match
            matchup = frozenset([
                (r.home_team or "").strip().lower(),
                (r.away_team or "").strip().lower(),
            ])
            if matchup in seen_matchups:
                continue
            seen_matchups.add(matchup)
            deduped.append({col: getattr(r, col) for col in columns})

        # Order deduped by commence_time (upcoming first)
        deduped.sort(
            key=lambda x: (x.get("commence_time") or "", x.get("id") or 0),
            reverse=True,
        )
        return deduped

    def get_value_bet_history(self) -> list[dict]:
        """Returns the full value bet history."""
        rows = (
            self._session.query(ValueBetHistory)
            .order_by(ValueBetHistory.id.desc())
            .all()
        )
        columns = [c.key for c in ValueBetHistory.__table__.columns]
        return [{col: getattr(r, col) for col in columns} for r in rows]

    def get_track_record_summary(self) -> dict:
        """Returns aggregated track record statistics."""
        from football_moneyball.domain.track_record import calculate_track_record
        preds = self.get_prediction_history()
        return calculate_track_record(preds)

    # =====================================================================
    # v0.6.0 — Odds persistence
    # =====================================================================

    def save_odds(self, odds_data: list[dict]) -> None:
        """Persists odds into the match_odds table.

        Receives a list of games in the normalized format from the odds_provider:
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
        """Fetches odds from the PG cache if they are recent enough."""
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
        """Fetches odds for a match by team names."""
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
        """Persists match-level stats (corners, cards, fouls, HT score)."""
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
            # v1.7.0
            "home_xg": float(stats.get("home_xg", 0) or 0),
            "away_xg": float(stats.get("away_xg", 0) or 0),
            "home_big_chances": int(stats.get("home_big_chances", 0) or 0),
            "away_big_chances": int(stats.get("away_big_chances", 0) or 0),
            "home_big_chances_scored": int(stats.get("home_big_chances_scored", 0) or 0),
            "away_big_chances_scored": int(stats.get("away_big_chances_scored", 0) or 0),
            "home_touches_box": int(stats.get("home_touches_box", 0) or 0),
            "away_touches_box": int(stats.get("away_touches_box", 0) or 0),
            "home_final_third_entries": int(stats.get("home_final_third_entries", 0) or 0),
            "away_final_third_entries": int(stats.get("away_final_third_entries", 0) or 0),
            "home_long_balls_pct": float(stats.get("home_long_balls_pct", 0) or 0),
            "away_long_balls_pct": float(stats.get("away_long_balls_pct", 0) or 0),
            "home_aerial_won_pct": float(stats.get("home_aerial_won_pct", 0) or 0),
            "away_aerial_won_pct": float(stats.get("away_aerial_won_pct", 0) or 0),
            "home_goals_prevented": float(stats.get("home_goals_prevented", 0) or 0),
            "away_goals_prevented": float(stats.get("away_goals_prevented", 0) or 0),
            "home_passes": int(stats.get("home_passes", 0) or 0),
            "away_passes": int(stats.get("away_passes", 0) or 0),
            "home_pass_accuracy": float(stats.get("home_pass_accuracy", 0) or 0),
            "away_pass_accuracy": float(stats.get("away_pass_accuracy", 0) or 0),
            "home_dispossessed": int(stats.get("home_dispossessed", 0) or 0),
            "away_dispossessed": int(stats.get("away_dispossessed", 0) or 0),
        }
        existing = self._session.get(MatchStats, match_id)
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
        else:
            self._session.add(MatchStats(**data))
        self._session.commit()

    def get_all_match_stats(
        self, competition: str = "Brasileirão Série A",
        seasons: list[str] | None = None,
    ) -> "pd.DataFrame":
        """Returns match_stats from all seasons as a DataFrame."""
        import pandas as pd
        from sqlalchemy import text
        q = text("""
            SELECT ms.* FROM match_stats ms
            JOIN matches m ON m.match_id = ms.match_id
            WHERE m.competition = :comp
        """)
        params: dict = {"comp": competition}
        if seasons:
            q = text("""
                SELECT ms.* FROM match_stats ms
                JOIN matches m ON m.match_id = ms.match_id
                WHERE m.competition = :comp AND m.season = ANY(:seasons)
            """)
            params["seasons"] = seasons
        rows = self._session.execute(q, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r._mapping) for r in rows])

    def get_all_coach_data_for_training(self) -> dict:
        """Returns coach data indexed by (team, match_id) for leak-proof training.

        Returns dict: {(team, match_id) -> {tenure_days, win_rate, changed_30d}}
        """
        from sqlalchemy import text

        # All coaches with start dates
        coaches = self._session.execute(text(
            "SELECT team, coach_name, start_match_date FROM team_coaches ORDER BY start_match_date"
        )).fetchall()

        if not coaches:
            return {}

        # All matches with dates
        matches = self._session.execute(text("""
            SELECT m.match_id, m.match_date, m.home_team, m.away_team,
                   m.home_score, m.away_score
            FROM matches m
            WHERE m.home_score IS NOT NULL
            ORDER BY m.match_date
        """)).fetchall()

        if not matches:
            return {}

        from datetime import datetime

        # Build coach timeline per team
        coach_timeline: dict[str, list] = {}
        for c in coaches:
            coach_timeline.setdefault(c.team, []).append({
                "name": c.coach_name,
                "start": str(c.start_match_date)[:10],
            })

        # For each match, find coach and compute features
        result: dict = {}
        team_wins: dict[str, dict] = {}  # team -> {coach_start -> {wins, total}}

        for m in matches:
            md = str(m.match_date or "")[:10]
            for team, is_home in [(m.home_team, True), (m.away_team, False)]:
                if not team:
                    continue
                # Find active coach for this team at this date
                timeline = coach_timeline.get(team, [])
                active_coach = None
                for c in reversed(timeline):
                    if c["start"] <= md:
                        active_coach = c
                        break
                if not active_coach:
                    continue

                # Track wins under this coach
                key = f"{team}:{active_coach['start']}"
                if key not in team_wins:
                    team_wins[key] = {"wins": 0, "total": 0, "start": active_coach["start"]}
                tw = team_wins[key]

                # Compute features BEFORE updating (leak-proof)
                try:
                    start_d = datetime.fromisoformat(active_coach["start"]).date()
                    match_d = datetime.fromisoformat(md).date()
                    tenure = (match_d - start_d).days
                except Exception:
                    tenure = 100

                wr = tw["wins"] / tw["total"] if tw["total"] > 0 else 0.5

                result[(team, m.match_id)] = {
                    "tenure_days": float(min(tenure, 365)),
                    "win_rate": float(wr),
                    "changed_30d": 1.0 if tenure < 30 else 0.0,
                }

                # NOW update stats (after feature extraction)
                tw["total"] += 1
                won = (is_home and (m.home_score or 0) > (m.away_score or 0)) or \
                      (not is_home and (m.away_score or 0) > (m.home_score or 0))
                if won:
                    tw["wins"] += 1

        return result

    def get_all_standings_for_training(self) -> dict:
        """Returns standings indexed by match_id for training.

        Returns dict: {match_id -> {home_position, away_position, position_gap}}
        """
        from sqlalchemy import text

        # Get matches with their teams
        matches = self._session.execute(text("""
            SELECT m.match_id, m.match_date, m.home_team, m.away_team
            FROM matches m
            WHERE m.home_score IS NOT NULL
            ORDER BY m.match_date
        """)).fetchall()

        if not matches:
            return {}

        # Get all standings snapshots
        standings = self._session.execute(text(
            "SELECT team, position, points, snapshot_date FROM league_standings ORDER BY snapshot_date"
        )).fetchall()

        if not standings:
            return {}

        # Build latest standing per team at each point
        # Simple approach: for each match, find latest standing <= match_date
        team_latest: dict[str, dict] = {}
        stand_list = [{"team": s.team, "position": s.position, "points": s.points,
                       "date": str(s.snapshot_date or "")[:10]} for s in standings]

        result: dict = {}
        si = 0  # standings index

        for m in matches:
            md = str(m.match_date or "")[:10]
            # Advance standings pointer
            while si < len(stand_list) and stand_list[si]["date"] <= md:
                s = stand_list[si]
                team_latest[s["team"]] = {"position": s["position"], "points": s["points"]}
                si += 1

            hp = team_latest.get(m.home_team, {}).get("position", 10)
            ap = team_latest.get(m.away_team, {}).get("position", 10)
            result[m.match_id] = {
                "home_position": float(hp),
                "away_position": float(ap),
                "position_gap": float(abs(hp - ap)),
            }

        return result

    def save_referee_stats(self, referee: dict) -> None:
        """Upserts referee statistics."""
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
        """Returns averages of goals/xg/corners/cards/shots/fouls over the team's last N games.

        Fuzzy match applied to the team name (resolves 'Sao Paulo' -> 'Sao Paulo').

        Also considers what the team CONCEDED (to compute opponent factor).

        Returns
        -------
        dict
            Keys: goals_for, goals_against, xg_for, xg_against,
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
        """Returns league averages: corners/game, cards/game, shots/game, HT goals."""
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
        """Days since the team's last game before the reference date.

        Parameters
        ----------
        team : str
            Team name.
        reference_date : str
            ISO format (YYYY-MM-DD or with timestamp).

        Returns
        -------
        int
            Days since the last game (minimum 1, default 7 if no history).
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
        """Returns advanced team aggregates for feature engineering v1.5.0.

        Extends get_team_stats_aggregates with:
        goal_diff_ema, xg_overperf, xga_overperf,
        creation_index, defensive_intensity, touches_per_match.

        Returns a dict with all keys that build_rich_team_features expects.
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
        """Returns all resolved matches with rich stats for ML training v1.5.0.

        Join of matches + match_stats + aggregated player_match_metrics.

        Returns
        -------
        pd.DataFrame
            Columns: match_id, match_date, home_team, away_team,
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

    def get_team_style_aggregates(
        self, team: str, season: str | None = None, last_n: int = 5,
    ) -> dict:
        """Returns playing-style aggregates (v1.8.0).

        Features: finishing_efficiency, sot_rate, gk_quality, possession_avg,
        long_balls_pct, big_chance_conversion.
        """
        team = _fuzzy_team_match(self._session, team)
        query = text("""
            WITH recent AS (
                SELECT m.match_id, m.home_team, m.away_team,
                       m.home_score, m.away_score,
                       CASE WHEN m.home_team = :team THEN 'home' ELSE 'away' END AS side
                FROM matches m
                WHERE (m.home_team = :team OR m.away_team = :team)
                  AND (:season IS NULL OR m.season = :season)
                  AND m.home_score IS NOT NULL
                ORDER BY m.match_date DESC, m.match_id DESC
                LIMIT :last_n
            )
            SELECT
                -- Goals scored
                AVG(CASE WHEN r.side='home' THEN r.home_score ELSE r.away_score END) AS goals_avg,
                -- Big chances (team perspective)
                AVG(CASE WHEN r.side='home' THEN ms.home_big_chances ELSE ms.away_big_chances END) AS big_chances_avg,
                AVG(CASE WHEN r.side='home' THEN ms.home_big_chances_scored ELSE ms.away_big_chances_scored END) AS bc_scored_avg,
                -- Shots
                AVG(CASE WHEN r.side='home' THEN ms.home_shots ELSE ms.away_shots END) AS shots_avg,
                AVG(CASE WHEN r.side='home' THEN ms.home_sot ELSE ms.away_sot END) AS sot_avg,
                -- Possession
                AVG(CASE WHEN r.side='home' THEN ms.home_possession ELSE ms.away_possession END) AS possession_avg,
                -- Long balls (directness)
                AVG(CASE WHEN r.side='home' THEN ms.home_long_balls_pct ELSE ms.away_long_balls_pct END) AS long_balls_avg,
                -- Goals prevented (GK quality - team's own)
                AVG(CASE WHEN r.side='home' THEN ms.home_goals_prevented ELSE ms.away_goals_prevented END) AS gk_quality_avg,
                -- Touches box
                AVG(CASE WHEN r.side='home' THEN ms.home_touches_box ELSE ms.away_touches_box END) AS touches_box_avg,
                COUNT(*) AS n
            FROM recent r
            LEFT JOIN match_stats ms ON ms.match_id = r.match_id
        """)
        try:
            result = self._session.execute(query, {
                "team": team, "season": season, "last_n": last_n,
            }).fetchone()
        except Exception:
            result = None

        if not result or not result.n:
            return {
                "finishing_efficiency": 0.35,  # typical avg
                "sot_rate": 0.35,
                "gk_quality": 0.0,
                "possession_avg": 50.0,
                "long_balls_pct": 30.0,
                "big_chance_conversion": 0.35,
                "touches_box_avg": 20.0,
            }

        goals = float(result.goals_avg or 0)
        big_chances = float(result.big_chances_avg or 0)
        bc_scored = float(result.bc_scored_avg or 0)
        shots = float(result.shots_avg or 0)
        sot = float(result.sot_avg or 0)

        finishing = (goals / big_chances) if big_chances > 0 else 0.35
        sot_rate = (sot / shots) if shots > 0 else 0.35
        bc_conv = (bc_scored / big_chances) if big_chances > 0 else 0.35

        return {
            "finishing_efficiency": round(min(finishing, 2.0), 3),
            "sot_rate": round(sot_rate, 3),
            "gk_quality": round(float(result.gk_quality_avg or 0), 3),
            "possession_avg": round(float(result.possession_avg or 50), 1),
            "long_balls_pct": round(float(result.long_balls_avg or 30), 1),
            "big_chance_conversion": round(bc_conv, 3),
            "touches_box_avg": round(float(result.touches_box_avg or 20), 1),
        }

    def get_round_for_date(
        self, commence_time: str, season: str | None = None,
    ) -> int | None:
        """Estimates the matchday for a date based on already-ingested games.

        Uses the most recent matchday before commence_time + 1 as the
        next one. If commence_time is before the latest game, uses the
        matchday of that date.

        Parameters
        ----------
        commence_time : str
            ISO format (e.g. '2026-04-12T21:30:00Z').
        season : str, optional

        Returns
        -------
        int | None
            Estimated matchday number.
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

            # If target_date is AFTER the closest match, estimate matchdays ahead
            if date_only > closest_date:
                import math
                from datetime import datetime
                d1 = datetime.fromisoformat(date_only)
                d2 = datetime.fromisoformat(closest_date)
                days_diff = (d1 - d2).days
                # Each matchday ~7 days in Brasileirao - use ceil to avoid underestimating
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
        """Upserts a coach-team relationship."""
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
        """Upserts the list of absent players for a game."""
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
        """Persists a standings snapshot."""
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

    # Context queries (for feature engineering)

    def get_coach_change_info(self, team: str, ref_date: str) -> dict:
        """Returns info about the team's coach at the reference date.

        Returns
        -------
        dict
            coach_name, games_since_change, coach_change_recent (<30d),
            coach_win_rate (for this team-coach relationship).
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

        # Count games coached + wins under this coach up to ref_date
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

        # Days since the coach's start
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
        """Counts absences among top N by xG/90 + lost xG contribution.

        Identifies the team's top N players by historical xG/90.
        Returns how many are absent for the target match/reference date.
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
        """Number of team games in a window around the date."""
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
        """Returns position/points gap between the teams at the reference date."""
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
        """Looks up a referee by name (fuzzy: LIKE)."""
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
    # v1.10.0 — H2H + Referee + Market
    # =====================================================================

    def get_h2h_history(
        self,
        home_team: str,
        away_team: str,
        ref_date: str | None = None,
        last_n: int = 5,
    ) -> list[dict]:
        """Returns the last N matches between the two teams (either venue).

        Parameters
        ----------
        home_team, away_team : str
        ref_date : str | None
            ISO date (YYYY-MM-DD). Only considers matches BEFORE that date.
            None = all matches.
        last_n : int
            Maximum number of matches to return.
        """
        home_team = _fuzzy_team_match(self._session, home_team)
        away_team = _fuzzy_team_match(self._session, away_team)

        query = text("""
            SELECT home_team, away_team, home_score, away_score, match_date
            FROM matches
            WHERE home_score IS NOT NULL
              AND ((home_team = :home AND away_team = :away)
                OR (home_team = :away AND away_team = :home))
              AND (:ref_date IS NULL OR match_date < CAST(:ref_date AS DATE))
            ORDER BY match_date DESC
            LIMIT :limit
        """)
        ref = ref_date[:10] if ref_date else None
        try:
            rows = self._session.execute(query, {
                "home": home_team, "away": away_team,
                "ref_date": ref, "limit": last_n,
            }).fetchall()
        except Exception:
            return []

        return [
            {
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_goals": r.home_score,
                "away_goals": r.away_score,
                "match_date": str(r.match_date),
            }
            for r in rows
        ]

    def get_referee_for_match(self, match_id: int) -> dict | None:
        """Returns stats of the referee assigned to a match (via match_stats.referee_id)."""
        query = text("""
            SELECT rs.referee_id, rs.name, rs.matches, rs.yellow_total,
                   rs.red_total, rs.yellowred_total, rs.cards_per_game
            FROM match_stats ms
            JOIN referee_stats rs ON ms.referee_id = rs.referee_id
            WHERE ms.match_id = :mid
            LIMIT 1
        """)
        try:
            row = self._session.execute(query, {"mid": match_id}).fetchone()
        except Exception:
            return None
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

    def get_market_odds_consensus(
        self,
        home_team: str,
        away_team: str,
        preferred_bookmakers: list[str] | None = None,
    ) -> list[dict] | None:
        """Returns h2h odds from several bookmakers for consensus devig.

        Parameters
        ----------
        home_team, away_team : str
        preferred_bookmakers : list[str] | None
            If provided, filters only those bookmakers. Useful for using
            only Pinnacle/Betfair.

        Returns
        -------
        list[dict] with odds_home, odds_draw, odds_away from each bookmaker. None if empty.
        """
        query_base = """
            SELECT bookmaker, outcome, odds
            FROM match_odds
            WHERE market = 'h2h'
              AND home_team = :home AND away_team = :away
        """
        params = {"home": home_team, "away": away_team}
        if preferred_bookmakers:
            query_base += " AND bookmaker = ANY(:books)"
            params["books"] = preferred_bookmakers

        try:
            rows = self._session.execute(text(query_base), params).fetchall()
        except Exception:
            return None

        if not rows:
            return None

        # Group by bookmaker
        by_book: dict[str, dict[str, float]] = {}
        for r in rows:
            book = r.bookmaker
            if book not in by_book:
                by_book[book] = {}
            if r.outcome == "Home" or r.outcome == home_team:
                by_book[book]["odds_home"] = r.odds
            elif r.outcome == "Away" or r.outcome == away_team:
                by_book[book]["odds_away"] = r.odds
            elif r.outcome == "Draw":
                by_book[book]["odds_draw"] = r.odds

        # Only return bookmakers with all 3 outcomes
        return [
            odds for odds in by_book.values()
            if "odds_home" in odds and "odds_draw" in odds and "odds_away" in odds
        ]

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def close(self) -> None:
        """Closes the SQLAlchemy session."""
        self._session.close()
