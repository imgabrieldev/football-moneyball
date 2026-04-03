"""Módulo de banco de dados para o Football Moneyball.

Gerencia conexão, modelos ORM e operações CRUD usando SQLAlchemy e pgvector.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Integer,
    String,
    Float,
    Date,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    Session,
    sessionmaker,
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://moneyball:moneyball@localhost:5432/moneyball",
)


# ---------------------------------------------------------------------------
# Base declarativa
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Modelos ORM
# ---------------------------------------------------------------------------

class Match(Base):
    """Partidas de futebol."""

    __tablename__ = "matches"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition: Mapped[Optional[str]] = mapped_column(String)
    season: Mapped[Optional[str]] = mapped_column(String)
    match_date: Mapped[Optional[str]] = mapped_column(Date)
    home_team: Mapped[Optional[str]] = mapped_column(String)
    away_team: Mapped[Optional[str]] = mapped_column(String)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)


class PlayerMatchMetrics(Base):
    """Métricas individuais de cada jogador por partida."""

    __tablename__ = "player_match_metrics"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_name: Mapped[Optional[str]] = mapped_column(String)
    team: Mapped[Optional[str]] = mapped_column(String)

    # Métricas numéricas
    minutes_played: Mapped[Optional[float]] = mapped_column(Float)
    goals: Mapped[Optional[float]] = mapped_column(Float)
    assists: Mapped[Optional[float]] = mapped_column(Float)
    shots: Mapped[Optional[float]] = mapped_column(Float)
    shots_on_target: Mapped[Optional[float]] = mapped_column(Float)
    xg: Mapped[Optional[float]] = mapped_column(Float)
    xa: Mapped[Optional[float]] = mapped_column(Float)
    passes: Mapped[Optional[float]] = mapped_column(Float)
    passes_completed: Mapped[Optional[float]] = mapped_column(Float)
    pass_pct: Mapped[Optional[float]] = mapped_column(Float)
    progressive_passes: Mapped[Optional[float]] = mapped_column(Float)
    progressive_carries: Mapped[Optional[float]] = mapped_column(Float)
    key_passes: Mapped[Optional[float]] = mapped_column(Float)
    through_balls: Mapped[Optional[float]] = mapped_column(Float)
    crosses: Mapped[Optional[float]] = mapped_column(Float)
    tackles: Mapped[Optional[float]] = mapped_column(Float)
    interceptions: Mapped[Optional[float]] = mapped_column(Float)
    blocks: Mapped[Optional[float]] = mapped_column(Float)
    clearances: Mapped[Optional[float]] = mapped_column(Float)
    aerials_won: Mapped[Optional[float]] = mapped_column(Float)
    aerials_lost: Mapped[Optional[float]] = mapped_column(Float)
    fouls_committed: Mapped[Optional[float]] = mapped_column(Float)
    fouls_won: Mapped[Optional[float]] = mapped_column(Float)
    dribbles_attempted: Mapped[Optional[float]] = mapped_column(Float)
    dribbles_completed: Mapped[Optional[float]] = mapped_column(Float)
    touches: Mapped[Optional[float]] = mapped_column(Float)
    carries: Mapped[Optional[float]] = mapped_column(Float)
    dispossessed: Mapped[Optional[float]] = mapped_column(Float)
    pressures: Mapped[Optional[float]] = mapped_column(Float)
    pressure_regains: Mapped[Optional[float]] = mapped_column(Float)

    # v0.2.0 — expanded metrics
    progressive_receptions: Mapped[Optional[float]] = mapped_column(Float)
    big_chances: Mapped[Optional[float]] = mapped_column(Float)
    big_chances_missed: Mapped[Optional[float]] = mapped_column(Float)
    passes_short: Mapped[Optional[float]] = mapped_column(Float)
    passes_short_completed: Mapped[Optional[float]] = mapped_column(Float)
    passes_medium: Mapped[Optional[float]] = mapped_column(Float)
    passes_medium_completed: Mapped[Optional[float]] = mapped_column(Float)
    passes_long: Mapped[Optional[float]] = mapped_column(Float)
    passes_long_completed: Mapped[Optional[float]] = mapped_column(Float)
    passes_under_pressure: Mapped[Optional[float]] = mapped_column(Float)
    passes_under_pressure_completed: Mapped[Optional[float]] = mapped_column(Float)
    switches_of_play: Mapped[Optional[float]] = mapped_column(Float)
    ground_duels_won: Mapped[Optional[float]] = mapped_column(Float)
    ground_duels_total: Mapped[Optional[float]] = mapped_column(Float)
    tackle_success_rate: Mapped[Optional[float]] = mapped_column(Float)
    xt_generated: Mapped[Optional[float]] = mapped_column(Float)
    vaep_generated: Mapped[Optional[float]] = mapped_column(Float)
    pressing_success_rate_individual: Mapped[Optional[float]] = mapped_column(Float)


class PassNetwork(Base):
    """Arestas da rede de passes entre jogadores em uma partida."""

    __tablename__ = "pass_networks"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    passer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receiver_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    passer_name: Mapped[Optional[str]] = mapped_column(String)
    receiver_name: Mapped[Optional[str]] = mapped_column(String)
    weight: Mapped[Optional[int]] = mapped_column(Integer)
    features = mapped_column(JSONB)


class PlayerEmbedding(Base):
    """Embeddings vetoriais de jogadores por temporada, gerados via clustering."""

    __tablename__ = "player_embeddings"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[str] = mapped_column(String, primary_key=True)
    player_name: Mapped[Optional[str]] = mapped_column(String)
    team: Mapped[Optional[str]] = mapped_column(String)
    competition: Mapped[Optional[str]] = mapped_column(String)
    embedding = mapped_column(Vector(16))
    cluster_label: Mapped[Optional[int]] = mapped_column(Integer)
    archetype: Mapped[Optional[str]] = mapped_column(String)
    position_group: Mapped[Optional[str]] = mapped_column(String)


class Stint(Base):
    """Períodos contínuos de jogo com a mesma formação de jogadores em campo."""

    __tablename__ = "stints"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stint_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    home_player_ids = mapped_column(ARRAY(Integer))
    away_player_ids = mapped_column(ARRAY(Integer))
    duration_minutes: Mapped[Optional[float]] = mapped_column(Float)
    home_xg: Mapped[Optional[float]] = mapped_column(Float)
    away_xg: Mapped[Optional[float]] = mapped_column(Float)
    xg_diff: Mapped[Optional[float]] = mapped_column(Float)
    boundary_type: Mapped[Optional[str]] = mapped_column(String)


class ActionValue(Base):
    """Valores de ação (xT e VAEP) por evento de uma partida."""

    __tablename__ = "action_values"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[Optional[int]] = mapped_column(Integer)
    player_name: Mapped[Optional[str]] = mapped_column(String)
    team: Mapped[Optional[str]] = mapped_column(String)
    action_type: Mapped[Optional[str]] = mapped_column(String)
    start_x: Mapped[Optional[float]] = mapped_column(Float)
    start_y: Mapped[Optional[float]] = mapped_column(Float)
    end_x: Mapped[Optional[float]] = mapped_column(Float)
    end_y: Mapped[Optional[float]] = mapped_column(Float)
    xt_value: Mapped[Optional[float]] = mapped_column(Float)
    vaep_value: Mapped[Optional[float]] = mapped_column(Float)
    vaep_offensive: Mapped[Optional[float]] = mapped_column(Float)
    vaep_defensive: Mapped[Optional[float]] = mapped_column(Float)


class PressingMetrics(Base):
    """Métricas de pressing por time por partida."""

    __tablename__ = "pressing_metrics"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team: Mapped[str] = mapped_column(String, primary_key=True)
    ppda: Mapped[Optional[float]] = mapped_column(Float)
    pressing_success_rate: Mapped[Optional[float]] = mapped_column(Float)
    counter_pressing_fraction: Mapped[Optional[float]] = mapped_column(Float)
    high_turnovers: Mapped[Optional[int]] = mapped_column(Integer)
    shot_ending_high_turnovers: Mapped[Optional[int]] = mapped_column(Integer)
    pressing_zone_1: Mapped[Optional[float]] = mapped_column(Float)
    pressing_zone_2: Mapped[Optional[float]] = mapped_column(Float)
    pressing_zone_3: Mapped[Optional[float]] = mapped_column(Float)
    pressing_zone_4: Mapped[Optional[float]] = mapped_column(Float)
    pressing_zone_5: Mapped[Optional[float]] = mapped_column(Float)
    pressing_zone_6: Mapped[Optional[float]] = mapped_column(Float)


# ---------------------------------------------------------------------------
# Engine / Session
# ---------------------------------------------------------------------------

def get_engine():
    """Cria e retorna o engine SQLAlchemy a partir da DATABASE_URL."""
    return create_engine(DATABASE_URL)


def get_session() -> Session:
    """Retorna uma nova sessão vinculada ao engine padrão."""
    engine = get_engine()
    return sessionmaker(bind=engine)()


def init_db(engine) -> None:
    """Cria todas as tabelas definidas nos modelos ORM."""
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Helpers de consulta e upsert
# ---------------------------------------------------------------------------

def match_exists(session: Session, match_id: int) -> bool:
    """Verifica se uma partida já está cadastrada no banco."""
    return session.query(Match).filter_by(match_id=match_id).first() is not None


def upsert_match(session: Session, match_data: dict) -> None:
    """Insere ou atualiza os dados de uma partida.

    Recebe um dicionário com as chaves correspondentes às colunas da tabela matches.
    """
    existing = session.get(Match, match_data["match_id"])
    if existing:
        for key, value in match_data.items():
            setattr(existing, key, value)
    else:
        session.add(Match(**match_data))
    session.commit()


def upsert_player_metrics(session: Session, metrics_df: pd.DataFrame, match_id: int) -> None:
    """Insere ou atualiza métricas de jogadores para uma partida.

    O DataFrame deve conter uma coluna 'player_id' e colunas compatíveis
    com os campos de PlayerMatchMetrics.
    """
    metrics_df = metrics_df.copy()
    metrics_df["match_id"] = match_id

    columns = {c.key for c in PlayerMatchMetrics.__table__.columns}

    for _, row in metrics_df.iterrows():
        data = {k: v for k, v in row.to_dict().items() if k in columns}
        existing = session.get(PlayerMatchMetrics, (data["match_id"], data["player_id"]))
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            session.add(PlayerMatchMetrics(**data))

    session.commit()


def upsert_pass_network(session: Session, edges_df: pd.DataFrame, match_id: int) -> None:
    """Insere ou atualiza as arestas da rede de passes de uma partida.

    O DataFrame deve conter as colunas 'passer_id', 'receiver_id', 'weight'
    e opcionalmente 'features'.
    """
    edges_df = edges_df.copy()
    edges_df["match_id"] = match_id

    columns = {c.key for c in PassNetwork.__table__.columns}

    for _, row in edges_df.iterrows():
        data = {k: v for k, v in row.to_dict().items() if k in columns}
        pk = (data["match_id"], data["passer_id"], data["receiver_id"])
        existing = session.get(PassNetwork, pk)
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            session.add(PassNetwork(**data))

    session.commit()


def upsert_embeddings(session: Session, embeddings_df: pd.DataFrame) -> None:
    """Insere ou atualiza embeddings vetoriais de jogadores.

    O DataFrame deve conter 'player_id', 'season', 'embedding' (lista de floats),
    e opcionalmente 'player_name', 'cluster_label' e 'archetype'.
    """
    columns = {c.key for c in PlayerEmbedding.__table__.columns}

    for _, row in embeddings_df.iterrows():
        data = {k: v for k, v in row.to_dict().items() if k in columns}
        existing = session.get(PlayerEmbedding, (data["player_id"], data["season"]))
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            session.add(PlayerEmbedding(**data))

    session.commit()


def upsert_stints(session: Session, stints_df: pd.DataFrame, match_id: int) -> None:
    """Insere ou atualiza os stints (períodos de jogo) de uma partida.

    O DataFrame deve conter 'stint_number' e colunas compatíveis com o modelo Stint.
    """
    stints_df = stints_df.copy()
    stints_df["match_id"] = match_id

    columns = {c.key for c in Stint.__table__.columns}

    for _, row in stints_df.iterrows():
        data = {k: v for k, v in row.to_dict().items() if k in columns}
        existing = session.get(Stint, (data["match_id"], data["stint_number"]))
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            session.add(Stint(**data))

    session.commit()


def upsert_action_values(session: Session, values_df: pd.DataFrame, match_id: int) -> None:
    """Insere ou atualiza valores de ação (xT/VAEP) de uma partida."""
    values_df = values_df.copy()
    values_df["match_id"] = match_id
    columns = {c.key for c in ActionValue.__table__.columns}
    for _, row in values_df.iterrows():
        data = {k: v for k, v in row.to_dict().items() if k in columns}
        existing = session.get(ActionValue, (data["match_id"], data["event_index"]))
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            session.add(ActionValue(**data))
    session.commit()


def upsert_pressing_metrics(session: Session, metrics_df: pd.DataFrame, match_id: int) -> None:
    """Insere ou atualiza métricas de pressing de uma partida."""
    metrics_df = metrics_df.copy()
    metrics_df["match_id"] = match_id
    columns = {c.key for c in PressingMetrics.__table__.columns}
    for _, row in metrics_df.iterrows():
        data = {k: v for k, v in row.to_dict().items() if k in columns}
        existing = session.get(PressingMetrics, (data["match_id"], data["team"]))
        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
        else:
            session.add(PressingMetrics(**data))
    session.commit()


def find_similar_players(
    session: Session,
    player_name: str,
    season: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Busca jogadores com embeddings similares usando distância cosseno do pgvector.

    Retorna um DataFrame com os jogadores mais similares, ordenados por distância.
    """
    query = text("""
        SELECT
            pe2.player_id,
            pe2.player_name,
            pe2.season,
            pe2.cluster_label,
            pe2.archetype,
            pe2.embedding <=> pe1.embedding AS distance
        FROM player_embeddings pe1
        JOIN player_embeddings pe2
            ON pe2.player_id != pe1.player_id
        WHERE pe1.player_name = :player_name
          AND pe1.season = :season
        ORDER BY distance
        LIMIT :limit
    """)

    result = session.execute(query, {
        "player_name": player_name,
        "season": season,
        "limit": limit,
    })

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["player_id", "player_name", "season",
                                       "cluster_label", "archetype", "distance"])


def get_player_metrics(
    session: Session,
    player_name: str,
    season: str = None,
) -> pd.DataFrame:
    """Retorna as métricas de um jogador, opcionalmente filtradas por temporada.

    Faz join com a tabela matches para obter a temporada quando o filtro é aplicado.
    """
    query = session.query(PlayerMatchMetrics).filter(
        PlayerMatchMetrics.player_name == player_name
    )

    if season is not None:
        query = (
            query
            .join(Match, Match.match_id == PlayerMatchMetrics.match_id)
            .filter(Match.season == season)
        )

    rows = query.all()
    if not rows:
        return pd.DataFrame()

    columns = [c.key for c in PlayerMatchMetrics.__table__.columns]
    data = [{col: getattr(r, col) for col in columns} for r in rows]
    return pd.DataFrame(data)
