"""Modelos ORM e gerenciamento de sessao do Football Moneyball.

Define todos os modelos SQLAlchemy mapeados para o schema PostgreSQL + pgvector,
alem de funcoes para criacao de engine, sessao e inicializacao do banco.
"""

from __future__ import annotations

import os
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Integer,
    String,
    Float,
    Date,
    Text,
    create_engine,
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
    """Metricas individuais de cada jogador por partida."""

    __tablename__ = "player_match_metrics"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_name: Mapped[Optional[str]] = mapped_column(String)
    team: Mapped[Optional[str]] = mapped_column(String)

    # Metricas numericas
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
    """Periodos continuos de jogo com a mesma formacao de jogadores em campo."""

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
    """Valores de acao (xT e VAEP) por evento de uma partida."""

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
    """Metricas de pressing por time por partida."""

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
    """Retorna uma nova sessao vinculada ao engine padrao."""
    engine = get_engine()
    return sessionmaker(bind=engine)()


def init_db(engine) -> None:
    """Cria todas as tabelas definidas nos modelos ORM."""
    Base.metadata.create_all(engine)
