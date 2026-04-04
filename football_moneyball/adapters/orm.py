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


class MatchOdds(Base):
    """Odds de casas de apostas por partida."""

    __tablename__ = "match_odds"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker: Mapped[str] = mapped_column(String, primary_key=True)
    market: Mapped[str] = mapped_column(String, primary_key=True)
    outcome: Mapped[str] = mapped_column(String, primary_key=True)
    point: Mapped[float] = mapped_column(Float, primary_key=True, default=0.0)
    odds: Mapped[Optional[float]] = mapped_column(Float)
    implied_prob: Mapped[Optional[float]] = mapped_column(Float)
    fetched_at: Mapped[Optional[str]] = mapped_column(String)
    commence_time: Mapped[Optional[str]] = mapped_column(String)


class MatchPrediction(Base):
    """Previsoes do modelo por partida."""

    __tablename__ = "match_predictions"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    home_team: Mapped[Optional[str]] = mapped_column(String)
    away_team: Mapped[Optional[str]] = mapped_column(String)
    home_xg_expected: Mapped[Optional[float]] = mapped_column(Float)
    away_xg_expected: Mapped[Optional[float]] = mapped_column(Float)
    home_win_prob: Mapped[Optional[float]] = mapped_column(Float)
    draw_prob: Mapped[Optional[float]] = mapped_column(Float)
    away_win_prob: Mapped[Optional[float]] = mapped_column(Float)
    over_25_prob: Mapped[Optional[float]] = mapped_column(Float)
    btts_prob: Mapped[Optional[float]] = mapped_column(Float)
    most_likely_score: Mapped[Optional[str]] = mapped_column(String)
    simulations: Mapped[Optional[int]] = mapped_column(Integer)
    predicted_at: Mapped[Optional[str]] = mapped_column(String)
    commence_time: Mapped[Optional[str]] = mapped_column(String)
    # v1.1.0 — player-aware
    lineup_type: Mapped[Optional[str]] = mapped_column(String)
    model_version: Mapped[Optional[str]] = mapped_column(String)
    # v1.2.0 — multi-output markets (corners, cards, shots, HT)
    multi_markets = mapped_column(JSONB)
    # v1.4.0 — player props (marcador, assistencia, chutes individuais)
    player_props = mapped_column(JSONB)
    # v1.3.0 — ML flag
    ml_used: Mapped[Optional[bool]] = mapped_column(Integer)  # 0/1 as int


class ValueBet(Base):
    """Value bets identificadas pelo modelo."""

    __tablename__ = "value_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[Optional[int]] = mapped_column(Integer)
    market: Mapped[Optional[str]] = mapped_column(String)
    outcome: Mapped[Optional[str]] = mapped_column(String)
    model_prob: Mapped[Optional[float]] = mapped_column(Float)
    best_odds: Mapped[Optional[float]] = mapped_column(Float)
    bookmaker: Mapped[Optional[str]] = mapped_column(String)
    implied_prob: Mapped[Optional[float]] = mapped_column(Float)
    edge: Mapped[Optional[float]] = mapped_column(Float)
    kelly_fraction: Mapped[Optional[float]] = mapped_column(Float)
    recommended_stake: Mapped[Optional[float]] = mapped_column(Float)
    actual_result: Mapped[Optional[str]] = mapped_column(String)
    profit: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[Optional[str]] = mapped_column(String)


class BacktestResult(Base):
    """Resultados de backtesting."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[Optional[str]] = mapped_column(String)
    matches_analyzed: Mapped[Optional[int]] = mapped_column(Integer)
    bets_placed: Mapped[Optional[int]] = mapped_column(Integer)
    total_staked: Mapped[Optional[float]] = mapped_column(Float)
    total_return: Mapped[Optional[float]] = mapped_column(Float)
    roi: Mapped[Optional[float]] = mapped_column(Float)
    hit_rate: Mapped[Optional[float]] = mapped_column(Float)
    brier_score: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    config = mapped_column(JSONB)


class PredictionHistory(Base):
    """Historico imutavel de previsoes."""

    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_key: Mapped[Optional[int]] = mapped_column(Integer)
    home_team: Mapped[Optional[str]] = mapped_column(String)
    away_team: Mapped[Optional[str]] = mapped_column(String)
    commence_time: Mapped[Optional[str]] = mapped_column(String)
    round: Mapped[Optional[int]] = mapped_column(Integer)
    home_win_prob: Mapped[Optional[float]] = mapped_column(Float)
    draw_prob: Mapped[Optional[float]] = mapped_column(Float)
    away_win_prob: Mapped[Optional[float]] = mapped_column(Float)
    over_25_prob: Mapped[Optional[float]] = mapped_column(Float)
    btts_prob: Mapped[Optional[float]] = mapped_column(Float)
    home_xg_expected: Mapped[Optional[float]] = mapped_column(Float)
    away_xg_expected: Mapped[Optional[float]] = mapped_column(Float)
    most_likely_score: Mapped[Optional[str]] = mapped_column(String)
    predicted_at: Mapped[Optional[str]] = mapped_column(String)
    actual_home_goals: Mapped[Optional[int]] = mapped_column(Integer)
    actual_away_goals: Mapped[Optional[int]] = mapped_column(Integer)
    actual_outcome: Mapped[Optional[str]] = mapped_column(String)
    resolved_at: Mapped[Optional[str]] = mapped_column(String)
    status: Mapped[Optional[str]] = mapped_column(String, default="pending")
    correct_1x2: Mapped[Optional[bool]] = mapped_column(Integer)  # SQLite compat
    correct_over_under: Mapped[Optional[bool]] = mapped_column(Integer)
    brier_score: Mapped[Optional[float]] = mapped_column(Float)
    # v1.1.0 — player-aware
    lineup_type: Mapped[Optional[str]] = mapped_column(String)
    model_version: Mapped[Optional[str]] = mapped_column(String)


class MatchStats(Base):
    """Estatisticas match-level: corners, cards, fouls, shots, HT score."""

    __tablename__ = "match_stats"

    match_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    home_corners: Mapped[Optional[int]] = mapped_column(Integer)
    away_corners: Mapped[Optional[int]] = mapped_column(Integer)
    home_yellow: Mapped[Optional[int]] = mapped_column(Integer)
    away_yellow: Mapped[Optional[int]] = mapped_column(Integer)
    home_red: Mapped[Optional[int]] = mapped_column(Integer)
    away_red: Mapped[Optional[int]] = mapped_column(Integer)
    home_fouls: Mapped[Optional[int]] = mapped_column(Integer)
    away_fouls: Mapped[Optional[int]] = mapped_column(Integer)
    home_shots: Mapped[Optional[int]] = mapped_column(Integer)
    away_shots: Mapped[Optional[int]] = mapped_column(Integer)
    home_sot: Mapped[Optional[int]] = mapped_column(Integer)
    away_sot: Mapped[Optional[int]] = mapped_column(Integer)
    home_saves: Mapped[Optional[int]] = mapped_column(Integer)
    away_saves: Mapped[Optional[int]] = mapped_column(Integer)
    home_possession: Mapped[Optional[float]] = mapped_column(Float)
    away_possession: Mapped[Optional[float]] = mapped_column(Float)
    ht_home_score: Mapped[Optional[int]] = mapped_column(Integer)
    ht_away_score: Mapped[Optional[int]] = mapped_column(Integer)
    referee_id: Mapped[Optional[int]] = mapped_column(Integer)
    referee_name: Mapped[Optional[str]] = mapped_column(String)


class RefereeStats(Base):
    """Estatisticas de arbitros (totais de carreira via Sofascore)."""

    __tablename__ = "referee_stats"

    referee_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String)
    matches: Mapped[Optional[int]] = mapped_column(Integer)
    yellow_total: Mapped[Optional[int]] = mapped_column(Integer)
    red_total: Mapped[Optional[int]] = mapped_column(Integer)
    yellowred_total: Mapped[Optional[int]] = mapped_column(Integer)
    cards_per_game: Mapped[Optional[float]] = mapped_column(Float)
    last_updated: Mapped[Optional[str]] = mapped_column(String)


class MatchLineup(Base):
    """Escalacao (provavel ou confirmada) de uma partida."""

    __tablename__ = "match_lineups"

    match_key: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team: Mapped[Optional[str]] = mapped_column(String)
    side: Mapped[Optional[str]] = mapped_column(String)  # 'home' | 'away'
    player_name: Mapped[Optional[str]] = mapped_column(String)
    position: Mapped[Optional[str]] = mapped_column(String)
    is_starter: Mapped[Optional[bool]] = mapped_column(Integer)  # SQLite compat
    jersey_number: Mapped[Optional[int]] = mapped_column(Integer)
    source: Mapped[Optional[str]] = mapped_column(String)  # 'probable' | 'confirmed'
    fetched_at: Mapped[Optional[str]] = mapped_column(String)


class ValueBetHistory(Base):
    """Historico imutavel de value bets."""

    __tablename__ = "value_bet_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[Optional[int]] = mapped_column(Integer)
    match_key: Mapped[Optional[int]] = mapped_column(Integer)
    home_team: Mapped[Optional[str]] = mapped_column(String)
    away_team: Mapped[Optional[str]] = mapped_column(String)
    market: Mapped[Optional[str]] = mapped_column(String)
    outcome: Mapped[Optional[str]] = mapped_column(String)
    model_prob: Mapped[Optional[float]] = mapped_column(Float)
    best_odds: Mapped[Optional[float]] = mapped_column(Float)
    bookmaker: Mapped[Optional[str]] = mapped_column(String)
    edge: Mapped[Optional[float]] = mapped_column(Float)
    kelly_stake: Mapped[Optional[float]] = mapped_column(Float)
    won: Mapped[Optional[bool]] = mapped_column(Integer)
    profit: Mapped[Optional[float]] = mapped_column(Float)
    resolved_at: Mapped[Optional[str]] = mapped_column(String)


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
    """Cria todas as tabelas definidas nos modelos ORM e aplica migracoes."""
    Base.metadata.create_all(engine)
    apply_migrations(engine)


# ---------------------------------------------------------------------------
# Migracoes idempotentes (ALTER TABLE em tabelas existentes)
# ---------------------------------------------------------------------------

def apply_migrations(engine) -> None:
    """Aplica ALTER TABLEs idempotentes em tabelas existentes.

    Use quando adicionar colunas novas em tabelas ja criadas. PostgreSQL
    16 suporta ``ADD COLUMN IF NOT EXISTS`` nativamente.
    """
    migrations = [
        # v1.1.0 — player-aware predictions
        "ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS lineup_type VARCHAR",
        "ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS model_version VARCHAR",
        "ALTER TABLE match_predictions ADD COLUMN IF NOT EXISTS lineup_type VARCHAR",
        "ALTER TABLE match_predictions ADD COLUMN IF NOT EXISTS model_version VARCHAR",
        # v1.2.0
        "ALTER TABLE match_predictions ADD COLUMN IF NOT EXISTS multi_markets JSONB",
        # v1.4.0
        "ALTER TABLE match_predictions ADD COLUMN IF NOT EXISTS player_props JSONB",
        "ALTER TABLE match_predictions ADD COLUMN IF NOT EXISTS ml_used INTEGER",
    ]

    with engine.connect() as conn:
        for sql in migrations:
            try:
                from sqlalchemy import text
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                # Ignora erros (tabela ainda nao existe, etc)
                conn.rollback()
