"""Adapters for Football Moneyball (hexagonal architecture).

Exposes the data providers and the PostgreSQL repository.
"""

from football_moneyball.adapters.orm import (
    Base,
    Match,
    PlayerMatchMetrics,
    PassNetwork,
    PlayerEmbedding,
    Stint,
    ActionValue,
    PressingMetrics,
    get_engine,
    get_session,
    init_db,
)
from football_moneyball.adapters.postgres_repository import PostgresRepository
from football_moneyball.adapters.statsbomb_provider import StatsBombProvider
from football_moneyball.adapters.sofascore_provider import SofascoreProvider

__all__ = [
    # ORM
    "Base",
    "Match",
    "PlayerMatchMetrics",
    "PassNetwork",
    "PlayerEmbedding",
    "Stint",
    "ActionValue",
    "PressingMetrics",
    "get_engine",
    "get_session",
    "init_db",
    # Repository
    "PostgresRepository",
    # Providers
    "StatsBombProvider",
    "SofascoreProvider",
]
