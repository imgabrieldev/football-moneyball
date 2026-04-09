"""Centralized configuration and Dependency Injection.

Single entry point to create providers, repositories and visualizers
based on configuration or parameters.
"""

from __future__ import annotations

import os


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://moneyball:moneyball@localhost:5432/moneyball",
)

DEFAULT_PROVIDER = os.getenv("MONEYBALL_PROVIDER", "statsbomb")


def get_provider(name: str | None = None):
    """Return the configured DataProvider.

    Parameters
    ----------
    name : str, optional
        Provider name ('statsbomb' or 'sofascore').
        If None, uses DEFAULT_PROVIDER.

    Returns
    -------
    DataProvider
        Instance of the selected provider.
    """
    name = name or DEFAULT_PROVIDER

    if name == "statsbomb":
        from football_moneyball.adapters.statsbomb_provider import StatsBombProvider
        return StatsBombProvider()
    elif name == "sofascore":
        from football_moneyball.adapters.sofascore_provider import SofascoreProvider
        return SofascoreProvider()
    else:
        raise ValueError(f"Unknown provider: {name}. Use 'statsbomb' or 'sofascore'.")


def get_repository():
    """Return the configured MatchRepository (PostgreSQL).

    Returns
    -------
    PostgresRepository
        Instance connected to the database.
    """
    from football_moneyball.adapters.orm import get_session
    from football_moneyball.adapters.postgres_repository import PostgresRepository
    return PostgresRepository(get_session())


def get_odds_provider():
    """Return the configured OddsProvider (The Odds API).

    Requires ODDS_API_KEY to be set.

    Returns
    -------
    TheOddsAPIProvider
        Instance of the odds provider.
    """
    from football_moneyball.adapters.odds_provider import TheOddsAPIProvider
    return TheOddsAPIProvider()


def get_visualizer():
    """Return the configured Visualizer (Matplotlib).

    Returns
    -------
    MatplotlibVisualizer
        Instance of the visualizer.
    """
    from football_moneyball.adapters.matplotlib_viz import MatplotlibVisualizer
    return MatplotlibVisualizer()
