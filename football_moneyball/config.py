"""Configuracao centralizada e Dependency Injection.

Ponto unico para criar providers, repositories e visualizers
com base em configuracao ou parametros.
"""

from __future__ import annotations

import os


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://moneyball:moneyball@localhost:5432/moneyball",
)

DEFAULT_PROVIDER = os.getenv("MONEYBALL_PROVIDER", "statsbomb")


def get_provider(name: str | None = None):
    """Retorna o DataProvider configurado.

    Parameters
    ----------
    name : str, optional
        Nome do provider ('statsbomb' ou 'sofascore').
        Se None, usa DEFAULT_PROVIDER.

    Returns
    -------
    DataProvider
        Instancia do provider selecionado.
    """
    name = name or DEFAULT_PROVIDER

    if name == "statsbomb":
        from football_moneyball.adapters.statsbomb_provider import StatsBombProvider
        return StatsBombProvider()
    elif name == "sofascore":
        from football_moneyball.adapters.sofascore_provider import SofascoreProvider
        return SofascoreProvider()
    else:
        raise ValueError(f"Provider desconhecido: {name}. Use 'statsbomb' ou 'sofascore'.")


def get_repository():
    """Retorna o MatchRepository configurado (PostgreSQL).

    Returns
    -------
    PostgresRepository
        Instancia conectada ao banco de dados.
    """
    from football_moneyball.adapters.orm import get_session
    from football_moneyball.adapters.postgres_repository import PostgresRepository
    return PostgresRepository(get_session())


def get_odds_provider():
    """Retorna o OddsProvider configurado (Betfair Exchange).

    Requer variaveis de ambiente BETFAIR_USERNAME, BETFAIR_PASSWORD,
    BETFAIR_APP_KEY configuradas.

    Returns
    -------
    BetfairProvider
        Instancia do provider de odds.
    """
    from football_moneyball.adapters.odds_provider import BetfairProvider
    return BetfairProvider()


def get_visualizer():
    """Retorna o Visualizer configurado (Matplotlib).

    Returns
    -------
    MatplotlibVisualizer
        Instancia do visualizer.
    """
    from football_moneyball.adapters.matplotlib_viz import MatplotlibVisualizer
    return MatplotlibVisualizer()
