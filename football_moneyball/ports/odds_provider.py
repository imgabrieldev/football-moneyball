"""Port para provedores de odds de apostas."""

from __future__ import annotations
from typing import Protocol


class OddsProvider(Protocol):
    """Interface para fontes de odds de casas de apostas."""

    def get_match_odds(self, home_team: str, away_team: str, markets: list[str] | None = None) -> list[dict]:
        """Retorna odds de uma partida por time names."""
        ...

    def get_upcoming_odds(self, sport: str = "soccer_brazil_campeonato", markets: list[str] | None = None) -> list[dict]:
        """Retorna odds de proximas partidas."""
        ...

    def get_historical_odds(self, sport: str, date: str, markets: list[str] | None = None) -> list[dict]:
        """Retorna odds historicas de uma data especifica."""
        ...
