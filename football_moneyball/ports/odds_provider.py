"""Port for betting odds providers."""

from __future__ import annotations
from typing import Protocol


class OddsProvider(Protocol):
    """Interface for bookmaker odds sources."""

    def get_match_odds(self, home_team: str, away_team: str, markets: list[str] | None = None) -> list[dict]:
        """Return odds for a match by team names."""
        ...

    def get_upcoming_odds(self, sport: str = "soccer_brazil_campeonato", markets: list[str] | None = None) -> list[dict]:
        """Return odds for upcoming matches."""
        ...

    def get_historical_odds(self, sport: str, date: str, markets: list[str] | None = None) -> list[dict]:
        """Return historical odds for a specific date."""
        ...
