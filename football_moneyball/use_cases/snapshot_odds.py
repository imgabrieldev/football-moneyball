"""Use case: snapshot of odds in the PostgreSQL."""

from __future__ import annotations
from typing import Any
import logging

logger = logging.getLogger(__name__)


class SnapshotOdds:
    """Busca odds atuais and persiste in the PostgreSQL.

    Parameters
    ----------
    odds_provider : TheOddsAPIProvider
    repo : MatchRepository
    """

    def __init__(self, odds_provider, repo) -> None:
        self.odds_provider = odds_provider
        self.repo = repo

    def execute(self) -> dict[str, Any]:
        """Busca odds of the API and salva in the banco.

        Forca refresh (ignora cache) for garantir data frescos.
        """
        # Force fetch from API (bypass cache)
        odds = self.odds_provider.get_upcoming_odds()

        if not odds:
            return {"error": "Nenhuma odd retornada by the API.", "matches": 0}

        # Persist
        self.repo.save_odds(odds)

        total_markets = sum(
            len(m)
            for game in odds
            for bm in game.get("bookmakers", [])
            for m in [bm.get("markets", [])]
        )

        return {
            "matches": len(odds),
            "bookmakers": len(set(
                bm.get("name", "")
                for game in odds
                for bm in game.get("bookmakers", [])
            )),
            "total_odds": total_markets,
        }
