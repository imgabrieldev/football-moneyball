"""The Odds API adapter - fetches odds from bookmakers.

Persistence via PostgreSQL (read-through/write-through).
No dependency on local filesystem.

Setup: export ODDS_API_KEY=your_key
Free signup: https://the-odds-api.com
"""

from __future__ import annotations

import os
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_SPORT = "soccer_brazil_campeonato"
DEFAULT_MARKETS = ["h2h", "totals"]
BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_TTL_HOURS = 24


class TheOddsAPIProvider:
    """Odds provider via The Odds API with cache on PostgreSQL."""

    def __init__(
        self,
        api_key: str | None = None,
        sport: str = DEFAULT_SPORT,
        repo: Any = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.sport = sport
        self.repo = repo

    def _has_api_key(self) -> bool:
        """Checks whether an API key is configured."""
        return bool(self.api_key)

    def _get(self, path: str, params: dict | None = None) -> dict | list | None:
        """Performs a GET on the API."""
        params = params or {}
        params["apiKey"] = self.api_key
        r = requests.get(f"{BASE_URL}/{path}", params=params)

        # Log remaining quota
        remaining = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        if remaining:
            logger.info(f"Odds API quota: {remaining} remaining ({used} used)")

        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            raise ValueError("Invalid ODDS_API_KEY.")
        elif r.status_code == 429:
            raise ValueError("Odds API quota exhausted for this month.")
        else:
            logger.warning(f"Odds API error {r.status_code}: {r.text[:200]}")
            return None

    def get_sports(self) -> list[dict]:
        """Lists available sports."""
        data = self._get("sports")
        return data if isinstance(data, list) else []

    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        markets: list[str] | None = None,
    ) -> list[dict]:
        """Fetches odds for a match by team names.

        Returns
        -------
        list[dict]
            Odds in the format: [{"name": "bookmaker", "markets": [...]}]
        """
        all_odds = self.get_upcoming_odds(markets=markets)
        home_lower = home_team.lower()
        away_lower = away_team.lower()

        for game in all_odds:
            game_home = game.get("home_team", "").lower()
            game_away = game.get("away_team", "").lower()
            if (home_lower in game_home or game_home in home_lower or
                away_lower in game_away or game_away in away_lower):
                return game.get("bookmakers", [])
        return []

    def get_upcoming_odds(
        self,
        sport: str | None = None,
        markets: list[str] | None = None,
    ) -> list[dict]:
        """Fetches odds for ALL upcoming matches.

        If a repo exists, tries the PostgreSQL cache first (TTL 24h).
        Otherwise, fetches directly from the API.

        Returns
        -------
        list[dict]
            List of matches with normalized odds.
        """
        sport = sport or self.sport
        markets = markets or DEFAULT_MARKETS

        # Read-through: check PG cache
        if self.repo is not None:
            cached = self.repo.get_cached_odds(max_age_hours=CACHE_TTL_HOURS)
            if cached is not None:
                logger.info(f"Odds cache hit (PG): {len(cached)} games")
                return cached

        # Cache miss - fetch from API
        if not self._has_api_key():
            logger.warning("No ODDS_API_KEY and no cache.")
            return []

        data = self._get(f"sports/{sport}/odds", {
            "regions": "eu,uk",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        })

        if not data or not isinstance(data, list):
            return []

        results = self._normalize_odds(data)

        # Write-through: save to PG
        if self.repo is not None and results:
            self.repo.save_odds(results)
            logger.info(f"Odds saved to PG: {len(results)} games")

        return results

    def _normalize_odds(self, data: list[dict]) -> list[dict]:
        """Normalizes the API response to the internal format."""
        results = []
        for game in data:
            normalized = {
                "id": game.get("id"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "commence_time": game.get("commence_time"),
                "bookmakers": [],
            }

            for bm in game.get("bookmakers", []):
                bm_data = {
                    "name": bm.get("key", bm.get("title", "")),
                    "markets": [],
                }
                for market in bm.get("markets", []):
                    market_key = market.get("key", "")
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price", 0)
                        bm_data["markets"].append({
                            "market": market_key,
                            "outcome": outcome.get("name", ""),
                            "point": outcome.get("point", 0.0),
                            "odds": price,
                            "implied_prob": round(1.0 / price, 4) if price > 0 else 0,
                        })
                normalized["bookmakers"].append(bm_data)

            results.append(normalized)

        return results

    def get_historical_odds(
        self,
        sport: str | None = None,
        date: str = "",
        markets: list[str] | None = None,
    ) -> list[dict]:
        """Fetches historical odds for a date (ISO 8601).

        Parameters
        ----------
        date : str
            Date in ISO 8601 format (e.g. '2026-03-18T12:00:00Z').

        Returns
        -------
        list[dict]
            Odds for that date.
        """
        sport = sport or self.sport
        markets = markets or DEFAULT_MARKETS

        data = self._get(f"historical/sports/{sport}/odds", {
            "regions": "eu,uk",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "date": date,
        })

        if not data:
            return []
        return data.get("data", []) if isinstance(data, dict) else data
