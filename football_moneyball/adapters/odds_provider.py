"""Adapter The Odds API — busca odds de casas de apostas."""

from __future__ import annotations

import os
import time
from typing import Any

import requests


DEFAULT_SPORT = "soccer_brazil_campeonato"
DEFAULT_MARKETS = ["h2h", "totals", "btts"]
BASE_URL = "https://api.the-odds-api.com/v4"


class TheOddsAPIProvider:
    """Provedor de odds via The Odds API (the-odds-api.com)."""

    def __init__(self, api_key: str | None = None, sport: str = DEFAULT_SPORT):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.sport = sport
        self._request_delay = 1.0

    def _get(self, path: str, params: dict | None = None) -> dict | list | None:
        """Faz GET com rate limiting."""
        time.sleep(self._request_delay)
        params = params or {}
        params["apiKey"] = self.api_key
        r = requests.get(f"{BASE_URL}/{path}", params=params)
        if r.status_code == 200:
            return r.json()
        return None

    def get_match_odds(
        self, home_team: str, away_team: str, markets: list[str] | None = None
    ) -> list[dict]:
        """Busca odds de uma partida especifica por nomes dos times."""
        all_odds = self.get_upcoming_odds(markets=markets)
        # Filter by team names (fuzzy match)
        for game in all_odds:
            if (home_team.lower() in game.get("home_team", "").lower() or
                away_team.lower() in game.get("away_team", "").lower()):
                return game.get("bookmakers", [])
        return []

    def get_upcoming_odds(
        self, sport: str | None = None, markets: list[str] | None = None
    ) -> list[dict]:
        """Busca odds de proximas partidas."""
        sport = sport or self.sport
        markets = markets or DEFAULT_MARKETS
        data = self._get(f"sports/{sport}/odds", {
            "regions": "eu",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        })
        if not data or not isinstance(data, list):
            return []

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
                bm_data = {"name": bm.get("key", bm.get("title", ""))}
                for market in bm.get("markets", []):
                    market_key = market.get("key", "")
                    for outcome in market.get("outcomes", []):
                        bm_data.setdefault("markets", []).append({
                            "market": market_key,
                            "outcome": outcome.get("name", ""),
                            "point": outcome.get("point", 0.0),
                            "odds": outcome.get("price", 0.0),
                            "implied_prob": round(1.0 / outcome["price"], 4) if outcome.get("price", 0) > 0 else 0,
                        })
                normalized["bookmakers"].append(bm_data)
            results.append(normalized)
        return results

    def get_historical_odds(
        self, sport: str | None = None, date: str = "", markets: list[str] | None = None
    ) -> list[dict]:
        """Busca odds historicas de uma data (ISO 8601)."""
        sport = sport or self.sport
        markets = markets or DEFAULT_MARKETS
        data = self._get(f"historical/sports/{sport}/odds", {
            "regions": "eu",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "date": date,
        })
        if not data:
            return []
        return data.get("data", []) if isinstance(data, dict) else data
