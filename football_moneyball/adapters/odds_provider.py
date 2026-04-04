"""Adapter The Odds API — busca odds de casas de apostas.

Puxa odds de ~30 bookmakers (Bet365, Betano, Pinnacle, etc.)
incluindo odds da Betfair Exchange, tudo numa unica chamada.

Configurar: export ODDS_API_KEY=sua_chave
Cadastro gratis: https://the-odds-api.com
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


class TheOddsAPIProvider:
    """Provedor de odds via The Odds API.

    Agrega odds de ~30 casas de apostas incluindo Bet365, Betano,
    Pinnacle e Betfair Exchange. 500 requests/mes gratis.

    Parameters
    ----------
    api_key : str, optional
        API key. Default: env ODDS_API_KEY.
    sport : str, optional
        Sport key. Default: soccer_brazil_campeonato.
    """

    def __init__(
        self,
        api_key: str | None = None,
        sport: str = DEFAULT_SPORT,
    ) -> None:
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.sport = sport

        if not self.api_key:
            raise ValueError(
                "ODDS_API_KEY nao configurada. "
                "Cadastre em https://the-odds-api.com e setar: "
                "export ODDS_API_KEY=sua_chave"
            )

    def _get(self, path: str, params: dict | None = None) -> dict | list | None:
        """Faz GET na API."""
        params = params or {}
        params["apiKey"] = self.api_key
        r = requests.get(f"{BASE_URL}/{path}", params=params)

        # Log remaining quota
        remaining = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        if remaining:
            logger.info(f"Odds API quota: {remaining} restantes ({used} usadas)")

        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            raise ValueError("ODDS_API_KEY invalida.")
        elif r.status_code == 429:
            raise ValueError("Quota da Odds API esgotada neste mes.")
        else:
            logger.warning(f"Odds API erro {r.status_code}: {r.text[:200]}")
            return None

    def get_sports(self) -> list[dict]:
        """Lista esportes disponiveis."""
        data = self._get("sports")
        return data if isinstance(data, list) else []

    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        markets: list[str] | None = None,
    ) -> list[dict]:
        """Busca odds de uma partida por nomes dos times.

        Returns
        -------
        list[dict]
            Odds no formato: [{"name": "bookmaker", "markets": [...]}]
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
        """Busca odds de TODAS as proximas partidas numa unica chamada.

        1 request = todas as partidas com odds de todos os bookmakers.

        Returns
        -------
        list[dict]
            Lista de partidas com odds normalizadas.
        """
        sport = sport or self.sport
        markets = markets or DEFAULT_MARKETS

        data = self._get(f"sports/{sport}/odds", {
            "regions": "eu,uk",
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
        """Busca odds historicas de uma data (ISO 8601).

        Parameters
        ----------
        date : str
            Data no formato ISO 8601 (ex: '2026-03-18T12:00:00Z').

        Returns
        -------
        list[dict]
            Odds daquela data.
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
