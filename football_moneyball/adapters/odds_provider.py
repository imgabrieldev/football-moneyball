"""Adapter The Odds API — busca odds de casas de apostas.

Puxa odds de ~30 bookmakers (Bet365, Betano, Pinnacle, etc.)
tudo numa unica chamada. Implementa cache local read-through/
write-through para evitar gastar creditos da API em desenvolvimento.

Cache: data/odds_cache.json (write-through no fetch, read-through no get)
Configurar: export ODDS_API_KEY=sua_chave
Cadastro gratis: https://the-odds-api.com
"""

from __future__ import annotations

import json
import os
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_SPORT = "soccer_brazil_campeonato"
DEFAULT_MARKETS = ["h2h", "totals"]
BASE_URL = "https://api.the-odds-api.com/v4"


CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_TTL_HOURS = 24  # cache valido por 24 horas


class TheOddsAPIProvider:
    """Provedor de odds via The Odds API com cache local.

    Implementa read-through/write-through cache em data/odds_cache.json.
    Se o cache existir e nao estiver expirado, usa o cache.
    Caso contrario, busca da API e atualiza o cache.

    Parameters
    ----------
    api_key : str, optional
        API key. Default: env ODDS_API_KEY.
    sport : str, optional
        Sport key. Default: soccer_brazil_campeonato.
    cache_dir : Path, optional
        Diretorio do cache. Default: data/
    force_refresh : bool
        Ignorar cache e buscar da API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        sport: str = DEFAULT_SPORT,
        cache_dir: Path | None = None,
        force_refresh: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.sport = sport
        self.cache_dir = cache_dir or CACHE_DIR
        self.force_refresh = force_refresh
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        """Retorna o caminho do arquivo de cache."""
        return self.cache_dir / f"odds_{key}.json"

    def _read_cache(self, key: str) -> list[dict] | None:
        """Le do cache se existir e nao estiver expirado."""
        if self.force_refresh:
            return None

        path = self._cache_path(key)
        if not path.exists():
            return None

        # Verificar TTL
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > CACHE_TTL_HOURS:
            logger.info(f"Cache expirado ({age_hours:.1f}h > {CACHE_TTL_HOURS}h)")
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Cache hit: {path.name} ({len(data)} items, {age_hours:.1f}h)")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache corrompido: {e}")
            return None

    def _write_cache(self, key: str, data: list[dict]) -> None:
        """Escreve no cache (write-through)."""
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Cache escrito: {path.name} ({len(data)} items)")
        except OSError as e:
            logger.warning(f"Erro ao escrever cache: {e}")

    def _has_api_key(self) -> bool:
        """Verifica se tem API key configurada."""
        return bool(self.api_key)

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
        """Busca odds de TODAS as proximas partidas.

        Read-through: le do cache se valido, senao busca da API e cacheia.
        1 request = todas as partidas com odds de todos os bookmakers.

        Returns
        -------
        list[dict]
            Lista de partidas com odds normalizadas.
        """
        sport = sport or self.sport
        markets = markets or DEFAULT_MARKETS
        cache_key = f"upcoming_{sport}"

        # Read-through: tentar cache primeiro
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        # Cache miss — buscar da API
        if not self._has_api_key():
            logger.warning("Sem ODDS_API_KEY e sem cache. Retornando vazio.")
            return []

        data = self._get(f"sports/{sport}/odds", {
            "regions": "eu,uk",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        })

        if not data or not isinstance(data, list):
            return []

        results = self._normalize_odds(data)

        # Write-through: salvar no cache
        self._write_cache(cache_key, results)

        return results

    def _normalize_odds(self, data: list[dict]) -> list[dict]:
        """Normaliza response da API para formato interno."""
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
