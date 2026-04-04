"""Adapter Betfair Exchange — busca odds via API oficial.

Usa betfairlightweight para acessar a Betfair Exchange API.
Odds de exchange sao o benchmark mais eficiente do mercado —
refletem probabilidade real sem margem de bookmaker.

Configuracao via variaveis de ambiente:
    BETFAIR_USERNAME — usuario Betfair
    BETFAIR_PASSWORD — senha
    BETFAIR_APP_KEY — app key (gerar em developer.betfair.com)
    BETFAIR_CERTS_PATH — caminho para certificados SSL (opcional)
"""

from __future__ import annotations

import os
import logging
from typing import Any

import betfairlightweight
from betfairlightweight import filters

logger = logging.getLogger(__name__)

# Betfair event type IDs
SOCCER_EVENT_TYPE_ID = "1"

# Competition IDs conhecidos (Betfair)
BRASILEIRAO_COMPETITION_ID = 13196  # Brasileirão Série A

# Market types
MATCH_ODDS = "MATCH_ODDS"           # 1X2
OVER_UNDER_25 = "OVER_UNDER_25"     # Over/Under 2.5
BOTH_TEAMS_TO_SCORE = "BOTH_TEAMS_TO_SCORE"  # BTTS


class BetfairProvider:
    """Provedor de odds via Betfair Exchange API.

    Requer conta Betfair com app key. Gere em developer.betfair.com.

    Parameters
    ----------
    username : str, optional
        Usuario Betfair. Default: env BETFAIR_USERNAME.
    password : str, optional
        Senha. Default: env BETFAIR_PASSWORD.
    app_key : str, optional
        App key. Default: env BETFAIR_APP_KEY.
    certs_path : str, optional
        Caminho para certificados SSL. Default: env BETFAIR_CERTS_PATH.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        app_key: str | None = None,
        certs_path: str | None = None,
    ) -> None:
        self.username = username or os.getenv("BETFAIR_USERNAME", "")
        self.password = password or os.getenv("BETFAIR_PASSWORD", "")
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY", "")
        self.certs_path = certs_path or os.getenv("BETFAIR_CERTS_PATH", "")
        self._client: betfairlightweight.APIClient | None = None
        self._logged_in = False

    def _ensure_login(self) -> betfairlightweight.APIClient:
        """Garante que o client esta autenticado."""
        if self._client is not None and self._logged_in:
            return self._client

        if not self.username or not self.app_key:
            raise ValueError(
                "Credenciais Betfair nao configuradas. "
                "Setar BETFAIR_USERNAME, BETFAIR_PASSWORD e BETFAIR_APP_KEY."
            )

        self._client = betfairlightweight.APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key,
            certs=self.certs_path if self.certs_path else None,
        )

        try:
            if self.certs_path:
                self._client.login()
            else:
                self._client.login_interactive()
            self._logged_in = True
            logger.info("Login na Betfair realizado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao fazer login na Betfair: {e}")
            raise

        return self._client

    def get_competitions(self, event_type_id: str = SOCCER_EVENT_TYPE_ID) -> list[dict]:
        """Lista competicoes de futebol disponiveis na Betfair.

        Returns
        -------
        list[dict]
            Lista com competition_id, name, market_count.
        """
        client = self._ensure_login()
        competition_filter = filters.market_filter(
            event_type_ids=[event_type_id]
        )
        competitions = client.betting.list_competitions(filter=competition_filter)

        return [
            {
                "competition_id": comp.competition.id,
                "name": comp.competition.name,
                "market_count": comp.market_count,
                "region": comp.competition_region,
            }
            for comp in competitions
        ]

    def get_upcoming_events(
        self,
        competition_id: int = BRASILEIRAO_COMPETITION_ID,
    ) -> list[dict]:
        """Lista proximos eventos (partidas) de uma competicao.

        Returns
        -------
        list[dict]
            Lista com event_id, name, open_date, home_team, away_team.
        """
        client = self._ensure_login()
        event_filter = filters.market_filter(
            event_type_ids=[SOCCER_EVENT_TYPE_ID],
            competition_ids=[str(competition_id)],
        )
        events = client.betting.list_events(filter=event_filter)

        results = []
        for ev in events:
            name = ev.event.name or ""
            teams = name.split(" v ") if " v " in name else name.split(" vs ")
            home = teams[0].strip() if len(teams) >= 2 else name
            away = teams[1].strip() if len(teams) >= 2 else ""

            results.append({
                "event_id": ev.event.id,
                "name": name,
                "home_team": home,
                "away_team": away,
                "open_date": str(ev.event.open_date) if ev.event.open_date else "",
                "market_count": ev.market_count,
            })

        return results

    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        markets: list[str] | None = None,
    ) -> list[dict]:
        """Busca odds de uma partida por nomes dos times.

        Procura o evento na Betfair, lista os mercados e retorna
        odds no formato normalizado.

        Returns
        -------
        list[dict]
            Odds no formato: [{"name": "betfair_exchange", "markets": [...]}]
        """
        client = self._ensure_login()
        market_types = markets or [MATCH_ODDS, OVER_UNDER_25, BOTH_TEAMS_TO_SCORE]

        # Buscar evento
        search_term = f"{home_team} {away_team}"
        event_filter = filters.market_filter(
            event_type_ids=[SOCCER_EVENT_TYPE_ID],
            text_query=search_term,
        )
        events = client.betting.list_events(filter=event_filter)

        if not events:
            logger.warning(f"Nenhum evento encontrado para '{search_term}'")
            return []

        event_id = events[0].event.id

        # Buscar mercados do evento
        market_filter = filters.market_filter(
            event_ids=[event_id],
            market_type_codes=market_types,
        )
        catalogues = client.betting.list_market_catalogue(
            filter=market_filter,
            max_results=10,
            market_projection=["RUNNER_DESCRIPTION"],
        )

        if not catalogues:
            return []

        # Buscar preços
        market_ids = [cat.market_id for cat in catalogues]
        price_filter = filters.price_projection(
            price_data=["EX_BEST_OFFERS"]
        )
        books = client.betting.list_market_book(
            market_ids=market_ids,
            price_projection=price_filter,
        )

        # Mapear runner IDs para nomes
        runner_names: dict[str, dict[int, str]] = {}
        market_types_map: dict[str, str] = {}
        for cat in catalogues:
            runner_names[cat.market_id] = {
                runner.selection_id: runner.runner_name
                for runner in (cat.runners or [])
            }
            market_types_map[cat.market_id] = cat.market_name or ""

        # Normalizar para nosso formato
        all_markets = []
        for book in books:
            mid = book.market_id
            names = runner_names.get(mid, {})
            market_name = market_types_map.get(mid, "")

            # Mapear market name → nosso market key
            if "Match Odds" in market_name:
                market_key = "h2h"
            elif "Over/Under" in market_name:
                market_key = "totals"
            elif "Both" in market_name:
                market_key = "btts"
            else:
                market_key = market_name.lower().replace(" ", "_")

            for runner in (book.runners or []):
                if not runner.ex or not runner.ex.available_to_back:
                    continue

                best_back = runner.ex.available_to_back[0]
                runner_name = names.get(runner.selection_id, str(runner.selection_id))

                # Normalizar outcome name
                outcome = runner_name
                if market_key == "h2h":
                    if runner_name == home_team or "Home" in runner_name:
                        outcome = "Home"
                    elif runner_name == away_team or "Away" in runner_name:
                        outcome = "Away"
                    elif runner_name == "The Draw" or "Draw" in runner_name:
                        outcome = "Draw"

                odds = best_back.price
                all_markets.append({
                    "market": market_key,
                    "outcome": outcome,
                    "odds": odds,
                    "point": 2.5 if "Over/Under" in market_name else 0.0,
                    "implied_prob": round(1.0 / odds, 4) if odds > 0 else 0,
                    "size": best_back.size,  # liquidez disponivel
                })

        return [{"name": "betfair_exchange", "markets": all_markets}]

    def get_upcoming_odds(
        self,
        sport: str | None = None,
        markets: list[str] | None = None,
        competition_id: int = BRASILEIRAO_COMPETITION_ID,
    ) -> list[dict]:
        """Busca odds de todas as proximas partidas de uma competicao.

        Returns
        -------
        list[dict]
            Lista de partidas com odds no formato normalizado.
        """
        events = self.get_upcoming_events(competition_id)
        results = []

        for ev in events:
            try:
                odds = self.get_match_odds(
                    ev["home_team"], ev["away_team"], markets
                )
                results.append({
                    "id": ev["event_id"],
                    "home_team": ev["home_team"],
                    "away_team": ev["away_team"],
                    "commence_time": ev["open_date"],
                    "bookmakers": odds,
                })
            except Exception as e:
                logger.warning(f"Erro ao buscar odds de {ev['name']}: {e}")
                continue

        return results

    def get_historical_odds(
        self,
        sport: str | None = None,
        date: str = "",
        markets: list[str] | None = None,
    ) -> list[dict]:
        """Betfair nao oferece odds historicas via API padrao.

        Para historico, usar Betfair Historical Data
        (https://historicdata.betfair.com/).
        """
        logger.warning(
            "Betfair Exchange API nao suporta odds historicas diretamente. "
            "Use historicdata.betfair.com para dados historicos."
        )
        return []

    def close(self) -> None:
        """Encerra sessao com a Betfair."""
        if self._client and self._logged_in:
            try:
                self._client.logout()
                self._logged_in = False
                logger.info("Logout da Betfair realizado.")
            except Exception:
                pass
