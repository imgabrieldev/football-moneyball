"""Adapter Sofascore — provedor de dados via Sofascore API.

Encapsula todas as chamadas a API do Sofascore, convertendo os dados
para o formato padrao do Football Moneyball (compativel com StatsBomb-like
schema).
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Constantes da API
# ---------------------------------------------------------------------------

SOFASCORE_BASE = "https://www.sofascore.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

# Mapeamento de posicoes Sofascore → grupos posicionais
POSITION_MAP = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

# Rate limit: 1 requisicao por segundo
REQUEST_DELAY = 1.0


class SofascoreProvider:
    """Provedor de dados usando Sofascore API.

    Converte dados do Sofascore para o formato padronizado do Football
    Moneyball, mantendo compatibilidade com o schema StatsBomb-like.

    Parameters
    ----------
    tournament_id : int
        ID do torneio no Sofascore (ex.: 325 para Brasileirao Serie A).
    season_id : int
        ID da temporada no Sofascore (ex.: 87678 para 2026).
    competition_name : str
        Nome da competicao para usar nos registros persistidos.
    season_name : str
        Nome da temporada para usar nos registros persistidos.
    request_delay : float
        Intervalo entre requisicoes em segundos (rate limiting).
    """

    # Valores padrao para Brasileirao 2026
    TOURNAMENT_ID = 325
    SEASON_ID = 87678

    def __init__(
        self,
        tournament_id: int = 325,
        season_id: int = 87678,
        competition_name: str = "Brasileirao Serie A",
        season_name: str = "2026",
        request_delay: float = REQUEST_DELAY,
    ) -> None:
        self.tournament_id = tournament_id
        self.season_id = season_id
        self.competition_name = competition_name
        self.season_name = season_name
        self.request_delay = request_delay

    # -----------------------------------------------------------------
    # Metodo HTTP privado
    # -----------------------------------------------------------------

    def _api_get(self, path: str) -> dict | None:
        """Faz GET na API do Sofascore com rate limiting."""
        time.sleep(self.request_delay)
        r = requests.get(f"{SOFASCORE_BASE}/{path}", headers=HEADERS)
        if r.status_code == 200:
            return r.json()
        return None

    # -----------------------------------------------------------------
    # Interface publica (compativel com DataProvider)
    # -----------------------------------------------------------------

    def get_match_events(self, match_id: int) -> pd.DataFrame:
        """Busca lineups + shotmap e retorna DataFrame normalizado.

        Combina dados de lineups (estatisticas individuais) e shotmap
        (finalizacoes) em um DataFrame com colunas compativeis com o
        schema de PlayerMatchMetrics.
        """
        lineups_data = self._api_get(f"event/{match_id}/lineups")
        shotmap_data = self._api_get(f"event/{match_id}/shotmap")

        if not lineups_data:
            warnings.warn(
                f"Sem lineups para match_id={match_id} no Sofascore."
            )
            return pd.DataFrame()

        shotmap = (shotmap_data or {}).get("shotmap", [])
        return self._convert_player_stats(lineups_data, shotmap, match_id)

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        """Retorna os lineups de uma partida, indexados por lado (home/away).

        Converte o formato Sofascore para um dict de DataFrames com colunas
        ``player_id``, ``player_name``, ``position``.
        """
        lineups_data = self._api_get(f"event/{match_id}/lineups")
        if not lineups_data:
            return {}

        result: dict[str, pd.DataFrame] = {}
        for side in ["home", "away"]:
            side_data = lineups_data.get(side, {})
            players = side_data.get("players", [])
            rows = []
            for p in players:
                player_info = p.get("player", {})
                rows.append({
                    "player_id": player_info.get("id"),
                    "player_name": player_info.get("name", ""),
                    "position": player_info.get("position", ""),
                })
            result[side] = pd.DataFrame(rows)

        return result

    def get_competitions(self) -> pd.DataFrame:
        """Retorna a lista de torneios disponiveis (formato simplificado).

        Retorna um DataFrame com o torneio configurado nesta instancia.
        A Sofascore API nao expoe uma lista publica de torneios da mesma
        forma que o StatsBomb, entao retornamos o torneio configurado.
        """
        return pd.DataFrame([{
            "competition_id": self.tournament_id,
            "competition_name": self.competition_name,
            "season_id": self.season_id,
            "season_name": self.season_name,
        }])

    def get_matches(
        self,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> pd.DataFrame:
        """Busca todos os jogos finalizados do torneio/temporada.

        Parameters
        ----------
        competition_id : int | None
            ID do torneio. Se None, usa o tournament_id da instancia.
        season_id : int | None
            ID da temporada. Se None, usa o season_id da instancia.
        """
        tid = competition_id or self.tournament_id
        sid = season_id or self.season_id

        all_matches: list[dict] = []
        page = 0
        while True:
            data = self._api_get(
                f"unique-tournament/{tid}/season/{sid}/events/last/{page}"
            )
            if not data:
                break
            events = data.get("events", [])
            if not events:
                break
            # Filtrar apenas jogos finalizados
            finished = [
                e for e in events
                if e.get("status", {}).get("type") == "finished"
            ]
            all_matches.extend(finished)
            page += 1
            if len(events) < 20:
                break

        if not all_matches:
            return pd.DataFrame()

        rows = [self._convert_match_info(m) for m in all_matches]
        return pd.DataFrame(rows)

    def get_match_info(self, match_id: int) -> dict[str, Any]:
        """Retorna metadados de uma partida especifica.

        Busca os dados do evento no Sofascore e converte para o formato
        padrao do Football Moneyball.
        """
        data = self._api_get(f"event/{match_id}")
        if not data or "event" not in data:
            warnings.warn(
                f"Partida {match_id} nao encontrada no Sofascore."
            )
            return {"match_id": match_id}

        match = data["event"]
        return self._convert_match_info(match)

    # -----------------------------------------------------------------
    # Conversoes privadas
    # -----------------------------------------------------------------

    def _convert_match_info(self, match: dict) -> dict[str, Any]:
        """Converte info de partida Sofascore para o schema interno."""
        ts = match.get("startTimestamp", 0)
        match_date = (
            datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else None
        )
        return {
            "match_id": match["id"],
            "competition": self.competition_name,
            "season": self.season_name,
            "match_date": match_date,
            "home_team": match.get("homeTeam", {}).get("name", ""),
            "away_team": match.get("awayTeam", {}).get("name", ""),
            "home_score": match.get("homeScore", {}).get("current", 0),
            "away_score": match.get("awayScore", {}).get("current", 0),
        }

    def _convert_player_stats(
        self,
        lineups: dict,
        shotmap: list[dict],
        match_id: int,
    ) -> pd.DataFrame:
        """Converte lineups + shotmap do Sofascore para DataFrame de metricas.

        Produz um DataFrame com colunas compativeis com o schema de
        PlayerMatchMetrics.
        """
        rows: list[dict] = []

        # Indexar shotmap por player_id para lookup de xG/gols
        player_shots: dict[int, list[dict]] = {}
        for shot in shotmap:
            pid = shot.get("player", {}).get("id")
            if pid:
                player_shots.setdefault(pid, []).append(shot)

        for side in ["home", "away"]:
            side_data = lineups.get(side, {})
            players = side_data.get("players", [])

            for p in players:
                player_info = p.get("player", {})
                pid = player_info.get("id")
                pname = player_info.get("name", "")
                stats = p.get("statistics", {})

                if not stats.get("minutesPlayed"):
                    continue

                # Shots do shotmap
                p_shots = player_shots.get(pid, [])
                goals = sum(
                    1 for s in p_shots if s.get("shotType") == "goal"
                )
                shots_on_target = sum(
                    1 for s in p_shots
                    if s.get("shotType") in ("goal", "save")
                )
                xg_total = sum(s.get("xg", 0) or 0 for s in p_shots)
                big_chances = sum(
                    1 for s in p_shots if (s.get("xg", 0) or 0) >= 0.3
                )
                big_chances_missed = sum(
                    1 for s in p_shots
                    if (s.get("xg", 0) or 0) >= 0.3
                    and s.get("shotType") != "goal"
                )

                row = {
                    "player_id": pid,
                    "player_name": pname,
                    "team": side,
                    "minutes_played": stats.get("minutesPlayed", 0),
                    "goals": goals,
                    "assists": stats.get("goalAssist", 0),
                    "shots": stats.get("totalShots", len(p_shots)),
                    "shots_on_target": shots_on_target,
                    "xg": round(stats.get("expectedGoals", xg_total), 4),
                    "xa": round(stats.get("expectedAssists", 0) or 0, 4),
                    "passes": stats.get("totalPass", 0),
                    "passes_completed": stats.get("accuratePass", 0),
                    "pass_pct": round(
                        stats.get("accuratePass", 0)
                        / max(stats.get("totalPass", 1), 1)
                        * 100,
                        1,
                    ),
                    "progressive_passes": 0,
                    "progressive_carries": stats.get(
                        "progressiveBallCarriesCount", 0
                    ),
                    "key_passes": stats.get("keyPass", 0),
                    "through_balls": 0,
                    "crosses": stats.get("totalCross", 0),
                    "tackles": stats.get("totalTackle", 0),
                    "interceptions": stats.get("interceptionWon", 0),
                    "blocks": 0,
                    "clearances": stats.get("totalClearance", 0),
                    "aerials_won": stats.get("aerialWon", 0),
                    "aerials_lost": stats.get("aerialLost", 0),
                    "fouls_committed": stats.get("fouls", 0),
                    "fouls_won": stats.get("wasFouled", 0),
                    "dribbles_attempted": (
                        stats.get("duelWon", 0) + stats.get("duelLost", 0)
                    ),
                    "dribbles_completed": stats.get("duelWon", 0),
                    "touches": stats.get("touches", 0),
                    "carries": stats.get("ballCarriesCount", 0),
                    "dispossessed": (
                        stats.get("dispossessed", 0)
                        + stats.get("possessionLostCtrl", 0)
                    ),
                    "pressures": 0,
                    "pressure_regains": stats.get("ballRecovery", 0),
                    # v0.2.0 metrics
                    "progressive_receptions": 0,
                    "big_chances": (
                        big_chances + stats.get("bigChanceCreated", 0)
                    ),
                    "big_chances_missed": big_chances_missed,
                    "passes_short": 0,
                    "passes_short_completed": 0,
                    "passes_medium": 0,
                    "passes_medium_completed": 0,
                    "passes_long": stats.get("totalLongBalls", 0),
                    "passes_long_completed": stats.get(
                        "accurateLongBalls", 0
                    ),
                    "passes_under_pressure": 0,
                    "passes_under_pressure_completed": 0,
                    "switches_of_play": 0,
                    "ground_duels_won": stats.get("wonTackle", 0),
                    "ground_duels_total": stats.get("totalTackle", 0),
                    "tackle_success_rate": round(
                        stats.get("wonTackle", 0)
                        / max(stats.get("totalTackle", 1), 1)
                        * 100,
                        1,
                    ),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def convert_shotmap_to_actions(
        self,
        shotmap: list[dict],
        match_id: int,
    ) -> pd.DataFrame:
        """Converte shotmap do Sofascore para DataFrame de ActionValue.

        Normaliza coordenadas e estrutura para o formato compativel com
        a tabela action_values.
        """
        rows: list[dict] = []
        for i, shot in enumerate(shotmap):
            coords = shot.get("playerCoordinates", {})
            gm_coords = shot.get("goalMouthCoordinates", {})
            rows.append({
                "event_index": i,
                "player_id": shot.get("player", {}).get("id"),
                "player_name": shot.get("player", {}).get("name"),
                "team": "home" if shot.get("isHome") else "away",
                "action_type": "Shot",
                "start_x": coords.get("x"),
                "start_y": coords.get("y"),
                "end_x": gm_coords.get("x"),
                "end_y": gm_coords.get("y"),
                "xt_value": None,
                "vaep_value": None,
                "vaep_offensive": shot.get("xg"),
                "vaep_defensive": None,
            })
        return pd.DataFrame(rows)
