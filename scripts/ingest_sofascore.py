"""Script para ingerir dados do Brasileirão via Sofascore API.

Busca todos os jogos finalizados, extrai métricas de jogadores e shotmap,
converte para o schema do Football Moneyball e persiste no PostgreSQL.
"""

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import requests

from football_moneyball.db import (
    get_session,
    upsert_match,
    upsert_player_metrics,
    upsert_pressing_metrics,
    upsert_action_values,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOFASCORE_BASE = "https://www.sofascore.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

TOURNAMENT_ID = 325   # Brasileirão Série A
SEASON_ID = 87678     # 2026
COMPETITION_NAME = "Brasileirão Série A"
SEASON_NAME = "2026"

POSITION_MAP = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

# Rate limit: 1 request per second
REQUEST_DELAY = 1.0


def _api_get(path: str) -> dict | None:
    """Faz GET na API do Sofascore com rate limiting."""
    time.sleep(REQUEST_DELAY)
    r = requests.get(f"{SOFASCORE_BASE}/{path}", headers=HEADERS)
    if r.status_code == 200:
        return r.json()
    return None


# ---------------------------------------------------------------------------
# Fetch matches
# ---------------------------------------------------------------------------

def fetch_all_matches() -> list[dict]:
    """Busca todos os jogos finalizados do Brasileirão 2026."""
    all_matches = []
    page = 0
    while True:
        data = _api_get(
            f"unique-tournament/{TOURNAMENT_ID}/season/{SEASON_ID}/events/last/{page}"
        )
        if not data:
            break
        events = data.get("events", [])
        if not events:
            break
        # Filtrar apenas jogos finalizados
        finished = [e for e in events if e.get("status", {}).get("type") == "finished"]
        all_matches.extend(finished)
        page += 1
        if len(events) < 20:
            break
    return all_matches


# ---------------------------------------------------------------------------
# Convert match data
# ---------------------------------------------------------------------------

def convert_match_info(match: dict) -> dict:
    """Converte info de partida Sofascore → nosso schema."""
    from datetime import datetime
    ts = match.get("startTimestamp", 0)
    match_date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else None
    return {
        "match_id": match["id"],
        "competition": COMPETITION_NAME,
        "season": SEASON_NAME,
        "match_date": match_date,
        "home_team": match["homeTeam"]["name"],
        "away_team": match["awayTeam"]["name"],
        "home_score": match.get("homeScore", {}).get("current", 0),
        "away_score": match.get("awayScore", {}).get("current", 0),
    }


def convert_player_stats(
    lineups: dict, shotmap: list[dict], match_id: int
) -> pd.DataFrame:
    """Converte lineups + shotmap do Sofascore → PlayerMatchMetrics DataFrame."""
    rows = []

    # Index shotmap by player_id for xG/goals lookup
    player_shots: dict[int, list[dict]] = {}
    for shot in shotmap:
        pid = shot.get("player", {}).get("id")
        if pid:
            player_shots.setdefault(pid, []).append(shot)

    for side in ["home", "away"]:
        side_data = lineups.get(side, {})
        team_name = side  # Will be overridden
        players = side_data.get("players", [])

        for p in players:
            player_info = p.get("player", {})
            pid = player_info.get("id")
            pname = player_info.get("name", "")
            stats = p.get("statistics", {})

            if not stats.get("minutesPlayed"):
                continue

            # Shots from shotmap
            p_shots = player_shots.get(pid, [])
            goals = sum(1 for s in p_shots if s.get("shotType") == "goal")
            shots_on_target = sum(
                1 for s in p_shots
                if s.get("shotType") in ("goal", "save")
            )
            xg_total = sum(s.get("xg", 0) or 0 for s in p_shots)
            big_chances = sum(1 for s in p_shots if (s.get("xg", 0) or 0) >= 0.3)
            big_chances_missed = sum(
                1 for s in p_shots
                if (s.get("xg", 0) or 0) >= 0.3 and s.get("shotType") != "goal"
            )

            row = {
                "player_id": pid,
                "player_name": pname,
                "team": side,  # placeholder, updated below
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
                    stats.get("accuratePass", 0) / max(stats.get("totalPass", 1), 1) * 100, 1
                ),
                "progressive_passes": 0,  # Sofascore has totalProgression but not pass-specific
                "progressive_carries": stats.get("progressiveBallCarriesCount", 0),
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
                "dribbles_attempted": stats.get("duelWon", 0) + stats.get("duelLost", 0),
                "dribbles_completed": stats.get("duelWon", 0),
                "touches": stats.get("touches", 0),
                "carries": stats.get("ballCarriesCount", 0),
                "dispossessed": stats.get("dispossessed", 0) + stats.get("possessionLostCtrl", 0),
                "pressures": 0,
                "pressure_regains": stats.get("ballRecovery", 0),
                # v0.2.0 metrics
                "progressive_receptions": 0,
                "big_chances": big_chances + stats.get("bigChanceCreated", 0),
                "big_chances_missed": big_chances_missed,
                "passes_short": 0,
                "passes_short_completed": 0,
                "passes_medium": 0,
                "passes_medium_completed": 0,
                "passes_long": stats.get("totalLongBalls", 0),
                "passes_long_completed": stats.get("accurateLongBalls", 0),
                "passes_under_pressure": 0,
                "passes_under_pressure_completed": 0,
                "switches_of_play": 0,
                "ground_duels_won": stats.get("wonTackle", 0),
                "ground_duels_total": stats.get("totalTackle", 0),
                "tackle_success_rate": round(
                    stats.get("wonTackle", 0) / max(stats.get("totalTackle", 1), 1) * 100, 1
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def convert_shotmap_to_actions(shotmap: list[dict], match_id: int) -> pd.DataFrame:
    """Converte shotmap do Sofascore → ActionValue DataFrame."""
    rows = []
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


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_brasileirao():
    """Pipeline completo de ingestão do Brasileirão 2026."""
    session = get_session()

    try:
        print("Buscando partidas do Brasileirão 2026...")
        matches = fetch_all_matches()
        print(f"Encontradas {len(matches)} partidas finalizadas.")

        for i, match in enumerate(matches):
            mid = match["id"]
            home = match["homeTeam"]["name"]
            away = match["awayTeam"]["name"]
            hs = match.get("homeScore", {}).get("current", 0)
            aws = match.get("awayScore", {}).get("current", 0)

            print(f"[{i+1}/{len(matches)}] {home} {hs}x{aws} {away} (ID: {mid})")

            # Match info
            match_info = convert_match_info(match)
            upsert_match(session, match_info)

            # Lineups + stats
            lineups_data = _api_get(f"event/{mid}/lineups")
            shotmap_data = _api_get(f"event/{mid}/shotmap")

            if not lineups_data:
                print(f"  Sem lineups, pulando...")
                continue

            shotmap = (shotmap_data or {}).get("shotmap", [])

            # Convert and set team names
            metrics_df = convert_player_stats(lineups_data, shotmap, mid)
            if metrics_df.empty:
                continue

            # Fix team names from match info
            metrics_df.loc[metrics_df["team"] == "home", "team"] = home
            metrics_df.loc[metrics_df["team"] == "away", "team"] = away

            upsert_player_metrics(session, metrics_df, mid)

            # Action values from shotmap
            if shotmap:
                actions_df = convert_shotmap_to_actions(shotmap, mid)
                actions_df.loc[actions_df["team"] == "home", "team"] = home
                actions_df.loc[actions_df["team"] == "away", "team"] = away
                upsert_action_values(session, actions_df, mid)

            print(f"  OK: {len(metrics_df)} jogadores, {len(shotmap)} chutes")

        print(f"\nIngestão completa: {len(matches)} partidas processadas.")

    finally:
        session.close()


if __name__ == "__main__":
    ingest_brasileirao()
