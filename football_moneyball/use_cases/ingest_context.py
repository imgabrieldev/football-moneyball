"""Use case: ingestao de contexto (coaches, injuries, standings)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class IngestContext:
    """Ingere dados contextuais pra v1.6.0 features.

    Parameters
    ----------
    provider : DataProvider (SofascoreProvider)
    repo : MatchRepository
    """

    def __init__(self, provider, repo) -> None:
        self.provider = provider
        self.repo = repo

    def execute(
        self,
        competition: str = "Brasileirão Série A",
        season: str = "2026",
        backfill: bool = False,
    ) -> dict[str, Any]:
        """Ingere managers + missing players + standings.

        Parameters
        ----------
        backfill : bool
            Se True, processa TODOS os matches historicos.
            Se False, so matches recentes (ultimas 30 dias).

        Returns
        -------
        dict com contadores.
        """
        from sqlalchemy import text

        now_iso = datetime.now().isoformat()
        today = now_iso[:10]

        # 1. Buscar matches pra processar
        if backfill:
            sql = text("""
                SELECT match_id, match_date, home_team, away_team
                FROM matches WHERE home_score IS NOT NULL
                ORDER BY match_date
            """)
        else:
            sql = text("""
                SELECT match_id, match_date, home_team, away_team
                FROM matches
                WHERE home_score IS NOT NULL
                  AND match_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY match_date
            """)
        matches = self.repo._session.execute(sql).fetchall()
        logger.info(f"Processing {len(matches)} matches")

        managers_saved = 0
        injuries_saved = 0
        errors = 0

        # Track coach changes per team
        coach_history: dict[str, list] = {}  # team -> [(match_date, coach_id, name)]

        for m in matches:
            mid = int(m.match_id)
            match_date = str(m.match_date)
            home_team = str(m.home_team or "")
            away_team = str(m.away_team or "")

            # Managers
            try:
                mgr = self.provider.get_event_managers(mid)
                if mgr:
                    for side, team_name in [("home", home_team), ("away", away_team)]:
                        side_mgr = mgr.get(side)
                        if not side_mgr or not side_mgr.get("id"):
                            continue
                        coach_id = side_mgr["id"]
                        coach_name = side_mgr["name"]
                        coach_history.setdefault(team_name, []).append(
                            (match_date, coach_id, coach_name)
                        )
                        managers_saved += 1
            except Exception as e:
                logger.warning(f"manager err {mid}: {e}")
                errors += 1

            # Missing players
            try:
                missing = self.provider.get_missing_players(mid)
                if missing.get("home"):
                    self.repo.save_player_injuries(mid, home_team, missing["home"])
                    injuries_saved += len(missing["home"])
                if missing.get("away"):
                    self.repo.save_player_injuries(mid, away_team, missing["away"])
                    injuries_saved += len(missing["away"])
            except Exception as e:
                logger.warning(f"missing players err {mid}: {e}")
                errors += 1

        # 2. Detect coach changes + persist team_coaches
        coaches_persisted = 0
        for team, history in coach_history.items():
            history.sort(key=lambda x: x[0])  # cronologico
            current_coach_id = None
            current_start = None
            for match_date, coach_id, coach_name in history:
                if coach_id != current_coach_id:
                    # Close previous coach relationship
                    if current_coach_id is not None:
                        # update end_date of previous
                        pass  # save_team_coach handles upsert
                    # Start new
                    current_coach_id = coach_id
                    current_start = match_date
                    self.repo.save_team_coach(
                        team=team,
                        coach_id=coach_id,
                        coach_name=coach_name,
                        start_match_date=current_start,
                    )
                    coaches_persisted += 1

        # 3. Standings snapshot
        standings_saved = 0
        try:
            standings = self.provider.get_standings()
            if standings:
                self.repo.save_league_standing(
                    standings, today, competition, season,
                )
                standings_saved = len(standings)
        except Exception as e:
            logger.warning(f"standings err: {e}")
            errors += 1

        return {
            "matches_processed": len(matches),
            "managers_found": managers_saved,
            "injuries_saved": injuries_saved,
            "coaches_persisted": coaches_persisted,
            "standings_saved": standings_saved,
            "errors": errors,
        }
