"""Use case: ingestao de escalacoes (provaveis ou confirmadas)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class IngestLineups:
    """Busca e persiste lineups de partidas do Sofascore.

    Usado pra ingerir escalacao confirmada (~1h antes do jogo) ou
    probable XI inferida via historico. Pode ser chamado por cronjob
    periodico ou manualmente.

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
        match_ids: list[int] | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
    ) -> dict[str, Any]:
        """Ingere lineups de partidas especificas.

        Parameters
        ----------
        match_ids : list[int], optional
            Lista de match_ids do Sofascore pra ingerir. Se fornecido,
            tem precedencia.
        home_team, away_team : str, optional
            Se match_ids nao fornecido, usa o hash(home-away) como chave.

        Returns
        -------
        dict
            Chaves: ingested, errors, details.
        """
        if not match_ids and not (home_team and away_team):
            return {"error": "Precisa match_ids OU home_team+away_team"}

        now = datetime.now().isoformat()
        ingested = 0
        errors = 0
        details: list[dict] = []

        # Modo 1: match_ids do Sofascore
        if match_ids:
            for mid in match_ids:
                try:
                    lineups = self.provider.get_lineups(mid)
                    if not lineups:
                        continue
                    match_info = self.provider.get_match_info(mid)
                    home = match_info.get("home_team", "")
                    away = match_info.get("away_team", "")
                    match_key = abs(hash(f"{home}-{away}")) % (10**9)

                    rows = self._convert_lineups(
                        lineups, match_key, home, away, now, "confirmed"
                    )
                    if rows:
                        self.repo.save_match_lineups(rows)
                        ingested += 1
                        details.append({
                            "match_id": mid,
                            "home": home,
                            "away": away,
                            "players": len(rows),
                        })
                except Exception as e:
                    logger.warning(f"Erro lineup {mid}: {e}")
                    errors += 1

        return {
            "ingested": ingested,
            "errors": errors,
            "details": details,
        }

    def _convert_lineups(
        self,
        lineups: dict,
        match_key: int,
        home_team: str,
        away_team: str,
        fetched_at: str,
        source: str,
    ) -> list[dict]:
        """Converte output do provider pro formato do repositorio."""
        rows: list[dict] = []
        for side in ("home", "away"):
            side_df = lineups.get(side)
            if side_df is None or side_df.empty:
                continue
            team_name = home_team if side == "home" else away_team
            for _, r in side_df.iterrows():
                pid = r.get("player_id")
                if pid is None:
                    continue
                rows.append({
                    "match_key": match_key,
                    "player_id": int(pid),
                    "team": team_name,
                    "side": side,
                    "player_name": str(r.get("player_name", "")),
                    "position": str(r.get("position", "")),
                    "is_starter": True,
                    "jersey_number": 0,
                    "source": source,
                    "fetched_at": fetched_at,
                })
        return rows
