"""Use case: ingestao delta de partidas do Sofascore."""

from __future__ import annotations
from typing import Any
import logging

logger = logging.getLogger(__name__)


class IngestMatches:
    """Ingere partidas novas do Sofascore (delta — so jogos que faltam).

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
        competition_id: int = 325,
        season_id: int = 87678,
    ) -> dict[str, Any]:
        """Executa ingestao delta.

        Busca partidas finalizadas do Sofascore, compara com o banco,
        e ingere apenas as novas.
        """
        from football_moneyball.domain import metrics

        # Buscar partidas do provider
        all_matches = self.provider.get_matches(competition_id, season_id)
        if all_matches.empty:
            return {"error": "Nenhuma partida encontrada no provider.", "ingested": 0}

        ingested = 0
        skipped = 0
        errors = 0

        for _, match_row in all_matches.iterrows():
            mid = int(match_row["match_id"])

            if self.repo.match_exists(mid):
                skipped += 1
                continue

            try:
                # Fetch events and extract metrics
                events = self.provider.get_match_events(mid)
                if events.empty:
                    continue

                # For Sofascore, events ARE the lineups+stats (not raw events)
                # The provider returns data already in metrics format
                match_info = self.provider.get_match_info(mid)
                match_info["competition"] = competition
                match_info["season"] = season
                self.repo.save_match(match_info)

                # If provider returns pre-computed metrics (Sofascore style)
                if "player_name" in events.columns and "xg" in events.columns:
                    self.repo.save_player_metrics(events, mid)
                else:
                    # StatsBomb style: raw events → compute metrics
                    metrics_df = metrics.extract_match_metrics(events)
                    if not metrics_df.empty:
                        self.repo.save_player_metrics(metrics_df, mid)

                # v1.2.0 — Ingerir match_stats + referee + HT score
                try:
                    if hasattr(self.provider, "get_match_stats"):
                        stats = self.provider.get_match_stats(mid)
                        if stats:
                            ht_home, ht_away = self.provider.get_ht_scores(mid)
                            stats["ht_home_score"] = ht_home
                            stats["ht_away_score"] = ht_away
                            referee = self.provider.get_referee_info(mid)
                            if referee:
                                stats["referee_id"] = referee["referee_id"]
                                stats["referee_name"] = referee["name"]
                                self.repo.save_referee_stats(referee)
                            self.repo.save_match_stats(mid, stats)
                except Exception as e:
                    logger.warning(f"Erro match_stats {mid}: {e}")

                ingested += 1
                logger.info(f"Ingerido: {match_info.get('home_team', '')} vs {match_info.get('away_team', '')} (ID: {mid})")

            except Exception as e:
                logger.warning(f"Erro ao ingerir match {mid}: {e}")
                errors += 1

        return {
            "ingested": ingested,
            "skipped": skipped,
            "errors": errors,
            "total": len(all_matches),
        }
