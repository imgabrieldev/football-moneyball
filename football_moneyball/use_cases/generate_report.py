"""Use case: geracao de scout report completo."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class GenerateReport:
    """Gera relatorio de scouting de um jogador.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio para buscar dados.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        player_name: str,
        season: str | None = None,
        team_target: str | None = None,
    ) -> dict[str, Any]:
        """Monta o relatorio completo.

        Returns
        -------
        dict
            Relatorio com metricas, percentis, arquetipo, similares,
            pressing e compatibilidade.
        """
        metrics_df = self.repo.get_player_metrics(player_name, season)
        if metrics_df.empty:
            return {"error": f"Jogador '{player_name}' nao encontrado."}

        # Aggregate
        numeric_cols = metrics_df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("player_id", "match_id")]
        agg = metrics_df[numeric_cols].sum()
        matches_played = len(metrics_df)
        per90 = {}
        mins = float(agg.get("minutes_played", 1) or 1)
        for k, v in agg.items():
            per90[k] = round(float(v) / (mins / 90), 2)

        # Percentiles
        percentiles = {}
        try:
            all_metrics = self.repo.get_all_metrics(None, season)
            if not all_metrics.empty:
                all_agg = all_metrics.groupby("player_id")[numeric_cols].sum()
                for col in numeric_cols:
                    if col in all_agg.columns and col in agg.index:
                        rank = (all_agg[col] < float(agg[col])).sum()
                        percentiles[col] = round(rank / len(all_agg) * 100, 1)
        except Exception:
            pass

        # Embedding / archetype
        emb = self.repo.get_embedding(player_name, season)
        archetype = emb.archetype if emb else "N/A"

        # Similar players
        similar_df = self.repo.find_similar_players(player_name, season or "", limit=5)

        # Pressing
        pressing_data = None
        try:
            team = metrics_df["team"].iloc[0] if "team" in metrics_df.columns else None
            if team:
                pressing_rows = self.repo.get_pressing_metrics(team, season)
                if pressing_rows:
                    pressing_data = {
                        "ppda": round(np.mean([p.ppda for p in pressing_rows if p.ppda]), 2),
                        "pressing_success_rate": round(
                            np.mean([p.pressing_success_rate for p in pressing_rows if p.pressing_success_rate]), 1
                        ),
                        "counter_pressing_fraction": round(
                            np.mean([p.counter_pressing_fraction for p in pressing_rows if p.counter_pressing_fraction]), 1
                        ),
                    }
        except Exception:
            pass

        report = {
            "jogador": player_name,
            "temporada": season or "Todas",
            "partidas": matches_played,
            "arquetipo": archetype,
            "metricas_totais": {k: float(v) for k, v in agg.items()},
            "metricas_por_90": per90,
            "percentis": percentiles,
            "similares": similar_df.to_dict("records") if not similar_df.empty else [],
            "pressing": pressing_data,
        }

        # Compatibility
        if team_target:
            report["time_alvo"] = team_target
            try:
                compat_df = self.repo.find_complementary_players(
                    player_name, season or "", limit=10
                )
                report["compatibilidade"] = compat_df.to_dict("records") if not compat_df.empty else []
            except Exception:
                report["compatibilidade"] = []

        return report
