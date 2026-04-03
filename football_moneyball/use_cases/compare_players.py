"""Use case: comparacao de dois jogadores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ComparePlayers:
    """Compara metricas agregadas de dois jogadores.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio para buscar metricas.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        player_a: str,
        player_b: str,
        season: str | None = None,
    ) -> dict[str, Any]:
        """Busca e agrega metricas de dois jogadores.

        Returns
        -------
        dict
            Chaves: 'agg_a', 'agg_b', 'metrics_a_df', 'metrics_b_df',
            'similarity'.
        """
        metrics_a = self.repo.get_player_metrics(player_a, season)
        metrics_b = self.repo.get_player_metrics(player_b, season)

        if metrics_a.empty:
            return {"error": f"Jogador '{player_a}' nao encontrado."}
        if metrics_b.empty:
            return {"error": f"Jogador '{player_b}' nao encontrado."}

        numeric_cols = metrics_a.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("player_id", "match_id")]

        agg_a = metrics_a[numeric_cols].sum()
        agg_a["partidas"] = len(metrics_a)
        agg_b = metrics_b[numeric_cols].sum()
        agg_b["partidas"] = len(metrics_b)

        # Cosine similarity from embeddings
        similarity = None
        try:
            emb_a = self.repo.get_embedding(player_a, season)
            emb_b = self.repo.get_embedding(player_b, season)
            if emb_a is not None and emb_b is not None:
                vec_a = np.array(emb_a.embedding)
                vec_b = np.array(emb_b.embedding)
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                if norm_a > 0 and norm_b > 0:
                    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        except Exception:
            pass

        return {
            "agg_a": agg_a,
            "agg_b": agg_b,
            "metrics_a_df": metrics_a,
            "metrics_b_df": metrics_b,
            "similarity": similarity,
        }
