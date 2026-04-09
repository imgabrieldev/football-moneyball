"""Use case: busca of players similares via pgvector."""

from __future__ import annotations

from typing import Any

import pandas as pd


class FindSimilar:
    """Busca players with perfil similar via embeddings.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio with acesso a pgvector.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        player_name: str,
        season: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Busca players similares.

        Returns
        -------
        dict
            Chaves: 'similar_df', 'season', 'player'.
        """
        # Resolve season if not provided
        if season is None:
            emb = self.repo.get_embedding(player_name)
            if emb is None:
                return {"error": f"Player '{player_name}' nao encontrado in the embeddings."}
            season = emb.season

        similar_df = self.repo.find_similar_players(player_name, season, limit)

        return {
            "similar_df": similar_df,
            "season": season,
            "player": player_name,
        }
