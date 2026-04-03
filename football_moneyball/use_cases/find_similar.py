"""Use case: busca de jogadores similares via pgvector."""

from __future__ import annotations

from typing import Any

import pandas as pd


class FindSimilar:
    """Busca jogadores com perfil similar via embeddings.

    Parameters
    ----------
    repo : MatchRepository
        Repositorio com acesso a pgvector.
    """

    def __init__(self, repo) -> None:
        self.repo = repo

    def execute(
        self,
        player_name: str,
        season: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Busca jogadores similares.

        Returns
        -------
        dict
            Chaves: 'similar_df', 'season', 'player'.
        """
        # Resolve season if not provided
        if season is None:
            emb = self.repo.get_embedding(player_name)
            if emb is None:
                return {"error": f"Jogador '{player_name}' nao encontrado nos embeddings."}
            season = emb.season

        similar_df = self.repo.find_similar_players(player_name, season, limit)

        return {
            "similar_df": similar_df,
            "season": season,
            "player": player_name,
        }
