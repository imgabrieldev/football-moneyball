"""Use case: analise of a match individual.

Extract metrics, pressing, network of passes and persiste in the banco.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_moneyball.domain import metrics, pressing, network


class AnalyzeMatch:
    """Orquestra a analise complete of a match.

    Parameters
    ----------
    provider : DataProvider
        Fonte of data (StatsBomb, Sofascore, etc).
    repo : MatchRepository
        Repositorio for persistencia.
    """

    def __init__(self, provider, repo) -> None:
        self.provider = provider
        self.repo = repo

    def execute(
        self, match_id: int, refresh: bool = False
    ) -> dict[str, Any]:
        """Runs the analise of a match.

        If a match ja existe in the banco and refresh=False, returns data
        of the cache. Caso contrario, extrai of the provider, calcula metrics
        and persiste.

        Parameters
        ----------
        match_id : int
            ID of the match.
        refresh : bool
            Forcar reprocessamento same if ja existir.

        Returns
        -------
        dict
            Dicionario with chaves: 'metrics_df', 'pressing_df',
            'partnerships', 'graph', 'edges_df'.
        """
        # Check cache
        if not refresh and self.repo.match_exists(match_id):
            metrics_df = self.repo.get_match_data(match_id)
            return {"metrics_df": metrics_df, "from_cache": True}

        # Fetch from provider
        events = self.provider.get_match_events(match_id)
        if events.empty:
            return {"metrics_df": pd.DataFrame(), "error": "Nenhum evento encontrado"}

        # Extract metrics
        metrics_df = metrics.extract_match_metrics(events)

        # Build pass network
        graph, edges_df = network.build_pass_network(events)
        edge_features = network.compute_edge_features(graph)
        feat_list = []
        for _, row in edges_df.iterrows():
            key = (row["passer_id"], row["receiver_id"])
            feat_list.append(edge_features.get(key, {}))
        edges_df["features"] = feat_list

        # Pressing metrics
        pressing_df = pressing.compute_match_pressing(events)

        # Persist
        match_info = self.provider.get_match_info(match_id)
        self.repo.save_match(match_info)
        self.repo.save_player_metrics(metrics_df, match_id)
        self.repo.save_pass_network(edges_df, match_id)
        if not pressing_df.empty:
            self.repo.save_pressing_metrics(pressing_df, match_id)

        # Partnerships
        partnerships = network.identify_key_partnerships(graph, top_n=5)

        return {
            "metrics_df": metrics_df,
            "pressing_df": pressing_df,
            "partnerships": partnerships,
            "graph": graph,
            "edges_df": edges_df,
            "from_cache": False,
        }
