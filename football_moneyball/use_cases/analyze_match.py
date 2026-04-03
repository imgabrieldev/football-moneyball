"""Use case: analise de uma partida individual.

Extrai metricas, pressing, rede de passes e persiste no banco.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_moneyball.domain import metrics, pressing, network


class AnalyzeMatch:
    """Orquestra a analise completa de uma partida.

    Parameters
    ----------
    provider : DataProvider
        Fonte de dados (StatsBomb, Sofascore, etc).
    repo : MatchRepository
        Repositorio para persistencia.
    """

    def __init__(self, provider, repo) -> None:
        self.provider = provider
        self.repo = repo

    def execute(
        self, match_id: int, refresh: bool = False
    ) -> dict[str, Any]:
        """Executa a analise de uma partida.

        Se a partida ja existe no banco e refresh=False, retorna dados
        do cache. Caso contrario, extrai do provider, calcula metricas
        e persiste.

        Parameters
        ----------
        match_id : int
            ID da partida.
        refresh : bool
            Forcar reprocessamento mesmo se ja existir.

        Returns
        -------
        dict
            Dicionario com chaves: 'metrics_df', 'pressing_df',
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
