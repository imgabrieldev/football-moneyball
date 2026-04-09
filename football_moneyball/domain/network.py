"""Modulo of dominio for thenalise of networks of passes.

Builds graphs of network of passes using networkx, calcula metrics
of centralidade and identifies parcerias-chave between players.
Logica pura sobre DataFrames and graphs — without dependencias of I/O externo.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_pass_network(
    events: pd.DataFrame, team: str | None = None
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Builds a graph direcionado of network of passes from eventos.

    Receives a DataFrame of eventos already loaded and builds o graph
    of network of passes with nos (players) and edges (passes) weighted.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame of eventos StatsBomb (retornado by sb.events() ou
        equivalente).
    team : str, opcional
        Nome of the time for filtrar. If None, inclui todos os times.

    Returns
    -------
    tuple[nx.DiGraph, pd.DataFrame]
        Grafo direcionado with nos (players) and edges (passes) weighted,
        and a DataFrame with the colunas: passer_id, receiver_id, passer_name,
        receiver_name, weight.
    """
    # Filtrar only passes completedos (pass_outcome NaN = sucesso)
    passes = events[events["type"] == "Pass"].copy()
    passes = passes[passes["pass_outcome"].isna()]

    if team is not None:
        passes = passes[passes["team"] == team]

    # Extrair passer and receiver
    passes["passer_id"] = passes["player_id"]
    passes["passer_name"] = passes["player"]
    passes["receiver_id"] = passes["pass_recipient_id"]
    passes["receiver_name"] = passes["pass_recipient"]

    # Remover linhas without receptor identificado
    passes = passes.dropna(subset=["receiver_id"])

    # Count passes between each par (passer, receiver)
    edges_df = (
        passes.groupby(
            ["passer_id", "receiver_id", "passer_name", "receiver_name"]
        )
        .size()
        .reset_index(name="weight")
    )

    # Calcular posicoes means of the players from the localizacoes of the eventos
    player_events = events[events["player_id"].notna()].copy()
    if team is not None:
        player_events = player_events[player_events["team"] == team]

    # Extrair coordenadas x and y of the column location
    player_events = player_events[player_events["location"].notna()].copy()
    player_events["loc_x"] = player_events["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 2 else None
    )
    player_events["loc_y"] = player_events["location"].apply(
        lambda loc: loc[1] if isinstance(loc, list) and len(loc) >= 2 else None
    )

    avg_positions = (
        player_events.groupby(["player_id", "player", "position"])
        .agg(avg_x=("loc_x", "mean"), avg_y=("loc_y", "mean"))
        .reset_index()
    )

    # Construir o graph direcionado
    G = nx.DiGraph()

    # Adicionar in the with atributos
    for _, row in avg_positions.iterrows():
        G.add_node(
            row["player_id"],
            player_name=row["player"],
            position=row["position"],
            avg_x=row["avg_x"],
            avg_y=row["avg_y"],
        )

    # Adicionar edges with weight
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["passer_id"],
            row["receiver_id"],
            weight=row["weight"],
            passer_name=row["passer_name"],
            receiver_name=row["receiver_name"],
        )

    return G, edges_df


def compute_network_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """Compute metrics of centralidade by in the of the graph of passes.

    Parameters
    ----------
    G : nx.DiGraph
        Grafo direcionado of network of passes.

    Returns
    -------
    pd.DataFrame
        DataFrame with player_id, player_name and as metrics:
        degree_centrality, betweenness_centrality, eigenvector_centrality,
        closeness_centrality, pagerank, in_degree, out_degree.
    """
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, weight="weight")
    closeness_cent = nx.closeness_centrality(G, distance=None)
    pagerank = nx.pagerank(G, weight="weight")

    try:
        eigenvector_cent = nx.eigenvector_centrality(
            G, weight="weight", max_iter=1000
        )
    except nx.PowerIterationFailedConvergence:
        eigenvector_cent = {node: 0.0 for node in G.nodes()}

    # Graus weighted (in and out)
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    records = []
    for node in G.nodes():
        player_name = G.nodes[node].get("player_name", "")
        records.append(
            {
                "player_id": node,
                "player_name": player_name,
                "degree_centrality": degree_cent.get(node, 0.0),
                "betweenness_centrality": betweenness_cent.get(node, 0.0),
                "eigenvector_centrality": eigenvector_cent.get(node, 0.0),
                "closeness_centrality": closeness_cent.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
            }
        )

    return pd.DataFrame(records)


def compute_edge_features(G: nx.DiGraph) -> dict:
    """Compute features by edge for thermazenamento in JSONB.

    For each edge calcula:
    - normalized_weight: weight normalizado by the total of passes.
    - is_reciprocal: if existe edge reversa.
    - pair_pass_share: sum of the weights of the par sobre o total of passes.

    Parameters
    ----------
    G : nx.DiGraph
        Grafo direcionado of network of passes.

    Returns
    -------
    dict
        Dicionario {(passer_id, receiver_id): features_dict}.
    """
    total_passes = sum(data["weight"] for _, _, data in G.edges(data=True))

    if total_passes == 0:
        return {}

    features = {}
    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        normalized_weight = weight / total_passes
        is_reciprocal = G.has_edge(v, u)

        reverse_weight = G[v][u]["weight"] if is_reciprocal else 0
        pair_pass_share = (weight + reverse_weight) / total_passes

        features[(u, v)] = {
            "normalized_weight": normalized_weight,
            "is_reciprocal": is_reciprocal,
            "pair_pass_share": pair_pass_share,
        }

    return features


def identify_key_partnerships(
    G: nx.DiGraph, top_n: int = 5
) -> list[dict]:
    """Identifies as parcerias of passe mais fortes in the graph.

    Parameters
    ----------
    G : nx.DiGraph
        Grafo direcionado of network of passes.
    top_n : int
        Numero of parcerias a retornar (default: 5).

    Returns
    -------
    list[dict]
        Lista of dicionarios with passer, receiver, weight e
        normalized_weight, ordenados by weight decrescente.
    """
    total_passes = sum(data["weight"] for _, _, data in G.edges(data=True))

    partnerships = []
    for u, v, data in G.edges(data=True):
        weight = data["weight"]
        passer_name = G.nodes[u].get("player_name", str(u))
        receiver_name = G.nodes[v].get("player_name", str(v))

        normalized_weight = weight / total_passes if total_passes > 0 else 0.0

        partnerships.append(
            {
                "passer": passer_name,
                "receiver": receiver_name,
                "weight": weight,
                "normalized_weight": normalized_weight,
            }
        )

    partnerships.sort(key=lambda x: x["weight"], reverse=True)

    return partnerships[:top_n]
