"""Modulo de dominio para embeddings de jogadores com consciencia posicional.

Gera representacoes vetoriais (embeddings) do estilo de jogo de cada jogador
usando PCA por grupo posicional sobre metricas agregadas, realiza clusterizacao
para identificar arquetipos taticos.
Logica pura sobre DataFrames e arrays — sem dependencias de I/O externo
(sqlalchemy, pgvector).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from football_moneyball.domain.constants import (
    GROUP_ARCHETYPES,
    PER90_COLUMNS,
)

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Construcao de perfis agregados
# =========================================================================

def build_player_profiles(
    metrics_df: pd.DataFrame,
    position_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Constroi perfis agregados de jogadores a partir de metricas por partida.

    Recebe um DataFrame de metricas (tipicamente lido do banco ou agregado
    de multiplas partidas), agrega por jogador (media entre partidas) e
    normaliza as metricas aplicaveis por 90 minutos.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame com metricas por jogador por partida. Deve conter colunas
        ``player_id``, ``player_name``, ``team`` e colunas de metricas
        numericas.
    position_map : dict[int, str] | None
        Dicionario ``player_id -> position_group`` (``'GK'``, ``'DEF'``,
        ``'MID'``, ``'FWD'``).  Se fornecido, adiciona coluna
        ``position_group`` ao resultado.  Se ``None``, assume ``'MID'`` para
        todos os jogadores.

    Returns
    -------
    pd.DataFrame
        DataFrame com ``player_id``, ``player_name``, ``team``,
        ``position_group`` e colunas de metricas agregadas.
    """
    if metrics_df.empty:
        return metrics_df

    # Colunas de identificacao
    id_cols = ["player_id", "player_name", "team"]
    metric_cols = [c for c in metrics_df.columns if c not in id_cols + ["match_id"]]

    # Agregar por jogador (media entre partidas)
    agg_df = metrics_df.groupby(id_cols, as_index=False)[metric_cols].mean()

    # Normalizacao por 90 minutos
    if "minutes_played" in agg_df.columns:
        per90_cols_present = [c for c in PER90_COLUMNS if c in agg_df.columns]
        for col in per90_cols_present:
            agg_df[col] = agg_df[col] / (agg_df["minutes_played"] / 90.0)
        # Substituir inf/NaN resultantes de divisao por zero
        agg_df.replace([np.inf, -np.inf], 0.0, inplace=True)
        agg_df.fillna(0.0, inplace=True)

    # Atribuir grupo posicional
    if position_map is not None:
        agg_df["position_group"] = agg_df["player_id"].map(position_map).fillna("MID")
    else:
        agg_df["position_group"] = "MID"

    return agg_df


# =========================================================================
# 2. Geracao de embeddings via PCA (por grupo posicional)
# =========================================================================

def generate_embeddings(
    profiles_df: pd.DataFrame,
    n_components: int = 16,
) -> tuple[pd.DataFrame, dict[str, PCA]]:
    """Gera embeddings de estilo de jogo via StandardScaler + PCA por grupo.

    Agrupa os jogadores por ``position_group`` e executa PCA independente
    para cada grupo, produzindo embeddings posicionalmente contextualizados.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        DataFrame retornado por :func:`build_player_profiles`.  Deve conter
        a coluna ``position_group``.
    n_components : int
        Numero de componentes principais a manter (por grupo).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, PCA]]
        - DataFrame com ``player_id``, ``player_name``, ``team``,
          ``position_group`` e coluna ``embedding`` (lista de floats).
        - Dicionario de objetos PCA ajustados, indexados por
          ``position_group``.  Cada PCA contem atributos extras
          ``feature_names_``, ``scaler_`` e ``explained_variance_sum_``.
    """
    id_cols = ["player_id", "player_name", "team", "position_group"]

    # Se nao houver coluna position_group, criar com default
    if "position_group" not in profiles_df.columns:
        profiles_df = profiles_df.copy()
        profiles_df["position_group"] = "MID"

    numeric_cols = [
        c for c in profiles_df.select_dtypes(include=[np.number]).columns
        if c not in id_cols
    ]

    result_parts: list[pd.DataFrame] = []
    pca_dict: dict[str, PCA] = {}

    for group, group_df in profiles_df.groupby("position_group"):
        group = str(group)
        X = group_df[numeric_cols].fillna(0.0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        nc = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=nc)
        X_pca = pca.fit_transform(X_scaled)

        # Guardar referencias auxiliares no objeto PCA
        pca.feature_names_ = numeric_cols  # type: ignore[attr-defined]
        pca.scaler_ = scaler  # type: ignore[attr-defined]
        pca.explained_variance_sum_ = float(  # type: ignore[attr-defined]
            pca.explained_variance_ratio_.sum()
        )

        logger.info(
            "PCA [%s]: %d componentes, variancia explicada = %.2f%%",
            group, nc, pca.explained_variance_sum_ * 100,  # type: ignore[attr-defined]
        )

        part_df = group_df[id_cols].copy()
        part_df["embedding"] = [row.tolist() for row in X_pca]
        result_parts.append(part_df)
        pca_dict[group] = pca

    result_df = pd.concat(result_parts, ignore_index=True)
    return result_df, pca_dict


# =========================================================================
# 3. Clusterizacao e deteccao de arquetipos
# =========================================================================

def _find_optimal_k(X: np.ndarray, k_range: range = range(3, 10)) -> int:
    """Encontra o numero otimo de clusters via silhouette analysis."""
    if len(X) < max(k_range):
        k_range = range(2, max(3, len(X)))
    best_k = k_range.start
    best_score = -1.0
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _infer_archetype(
    centroid: np.ndarray,
    pca: PCA,
    position_group: str = "MID",
) -> str:
    """Infere o nome do arquetipo a partir do centroide do cluster.

    Reconstroi o centroide no espaco original das features via transformada
    inversa do PCA, identifica as 3 features com maior peso absoluto e mapeia
    para um arquetipo tatico baseado no grupo posicional.
    """
    reconstructed = pca.inverse_transform(centroid.reshape(1, -1))[0]
    feature_names: list[str] = pca.feature_names_  # type: ignore[attr-defined]

    top_indices = np.argsort(np.abs(reconstructed))[::-1][:3]
    top_features = [feature_names[i] for i in top_indices]

    archetype_map = GROUP_ARCHETYPES.get(position_group, GROUP_ARCHETYPES["MID"])

    # Contagem de votos por arquetipo
    votes: dict[str, int] = {}
    for feat in top_features:
        archetype = archetype_map.get(feat)
        if archetype:
            votes[archetype] = votes.get(archetype, 0) + 1

    if votes:
        return max(votes, key=lambda k: votes[k])

    # Fallback: primeiro arquetipo do grupo
    fallback_values = list(archetype_map.values())
    return fallback_values[0] if fallback_values else "MID"


def cluster_players(
    embeddings_df: pd.DataFrame,
    n_clusters: int = 6,
    pca: PCA | dict[str, PCA] | None = None,
) -> pd.DataFrame:
    """Agrupa jogadores em clusters taticos via KMeans.

    Para cada cluster, atribui um nome de arquetipo baseado na analise dos
    centroides (quais features originais mais contribuem).  Se ``pca`` for
    um dicionario de PCA por grupo posicional, clusteriza separadamente por
    grupo e usa arquetipos posicionais.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        DataFrame com coluna ``embedding`` (lista de floats).
    n_clusters : int
        Numero maximo de clusters (pode ser reduzido pelo silhouette
        analysis).
    pca : PCA | dict[str, PCA] | None
        Objeto PCA ajustado ou dicionario ``{position_group: PCA}``
        (necessario para inferir nomes de arquetipo).  Se ``None``, usa
        nomes genericos.

    Returns
    -------
    pd.DataFrame
        DataFrame original acrescido de ``cluster_label`` e ``archetype``.
    """
    has_groups = (
        "position_group" in embeddings_df.columns
        and isinstance(pca, dict)
    )

    if has_groups:
        return _cluster_by_group(embeddings_df, n_clusters, pca)  # type: ignore[arg-type]

    # Fallback: clusterizacao global (compatibilidade retroativa)
    X = np.array(embeddings_df["embedding"].tolist())
    n_clusters = min(n_clusters, len(X))

    optimal_k = _find_optimal_k(X, k_range=range(3, n_clusters + 1))
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    single_pca = pca if isinstance(pca, PCA) else None

    archetype_map: dict[int, str] = {}
    for i, centroid in enumerate(kmeans.cluster_centers_):
        if single_pca is not None and hasattr(single_pca, "feature_names_"):
            archetype_map[i] = _infer_archetype(centroid, single_pca)
        else:
            fallback_names = list(GROUP_ARCHETYPES["MID"].values())
            archetype_map[i] = fallback_names[i % len(fallback_names)]

    # Desambiguar nomes repetidos
    archetype_map = _disambiguate_archetypes(archetype_map)

    result_df = embeddings_df.copy()
    result_df["cluster_label"] = labels
    result_df["archetype"] = [archetype_map[label] for label in labels]
    return result_df


def _cluster_by_group(
    embeddings_df: pd.DataFrame,
    n_clusters: int,
    pca_dict: dict[str, PCA],
) -> pd.DataFrame:
    """Clusteriza jogadores separadamente por grupo posicional."""
    result_parts: list[pd.DataFrame] = []
    global_label_offset = 0

    for group, group_df in embeddings_df.groupby("position_group"):
        group = str(group)
        X = np.array(group_df["embedding"].tolist())

        if len(X) < 2:
            part = group_df.copy()
            part["cluster_label"] = global_label_offset
            group_pca = pca_dict.get(group)
            if group_pca is not None and hasattr(group_pca, "feature_names_"):
                part["archetype"] = _infer_archetype(
                    X[0] if len(X) == 1 else np.mean(X, axis=0),
                    group_pca,
                    position_group=group,
                )
            else:
                fallback = list(
                    GROUP_ARCHETYPES.get(group, GROUP_ARCHETYPES["MID"]).values()
                )
                part["archetype"] = fallback[0] if fallback else group
            global_label_offset += 1
            result_parts.append(part)
            continue

        group_k = min(n_clusters, len(X))
        optimal_k = _find_optimal_k(X, k_range=range(2, group_k + 1))
        kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        group_pca = pca_dict.get(group)

        archetype_map: dict[int, str] = {}
        for i, centroid in enumerate(kmeans.cluster_centers_):
            if group_pca is not None and hasattr(group_pca, "feature_names_"):
                archetype_map[i] = _infer_archetype(
                    centroid, group_pca, position_group=group,
                )
            else:
                fallback = list(
                    GROUP_ARCHETYPES.get(group, GROUP_ARCHETYPES["MID"]).values()
                )
                archetype_map[i] = fallback[i % len(fallback)]

        archetype_map = _disambiguate_archetypes(archetype_map)

        part = group_df.copy()
        part["cluster_label"] = [label + global_label_offset for label in labels]
        part["archetype"] = [archetype_map[label] for label in labels]
        global_label_offset += optimal_k
        result_parts.append(part)

    return pd.concat(result_parts, ignore_index=True)


def _disambiguate_archetypes(archetype_map: dict[int, str]) -> dict[int, str]:
    """Desambigua nomes de arquetipos repetidos adicionando sufixo numerico."""
    seen: dict[str, int] = {}
    for k, v in archetype_map.items():
        if v in seen:
            seen[v] += 1
            archetype_map[k] = f"{v} {seen[v]}"
        else:
            seen[v] = 1
    return archetype_map
