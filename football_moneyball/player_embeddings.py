"""Modulo de embeddings de jogadores com consciencia posicional.

Gera representacoes vetoriais (embeddings) do estilo de jogo de cada jogador
usando PCA por grupo posicional sobre metricas agregadas, realiza clusterizacao
para identificar arquetipos taticos e fornece busca por similaridade/
complementaridade via pgvector.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

from football_moneyball.player_metrics import POSITION_GROUP_MAP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colunas que devem ser normalizadas por 90 minutos
# ---------------------------------------------------------------------------
_PER90_COLUMNS = [
    "passes_completed",
    "passes_attempted",
    "tackles",
    "interceptions",
    "shots",
    "shots_on_target",
    "goals",
    "assists",
    "key_passes",
    "dribbles_completed",
    "dribbles_attempted",
    "aerial_duels_won",
    "aerial_duels_lost",
    "fouls_committed",
    "fouls_won",
    "crosses",
    "long_balls",
    "through_balls",
    "progressive_passes",
    "progressive_carries",
    "carries",
    "touches",
    "pressures",
    "blocks",
    "clearances",
    "recoveries",
    # v0.2.0 metrics
    "progressive_receptions",
    "big_chances",
    "passes_short",
    "passes_short_completed",
    "passes_medium",
    "passes_medium_completed",
    "passes_long",
    "passes_long_completed",
    "passes_under_pressure",
    "passes_under_pressure_completed",
    "switches_of_play",
    "ground_duels_won",
    "ground_duels_total",
]

# ---------------------------------------------------------------------------
# Arquetipos por grupo posicional
# ---------------------------------------------------------------------------
_GROUP_ARCHETYPES: dict[str, dict[str, str]] = {
    "DEF": {
        "progressive_passes": "Playmaking CB",
        "tackles": "Stopper",
        "carries": "Ball-Playing FB",
        "crosses": "Attacking FB",
        "interceptions": "Sweeper",
    },
    "MID": {
        "passes_completed": "Deep-Lying Playmaker",
        "tackles": "Box-to-Box",
        "interceptions": "Defensive Mid",
        "key_passes": "Creative AM",
        "progressive_carries": "Mezzala",
    },
    "FWD": {
        "aerials_won": "Target Man",
        "dribbles_completed": "Inside Forward",
        "goals": "Poacher",
        "assists": "Complete Forward",
        "xg": "Goal Threat",
    },
    "GK": {
        "passes_completed": "Sweeper Keeper",
        "passes_long_completed": "Distribution GK",
        "clearances": "Traditional GK",
    },
}


# =========================================================================
# 1. Construcao de perfis agregados
# =========================================================================

def build_player_profiles(
    session: Session,
    competition: str | None = None,
    season: str | None = None,
    position_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Constroi perfis agregados de jogadores a partir de metricas por partida.

    Consulta a tabela ``player_match_metrics`` no banco de dados, agrega por
    jogador (media entre partidas) e normaliza as metricas aplicaveis por
    90 minutos.

    Parameters
    ----------
    session:
        Sessao SQLAlchemy conectada ao banco.
    competition:
        Filtro opcional de competicao (ex.: ``"La Liga"``).
    season:
        Filtro opcional de temporada (ex.: ``"2023/2024"``).
    position_map:
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
    base_query = """
        SELECT pmm.*
        FROM player_match_metrics pmm
    """
    conditions: list[str] = []
    params: dict[str, Any] = {}

    if competition or season:
        base_query += " JOIN matches m ON pmm.match_id = m.match_id"
        if competition:
            conditions.append("m.competition = :competition")
            params["competition"] = competition
        if season:
            conditions.append("m.season = :season")
            params["season"] = season

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    df = pd.read_sql(text(base_query), session.bind, params=params)

    if df.empty:
        return df

    # Colunas de identificacao
    id_cols = ["player_id", "player_name", "team"]
    metric_cols = [c for c in df.columns if c not in id_cols + ["match_id"]]

    # Agregar por jogador (media entre partidas)
    agg_df = df.groupby(id_cols, as_index=False)[metric_cols].mean()

    # Normalizacao por 90 minutos
    if "minutes_played" in agg_df.columns:
        per90_cols_present = [c for c in _PER90_COLUMNS if c in agg_df.columns]
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
    profiles_df:
        DataFrame retornado por :func:`build_player_profiles`.  Deve conter
        a coluna ``position_group``.
    n_components:
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

    archetype_map = _GROUP_ARCHETYPES.get(position_group, _GROUP_ARCHETYPES["MID"])

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
    embeddings_df:
        DataFrame com coluna ``embedding`` (lista de floats).
    n_clusters:
        Numero maximo de clusters (pode ser reduzido pelo silhouette
        analysis).
    pca:
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
            fallback_names = list(_GROUP_ARCHETYPES["MID"].values())
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
                    _GROUP_ARCHETYPES.get(group, _GROUP_ARCHETYPES["MID"]).values()
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
                    _GROUP_ARCHETYPES.get(group, _GROUP_ARCHETYPES["MID"]).values()
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


# =========================================================================
# 4. Busca por similaridade (pgvector)
# =========================================================================

def find_similar(
    session: Session,
    player_name: str,
    season: str,
    limit: int = 10,
    cross_position: bool = False,
) -> pd.DataFrame:
    """Encontra jogadores com estilo de jogo mais parecido via pgvector.

    Utiliza distancia cosseno (operador ``<=>``) na tabela
    ``player_embeddings``.  Por padrao, filtra apenas jogadores do mesmo
    grupo posicional.

    Parameters
    ----------
    session:
        Sessao SQLAlchemy.
    player_name:
        Nome do jogador de referencia.
    season:
        Temporada para filtrar.
    limit:
        Numero maximo de resultados.
    cross_position:
        Se ``True``, busca entre todos os jogadores independente da posicao.
        Se ``False`` (padrao), filtra pelo mesmo ``position_group``.

    Returns
    -------
    pd.DataFrame
        Colunas: ``player_name``, ``team``, ``archetype``,
        ``position_group``, ``distance``, ``similarity``.
    """
    position_filter = ""
    if not cross_position:
        position_filter = (
            " AND pe2.position_group = ("
            "   SELECT pe1.position_group"
            "   FROM player_embeddings pe1"
            "   WHERE pe1.player_name = :name AND pe1.season = :season"
            "   LIMIT 1"
            " )"
        )

    query = text(f"""
        SELECT pe2.player_name, pe2.team, pe2.archetype,
               pe2.position_group,
               pe2.embedding <=> (
                   SELECT pe1.embedding
                   FROM player_embeddings pe1
                   WHERE pe1.player_name = :name AND pe1.season = :season
               ) AS distance
        FROM player_embeddings pe2
        WHERE pe2.season = :season AND pe2.player_name != :name
        {position_filter}
        ORDER BY distance
        LIMIT :limit
    """)

    result = session.execute(
        query,
        {"name": player_name, "season": season, "limit": limit},
    )
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if not df.empty:
        df["similarity"] = 1.0 - df["distance"]

    return df


# =========================================================================
# 5. Busca por complementaridade
# =========================================================================

def find_complementary(
    session: Session,
    player_name: str,
    season: str,
    limit: int = 10,
    cross_position: bool = False,
) -> pd.DataFrame:
    """Encontra jogadores com perfil complementar (mais dissimilar).

    Ordena pela maior distancia cosseno, retornando jogadores que cobrem
    caracteristicas opostas ao jogador de referencia.  Por padrao, filtra
    apenas jogadores do mesmo grupo posicional.

    Parameters
    ----------
    session:
        Sessao SQLAlchemy.
    player_name:
        Nome do jogador de referencia.
    season:
        Temporada para filtrar.
    limit:
        Numero maximo de resultados.
    cross_position:
        Se ``True``, busca entre todos os jogadores independente da posicao.
        Se ``False`` (padrao), filtra pelo mesmo ``position_group``.

    Returns
    -------
    pd.DataFrame
        Colunas: ``player_name``, ``team``, ``archetype``,
        ``position_group``, ``distance``, ``similarity``.
    """
    position_filter = ""
    if not cross_position:
        position_filter = (
            " AND pe2.position_group = ("
            "   SELECT pe1.position_group"
            "   FROM player_embeddings pe1"
            "   WHERE pe1.player_name = :name AND pe1.season = :season"
            "   LIMIT 1"
            " )"
        )

    query = text(f"""
        SELECT pe2.player_name, pe2.team, pe2.archetype,
               pe2.position_group,
               pe2.embedding <=> (
                   SELECT pe1.embedding
                   FROM player_embeddings pe1
                   WHERE pe1.player_name = :name AND pe1.season = :season
               ) AS distance
        FROM player_embeddings pe2
        WHERE pe2.season = :season AND pe2.player_name != :name
        {position_filter}
        ORDER BY distance DESC
        LIMIT :limit
    """)

    result = session.execute(
        query,
        {"name": player_name, "season": season, "limit": limit},
    )
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if not df.empty:
        df["similarity"] = 1.0 - df["distance"]

    return df


# =========================================================================
# 6. Recomendacao por perfil sintetico
# =========================================================================

def recommend_by_profile(
    session: Session,
    profile: dict[str, float],
    season: str,
    limit: int = 10,
    pca_pickle_path: str | None = None,
    pca: PCA | dict[str, PCA] | None = None,
    position_group: str | None = None,
) -> pd.DataFrame:
    """Recomenda jogadores mais proximos de um perfil tatico desejado.

    Recebe um dicionario de pesos de features, constroi um embedding sintetico
    usando o PCA/scaler salvos e consulta os vizinhos mais proximos via
    pgvector.

    Parameters
    ----------
    session:
        Sessao SQLAlchemy.
    profile:
        Dicionario ``{feature_name: valor_desejado}``, ex.:
        ``{"goals": 0.8, "tackles": 0.2, "key_passes": 0.9}``.
    season:
        Temporada para filtrar.
    limit:
        Numero maximo de resultados.
    pca_pickle_path:
        Caminho para o arquivo pickle contendo o PCA ajustado. Ignorado se
        ``pca`` for fornecido diretamente.
    pca:
        Objeto PCA ajustado ou dicionario ``{position_group: PCA}``
        (alternativa ao pickle).  Deve conter os atributos
        ``feature_names_`` e ``scaler_`` anexados durante
        :func:`generate_embeddings`.
    position_group:
        Grupo posicional para filtrar resultados e selecionar o PCA
        correto quando ``pca`` for um dicionario.  Se ``None``, usa
        ``'MID'`` como fallback.

    Returns
    -------
    pd.DataFrame
        Colunas: ``player_name``, ``team``, ``archetype``,
        ``position_group``, ``distance``, ``similarity``.
    """
    resolved_group = position_group or "MID"

    # Resolver PCA
    resolved_pca: PCA | None = None
    if pca is not None:
        if isinstance(pca, dict):
            resolved_pca = pca.get(resolved_group, pca.get("MID"))
        else:
            resolved_pca = pca

    if resolved_pca is None:
        if pca_pickle_path is None:
            raise ValueError(
                "E necessario fornecer 'pca' ou 'pca_pickle_path'."
            )
        with open(pca_pickle_path, "rb") as f:
            loaded = pickle.load(f)
            if isinstance(loaded, dict):
                resolved_pca = loaded.get(resolved_group, loaded.get("MID"))
            else:
                resolved_pca = loaded

    if resolved_pca is None:
        raise ValueError(
            f"PCA nao encontrado para o grupo posicional '{resolved_group}'."
        )

    feature_names: list[str] = resolved_pca.feature_names_  # type: ignore[attr-defined]
    scaler: StandardScaler = resolved_pca.scaler_  # type: ignore[attr-defined]

    # Construir vetor no espaco original das features
    raw_vector = np.zeros(len(feature_names))
    for feat, value in profile.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            raw_vector[idx] = value

    # Escalonar e projetar no espaco PCA
    scaled_vector = scaler.transform(raw_vector.reshape(1, -1))
    embedding = resolved_pca.transform(scaled_vector)[0].tolist()

    # Formatar como literal de array para pgvector
    embedding_literal = "[" + ",".join(str(v) for v in embedding) + "]"

    position_filter = ""
    params: dict[str, Any] = {
        "embedding": embedding_literal,
        "season": season,
        "limit": limit,
    }
    if position_group is not None:
        position_filter = " AND position_group = :pos_group"
        params["pos_group"] = position_group

    query = text(f"""
        SELECT player_name, team, archetype, position_group,
               embedding <=> :embedding::vector AS distance
        FROM player_embeddings
        WHERE season = :season
        {position_filter}
        ORDER BY distance
        LIMIT :limit
    """)

    result = session.execute(query, params)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if not df.empty:
        df["similarity"] = 1.0 - df["distance"]

    return df
