"""Módulo de embeddings de jogadores.

Gera representações vetoriais (embeddings) do estilo de jogo de cada jogador
usando PCA sobre métricas agregadas, realiza clusterização para identificar
arquétipos táticos e fornece busca por similaridade/complementaridade via
pgvector.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

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
]

# Mapeamento de índice de cluster → nome de arquétipo (fallback)
_ARCHETYPE_NAMES = [
    "Criador",
    "Finalizador",
    "Destruidor",
    "Construtor",
    "Ala Ofensivo",
    "Volante",
]

# Palavras-chave nas features para inferir arquétipo
_FEATURE_ARCHETYPE_MAP: dict[str, str] = {
    "goals": "Finalizador",
    "shots": "Finalizador",
    "shots_on_target": "Finalizador",
    "assists": "Criador",
    "key_passes": "Criador",
    "through_balls": "Criador",
    "tackles": "Destruidor",
    "interceptions": "Destruidor",
    "blocks": "Destruidor",
    "clearances": "Destruidor",
    "pressures": "Destruidor",
    "passes_completed": "Construtor",
    "progressive_passes": "Construtor",
    "long_balls": "Construtor",
    "pass_completion_rate": "Construtor",
    "dribbles_completed": "Ala Ofensivo",
    "crosses": "Ala Ofensivo",
    "progressive_carries": "Ala Ofensivo",
    "carries": "Ala Ofensivo",
    "recoveries": "Volante",
    "aerial_duels_won": "Volante",
    "fouls_committed": "Volante",
}


# =========================================================================
# 1. Construção de perfis agregados
# =========================================================================

def build_player_profiles(
    session: Session,
    competition: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Constrói perfis agregados de jogadores a partir de métricas por partida.

    Consulta a tabela ``player_match_metrics`` no banco de dados, agrega por
    jogador (média entre partidas) e normaliza as métricas aplicáveis por
    90 minutos.

    Parameters
    ----------
    session:
        Sessão SQLAlchemy conectada ao banco.
    competition:
        Filtro opcional de competição (ex.: ``"La Liga"``).
    season:
        Filtro opcional de temporada (ex.: ``"2023/2024"``).

    Returns
    -------
    pd.DataFrame
        DataFrame com ``player_id``, ``player_name``, ``team`` e colunas de
        métricas agregadas.
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

    # Colunas de identificação
    id_cols = ["player_id", "player_name", "team"]
    metric_cols = [c for c in df.columns if c not in id_cols + ["match_id"]]

    # Agregar por jogador (média entre partidas)
    agg_df = df.groupby(id_cols, as_index=False)[metric_cols].mean()

    # Normalização por 90 minutos
    if "minutes_played" in agg_df.columns:
        per90_cols_present = [c for c in _PER90_COLUMNS if c in agg_df.columns]
        for col in per90_cols_present:
            agg_df[col] = agg_df[col] / (agg_df["minutes_played"] / 90.0)
        # Substituir inf/NaN resultantes de divisão por zero
        agg_df.replace([np.inf, -np.inf], 0.0, inplace=True)
        agg_df.fillna(0.0, inplace=True)

    return agg_df


# =========================================================================
# 2. Geração de embeddings via PCA
# =========================================================================

def generate_embeddings(
    profiles_df: pd.DataFrame,
    n_components: int = 16,
) -> tuple[pd.DataFrame, PCA]:
    """Gera embeddings de estilo de jogo via StandardScaler + PCA.

    Parameters
    ----------
    profiles_df:
        DataFrame retornado por :func:`build_player_profiles`.
    n_components:
        Número de componentes principais a manter.

    Returns
    -------
    tuple[pd.DataFrame, PCA]
        - DataFrame com ``player_id``, ``player_name``, ``team`` e coluna
          ``embedding`` (lista de floats).
        - Objeto PCA ajustado (contém ``explained_variance_ratio_``).
    """
    id_cols = ["player_id", "player_name", "team"]
    numeric_cols = [
        c for c in profiles_df.select_dtypes(include=[np.number]).columns
        if c not in id_cols
    ]

    X = profiles_df[numeric_cols].fillna(0.0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Guardar referências auxiliares no objeto PCA para uso posterior
    pca.feature_names_ = numeric_cols  # type: ignore[attr-defined]
    pca.scaler_ = scaler  # type: ignore[attr-defined]

    result_df = profiles_df[id_cols].copy()
    result_df["embedding"] = [row.tolist() for row in X_pca]

    return result_df, pca


# =========================================================================
# 3. Clusterização e detecção de arquétipos
# =========================================================================

def _infer_archetype(
    centroid: np.ndarray,
    pca: PCA,
) -> str:
    """Infere o nome do arquétipo a partir do centróide do cluster.

    Reconstrói o centróide no espaço original das features via transformada
    inversa do PCA, identifica as 3 features com maior peso absoluto e mapeia
    para um arquétipo tático.
    """
    reconstructed = pca.inverse_transform(centroid.reshape(1, -1))[0]
    feature_names: list[str] = pca.feature_names_  # type: ignore[attr-defined]

    top_indices = np.argsort(np.abs(reconstructed))[::-1][:3]
    top_features = [feature_names[i] for i in top_indices]

    # Contagem de votos por arquétipo
    votes: dict[str, int] = {}
    for feat in top_features:
        archetype = _FEATURE_ARCHETYPE_MAP.get(feat)
        if archetype:
            votes[archetype] = votes.get(archetype, 0) + 1

    if votes:
        return max(votes, key=lambda k: votes[k])

    # Fallback: usar nome genérico
    return "Construtor"


def cluster_players(
    embeddings_df: pd.DataFrame,
    n_clusters: int = 6,
    pca: PCA | None = None,
) -> pd.DataFrame:
    """Agrupa jogadores em clusters táticos via KMeans.

    Para cada cluster, atribui um nome de arquétipo baseado na análise dos
    centróides (quais features originais mais contribuem).

    Parameters
    ----------
    embeddings_df:
        DataFrame com coluna ``embedding`` (lista de floats).
    n_clusters:
        Número de clusters.
    pca:
        Objeto PCA ajustado (necessário para inferir nomes de arquétipo).
        Se ``None``, usa nomes genéricos.

    Returns
    -------
    pd.DataFrame
        DataFrame original acrescido de ``cluster_label`` e ``archetype``.
    """
    X = np.array(embeddings_df["embedding"].tolist())
    n_clusters = min(n_clusters, len(X))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    # Mapear clusters → nomes de arquétipo
    archetype_map: dict[int, str] = {}
    for i, centroid in enumerate(kmeans.cluster_centers_):
        if pca is not None and hasattr(pca, "feature_names_"):
            archetype_map[i] = _infer_archetype(centroid, pca)
        else:
            archetype_map[i] = _ARCHETYPE_NAMES[i % len(_ARCHETYPE_NAMES)]

    # Desambiguar nomes repetidos adicionando sufixo numérico
    seen: dict[str, int] = {}
    for k, v in archetype_map.items():
        if v in seen:
            seen[v] += 1
            archetype_map[k] = f"{v} {seen[v]}"
        else:
            seen[v] = 1

    result_df = embeddings_df.copy()
    result_df["cluster_label"] = labels
    result_df["archetype"] = [archetype_map[label] for label in labels]

    return result_df


# =========================================================================
# 4. Busca por similaridade (pgvector)
# =========================================================================

def find_similar(
    session: Session,
    player_name: str,
    season: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Encontra jogadores com estilo de jogo mais parecido via pgvector.

    Utiliza distância cosseno (operador ``<=>``) na tabela
    ``player_embeddings``.

    Parameters
    ----------
    session:
        Sessão SQLAlchemy.
    player_name:
        Nome do jogador de referência.
    season:
        Temporada para filtrar.
    limit:
        Número máximo de resultados.

    Returns
    -------
    pd.DataFrame
        Colunas: ``player_name``, ``team``, ``archetype``, ``distance``,
        ``similarity``.
    """
    query = text("""
        SELECT player_name, team, archetype,
               embedding <=> (
                   SELECT embedding
                   FROM player_embeddings
                   WHERE player_name = :name AND season = :season
               ) AS distance
        FROM player_embeddings
        WHERE season = :season AND player_name != :name
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
) -> pd.DataFrame:
    """Encontra jogadores com perfil complementar (mais dissimilar).

    Ordena pela maior distância cosseno, retornando jogadores que cobrem
    características opostas ao jogador de referência.

    Parameters
    ----------
    session:
        Sessão SQLAlchemy.
    player_name:
        Nome do jogador de referência.
    season:
        Temporada para filtrar.
    limit:
        Número máximo de resultados.

    Returns
    -------
    pd.DataFrame
        Colunas: ``player_name``, ``team``, ``archetype``, ``distance``,
        ``similarity``.
    """
    query = text("""
        SELECT player_name, team, archetype,
               embedding <=> (
                   SELECT embedding
                   FROM player_embeddings
                   WHERE player_name = :name AND season = :season
               ) AS distance
        FROM player_embeddings
        WHERE season = :season AND player_name != :name
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
# 6. Recomendação por perfil sintético
# =========================================================================

def recommend_by_profile(
    session: Session,
    profile: dict[str, float],
    season: str,
    limit: int = 10,
    pca_pickle_path: str | None = None,
    pca: PCA | None = None,
) -> pd.DataFrame:
    """Recomenda jogadores mais próximos de um perfil tático desejado.

    Recebe um dicionário de pesos de features, constrói um embedding sintético
    usando o PCA/scaler salvos e consulta os vizinhos mais próximos via
    pgvector.

    Parameters
    ----------
    session:
        Sessão SQLAlchemy.
    profile:
        Dicionário ``{feature_name: valor_desejado}``, ex.:
        ``{"goals": 0.8, "tackles": 0.2, "key_passes": 0.9}``.
    season:
        Temporada para filtrar.
    limit:
        Número máximo de resultados.
    pca_pickle_path:
        Caminho para o arquivo pickle contendo o PCA ajustado. Ignorado se
        ``pca`` for fornecido diretamente.
    pca:
        Objeto PCA ajustado (alternativa ao pickle). Deve conter os atributos
        ``feature_names_`` e ``scaler_`` anexados durante
        :func:`generate_embeddings`.

    Returns
    -------
    pd.DataFrame
        Colunas: ``player_name``, ``team``, ``archetype``, ``distance``,
        ``similarity``.
    """
    # Carregar PCA
    if pca is None:
        if pca_pickle_path is None:
            raise ValueError(
                "É necessário fornecer 'pca' ou 'pca_pickle_path'."
            )
        with open(pca_pickle_path, "rb") as f:
            pca = pickle.load(f)

    feature_names: list[str] = pca.feature_names_  # type: ignore[attr-defined]
    scaler: StandardScaler = pca.scaler_  # type: ignore[attr-defined]

    # Construir vetor no espaço original das features
    raw_vector = np.zeros(len(feature_names))
    for feat, value in profile.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            raw_vector[idx] = value

    # Escalonar e projetar no espaço PCA
    scaled_vector = scaler.transform(raw_vector.reshape(1, -1))
    embedding = pca.transform(scaled_vector)[0].tolist()

    # Formatar como literal de array para pgvector
    embedding_literal = "[" + ",".join(str(v) for v in embedding) + "]"

    query = text("""
        SELECT player_name, team, archetype,
               embedding <=> :embedding::vector AS distance
        FROM player_embeddings
        WHERE season = :season
        ORDER BY distance
        LIMIT :limit
    """)

    result = session.execute(
        query,
        {"embedding": embedding_literal, "season": season, "limit": limit},
    )
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if not df.empty:
        df["similarity"] = 1.0 - df["distance"]

    return df
