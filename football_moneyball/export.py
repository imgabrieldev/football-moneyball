"""Módulo de exportação de relatórios de scout.

Gera relatórios completos em formato Markdown e JSON, incluindo métricas
absolutas, percentis, arquétipo tático, jogadores similares, análise de
rede de passes, RAPM e compatibilidade com elencos-alvo.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from football_moneyball.db import (
    Match,
    PlayerMatchMetrics,
    PlayerEmbedding,
    PassNetwork,
    Stint,
    find_similar_players,
    get_player_metrics,
)

# ---------------------------------------------------------------------------
# Metric categories for display grouping
# ---------------------------------------------------------------------------

_METRIC_CATEGORIES = {
    "Ataque": [
        "goals", "shots", "shots_on_target", "xg",
        "dribbles_attempted", "dribbles_completed",
    ],
    "Criação": [
        "assists", "xa", "key_passes", "through_balls", "crosses",
        "progressive_passes", "progressive_carries",
    ],
    "Defesa": [
        "tackles", "interceptions",
        "blocks", "clearances", "aerials_won", "aerials_lost",
        "pressures", "pressure_regains",
    ],
    "Posse": [
        "passes", "passes_completed", "pass_pct",
        "touches", "carries", "dispossessed",
        "fouls_committed", "fouls_won",
    ],
}

# All metric columns from PlayerMatchMetrics (excluding identifiers)
_ALL_METRIC_COLS = [
    "minutes_played", "goals", "assists", "shots", "shots_on_target",
    "xg", "xa", "passes", "passes_completed", "pass_pct",
    "progressive_passes", "progressive_carries",
    "key_passes", "through_balls", "crosses",
    "tackles", "interceptions",
    "blocks", "clearances", "aerials_won", "aerials_lost",
    "fouls_committed", "fouls_won",
    "dribbles_attempted", "dribbles_completed",
    "touches", "carries", "dispossessed",
    "pressures", "pressure_regains",
]

# Friendly Portuguese labels for metrics
_METRIC_LABELS = {
    "minutes_played": "Minutos Jogados",
    "goals": "Gols",
    "assists": "Assistências",
    "shots": "Finalizações",
    "shots_on_target": "Finalizações no Alvo",
    "xg": "Gols Esperados (xG)",
    "xa": "Assistências Esperadas (xA)",
    "passes": "Passes Tentados",
    "passes_completed": "Passes Completados",
    "pass_pct": "Precisão de Passe (%)",
    "progressive_passes": "Passes Progressivos",
    "progressive_carries": "Conduções Progressivas",
    "key_passes": "Passes-Chave",
    "through_balls": "Bolas Enfiadas",
    "crosses": "Cruzamentos",
    "tackles": "Desarmes",
    "interceptions": "Interceptações",
    "blocks": "Bloqueios",
    "clearances": "Cortes",
    "aerials_won": "Duelos Aéreos Ganhos",
    "aerials_lost": "Duelos Aéreos Perdidos",
    "fouls_committed": "Faltas Cometidas",
    "fouls_won": "Faltas Sofridas",
    "dribbles_attempted": "Dribles Tentados",
    "dribbles_completed": "Dribles Completados",
    "touches": "Toques na Bola",
    "carries": "Conduções",
    "dispossessed": "Desapossado",
    "pressures": "Pressões",
    "pressure_regains": "Recuperações por Pressão",
}


# ---------------------------------------------------------------------------
# Numpy / pandas type conversion helper
# ---------------------------------------------------------------------------

def _convert_numpy(obj: Any) -> Any:
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Percentile bar helper
# ---------------------------------------------------------------------------

def _percentile_bar(pct: float, width: int = 10) -> str:
    """Build a text-based percentile bar like: ████████░░ 80%"""
    filled = round(pct / 100 * width)
    empty = width - filled
    return "█" * filled + "░" * empty + f" {pct:.0f}%"


# =========================================================================
# 1. generate_scout_report
# =========================================================================

def generate_scout_report(
    session: Session,
    player_name: str,
    season: str = None,
    target_team: str = None,
) -> dict:
    """Build a comprehensive scout report for a player.

    Parameters
    ----------
    session:
        SQLAlchemy session connected to the database.
    player_name:
        Name of the player to scout.
    season:
        Season filter (e.g. ``"2023/2024"``). If ``None``, uses all
        available seasons.
    target_team:
        If provided, compute compatibility scores between the player and
        every player in the target team's roster.

    Returns
    -------
    dict
        Full report dictionary with sections: ``player_info``, ``metrics``,
        ``percentiles``, ``archetype``, ``similar_players``,
        ``network_stats``, ``rapm``, and optionally ``compatibility``.
    """
    report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Player info & aggregated metrics
    # ------------------------------------------------------------------
    metrics_df = get_player_metrics(session, player_name, season)

    if metrics_df.empty:
        report["player_info"] = {
            "name": player_name,
            "team": None,
            "competition": None,
            "season": season,
        }
        report["metrics"] = {}
        report["percentiles"] = {}
    else:
        team = metrics_df["team"].mode().iloc[0] if not metrics_df["team"].mode().empty else None

        # Determine competition from matches table
        match_ids = metrics_df["match_id"].unique().tolist()
        competition = None
        if match_ids:
            match_row = session.query(Match).filter(
                Match.match_id == match_ids[0]
            ).first()
            if match_row:
                competition = match_row.competition

        # Determine the effective season
        effective_season = season
        if effective_season is None and match_ids:
            match_row = session.query(Match).filter(
                Match.match_id == match_ids[0]
            ).first()
            if match_row:
                effective_season = match_row.season

        report["player_info"] = {
            "name": player_name,
            "team": team,
            "competition": competition,
            "season": effective_season,
        }

        # Aggregate metrics (mean across matches)
        available_cols = [c for c in _ALL_METRIC_COLS if c in metrics_df.columns]
        agg = metrics_df[available_cols].mean()
        report["metrics"] = {col: round(float(agg[col]), 2) for col in available_cols if pd.notna(agg[col])}

        # ------------------------------------------------------------------
        # Percentiles vs all players in the same season
        # ------------------------------------------------------------------
        season_filter = effective_season
        if season_filter:
            all_query = (
                session.query(PlayerMatchMetrics)
                .join(Match, Match.match_id == PlayerMatchMetrics.match_id)
                .filter(Match.season == season_filter)
            )
        else:
            all_query = session.query(PlayerMatchMetrics)

        all_rows = all_query.all()
        if all_rows:
            columns = [c.key for c in PlayerMatchMetrics.__table__.columns]
            all_data = [{col: getattr(r, col) for col in columns} for r in all_rows]
            all_df = pd.DataFrame(all_data)

            # Aggregate per player
            id_cols = ["player_id", "player_name", "team"]
            metric_cols_present = [c for c in available_cols if c in all_df.columns]
            all_agg = all_df.groupby(id_cols, as_index=False)[metric_cols_present].mean()

            percentiles = {}
            for col in metric_cols_present:
                col_values = all_agg[col].dropna()
                if col_values.empty:
                    percentiles[col] = 0.0
                    continue
                player_val = report["metrics"].get(col, 0.0)
                # Manual percentile calculation
                count_below = (col_values < player_val).sum()
                count_equal = (col_values == player_val).sum()
                pct = ((count_below + 0.5 * count_equal) / len(col_values)) * 100
                percentiles[col] = round(float(pct), 1)

            report["percentiles"] = percentiles
        else:
            report["percentiles"] = {}

    # ------------------------------------------------------------------
    # Archetype from player_embeddings
    # ------------------------------------------------------------------
    emb_query = session.query(PlayerEmbedding).filter(
        PlayerEmbedding.player_name == player_name
    )
    if season:
        emb_query = emb_query.filter(PlayerEmbedding.season == season)

    embedding_row = emb_query.first()
    if embedding_row:
        report["archetype"] = {
            "cluster_label": embedding_row.cluster_label,
            "archetype": embedding_row.archetype,
        }
    else:
        report["archetype"] = {"cluster_label": None, "archetype": None}

    # ------------------------------------------------------------------
    # Similar players (top 5 via pgvector)
    # ------------------------------------------------------------------
    effective_season = report["player_info"].get("season")
    if effective_season:
        similar_df = find_similar_players(
            session, player_name, effective_season, limit=5
        )
        if not similar_df.empty:
            report["similar_players"] = similar_df.to_dict(orient="records")
        else:
            report["similar_players"] = []
    else:
        report["similar_players"] = []

    # ------------------------------------------------------------------
    # Network stats (averaged across matches)
    # ------------------------------------------------------------------
    if not metrics_df.empty:
        match_ids = metrics_df["match_id"].unique().tolist()
        network_records = []

        for mid in match_ids:
            # Retrieve pass network edges for this match
            edges = session.query(PassNetwork).filter(
                PassNetwork.match_id == mid
            ).all()

            if not edges:
                continue

            # Reconstruct a simple directed graph
            import networkx as nx

            G = nx.DiGraph()
            for edge in edges:
                G.add_edge(edge.passer_id, edge.receiver_id, weight=edge.weight or 1.0)

            # Identify the player's node (by player_id)
            player_ids = metrics_df.loc[
                metrics_df["match_id"] == mid, "player_id"
            ].unique()
            if len(player_ids) == 0:
                continue
            pid = int(player_ids[0])

            if pid not in G:
                continue

            dc = nx.degree_centrality(G).get(pid, 0.0)
            bc = nx.betweenness_centrality(G, weight="weight").get(pid, 0.0)
            pr = nx.pagerank(G, weight="weight").get(pid, 0.0)

            network_records.append({
                "degree_centrality": dc,
                "betweenness_centrality": bc,
                "pagerank": pr,
            })

        if network_records:
            net_df = pd.DataFrame(network_records)
            report["network_stats"] = {
                "degree_centrality": round(float(net_df["degree_centrality"].mean()), 4),
                "betweenness_centrality": round(float(net_df["betweenness_centrality"].mean()), 4),
                "pagerank": round(float(net_df["pagerank"].mean()), 4),
            }
        else:
            report["network_stats"] = {}
    else:
        report["network_stats"] = {}

    # ------------------------------------------------------------------
    # RAPM (if available from stints)
    # ------------------------------------------------------------------
    rapm_value = _compute_rapm_from_stints(session, player_name, effective_season)
    report["rapm"] = rapm_value

    # ------------------------------------------------------------------
    # Compatibility with target team
    # ------------------------------------------------------------------
    if target_team and effective_season and embedding_row:
        compatibility = _compute_compatibility(
            session, player_name, effective_season, target_team
        )
        report["compatibility"] = compatibility
    elif target_team:
        report["compatibility"] = []

    return report


# ---------------------------------------------------------------------------
# RAPM helper
# ---------------------------------------------------------------------------

def _compute_rapm_from_stints(
    session: Session,
    player_name: str,
    season: str = None,
) -> Optional[float]:
    """Attempt to retrieve or compute a simple RAPM proxy from stint data.

    Looks for pre-computed RAPM stored as a JSONB field or computes a
    simplified plus-minus from stint xG differentials.
    """
    # Find the player's id
    player_row = session.query(PlayerMatchMetrics).filter(
        PlayerMatchMetrics.player_name == player_name
    ).first()

    if not player_row:
        return None

    pid = player_row.player_id

    # Get match ids for the season
    if season:
        matches = (
            session.query(Match.match_id, Match.home_team, Match.away_team)
            .filter(Match.season == season)
            .all()
        )
    else:
        matches = session.query(Match.match_id, Match.home_team, Match.away_team).all()

    if not matches:
        return None

    plus_minus_values = []

    for match_id, home_team, away_team in matches:
        stints = session.query(Stint).filter(Stint.match_id == match_id).all()

        for stint in stints:
            home_ids = stint.home_player_ids or []
            away_ids = stint.away_player_ids or []
            duration = stint.duration_minutes or 0.0
            home_xg = stint.home_xg or 0.0
            away_xg = stint.away_xg or 0.0

            if duration <= 0:
                continue

            if pid in home_ids:
                # Player is on the home side
                plus_minus_values.append(home_xg - away_xg)
            elif pid in away_ids:
                # Player is on the away side
                plus_minus_values.append(away_xg - home_xg)

    if plus_minus_values:
        return round(float(np.mean(plus_minus_values)), 4)

    return None


# ---------------------------------------------------------------------------
# Compatibility helper
# ---------------------------------------------------------------------------

def _compute_compatibility(
    session: Session,
    player_name: str,
    season: str,
    target_team: str,
) -> list[dict]:
    """Compute compatibility scores between the player and target team roster.

    Uses cosine distance from the embedding space: lower distance means
    more similar style, higher distance means more complementary.
    """
    query = text("""
        SELECT
            pe2.player_name,
            pe2.archetype,
            pe2.embedding <=> pe1.embedding AS distance
        FROM player_embeddings pe1
        JOIN player_embeddings pe2
            ON pe2.player_id != pe1.player_id
            AND pe2.season = pe1.season
        WHERE pe1.player_name = :player_name
          AND pe1.season = :season
          AND pe2.player_name IN (
              SELECT DISTINCT pmm.player_name
              FROM player_match_metrics pmm
              JOIN matches m ON m.match_id = pmm.match_id
              WHERE pmm.team = :target_team AND m.season = :season
          )
        ORDER BY distance
    """)

    result = session.execute(query, {
        "player_name": player_name,
        "season": season,
        "target_team": target_team,
    })

    rows = result.fetchall()
    compatibility = []
    for row in rows:
        name, archetype, distance = row
        compatibility.append({
            "player_name": name,
            "archetype": archetype,
            "distance": round(float(distance), 4),
            "similarity": round(1.0 - float(distance), 4),
        })

    return compatibility


# =========================================================================
# 2. format_metrics_table
# =========================================================================

def format_metrics_table(metrics: dict, percentiles: dict) -> str:
    """Create a formatted metrics table grouped by category.

    Parameters
    ----------
    metrics:
        Dictionary of ``{metric_name: absolute_value}``.
    percentiles:
        Dictionary of ``{metric_name: percentile_value}``.

    Returns
    -------
    str
        A Markdown-formatted table string with columns for metric label,
        absolute value, and percentile bar.
    """
    lines = []

    for category, metric_keys in _METRIC_CATEGORIES.items():
        # Filter to metrics that actually have values
        category_metrics = [m for m in metric_keys if m in metrics]
        if not category_metrics:
            continue

        lines.append(f"\n### {category}\n")
        lines.append("| Métrica | Valor | Percentil |")
        lines.append("|---------|------:|-----------|")

        for m in category_metrics:
            label = _METRIC_LABELS.get(m, m)
            value = metrics[m]
            pct = percentiles.get(m, 0.0)
            bar = _percentile_bar(pct)
            lines.append(f"| {label} | {value:.2f} | {bar} |")

    return "\n".join(lines)


# =========================================================================
# 3. report_to_markdown
# =========================================================================

def report_to_markdown(report: dict) -> str:
    """Convert a scout report dictionary to a formatted Markdown string.

    Parameters
    ----------
    report:
        Report dictionary as returned by :func:`generate_scout_report`.

    Returns
    -------
    str
        Markdown-formatted report in Brazilian Portuguese.
    """
    info = report.get("player_info", {})
    player_name = info.get("name", "Desconhecido")

    lines = []
    lines.append(f"# Relatório de Scout: {player_name}\n")

    # -- Informações Gerais ------------------------------------------------
    lines.append("## Informações Gerais\n")
    lines.append(f"- **Jogador:** {player_name}")
    lines.append(f"- **Equipe:** {info.get('team', 'N/A')}")
    lines.append(f"- **Competição:** {info.get('competition', 'N/A')}")
    lines.append(f"- **Temporada:** {info.get('season', 'N/A')}")
    lines.append("")

    # -- Métricas ----------------------------------------------------------
    metrics = report.get("metrics", {})
    percentiles = report.get("percentiles", {})

    if metrics:
        lines.append("## Métricas (Absoluto | Percentil)\n")
        lines.append(format_metrics_table(metrics, percentiles))
        lines.append("")

    # -- Perfil Tático -----------------------------------------------------
    archetype_info = report.get("archetype", {})
    lines.append("## Perfil Tático\n")
    lines.append(f"- **Arquétipo:** {archetype_info.get('archetype', 'N/A')}")
    lines.append(f"- **Cluster:** {archetype_info.get('cluster_label', 'N/A')}")
    lines.append("")

    # -- Rede de Passes ----------------------------------------------------
    network = report.get("network_stats", {})
    lines.append("## Posição na Rede de Passes\n")
    if network:
        lines.append("| Métrica | Valor |")
        lines.append("|---------|------:|")
        lines.append(f"| Centralidade de Grau | {network.get('degree_centrality', 0.0):.4f} |")
        lines.append(f"| Centralidade de Intermediação | {network.get('betweenness_centrality', 0.0):.4f} |")
        lines.append(f"| PageRank | {network.get('pagerank', 0.0):.4f} |")
    else:
        lines.append("Dados de rede de passes indisponíveis.")
    lines.append("")

    # -- Jogadores Similares -----------------------------------------------
    similar = report.get("similar_players", [])
    lines.append("## Top 5 Jogadores Similares\n")
    if similar:
        lines.append("| # | Jogador | Arquétipo | Distância |")
        lines.append("|---|---------|-----------|----------:|")
        for i, sp in enumerate(similar[:5], 1):
            name = sp.get("player_name", "N/A")
            arch = sp.get("archetype", "N/A")
            dist = sp.get("distance", 0.0)
            lines.append(f"| {i} | {name} | {arch} | {dist:.4f} |")
    else:
        lines.append("Nenhum jogador similar encontrado.")
    lines.append("")

    # -- RAPM --------------------------------------------------------------
    rapm = report.get("rapm")
    lines.append("## RAPM - Impacto Individual\n")
    if rapm is not None:
        sign = "+" if rapm >= 0 else ""
        lines.append(f"**RAPM (xG +/-):** {sign}{rapm:.4f}")
    else:
        lines.append("Dados de RAPM indisponíveis.")
    lines.append("")

    # -- Compatibilidade ---------------------------------------------------
    compatibility = report.get("compatibility")
    if compatibility is not None:
        # Infer target team name from the first entry or from the report
        target_team = "Equipe-Alvo"
        lines.append(f"## Compatibilidade com {target_team}\n")
        if compatibility:
            lines.append("| Jogador | Arquétipo | Distância | Similaridade |")
            lines.append("|---------|-----------|----------:|-------------:|")
            for cp in compatibility:
                lines.append(
                    f"| {cp.get('player_name', 'N/A')} "
                    f"| {cp.get('archetype', 'N/A')} "
                    f"| {cp.get('distance', 0.0):.4f} "
                    f"| {cp.get('similarity', 0.0):.4f} |"
                )
        else:
            lines.append("Nenhum dado de compatibilidade encontrado.")
        lines.append("")

    return "\n".join(lines)


# =========================================================================
# 4. report_to_json
# =========================================================================

def report_to_json(report: dict) -> str:
    """Serialize a scout report dictionary to a JSON string.

    Converts numpy/pandas types to Python native types before serialization.

    Parameters
    ----------
    report:
        Report dictionary as returned by :func:`generate_scout_report`.

    Returns
    -------
    str
        JSON string with ``indent=2`` and ``ensure_ascii=False``.
    """
    cleaned = _convert_numpy(report)
    return json.dumps(cleaned, indent=2, ensure_ascii=False)


# =========================================================================
# 5. save_report
# =========================================================================

def save_report(
    report: dict,
    output_path: str,
    format: str = "markdown",
) -> None:
    """Save a scout report to a file.

    Parameters
    ----------
    report:
        Report dictionary as returned by :func:`generate_scout_report`.
    output_path:
        Destination file path.
    format:
        Output format: ``"markdown"`` or ``"json"``. If not specified,
        the format is auto-detected from the file extension (``.md`` for
        markdown, ``.json`` for JSON).
    """
    # Auto-detect format from extension
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    if ext == ".json":
        fmt = "json"
    elif ext in (".md", ".markdown"):
        fmt = "markdown"
    else:
        fmt = format

    if fmt == "json":
        content = report_to_json(report)
    else:
        content = report_to_markdown(report)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
