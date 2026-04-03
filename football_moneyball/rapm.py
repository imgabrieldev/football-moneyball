"""Módulo RAPM (Regularized Adjusted Plus-Minus) para o Football Moneyball.

Implementa o modelo de Plus-Minus Ajustado Regularizado usando regressão Ridge
para isolar o impacto individual de cada jogador, baseado em dados de eventos
do StatsBomb.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from statsbombpy import sb
from sqlalchemy.orm import Session

from football_moneyball.db import (
    Match,
    PlayerMatchMetrics,
    Stint,
    upsert_stints,
)


# ---------------------------------------------------------------------------
# Reconstrução de stints a partir de eventos StatsBomb
# ---------------------------------------------------------------------------

def reconstruct_stints(match_id: int) -> pd.DataFrame:
    """Reconstrói os stints (períodos com mesmos 22 jogadores) de uma partida.

    Busca os eventos da partida via StatsBomb e identifica os momentos em que
    a composição dos jogadores em campo muda (substituições, início de tempo).
    Para cada stint, calcula o diferencial de xG (casa - fora) e a duração
    em minutos.

    Parâmetros
    ----------
    match_id : int
        Identificador da partida no StatsBomb.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: match_id, stint_number, home_player_ids,
        away_player_ids, duration_minutes, home_xg, away_xg, xg_diff.
    """
    events = sb.events(match_id=match_id)
    lineups = sb.lineups(match_id=match_id)

    # Identificar times (home / away)
    teams = events["team"].unique()
    # O primeiro evento de tipo "Starting XI" nos dá a ordem home/away
    starting_xi_events = events[events["type"] == "Starting XI"].sort_values("index")

    if len(starting_xi_events) < 2:
        return pd.DataFrame()

    home_team = starting_xi_events.iloc[0]["team"]
    away_team = starting_xi_events.iloc[1]["team"]

    # Extrair escalações iniciais (titulares) a partir dos lineups do StatsBomb
    home_lineup = lineups[home_team]
    away_lineup = lineups[away_team]

    # Jogadores titulares são aqueles com posição listada nas tactics do Starting XI
    home_tactics = starting_xi_events.iloc[0].get("tactics", {})
    away_tactics = starting_xi_events.iloc[1].get("tactics", {})

    if isinstance(home_tactics, dict) and "lineup" in home_tactics:
        home_players = {
            p["player"]["id"] for p in home_tactics["lineup"]
        }
    else:
        # Fallback: pegar os primeiros 11 do lineup
        home_players = set(home_lineup["player_id"].head(11).tolist())

    if isinstance(away_tactics, dict) and "lineup" in away_tactics:
        away_players = {
            p["player"]["id"] for p in away_tactics["lineup"]
        }
    else:
        away_players = set(away_lineup["player_id"].head(11).tolist())

    # Ordenar eventos cronologicamente
    events = events.sort_values(["period", "minute", "second", "index"]).reset_index(
        drop=True
    )

    # Calcular timestamp contínuo em minutos (considerando períodos)
    period_offsets = {1: 0, 2: 45, 3: 90, 4: 105}
    events["timestamp_minutes"] = events.apply(
        lambda row: period_offsets.get(row["period"], 0)
        + row["minute"]
        + row.get("second", 0) / 60.0,
        axis=1,
    )

    # Identificar eventos que delimitam stints
    # Substituições
    subs = events[events["type"] == "Substitution"].copy()

    # Half Start (início de período)
    half_starts = events[events["type"] == "Half Start"].copy()

    # Coletar pontos de corte (timestamps onde a composição muda)
    boundary_times: list[float] = []

    # Início de cada período como boundary
    for _, hs in half_starts.iterrows():
        boundary_times.append(hs["timestamp_minutes"])

    # Cada substituição como boundary
    for _, sub_event in subs.iterrows():
        boundary_times.append(sub_event["timestamp_minutes"])

    boundary_times = sorted(set(boundary_times))

    # Se não houver boundaries, usar início e fim da partida
    if not boundary_times:
        boundary_times = [0.0]

    # Fim da partida
    match_end = events["timestamp_minutes"].max()

    # Construir intervalos de stints
    intervals: list[tuple[float, float]] = []
    for i in range(len(boundary_times)):
        start = boundary_times[i]
        end = boundary_times[i + 1] if i + 1 < len(boundary_times) else match_end
        if end > start:
            intervals.append((start, end))

    # Também incluir intervalo antes da primeira boundary se > 0
    if boundary_times[0] > 0:
        intervals.insert(0, (0.0, boundary_times[0]))

    # Processar substituições em ordem cronológica para rastrear composição
    subs_sorted = subs.sort_values("timestamp_minutes").reset_index(drop=True)

    # Mapear substituições por timestamp para saber o estado do elenco em cada stint
    sub_records: list[dict] = []
    for _, sub_event in subs_sorted.iterrows():
        sub_team = sub_event["team"]
        player_out = sub_event.get("player_id", None)
        # O jogador que entra está em substitution_replacement
        replacement_info = sub_event.get("substitution_replacement", {})
        if isinstance(replacement_info, dict):
            player_in = replacement_info.get("id", None)
        else:
            player_in = None

        sub_records.append(
            {
                "timestamp": sub_event["timestamp_minutes"],
                "team": sub_team,
                "player_out": player_out,
                "player_in": player_in,
            }
        )

    # Chutes com xG para calcular xG por stint
    shots = events[events["type"] == "Shot"].copy()
    if "shot_statsbomb_xg" in shots.columns:
        shots["xg_value"] = shots["shot_statsbomb_xg"].fillna(0.0)
    else:
        shots["xg_value"] = 0.0

    # Construir stints
    stints_data: list[dict] = []
    current_home = set(home_players)
    current_away = set(away_players)

    for stint_idx, (start_t, end_t) in enumerate(intervals):
        # Aplicar substituições que ocorreram exatamente no início deste stint
        for sub in sub_records:
            if abs(sub["timestamp"] - start_t) < 1e-6:
                if sub["team"] == home_team:
                    if sub["player_out"] in current_home and sub["player_in"]:
                        current_home.discard(sub["player_out"])
                        current_home.add(sub["player_in"])
                elif sub["team"] == away_team:
                    if sub["player_out"] in current_away and sub["player_in"]:
                        current_away.discard(sub["player_out"])
                        current_away.add(sub["player_in"])

        # Filtrar chutes dentro do intervalo do stint
        stint_shots = shots[
            (shots["timestamp_minutes"] >= start_t)
            & (shots["timestamp_minutes"] < end_t)
        ]

        home_xg = stint_shots[stint_shots["team"] == home_team]["xg_value"].sum()
        away_xg = stint_shots[stint_shots["team"] == away_team]["xg_value"].sum()

        duration = end_t - start_t

        stints_data.append(
            {
                "match_id": match_id,
                "stint_number": stint_idx,
                "home_player_ids": sorted(current_home),
                "away_player_ids": sorted(current_away),
                "duration_minutes": round(duration, 2),
                "home_xg": round(home_xg, 4),
                "away_xg": round(away_xg, 4),
                "xg_diff": round(home_xg - away_xg, 4),
            }
        )

    return pd.DataFrame(stints_data)


# ---------------------------------------------------------------------------
# Construção da matriz de design para RAPM
# ---------------------------------------------------------------------------

def build_rapm_matrix(
    stints_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Constrói a matriz de design X e o vetor alvo y para a regressão Ridge.

    Cada linha representa um stint. Para cada jogador, a coluna recebe +1 se
    ele estava em campo pelo time da casa, -1 se estava pelo time visitante,
    e 0 caso contrário. O vetor y é o diferencial de xG ponderado pela raiz
    quadrada da duração do stint.

    Parâmetros
    ----------
    stints_df : pd.DataFrame
        DataFrame de stints conforme retornado por ``reconstruct_stints``.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, list[int]]
        (X, y, player_ids) onde X é a matriz de design, y o vetor alvo
        ponderado e player_ids a lista ordenada de IDs de jogadores
        correspondente às colunas de X.
    """
    # Coletar todos os player_ids únicos
    all_player_ids: set[int] = set()
    for _, row in stints_df.iterrows():
        home_ids = row["home_player_ids"]
        away_ids = row["away_player_ids"]
        if isinstance(home_ids, (list, set)):
            all_player_ids.update(home_ids)
        if isinstance(away_ids, (list, set)):
            all_player_ids.update(away_ids)

    player_ids = sorted(all_player_ids)
    pid_to_idx = {pid: idx for idx, pid in enumerate(player_ids)}

    n_stints = len(stints_df)
    n_players = len(player_ids)

    X = np.zeros((n_stints, n_players), dtype=np.float64)
    y = np.zeros(n_stints, dtype=np.float64)

    for i, (_, row) in enumerate(stints_df.iterrows()):
        home_ids = row["home_player_ids"]
        away_ids = row["away_player_ids"]
        duration = row.get("duration_minutes", 1.0)
        xg_diff = row.get("xg_diff", 0.0)

        weight = np.sqrt(max(duration, 0.01))

        # +1 para jogadores da casa, -1 para visitantes (ponderados)
        if isinstance(home_ids, (list, set)):
            for pid in home_ids:
                if pid in pid_to_idx:
                    X[i, pid_to_idx[pid]] = weight

        if isinstance(away_ids, (list, set)):
            for pid in away_ids:
                if pid in pid_to_idx:
                    X[i, pid_to_idx[pid]] = -weight

        y[i] = xg_diff * weight

    return X, y, player_ids


# ---------------------------------------------------------------------------
# Ajuste do modelo Ridge (RAPM)
# ---------------------------------------------------------------------------

def fit_rapm(
    X: np.ndarray,
    y: np.ndarray,
    player_ids: list[int],
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Ajusta a regressão Ridge para estimar o RAPM de cada jogador.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de design (stints x jogadores).
    y : np.ndarray
        Vetor alvo (xG diferencial ponderado).
    player_ids : list[int]
        Lista de IDs de jogadores correspondente às colunas de X.
    alpha : float, optional
        Parâmetro de regularização da regressão Ridge. Padrão: 1.0.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: player_id, rapm_value, offensive_rapm,
        defensive_rapm.
    """
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y)

    results = pd.DataFrame(
        {
            "player_id": player_ids,
            "rapm_value": model.coef_,
        }
    )

    # Tentar calcular RAPM ofensivo/defensivo separadamente
    # Ofensivo: impacto no xG a favor; Defensivo: impacto no xG contra
    # Aproximação: usar o coeficiente total dividido proporcionalmente
    results["offensive_rapm"] = results["rapm_value"].clip(lower=0)
    results["defensive_rapm"] = results["rapm_value"].clip(upper=0)

    return results


# ---------------------------------------------------------------------------
# Validação cruzada para escolha do alpha ótimo
# ---------------------------------------------------------------------------

def cross_validate_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[list[float]] = None,
) -> float:
    """Encontra o melhor parâmetro de regularização alpha via validação cruzada.

    Utiliza RidgeCV do scikit-learn para testar diferentes valores de alpha
    e retornar aquele que minimiza o erro de validação cruzada.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de design.
    y : np.ndarray
        Vetor alvo.
    alphas : list[float], optional
        Lista de valores de alpha a testar. Se None, usa uma grade logarítmica
        padrão de 0.01 a 1000.

    Retorna
    -------
    float
        O valor de alpha com melhor desempenho na validação cruzada.
    """
    if alphas is None:
        alphas = np.logspace(-2, 3, num=50).tolist()

    model = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
    model.fit(X, y)

    return float(model.alpha_)


# ---------------------------------------------------------------------------
# Pipeline completo para uma temporada
# ---------------------------------------------------------------------------

def compute_season_rapm(
    session: Session,
    competition: str,
    season: str,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Calcula o RAPM para todos os jogadores de uma temporada.

    Orquestra o pipeline completo: busca partidas no banco de dados,
    reconstrói ou recupera stints, constrói a matriz de design e ajusta
    o modelo Ridge.

    Parâmetros
    ----------
    session : Session
        Sessão SQLAlchemy para acesso ao banco de dados.
    competition : str
        Nome da competição (ex: 'La Liga').
    season : str
        Temporada (ex: '2023/2024').
    alpha : float, optional
        Parâmetro de regularização. Padrão: 1.0.

    Retorna
    -------
    pd.DataFrame
        DataFrame com player_id, player_name, rapm_value, offensive_rapm,
        defensive_rapm, ordenado por rapm_value decrescente.
    """
    # Buscar todas as partidas da competição/temporada
    matches = (
        session.query(Match)
        .filter(Match.competition == competition, Match.season == season)
        .all()
    )

    if not matches:
        return pd.DataFrame()

    all_stints: list[pd.DataFrame] = []

    for match in matches:
        # Tentar buscar stints do banco de dados primeiro (cache)
        cached_stints = (
            session.query(Stint)
            .filter(Stint.match_id == match.match_id)
            .all()
        )

        if cached_stints:
            stint_rows = []
            for s in cached_stints:
                stint_rows.append(
                    {
                        "match_id": s.match_id,
                        "stint_number": s.stint_number,
                        "home_player_ids": s.home_player_ids or [],
                        "away_player_ids": s.away_player_ids or [],
                        "duration_minutes": s.duration_minutes or 0.0,
                        "home_xg": s.home_xg or 0.0,
                        "away_xg": s.away_xg or 0.0,
                        "xg_diff": (s.home_xg or 0.0) - (s.away_xg or 0.0),
                    }
                )
            stints_df = pd.DataFrame(stint_rows)
        else:
            # Reconstruir a partir do StatsBomb e persistir
            stints_df = reconstruct_stints(match.match_id)
            if not stints_df.empty:
                upsert_stints(session, stints_df, match.match_id)

        if not stints_df.empty:
            all_stints.append(stints_df)

    if not all_stints:
        return pd.DataFrame()

    combined_stints = pd.concat(all_stints, ignore_index=True)

    # Construir matriz e ajustar modelo
    X, y, player_ids = build_rapm_matrix(combined_stints)
    results = fit_rapm(X, y, player_ids, alpha=alpha)

    # Enriquecer com nomes dos jogadores a partir de player_match_metrics
    player_names = {}
    name_rows = (
        session.query(
            PlayerMatchMetrics.player_id, PlayerMatchMetrics.player_name
        )
        .filter(PlayerMatchMetrics.player_id.in_(player_ids))
        .distinct()
        .all()
    )
    for row in name_rows:
        player_names[row.player_id] = row.player_name

    results["player_name"] = results["player_id"].map(player_names)

    # Reordenar colunas e ordenar por RAPM decrescente
    cols = ["player_id", "player_name", "rapm_value", "offensive_rapm", "defensive_rapm"]
    results = results[cols].sort_values("rapm_value", ascending=False).reset_index(
        drop=True
    )

    return results
