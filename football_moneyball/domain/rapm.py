"""Modulo of dominio RAPM (Regularized Adjusted Plus-Minus).

Implementa o model of Plus-Minus Ajustado Regularizado usando regression Ridge
for isolate o impacto individual of each player.

Funcionalidades avancadas:
- Splints: stints delimitados tambem by goals (alem of substitutions and periodos)
- SPM Prior: prior informative baseado in box-score metrics (Statistical Plus-Minus)
- Offensive/Defensive Split: colunas separadas for impacto offensive and defensive
- Augmented Regression: incorporacao of the SPM prior via regression aumentada

Logica pura sobre DataFrames and arrays — without dependencias of I/O externo
(statsbombpy, sqlalchemy).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV


# ---------------------------------------------------------------------------
# Reconstrucao of stints (splints) from eventos
# ---------------------------------------------------------------------------

def reconstruct_stints(
    events: pd.DataFrame, lineups: dict
) -> pd.DataFrame:
    """Reconstroi os stints (periodos with same 22 players) of a match.

    Receives eventos and lineups ja carregados and identifies os momentos in que
    a composicao of the players in pitch muda (substitutions, inicio of tempo)
    or a goal is scored. For each stint, calcula o diferencial of xG
    (home - outside) is the duration in minutos.

    Gols criam novas boundaries ('splints'), permitindo analise mais granular
    of the impacto of the players in states of jogo diferentes (empate, lideranca,
    disadvantage). O pitch ``boundary_type`` records o that caused each boundary:
    'period_start', 'substitution' ou 'goal'.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame of eventos StatsBomb (retornado by sb.events() ou
        equivalente).
    lineups : dict
        Dicionario of lineups by time, in the formato retornado por
        sb.lineups() (chave = nome of the time, value = DataFrame com
        colunas player_id, positions, etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with colunas: stint_number, home_player_ids,
        away_player_ids, duration_minutes, home_xg, away_xg, xg_diff,
        boundary_type.
    """
    # Identificar times (home / away)
    # O first evento of tipo "Starting XI" in the of the a ordem home/away
    starting_xi_events = events[events["type"] == "Starting XI"].sort_values("index")

    if len(starting_xi_events) < 2:
        return pd.DataFrame()

    home_team = starting_xi_events.iloc[0]["team"]
    away_team = starting_xi_events.iloc[1]["team"]

    # Extrair escalacoes iniciais (titulares) from the lineups
    home_lineup = lineups[home_team]
    away_lineup = lineups[away_team]

    # Players titulares sao aqueles with posicao listada in the tactics of the Starting XI
    home_tactics = starting_xi_events.iloc[0].get("tactics", {})
    away_tactics = starting_xi_events.iloc[1].get("tactics", {})

    if isinstance(home_tactics, dict) and "lineup" in home_tactics:
        home_players = {
            p["player"]["id"] for p in home_tactics["lineup"]
        }
    else:
        # Fallback: pegar os first 11 of the lineup
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

    # Calcular timestamp continuo in minutos (considerando periodos)
    period_offsets = {1: 0, 2: 45, 3: 90, 4: 105}
    events["timestamp_minutes"] = events.apply(
        lambda row: period_offsets.get(row["period"], 0)
        + row["minute"]
        + row.get("second", 0) / 60.0,
        axis=1,
    )

    # Identificar eventos that delimitam stints
    # Substituicoes
    subs = events[events["type"] == "Substitution"].copy()

    # Half Start (inicio of periodo)
    half_starts = events[events["type"] == "Half Start"].copy()

    # Coletar points of corte (timestamps where a composicao muda)
    boundary_times: list[float] = []
    boundary_types: dict[float, str] = {}

    # Inicio of each periodo as boundary
    for _, hs in half_starts.iterrows():
        ts = hs["timestamp_minutes"]
        boundary_times.append(ts)
        boundary_types[ts] = "period_start"

    # Each substitution as boundary
    for _, sub_event in subs.iterrows():
        ts = sub_event["timestamp_minutes"]
        boundary_times.append(ts)
        boundary_types[ts] = "substitution"

    # Gols as boundaries (splints)
    if "shot_outcome" in events.columns:
        goal_events = events[
            (events["type"] == "Shot") & (events["shot_outcome"] == "Goal")
        ]
        for _, goal_event in goal_events.iterrows():
            ts = goal_event["timestamp_minutes"]
            boundary_times.append(ts)
            boundary_types[ts] = "goal"

    boundary_times = sorted(set(boundary_times))

    # If nao houver boundaries, usar inicio and fim of the match
    if not boundary_times:
        boundary_times = [0.0]
        boundary_types[0.0] = "period_start"

    # Fim of the match
    match_end = events["timestamp_minutes"].max()

    # Construir intervalos of stints
    intervals: list[tuple[float, float]] = []
    interval_boundary_types: list[str] = []
    for i in range(len(boundary_times)):
        start = boundary_times[i]
        end = boundary_times[i + 1] if i + 1 < len(boundary_times) else match_end
        if end > start:
            intervals.append((start, end))
            interval_boundary_types.append(boundary_types.get(start, "period_start"))

    # Tambem incluir intervalo antes of the first boundary se > 0
    if boundary_times[0] > 0:
        intervals.insert(0, (0.0, boundary_times[0]))
        interval_boundary_types.insert(0, "period_start")

    # Processar substitutions in ordem cronologica for rastrear composicao
    subs_sorted = subs.sort_values("timestamp_minutes").reset_index(drop=True)

    # Mapear substitutions by timestamp for saber o state of the roster in each stint
    sub_records: list[dict] = []
    for _, sub_event in subs_sorted.iterrows():
        sub_team = sub_event["team"]
        player_out = sub_event.get("player_id", None)
        # O player that entra esta in substitution_replacement
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

    # Chutes with xG for calcular xG by stint
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
        # Aplicar substitutions that ocorreram exatamente in the inicio deste stint
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

        # Filtrar shots within of the intervalo of the stint
        stint_shots = shots[
            (shots["timestamp_minutes"] >= start_t)
            & (shots["timestamp_minutes"] < end_t)
        ]

        home_xg = stint_shots[stint_shots["team"] == home_team]["xg_value"].sum()
        away_xg = stint_shots[stint_shots["team"] == away_team]["xg_value"].sum()

        duration = end_t - start_t

        b_type = (
            interval_boundary_types[stint_idx]
            if stint_idx < len(interval_boundary_types)
            else "period_start"
        )

        stints_data.append(
            {
                "stint_number": stint_idx,
                "home_player_ids": sorted(current_home),
                "away_player_ids": sorted(current_away),
                "duration_minutes": round(duration, 2),
                "home_xg": round(home_xg, 4),
                "away_xg": round(away_xg, 4),
                "xg_diff": round(home_xg - away_xg, 4),
                "boundary_type": b_type,
            }
        )

    return pd.DataFrame(stints_data)


# ---------------------------------------------------------------------------
# SPM Prior (Statistical Plus-Minus)
# ---------------------------------------------------------------------------

def compute_spm_prior(
    player_metrics_list: list,
    player_ids: list[int],
) -> np.ndarray:
    """Compute prior SPM (Statistical Plus-Minus) from box-score metrics.

    Usa metrics per-90 (goals, assists, xg, tackles, interceptions, key_passes)
    for generate a prior informative for each player. Esse prior is used na
    regression aumentada of the RAPM for estabilizar estimativas of players com
    poucos minutos amostrados.

    Parameters
    ----------
    player_metrics_list : list
        Lista of objetos/dicts with atributos: player_id, minutes_played,
        goals, assists, xg, key_passes, tackles, interceptions. Each
        elemento representa metrics of a match for a player.
    player_ids : list[int]
        Lista of IDs of players for the quais calcular o prior.

    Returns
    -------
    np.ndarray
        Array with the prior SPM for each player (same index of player_ids),
        centrado in 0.
    """
    if not player_metrics_list:
        return np.zeros(len(player_ids))

    # Aggregate per-90 stats
    pid_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    prior = np.zeros(len(player_ids))
    counts = np.zeros(len(player_ids))

    for m in player_metrics_list:
        # Support both dict-like and attribute access
        if isinstance(m, dict):
            _get = m.get
        else:
            _get = lambda key, default=None: getattr(m, key, default)  # noqa: E731

        idx = pid_to_idx.get(_get("player_id"))
        if idx is None:
            continue
        mins = _get("minutes_played") or 0.0
        if mins < 10:
            continue
        per90 = 90.0 / mins
        # Contribuicoes positivas (ofensivas and defensivas)
        score = (
            (_get("goals") or 0) * per90 * 1.0
            + (_get("assists") or 0) * per90 * 0.7
            + (_get("xg") or 0) * per90 * 0.5
            + (_get("key_passes") or 0) * per90 * 0.2
            # Contribuicoes defensivas (reduz xG adversario)
            + (_get("tackles") or 0) * per90 * 0.1
            + (_get("interceptions") or 0) * per90 * 0.15
        )
        prior[idx] += score
        counts[idx] += 1

    # Average across matches
    mask = counts > 0
    prior[mask] /= counts[mask]

    # Center around 0
    prior -= prior.mean()

    return prior


# ---------------------------------------------------------------------------
# Construcao of the matriz of design for RAPM
# ---------------------------------------------------------------------------

def build_rapm_matrix(
    stints_df: pd.DataFrame,
    offensive_defensive_split: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Builds the matriz of design X is the vetor alvo y for the regression Ridge.

    Each row representa a stint. In mode default, each player tem a column
    that recebe +weight if of the home, -weight if away team (weight = sqrt(duration)).
    O vetor y is the diferencial of xG weighted by the same raiz.

    In mode ``offensive_defensive_split``, each player gera duas colunas:
    a offensive (impacto in the xG a favor) and a defensive (impacto in the xG
    contra), permitindo separacao real of the RAPM offensive/defensive.

    Parameters
    ----------
    stints_df : pd.DataFrame
        DataFrame of stints conforme retornado por ``reconstruct_stints``.
    offensive_defensive_split : bool, optional
        If True, cria 2 colunas by player (offensive and defensive).
        Padrao: False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[int]]
        (X, y, player_ids) where X is the matriz of design, y o vetor alvo
        weighted and player_ids a lista ordenada of IDs of players
        correspondente as colunas of X. Quando ``offensive_defensive_split``
        and True, player_ids contem IDs intercalados: [p1_off, p1_def, p2_off,
        p2_def, ...] and X tem 2*n_players colunas.
    """
    # Coletar todos os player_ids unicos
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

    if offensive_defensive_split:
        # 2 colunas by player: [off, def] intercaladas
        n_cols = 2 * n_players
        X = np.zeros((n_stints, n_cols), dtype=np.float64)
        y = np.zeros(n_stints, dtype=np.float64)

        for i, (_, row) in enumerate(stints_df.iterrows()):
            home_ids = row["home_player_ids"]
            away_ids = row["away_player_ids"]
            duration = row.get("duration_minutes", 1.0)
            xg_diff = row.get("xg_diff", 0.0)

            weight = np.sqrt(max(duration, 0.01))

            # Players of the home: +weight in the column offensive
            if isinstance(home_ids, (list, set)):
                for pid in home_ids:
                    if pid in pid_to_idx:
                        base_idx = pid_to_idx[pid] * 2
                        X[i, base_idx] = weight       # offensive column

            # Players of the away team: -weight in the column defensive
            if isinstance(away_ids, (list, set)):
                for pid in away_ids:
                    if pid in pid_to_idx:
                        base_idx = pid_to_idx[pid] * 2
                        X[i, base_idx + 1] = -weight  # defensive column

            y[i] = xg_diff * weight

        # Gerar lista of player_ids with sufixos off/def
        split_player_ids = []
        for pid in player_ids:
            split_player_ids.append(pid)  # offensive
            split_player_ids.append(pid)  # defensive
        return X, y, split_player_ids

    else:
        # Modo default: 1 column by player
        X = np.zeros((n_stints, n_players), dtype=np.float64)
        y = np.zeros(n_stints, dtype=np.float64)

        for i, (_, row) in enumerate(stints_df.iterrows()):
            home_ids = row["home_player_ids"]
            away_ids = row["away_player_ids"]
            duration = row.get("duration_minutes", 1.0)
            xg_diff = row.get("xg_diff", 0.0)

            weight = np.sqrt(max(duration, 0.01))

            # +1 for players of the home, -1 for visitantes (weighted)
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
# Ajuste of the model Ridge (RAPM)
# ---------------------------------------------------------------------------

def fit_rapm(
    X: np.ndarray,
    y: np.ndarray,
    player_ids: list[int],
    alpha: float = 1.0,
    spm_prior: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fits a regression Ridge for estimate o RAPM of each player.

    When a prior SPM and fornecido, uses regression aumentada: empilha uma
    matriz identidade escalada abaixo of X is the prior escalado abaixo of y. Isso
    encolhe os coeficientes in direcao ao prior in vez of in direcao a zero,
    produzindo estimativas mais estaveis for players with poucos stints.

    Parameters
    ----------
    X : np.ndarray
        Matriz of design (stints x players).
    y : np.ndarray
        Vetor alvo (xG diferencial weighted).
    player_ids : list[int]
        Lista of IDs of players correspondente as colunas of X.
    alpha : float, optional
        Parametro of regularizacao of the regression Ridge. Padrao: 1.0.
    spm_prior : np.ndarray, optional
        Prior SPM for regression aumentada. Deve ter o same numero de
        elementos that colunas of X. If None, usa Ridge default (shrink p/ zero).

    Returns
    -------
    pd.DataFrame
        DataFrame with colunas: player_id, rapm_value, offensive_rapm,
        defensive_rapm.
    """
    if spm_prior is not None:
        # Regressao aumentada: encolhe in direcao ao prior SPM
        lambda_reg = alpha
        n_players = X.shape[1]
        X_aug = np.vstack([X, np.sqrt(lambda_reg) * np.eye(n_players)])
        y_aug = np.hstack([y, np.sqrt(lambda_reg) * spm_prior])
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_aug, y_aug)
    else:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, y)

    results = pd.DataFrame(
        {
            "player_id": player_ids,
            "rapm_value": model.coef_,
        }
    )

    # Tentar calcular RAPM offensive/defensive separadamente
    # Ofensivo: impacto in the xG a favor; Defensivo: impacto in the xG contra
    # Aproximacao: usar o coeficiente total dividido proporcionalmente
    results["offensive_rapm"] = results["rapm_value"].clip(lower=0)
    results["defensive_rapm"] = results["rapm_value"].clip(upper=0)

    return results


# ---------------------------------------------------------------------------
# Validacao cruzada for escolha of the alpha otimo
# ---------------------------------------------------------------------------

def cross_validate_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[list[float]] = None,
) -> float:
    """Encontra o melhor parameter of regularizacao alpha via validacao cruzada.

    Utiliza RidgeCV of the scikit-learn for testar diferentes values of alpha
    and retornar aquele that minimiza o erro of validacao cruzada.

    Parameters
    ----------
    X : np.ndarray
        Matriz of design.
    y : np.ndarray
        Vetor alvo.
    alphas : list[float], optional
        Lista of values of alpha a testar. If None, usa a grade logaritmica
        default of 0.01 a 1000.

    Returns
    -------
    float
        O value of alpha with melhor desempenho in the validacao cruzada.
    """
    if alphas is None:
        alphas = np.logspace(-2, 3, num=50).tolist()

    model = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
    model.fit(X, y)

    return float(model.alpha_)
