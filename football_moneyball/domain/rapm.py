"""Modulo de dominio RAPM (Regularized Adjusted Plus-Minus).

Implementa o modelo de Plus-Minus Ajustado Regularizado usando regressao Ridge
para isolar o impacto individual de cada jogador.

Funcionalidades avancadas:
- Splints: stints delimitados tambem por gols (alem de substituicoes e periodos)
- SPM Prior: prior informativo baseado em box-score metrics (Statistical Plus-Minus)
- Offensive/Defensive Split: colunas separadas para impacto ofensivo e defensivo
- Augmented Regression: incorporacao do SPM prior via regressao aumentada

Logica pura sobre DataFrames e arrays — sem dependencias de I/O externo
(statsbombpy, sqlalchemy).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV


# ---------------------------------------------------------------------------
# Reconstrucao de stints (splints) a partir de eventos
# ---------------------------------------------------------------------------

def reconstruct_stints(
    events: pd.DataFrame, lineups: dict
) -> pd.DataFrame:
    """Reconstroi os stints (periodos com mesmos 22 jogadores) de uma partida.

    Recebe eventos e lineups ja carregados e identifica os momentos em que
    a composicao dos jogadores em campo muda (substituicoes, inicio de tempo)
    ou um gol e marcado. Para cada stint, calcula o diferencial de xG
    (casa - fora) e a duracao em minutos.

    Gols criam novas fronteiras ('splints'), permitindo analise mais granular
    do impacto dos jogadores em estados de jogo diferentes (empate, lideranca,
    desvantagem). O campo ``boundary_type`` registra o que causou cada fronteira:
    'period_start', 'substitution' ou 'goal'.

    Parametros
    ----------
    events : pd.DataFrame
        DataFrame de eventos StatsBomb (retornado por sb.events() ou
        equivalente).
    lineups : dict
        Dicionario de lineups por time, no formato retornado por
        sb.lineups() (chave = nome do time, valor = DataFrame com
        colunas player_id, positions, etc.).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: stint_number, home_player_ids,
        away_player_ids, duration_minutes, home_xg, away_xg, xg_diff,
        boundary_type.
    """
    # Identificar times (home / away)
    # O primeiro evento de tipo "Starting XI" nos da a ordem home/away
    starting_xi_events = events[events["type"] == "Starting XI"].sort_values("index")

    if len(starting_xi_events) < 2:
        return pd.DataFrame()

    home_team = starting_xi_events.iloc[0]["team"]
    away_team = starting_xi_events.iloc[1]["team"]

    # Extrair escalacoes iniciais (titulares) a partir dos lineups
    home_lineup = lineups[home_team]
    away_lineup = lineups[away_team]

    # Jogadores titulares sao aqueles com posicao listada nas tactics do Starting XI
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

    # Calcular timestamp continuo em minutos (considerando periodos)
    period_offsets = {1: 0, 2: 45, 3: 90, 4: 105}
    events["timestamp_minutes"] = events.apply(
        lambda row: period_offsets.get(row["period"], 0)
        + row["minute"]
        + row.get("second", 0) / 60.0,
        axis=1,
    )

    # Identificar eventos que delimitam stints
    # Substituicoes
    subs = events[events["type"] == "Substitution"].copy()

    # Half Start (inicio de periodo)
    half_starts = events[events["type"] == "Half Start"].copy()

    # Coletar pontos de corte (timestamps onde a composicao muda)
    boundary_times: list[float] = []
    boundary_types: dict[float, str] = {}

    # Inicio de cada periodo como boundary
    for _, hs in half_starts.iterrows():
        ts = hs["timestamp_minutes"]
        boundary_times.append(ts)
        boundary_types[ts] = "period_start"

    # Cada substituicao como boundary
    for _, sub_event in subs.iterrows():
        ts = sub_event["timestamp_minutes"]
        boundary_times.append(ts)
        boundary_types[ts] = "substitution"

    # Gols como boundaries (splints)
    if "shot_outcome" in events.columns:
        goal_events = events[
            (events["type"] == "Shot") & (events["shot_outcome"] == "Goal")
        ]
        for _, goal_event in goal_events.iterrows():
            ts = goal_event["timestamp_minutes"]
            boundary_times.append(ts)
            boundary_types[ts] = "goal"

    boundary_times = sorted(set(boundary_times))

    # Se nao houver boundaries, usar inicio e fim da partida
    if not boundary_times:
        boundary_times = [0.0]
        boundary_types[0.0] = "period_start"

    # Fim da partida
    match_end = events["timestamp_minutes"].max()

    # Construir intervalos de stints
    intervals: list[tuple[float, float]] = []
    interval_boundary_types: list[str] = []
    for i in range(len(boundary_times)):
        start = boundary_times[i]
        end = boundary_times[i + 1] if i + 1 < len(boundary_times) else match_end
        if end > start:
            intervals.append((start, end))
            interval_boundary_types.append(boundary_types.get(start, "period_start"))

    # Tambem incluir intervalo antes da primeira boundary se > 0
    if boundary_times[0] > 0:
        intervals.insert(0, (0.0, boundary_times[0]))
        interval_boundary_types.insert(0, "period_start")

    # Processar substituicoes em ordem cronologica para rastrear composicao
    subs_sorted = subs.sort_values("timestamp_minutes").reset_index(drop=True)

    # Mapear substituicoes por timestamp para saber o estado do elenco em cada stint
    sub_records: list[dict] = []
    for _, sub_event in subs_sorted.iterrows():
        sub_team = sub_event["team"]
        player_out = sub_event.get("player_id", None)
        # O jogador que entra esta em substitution_replacement
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
        # Aplicar substituicoes que ocorreram exatamente no inicio deste stint
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
    """Calcula prior SPM (Statistical Plus-Minus) a partir de box-score metrics.

    Usa metricas per-90 (goals, assists, xg, tackles, interceptions, key_passes)
    para gerar um prior informativo para cada jogador. Esse prior e usado na
    regressao aumentada do RAPM para estabilizar estimativas de jogadores com
    poucos minutos amostrados.

    Parametros
    ----------
    player_metrics_list : list
        Lista de objetos/dicts com atributos: player_id, minutes_played,
        goals, assists, xg, key_passes, tackles, interceptions. Cada
        elemento representa metricas de uma partida para um jogador.
    player_ids : list[int]
        Lista de IDs de jogadores para os quais calcular o prior.

    Retorna
    -------
    np.ndarray
        Array com o prior SPM para cada jogador (mesmo indice de player_ids),
        centrado em 0.
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
        # Contribuicoes positivas (ofensivas e defensivas)
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
# Construcao da matriz de design para RAPM
# ---------------------------------------------------------------------------

def build_rapm_matrix(
    stints_df: pd.DataFrame,
    offensive_defensive_split: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Constroi a matriz de design X e o vetor alvo y para a regressao Ridge.

    Cada linha representa um stint. No modo padrao, cada jogador tem uma coluna
    que recebe +weight se da casa, -weight se visitante (weight = sqrt(duracao)).
    O vetor y e o diferencial de xG ponderado pela mesma raiz.

    No modo ``offensive_defensive_split``, cada jogador gera duas colunas:
    uma ofensiva (impacto no xG a favor) e uma defensiva (impacto no xG
    contra), permitindo separacao real do RAPM ofensivo/defensivo.

    Parametros
    ----------
    stints_df : pd.DataFrame
        DataFrame de stints conforme retornado por ``reconstruct_stints``.
    offensive_defensive_split : bool, optional
        Se True, cria 2 colunas por jogador (ofensiva e defensiva).
        Padrao: False.

    Retorna
    -------
    tuple[np.ndarray, np.ndarray, list[int]]
        (X, y, player_ids) onde X e a matriz de design, y o vetor alvo
        ponderado e player_ids a lista ordenada de IDs de jogadores
        correspondente as colunas de X. Quando ``offensive_defensive_split``
        e True, player_ids contem IDs intercalados: [p1_off, p1_def, p2_off,
        p2_def, ...] e X tem 2*n_players colunas.
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
        # 2 colunas por jogador: [off, def] intercaladas
        n_cols = 2 * n_players
        X = np.zeros((n_stints, n_cols), dtype=np.float64)
        y = np.zeros(n_stints, dtype=np.float64)

        for i, (_, row) in enumerate(stints_df.iterrows()):
            home_ids = row["home_player_ids"]
            away_ids = row["away_player_ids"]
            duration = row.get("duration_minutes", 1.0)
            xg_diff = row.get("xg_diff", 0.0)

            weight = np.sqrt(max(duration, 0.01))

            # Jogadores da casa: +weight na coluna ofensiva
            if isinstance(home_ids, (list, set)):
                for pid in home_ids:
                    if pid in pid_to_idx:
                        base_idx = pid_to_idx[pid] * 2
                        X[i, base_idx] = weight       # offensive column

            # Jogadores do visitante: -weight na coluna defensiva
            if isinstance(away_ids, (list, set)):
                for pid in away_ids:
                    if pid in pid_to_idx:
                        base_idx = pid_to_idx[pid] * 2
                        X[i, base_idx + 1] = -weight  # defensive column

            y[i] = xg_diff * weight

        # Gerar lista de player_ids com sufixos off/def
        split_player_ids = []
        for pid in player_ids:
            split_player_ids.append(pid)  # offensive
            split_player_ids.append(pid)  # defensive
        return X, y, split_player_ids

    else:
        # Modo padrao: 1 coluna por jogador
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
    spm_prior: np.ndarray | None = None,
) -> pd.DataFrame:
    """Ajusta a regressao Ridge para estimar o RAPM de cada jogador.

    Quando um prior SPM e fornecido, utiliza regressao aumentada: empilha uma
    matriz identidade escalada abaixo de X e o prior escalado abaixo de y. Isso
    encolhe os coeficientes em direcao ao prior em vez de em direcao a zero,
    produzindo estimativas mais estaveis para jogadores com poucos stints.

    Parametros
    ----------
    X : np.ndarray
        Matriz de design (stints x jogadores).
    y : np.ndarray
        Vetor alvo (xG diferencial ponderado).
    player_ids : list[int]
        Lista de IDs de jogadores correspondente as colunas de X.
    alpha : float, optional
        Parametro de regularizacao da regressao Ridge. Padrao: 1.0.
    spm_prior : np.ndarray, optional
        Prior SPM para regressao aumentada. Deve ter o mesmo numero de
        elementos que colunas de X. Se None, usa Ridge padrao (shrink p/ zero).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: player_id, rapm_value, offensive_rapm,
        defensive_rapm.
    """
    if spm_prior is not None:
        # Regressao aumentada: encolhe em direcao ao prior SPM
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

    # Tentar calcular RAPM ofensivo/defensivo separadamente
    # Ofensivo: impacto no xG a favor; Defensivo: impacto no xG contra
    # Aproximacao: usar o coeficiente total dividido proporcionalmente
    results["offensive_rapm"] = results["rapm_value"].clip(lower=0)
    results["defensive_rapm"] = results["rapm_value"].clip(upper=0)

    return results


# ---------------------------------------------------------------------------
# Validacao cruzada para escolha do alpha otimo
# ---------------------------------------------------------------------------

def cross_validate_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[list[float]] = None,
) -> float:
    """Encontra o melhor parametro de regularizacao alpha via validacao cruzada.

    Utiliza RidgeCV do scikit-learn para testar diferentes valores de alpha
    e retornar aquele que minimiza o erro de validacao cruzada.

    Parametros
    ----------
    X : np.ndarray
        Matriz de design.
    y : np.ndarray
        Vetor alvo.
    alphas : list[float], optional
        Lista de valores de alpha a testar. Se None, usa uma grade logaritmica
        padrao de 0.01 a 1000.

    Retorna
    -------
    float
        O valor de alpha com melhor desempenho na validacao cruzada.
    """
    if alphas is None:
        alphas = np.logspace(-2, 3, num=50).tolist()

    model = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
    model.fit(X, y)

    return float(model.alpha_)
