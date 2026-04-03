"""Modulo de metricas de pressing a partir de dados StatsBomb.

Calcula PPDA, pressing success rate, counter-pressing fraction,
high turnovers e distribuicao de pressing por zonas do campo.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsbombpy import sb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 6 horizontal zones across the pitch (StatsBomb coords: 0-120)
_ZONE_BOUNDARIES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120)]

_DEFENSIVE_ACTION_TYPES = {"Pressure", "Duel", "Interception", "Foul Committed", "Block"}

# High turnover threshold: within 40m of opponent goal (x >= 80 in StatsBomb)
_HIGH_TURNOVER_X_THRESHOLD = 80.0

# Time windows (seconds)
_COUNTERPRESS_WINDOW = 5.0
_PRESSING_RECOVERY_WINDOW = 5.0
_SHOT_AFTER_TURNOVER_WINDOW = 15.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_timestamp_seconds(events: pd.DataFrame) -> pd.Series:
    """Converte timestamp para segundos continuos considerando periodos."""
    period_offsets = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
    result = pd.Series(0.0, index=events.index)
    for idx, row in events.iterrows():
        offset = period_offsets.get(row.get("period", 1), 0)
        minute = row.get("minute", 0)
        second = row.get("second", 0)
        result[idx] = offset + minute * 60 + second
    return result


def _compute_ppda(events: pd.DataFrame, team: str, opponent: str) -> float:
    """Calcula PPDA (Passes Per Defensive Action).

    PPDA = passes do adversario / acoes defensivas do time.
    Menor PPDA = pressing mais intenso.
    Exclui passes no terco defensivo do adversario (x < 40).
    """
    # Opponent passes (excluding their own defensive third)
    opp_passes = events[
        (events["type"] == "Pass")
        & (events["team"] == opponent)
    ]
    if "location" in opp_passes.columns:
        opp_pass_locations = opp_passes["location"].apply(
            lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 1 else 0
        )
        # Opponent's defensive third is x < 40 from THEIR perspective
        # But in StatsBomb coords, home team attacks toward x=120
        # So we need to figure out which direction the opponent attacks
        # Simplification: exclude passes in first 40 yards (own defensive third)
        opp_passes = opp_passes[opp_pass_locations >= 40]

    n_opp_passes = len(opp_passes)

    # Team's defensive actions
    def_actions = events[
        (events["type"].isin(_DEFENSIVE_ACTION_TYPES))
        & (events["team"] == team)
    ]
    n_def_actions = len(def_actions)

    if n_def_actions == 0:
        return float("inf")

    return round(n_opp_passes / n_def_actions, 2)


def _compute_pressing_success(events: pd.DataFrame, team: str) -> float:
    """Calcula taxa de sucesso do pressing.

    Porcentagem de pressoes que resultam em recuperacao de bola pelo
    mesmo time dentro de 5 segundos.
    """
    pressures = events[
        (events["type"] == "Pressure") & (events["team"] == team)
    ].copy()

    if pressures.empty:
        return 0.0

    ts = _get_timestamp_seconds(events)
    pressure_ts = ts[pressures.index]

    recoveries = events[
        (events["type"] == "Ball Recovery") & (events["team"] == team)
    ]
    recovery_ts = ts[recoveries.index].values

    successful = 0
    for p_ts in pressure_ts:
        # Check if any recovery by same team within window
        time_diffs = recovery_ts - p_ts
        if np.any((time_diffs >= 0) & (time_diffs <= _PRESSING_RECOVERY_WINDOW)):
            successful += 1

    total = len(pressures)
    return round(successful / total * 100, 1) if total > 0 else 0.0


def _compute_counter_pressing_fraction(events: pd.DataFrame, team: str) -> float:
    """Calcula fracao de counter-pressing.

    Porcentagem de perdas de bola seguidas de pressao com counterpress=True
    dentro de 5 segundos.
    """
    # Identify turnovers: possession changes where team lost the ball
    # A turnover is when the team had the ball and the next event is by the opponent
    # Simpler approach: count events with counterpress=True for this team
    if "counterpress" not in events.columns:
        return 0.0

    # Count all turnovers (Dispossessed, Miscontrol, failed passes by team)
    turnover_types = {"Dispossessed", "Miscontrol"}
    turnovers = events[
        (events["type"].isin(turnover_types)) & (events["team"] == team)
    ]
    # Also count failed passes
    failed_passes = events[
        (events["type"] == "Pass")
        & (events["team"] == team)
    ]
    if "pass_outcome" in failed_passes.columns:
        failed_passes = failed_passes[
            failed_passes["pass_outcome"].notna()
            & (failed_passes["pass_outcome"] != "Complete")
        ]
    else:
        failed_passes = failed_passes.iloc[0:0]  # empty

    total_turnovers = len(turnovers) + len(failed_passes)

    if total_turnovers == 0:
        return 0.0

    # Count counterpress events for this team
    cp_events = events[
        (events["counterpress"] == True)  # noqa: E712
        & (events["team"] == team)
    ]
    n_counterpresses = len(cp_events)

    return round(min(n_counterpresses / total_turnovers * 100, 100.0), 1)


def _compute_high_turnovers(
    events: pd.DataFrame, team: str
) -> tuple[int, int]:
    """Calcula high turnovers e shot-ending high turnovers.

    High turnovers: recuperacoes de bola a <= 40m do gol adversario.
    Shot-ending: high turnovers seguidos de finalizacao em <= 15 segundos.
    """
    recoveries = events[
        (events["type"] == "Ball Recovery")
        & (events["team"] == team)
    ].copy()

    if recoveries.empty or "location" not in recoveries.columns:
        return 0, 0

    # Filter to high recoveries (x >= 80 in StatsBomb coords)
    rec_x = recoveries["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 1 else 0
    )
    high_recoveries = recoveries[rec_x >= _HIGH_TURNOVER_X_THRESHOLD]
    n_high = len(high_recoveries)

    if n_high == 0:
        return 0, 0

    # Check for shot-ending high turnovers
    ts = _get_timestamp_seconds(events)
    shots = events[(events["type"] == "Shot") & (events["team"] == team)]
    shot_ts = ts[shots.index].values

    n_shot_ending = 0
    for idx in high_recoveries.index:
        rec_ts = ts[idx]
        time_diffs = shot_ts - rec_ts
        if np.any((time_diffs >= 0) & (time_diffs <= _SHOT_AFTER_TURNOVER_WINDOW)):
            n_shot_ending += 1

    return n_high, n_shot_ending


def _compute_pressing_zones(events: pd.DataFrame, team: str) -> list[float]:
    """Calcula distribuicao de pressing por 6 zonas horizontais do campo.

    Retorna lista de 6 floats representando a porcentagem de pressoes
    em cada zona (soma = 100).
    """
    pressures = events[
        (events["type"] == "Pressure") & (events["team"] == team)
    ]

    if pressures.empty or "location" not in pressures.columns:
        return [0.0] * 6

    pressure_x = pressures["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 1 else 0
    )

    total = len(pressures)
    zones = []
    for low, high in _ZONE_BOUNDARIES:
        count = ((pressure_x >= low) & (pressure_x < high)).sum()
        zones.append(round(count / total * 100, 1) if total > 0 else 0.0)

    return zones


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_match_pressing(match_id: int) -> pd.DataFrame:
    """Calcula metricas de pressing por time para uma partida.

    Busca os eventos da partida via StatsBomb e calcula PPDA,
    pressing success rate, counter-pressing fraction, high turnovers
    e distribuicao de pressing por zonas para cada time.

    Parameters
    ----------
    match_id : int
        Identificador da partida no StatsBomb.

    Returns
    -------
    pd.DataFrame
        DataFrame com uma linha por time e colunas: team, ppda,
        pressing_success_rate, counter_pressing_fraction, high_turnovers,
        shot_ending_high_turnovers, pressing_zone_1..6.
    """
    events = sb.events(match_id=match_id)

    if events.empty:
        warnings.warn(f"Nenhum evento encontrado para match_id={match_id}")
        return pd.DataFrame()

    teams = [t for t in events["team"].dropna().unique() if isinstance(t, str)]
    if len(teams) < 2:
        return pd.DataFrame()

    results = []
    for team in teams:
        opponent = [t for t in teams if t != team][0]

        ppda = _compute_ppda(events, team, opponent)
        success_rate = _compute_pressing_success(events, team)
        cp_fraction = _compute_counter_pressing_fraction(events, team)
        high_to, shot_ht = _compute_high_turnovers(events, team)
        zones = _compute_pressing_zones(events, team)

        results.append({
            "team": team,
            "ppda": ppda,
            "pressing_success_rate": success_rate,
            "counter_pressing_fraction": cp_fraction,
            "high_turnovers": high_to,
            "shot_ending_high_turnovers": shot_ht,
            "pressing_zone_1": zones[0],
            "pressing_zone_2": zones[1],
            "pressing_zone_3": zones[2],
            "pressing_zone_4": zones[3],
            "pressing_zone_5": zones[4],
            "pressing_zone_6": zones[5],
        })

    return pd.DataFrame(results)
