"""Modulo de modelos de valor de posse (Possession Value).

Implementa Expected Threat (xT) via Markov chain iterativo sobre grid 16x12,
seguindo a metodologia de Karun Singh (2018). Valora cada acao no campo pelo
delta de ameaca entre posicao inicial e final.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsbombpy import sb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grid dimensions (Karun Singh standard)
XT_GRID_L = 16  # cells in x-dimension
XT_GRID_W = 12  # cells in y-dimension

# StatsBomb pitch dimensions
_PITCH_LENGTH = 120.0
_PITCH_WIDTH = 80.0

# Convergence
_XT_MAX_ITER = 50
_XT_EPS = 1e-5


# ---------------------------------------------------------------------------
# xT Model
# ---------------------------------------------------------------------------

class ExpectedThreat:
    """Modelo Expected Threat (xT) via Markov chain iterativo.

    Divide o campo em grid l x w e calcula o valor de ameaca de cada zona
    considerando probabilidade de chutar, probabilidade de gol ao chutar
    e probabilidade de transicao para outras zonas.

    Parameters
    ----------
    l : int
        Celulas no eixo x (default 16).
    w : int
        Celulas no eixo y (default 12).
    """

    def __init__(self, l: int = XT_GRID_L, w: int = XT_GRID_W) -> None:
        self.l = l
        self.w = w
        self.xt_grid: np.ndarray | None = None
        self._shoot_prob: np.ndarray | None = None
        self._move_prob: np.ndarray | None = None
        self._goal_prob: np.ndarray | None = None
        self._transition: np.ndarray | None = None

    def _loc_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Converte coordenadas StatsBomb (120x80) para indices do grid."""
        cx = min(int(x / _PITCH_LENGTH * self.l), self.l - 1)
        cy = min(int(y / _PITCH_WIDTH * self.w), self.w - 1)
        return max(cx, 0), max(cy, 0)

    def fit(self, events_list: list[pd.DataFrame]) -> "ExpectedThreat":
        """Treina o modelo xT a partir de eventos de multiplas partidas.

        Calcula as probabilidades de chutar, mover, gol e a matriz de
        transicao a partir dos eventos fornecidos.

        Parameters
        ----------
        events_list : list[pd.DataFrame]
            Lista de DataFrames de eventos (retornados por sb.events()).

        Returns
        -------
        ExpectedThreat
            Self (para encadeamento).
        """
        all_events = pd.concat(events_list, ignore_index=True)

        # Count actions per cell
        shoot_count = np.zeros((self.l, self.w))
        move_count = np.zeros((self.l, self.w))
        goal_count = np.zeros((self.l, self.w))
        transition_count = np.zeros((self.l, self.w, self.l, self.w))

        # Shots
        shots = all_events[all_events["type"] == "Shot"]
        for _, row in shots.iterrows():
            loc = row.get("location")
            if not isinstance(loc, list) or len(loc) < 2:
                continue
            cx, cy = self._loc_to_cell(loc[0], loc[1])
            shoot_count[cx, cy] += 1
            if row.get("shot_outcome") == "Goal":
                goal_count[cx, cy] += 1

        # Successful moves (passes + carries that maintain possession)
        for move_type, end_col in [
            ("Pass", "pass_end_location"),
            ("Carry", "carry_end_location"),
        ]:
            moves = all_events[all_events["type"] == move_type]
            if move_type == "Pass" and "pass_outcome" in moves.columns:
                # Only successful passes
                moves = moves[
                    moves["pass_outcome"].isna()
                    | (moves["pass_outcome"] == "Complete")
                ]

            for _, row in moves.iterrows():
                start = row.get("location")
                end = row.get(end_col)
                if (
                    not isinstance(start, list) or len(start) < 2
                    or not isinstance(end, list) or len(end) < 2
                ):
                    continue
                sx, sy = self._loc_to_cell(start[0], start[1])
                ex, ey = self._loc_to_cell(end[0], end[1])
                move_count[sx, sy] += 1
                transition_count[sx, sy, ex, ey] += 1

        # Compute probabilities
        total_actions = shoot_count + move_count
        total_actions[total_actions == 0] = 1  # avoid division by zero

        self._shoot_prob = shoot_count / total_actions
        self._move_prob = move_count / total_actions
        self._goal_prob = np.where(shoot_count > 0, goal_count / shoot_count, 0.0)

        # Transition matrix: P(move to (ex,ey) | move from (sx,sy))
        self._transition = np.zeros((self.l, self.w, self.l, self.w))
        for sx in range(self.l):
            for sy in range(self.w):
                total_moves = move_count[sx, sy]
                if total_moves > 0:
                    self._transition[sx, sy] = transition_count[sx, sy] / total_moves

        # Iterative solution
        self.xt_grid = np.zeros((self.l, self.w))
        for iteration in range(_XT_MAX_ITER):
            new_xt = np.zeros((self.l, self.w))
            for x in range(self.l):
                for y in range(self.w):
                    # xT(x,y) = s(x,y)*g(x,y) + m(x,y) * sum(T * xT)
                    shoot_val = self._shoot_prob[x, y] * self._goal_prob[x, y]
                    move_val = self._move_prob[x, y] * np.sum(
                        self._transition[x, y] * self.xt_grid
                    )
                    new_xt[x, y] = shoot_val + move_val

            if np.max(np.abs(new_xt - self.xt_grid)) < _XT_EPS:
                self.xt_grid = new_xt
                break
            self.xt_grid = new_xt

        return self

    def get_value(self, x: float, y: float) -> float:
        """Retorna o valor xT de uma posicao no campo.

        Parameters
        ----------
        x, y : float
            Coordenadas StatsBomb (120x80).

        Returns
        -------
        float
            Valor xT da zona correspondente.
        """
        if self.xt_grid is None:
            raise RuntimeError("Modelo nao treinado. Chame fit() primeiro.")
        cx, cy = self._loc_to_cell(x, y)
        return float(self.xt_grid[cx, cy])

    def rate_actions(self, events: pd.DataFrame) -> pd.Series:
        """Calcula o delta xT de cada acao em uma partida.

        Retorna xT(destino) - xT(origem) para passes e carries bem-sucedidos.
        Outras acoes recebem NaN.

        Parameters
        ----------
        events : pd.DataFrame
            Eventos de uma partida (retornados por sb.events()).

        Returns
        -------
        pd.Series
            Serie com valores xT alinhada ao indice do DataFrame.
        """
        if self.xt_grid is None:
            raise RuntimeError("Modelo nao treinado. Chame fit() primeiro.")

        xt_values = pd.Series(np.nan, index=events.index)

        for move_type, end_col in [
            ("Pass", "pass_end_location"),
            ("Carry", "carry_end_location"),
        ]:
            mask = events["type"] == move_type
            if move_type == "Pass" and "pass_outcome" in events.columns:
                mask = mask & (
                    events["pass_outcome"].isna()
                    | (events["pass_outcome"] == "Complete")
                )

            for idx in events[mask].index:
                row = events.loc[idx]
                start = row.get("location")
                end = row.get(end_col)
                if (
                    isinstance(start, list) and len(start) >= 2
                    and isinstance(end, list) and len(end) >= 2
                ):
                    start_xt = self.get_value(start[0], start[1])
                    end_xt = self.get_value(end[0], end[1])
                    xt_values[idx] = end_xt - start_xt

        return xt_values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_xt_model(match_ids: list[int]) -> ExpectedThreat:
    """Treina modelo xT a partir de uma lista de partidas.

    Busca eventos de cada partida via StatsBomb e ajusta o modelo
    Expected Threat sobre todos os dados combinados.

    Parameters
    ----------
    match_ids : list[int]
        Lista de IDs de partidas do StatsBomb.

    Returns
    -------
    ExpectedThreat
        Modelo xT treinado.
    """
    events_list = []
    for mid in match_ids:
        try:
            events = sb.events(match_id=mid)
            if not events.empty:
                events_list.append(events)
        except Exception as e:
            warnings.warn(f"Erro ao carregar eventos de match_id={mid}: {e}")

    if not events_list:
        raise ValueError("Nenhum evento disponivel para treinar o modelo xT.")

    model = ExpectedThreat()
    model.fit(events_list)
    return model


def compute_match_xt(
    model: ExpectedThreat, match_id: int
) -> pd.DataFrame:
    """Calcula valores xT por acao para uma partida.

    Parameters
    ----------
    model : ExpectedThreat
        Modelo xT treinado.
    match_id : int
        ID da partida.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: event_index, player_id, player_name, team,
        action_type, start_x, start_y, end_x, end_y, xt_value.
    """
    events = sb.events(match_id=match_id)
    if events.empty:
        return pd.DataFrame()

    xt_values = model.rate_actions(events)

    # Build result with only valued actions
    valued_mask = xt_values.notna()
    valued_events = events[valued_mask].copy()
    valued_events["xt_value"] = xt_values[valued_mask]

    rows = []
    for idx, row in valued_events.iterrows():
        loc = row.get("location", [None, None])
        end_loc = (
            row.get("pass_end_location")
            or row.get("carry_end_location")
            or [None, None]
        )
        rows.append({
            "event_index": int(row.get("index", idx)),
            "player_id": row.get("player_id"),
            "player_name": row.get("player"),
            "team": row.get("team"),
            "action_type": row.get("type"),
            "start_x": loc[0] if isinstance(loc, list) else None,
            "start_y": loc[1] if isinstance(loc, list) and len(loc) > 1 else None,
            "end_x": end_loc[0] if isinstance(end_loc, list) else None,
            "end_y": end_loc[1] if isinstance(end_loc, list) and len(end_loc) > 1 else None,
            "xt_value": row["xt_value"],
        })

    return pd.DataFrame(rows)


def aggregate_player_xt(action_values: pd.DataFrame) -> pd.DataFrame:
    """Agrega xT total gerado por cada jogador.

    Parameters
    ----------
    action_values : pd.DataFrame
        DataFrame retornado por compute_match_xt().

    Returns
    -------
    pd.DataFrame
        DataFrame com player_id, player_name, team, xt_generated.
    """
    if action_values.empty:
        return pd.DataFrame()

    return (
        action_values.groupby(["player_id", "player_name", "team"])
        .agg(xt_generated=("xt_value", "sum"))
        .reset_index()
        .sort_values("xt_generated", ascending=False)
    )
