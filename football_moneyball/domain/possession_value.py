"""Modulo of dominio for models of value of posse (Possession Value).

Implementa Expected Threat (xT) via Markov chain iterativo sobre grid 16x12,
seguindo a methodology of Karun Singh (2018). Values each action in the pitch pelo
delta of threat between posicao inicial and final.

Logica pura sobre DataFrames and arrays numpy — without dependencias of I/O externo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_moneyball.domain.constants import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    XT_EPS,
    XT_GRID_L,
    XT_GRID_W,
    XT_MAX_ITER,
)


# ---------------------------------------------------------------------------
# xT Model
# ---------------------------------------------------------------------------

class ExpectedThreat:
    """Model Expected Threat (xT) via Markov chain iterativo.

    Divide o pitch in grid l x w and calcula o value of threat of each zona
    considerando probability of chutar, probability of goal ao chutar
    and probability of transicao for theutras zonas.

    Parameters
    ----------
    l : int
        Celulas in the eixo x (default 16).
    w : int
        Celulas in the eixo y (default 12).
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
        """Converte coordenadas StatsBomb (120x80) for indices of the grid."""
        cx = min(int(x / PITCH_LENGTH * self.l), self.l - 1)
        cy = min(int(y / PITCH_WIDTH * self.w), self.w - 1)
        return max(cx, 0), max(cy, 0)

    def fit(self, events_list: list[pd.DataFrame]) -> "ExpectedThreat":
        """Treina o model xT from eventos of multiplas matches.

        Compute the probabilitys of chutar, mover, goal is the matriz de
        transicao from the eventos fornecidos.

        Parameters
        ----------
        events_list : list[pd.DataFrame]
            Lista of DataFrames of eventos (retornados by sb.events()).

        Returns
        -------
        ExpectedThreat
            Self (for encadeamento).
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
        for iteration in range(XT_MAX_ITER):
            new_xt = np.zeros((self.l, self.w))
            for x in range(self.l):
                for y in range(self.w):
                    # xT(x,y) = s(x,y)*g(x,y) + m(x,y) * sum(T * xT)
                    shoot_val = self._shoot_prob[x, y] * self._goal_prob[x, y]
                    move_val = self._move_prob[x, y] * np.sum(
                        self._transition[x, y] * self.xt_grid
                    )
                    new_xt[x, y] = shoot_val + move_val

            if np.max(np.abs(new_xt - self.xt_grid)) < XT_EPS:
                self.xt_grid = new_xt
                break
            self.xt_grid = new_xt

        return self

    def get_value(self, x: float, y: float) -> float:
        """Returns the value xT of a posicao in the pitch.

        Parameters
        ----------
        x, y : float
            Coordenadas StatsBomb (120x80).

        Returns
        -------
        float
            Valor xT of the zona correspondente.
        """
        if self.xt_grid is None:
            raise RuntimeError("Model nao trained. Chame fit() first.")
        cx, cy = self._loc_to_cell(x, y)
        return float(self.xt_grid[cx, cy])

    def rate_actions(self, events: pd.DataFrame) -> pd.Series:
        """Computes the delta xT of each action in a match.

        Returns xT(destino) - xT(origem) for passes and carries bem-sucedidos.
        Outras actions recebem NaN.

        Parameters
        ----------
        events : pd.DataFrame
            Eventos of a match (retornados by sb.events()).

        Returns
        -------
        pd.Series
            Serie with values xT alinhada ao index of the DataFrame.
        """
        if self.xt_grid is None:
            raise RuntimeError("Model nao trained. Chame fit() first.")

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
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_player_xt(action_values: pd.DataFrame) -> pd.DataFrame:
    """Aggregates xT total gerado by each player.

    Parameters
    ----------
    action_values : pd.DataFrame
        DataFrame retornado by compute_match_xt() (or equivalente),
        with colunas player_id, player_name, team, xt_value.

    Returns
    -------
    pd.DataFrame
        DataFrame with player_id, player_name, team, xt_generated.
    """
    if action_values.empty:
        return pd.DataFrame()

    return (
        action_values.groupby(["player_id", "player_name", "team"])
        .agg(xt_generated=("xt_value", "sum"))
        .reset_index()
        .sort_values("xt_generated", ascending=False)
    )
