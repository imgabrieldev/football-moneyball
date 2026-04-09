"""Port for data visualization."""

from __future__ import annotations

from typing import Any, Protocol

import networkx as nx
import numpy as np
import pandas as pd


class Visualizer(Protocol):
    """Interface for generating visualizations.

    Defines the contract that any visualization implementation must follow.
    The default implementation uses matplotlib + mplsoccer, but the system can
    be extended to other backends (plotly, bokeh, etc).

    All functions return a Figure object (matplotlib or equivalent).
    The save_path parameter, when provided, saves the figure to the given
    path.
    """

    def plot_pass_network(
        self,
        G: nx.DiGraph,
        team: str,
        match_info: dict | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Draw the pass network over a football pitch.

        Parameters
        ----------
        G : nx.DiGraph
            Directed graph with node attributes (avg_x, avg_y) and
            weighted edges (weight). StatsBomb coordinates: x 0-120,
            y 0-80.
        team : str
            Team name for the title.
        match_info : dict, optional
            Match metadata (opponent, date, score) for the title.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        Figure
            Figure object with the pass network visualization.
        """
        ...

    def plot_radar_comparison(
        self,
        player_a: dict,
        player_b: dict,
        metrics: list[str] | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Radar chart comparing two players.

        Parameters
        ----------
        player_a : dict
            Dictionary with 'name' (str) and percentile values (0-100)
            for each metric.
        player_b : dict
            Dictionary with 'name' (str) and percentile values (0-100)
            for each metric.
        metrics : list[str], optional
            List of metrics to compare. If None, uses default metrics.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        Figure
            Figure object with the comparison radar.
        """
        ...

    def plot_action_heatmap(
        self,
        events_df: pd.DataFrame,
        player_name: str,
        action_type: str | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Heatmap of a player's actions on the pitch.

        Parameters
        ----------
        events_df : pd.DataFrame
            Events DataFrame with columns 'player', 'type', 'location'
            (or 'x', 'y'). StatsBomb coordinates.
        player_name : str
            Player name to filter.
        action_type : str, optional
            Action type to filter (e.g. 'Pass', 'Shot', 'Carry').
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        Figure
            Figure object with the heatmap.
        """
        ...

    def plot_xt_heatmap(
        self,
        xt_grid: np.ndarray,
        l: int = 16,
        w: int = 12,
        save_path: str | None = None,
    ) -> Any:
        """Draw the Expected Threat (xT) surface over the pitch.

        Parameters
        ----------
        xt_grid : np.ndarray
            Matrix (l, w) with xT values per zone.
        l : int
            Cells on the x axis.
        w : int
            Cells on the y axis.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        Figure
            Figure object with the xT heatmap.
        """
        ...

    def plot_pressing_zones(
        self,
        pressing_data: dict,
        team: str,
        save_path: str | None = None,
    ) -> Any:
        """Visualize the pressing distribution by zone and key metrics.

        Parameters
        ----------
        pressing_data : dict
            Dictionary with keys: ppda, pressing_success_rate,
            counter_pressing_fraction, pressing_zone_1..6.
        team : str
            Team name.
        save_path : str, optional
            Path to save.

        Returns
        -------
        Figure
            Figure object with the pressing zones visualization.
        """
        ...

    def plot_shot_map(
        self,
        shots_df: pd.DataFrame,
        player_name: str,
        save_path: str | None = None,
    ) -> Any:
        """Shot map with size proportional to xG and color by outcome.

        Parameters
        ----------
        shots_df : pd.DataFrame
            Shot events DataFrame with columns: location,
            shot_outcome, shot_statsbomb_xg.
        player_name : str
            Player name.
        save_path : str, optional
            Path to save.

        Returns
        -------
        Figure
            Figure object with the shot map.
        """
        ...

    def plot_rapm_rankings(
        self,
        rapm_df: pd.DataFrame,
        top_n: int = 20,
        save_path: str | None = None,
    ) -> Any:
        """Horizontal bar chart with RAPM rankings.

        Parameters
        ----------
        rapm_df : pd.DataFrame
            DataFrame with columns 'player_name' and 'rapm_value'.
        top_n : int
            Number of players to display at the top and bottom of the ranking.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        Figure
            Figure object with the RAPM ranking.
        """
        ...

    def plot_synergy_graph(
        self,
        compatibility_df: pd.DataFrame,
        team: str | None = None,
        save_path: str | None = None,
    ) -> Any:
        """Synergy/compatibility graph between players.

        Parameters
        ----------
        compatibility_df : pd.DataFrame
            DataFrame with columns 'player_a', 'player_b' and 'score'
            representing the compatibility between pairs.
        team : str, optional
            Team name for the title.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        Figure
            Figure object with the synergy graph.
        """
        ...
