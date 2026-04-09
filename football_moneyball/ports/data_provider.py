"""Port for football data providers."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    """Interface for football data sources (StatsBomb, Sofascore, etc).

    Defines the contract that any data provider must implement to
    be used by Football Moneyball. The default implementation uses
    statsbombpy, but the system can be extended to other providers.
    """

    def get_match_events(self, match_id: int) -> pd.DataFrame:
        """Return events of a match as a DataFrame.

        The DataFrame must contain at least the columns: type, player,
        player_id, team, location, period, minute, second, timestamp.
        Additional columns depend on the event type (e.g. pass_end_location,
        shot_outcome, shot_statsbomb_xg).

        Parameters
        ----------
        match_id : int
            Match identifier.

        Returns
        -------
        pd.DataFrame
            DataFrame with all events of the match.
        """
        ...

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        """Return lineups per team with position data.

        The dictionary maps team name to a DataFrame with columns:
        player_id, player_name, jersey_number, positions (list of dicts
        with position_id and position).

        Parameters
        ----------
        match_id : int
            Match identifier.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping team_name -> DataFrame of players.
        """
        ...

    def get_competitions(self) -> pd.DataFrame:
        """List available competitions.

        Returns DataFrame with columns: competition_id, competition_name,
        season_id, season_name, and optionally match_available.

        Returns
        -------
        pd.DataFrame
            DataFrame with the available competitions.
        """
        ...

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """List matches of a competition/season.

        Returns DataFrame with columns: match_id, match_date, home_team,
        away_team, home_score, away_score, competition, season.

        Parameters
        ----------
        competition_id : int
            Competition identifier.
        season_id : int
            Season identifier.

        Returns
        -------
        pd.DataFrame
            DataFrame with info of all matches.
        """
        ...

    def get_match_info(self, match_id: int) -> dict:
        """Return metadata of a match (competition, score, teams).

        The returned dictionary must contain the keys: match_id, competition,
        season, match_date, home_team, away_team, home_score, away_score.

        Parameters
        ----------
        match_id : int
            Match identifier.

        Returns
        -------
        dict
            Dictionary with match metadata.
        """
        ...
