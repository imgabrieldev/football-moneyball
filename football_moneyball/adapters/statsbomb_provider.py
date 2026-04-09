"""StatsBomb adapter - data provider via statsbombpy.

Encapsulates all calls to the StatsBomb API (open data), exposing
a standardized interface for fetching events, lineups, competitions
and match metadata.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsbombpy import sb


class StatsBombProvider:
    """Data provider using StatsBomb open data."""

    def get_match_events(self, match_id: int) -> pd.DataFrame:
        """Returns all events of a match."""
        return sb.events(match_id=match_id)

    def get_lineups(self, match_id: int) -> dict[str, pd.DataFrame]:
        """Returns the lineups of a match, indexed by team."""
        return sb.lineups(match_id=match_id)

    def get_competitions(self) -> pd.DataFrame:
        """Returns the competitions available in the open data.

        Filters only competitions with available matches when the
        'match_available' column exists.
        """
        comps = sb.competitions()
        if "match_available" in comps.columns:
            return comps[comps["match_available"].notna()].reset_index(drop=True)
        return comps

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Returns all matches for a competition/season."""
        return sb.matches(competition_id=competition_id, season_id=season_id)

    def get_match_info(self, match_id: int) -> dict[str, Any]:
        """Returns metadata for a specific match.

        Fetches the match information (competition, season, teams, score)
        from StatsBomb open data. Iterates over all competitions until
        it finds the matching match.

        Parameters
        ----------
        match_id : int
            Match identifier in StatsBomb.

        Returns
        -------
        dict
            Dictionary with keys: ``match_id``, ``competition``, ``season``,
            ``match_date``, ``home_team``, ``away_team``, ``home_score``,
            ``away_score``.
        """
        comps = sb.competitions()
        for _, comp in comps.iterrows():
            try:
                matches = sb.matches(
                    competition_id=comp["competition_id"],
                    season_id=comp["season_id"],
                )
            except Exception:
                continue

            match_row = matches[matches["match_id"] == match_id]
            if not match_row.empty:
                m = match_row.iloc[0]
                return {
                    "match_id": int(m["match_id"]),
                    "competition": m.get(
                        "competition", comp.get("competition_name", "")
                    ),
                    "season": m.get("season", comp.get("season_name", "")),
                    "match_date": str(m.get("match_date", "")),
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "home_score": int(m.get("home_score", 0)),
                    "away_score": int(m.get("away_score", 0)),
                }

        warnings.warn(
            f"Match {match_id} not found in StatsBomb open data."
        )
        return {"match_id": match_id}
