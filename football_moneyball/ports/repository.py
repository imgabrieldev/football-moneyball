"""Port for data persistence."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class MatchRepository(Protocol):
    """Interface for persistence operations.

    Defines the contract that any repository implementation must follow.
    The default implementation uses PostgreSQL + pgvector via SQLAlchemy,
    but the system can be extended to other backends (SQLite, DuckDB, etc).
    """

    # ------------------------------------------------------------------
    # Match operations
    # ------------------------------------------------------------------

    def match_exists(self, match_id: int) -> bool:
        """Check whether a match is already stored in the database."""
        ...

    def save_match(self, match_data: dict) -> None:
        """Insert or update the data of a match.

        The dictionary must contain: match_id, competition, season,
        match_date, home_team, away_team, home_score, away_score.
        """
        ...

    def get_match_data(self, match_id: int) -> pd.DataFrame:
        """Return the data of a match as a DataFrame."""
        ...

    def get_season_matches(self, competition: str, season: str) -> list:
        """Return list of match_ids for a competition/season."""
        ...

    # ------------------------------------------------------------------
    # Player metrics
    # ------------------------------------------------------------------

    def save_player_metrics(self, metrics_df: pd.DataFrame, match_id: int) -> None:
        """Insert or update player metrics for a match.

        The DataFrame must contain player_id and columns compatible with
        PlayerMatchMetrics.
        """
        ...

    def get_player_metrics(
        self, player_name: str, season: str | None = None
    ) -> pd.DataFrame:
        """Return metrics of a player, optionally filtered by season."""
        ...

    def get_all_metrics(self, competition: str, season: str) -> pd.DataFrame:
        """Return metrics of all players of a competition/season."""
        ...

    # ------------------------------------------------------------------
    # Pass network
    # ------------------------------------------------------------------

    def save_pass_network(self, edges_df: pd.DataFrame, match_id: int) -> None:
        """Insert or update pass network edges of a match.

        The DataFrame must contain: passer_id, receiver_id, weight,
        and optionally passer_name, receiver_name, features.
        """
        ...

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def save_embeddings(self, embeddings_df: pd.DataFrame) -> None:
        """Insert or update vector embeddings of players.

        The DataFrame must contain: player_id, season, embedding (list of floats),
        and optionally player_name, cluster_label, archetype, position_group.
        """
        ...

    def get_embedding(
        self, player_name: str, season: str | None = None
    ) -> Any:
        """Return the embedding of a player for a season."""
        ...

    # ------------------------------------------------------------------
    # Similarity / Complementarity
    # ------------------------------------------------------------------

    def find_similar_players(
        self, player_name: str, season: str, limit: int = 10
    ) -> pd.DataFrame:
        """Search for players with similar embeddings via cosine distance.

        Returns DataFrame with: player_name, team, archetype,
        position_group, distance, similarity.
        """
        ...

    def find_complementary_players(
        self, player_name: str, season: str, limit: int = 10
    ) -> pd.DataFrame:
        """Search for players with a complementary profile (most dissimilar).

        Returns DataFrame with: player_name, team, archetype,
        position_group, distance, similarity.
        """
        ...

    # ------------------------------------------------------------------
    # Stints (RAPM)
    # ------------------------------------------------------------------

    def save_stints(self, stints_df: pd.DataFrame, match_id: int) -> None:
        """Insert or update stints (game periods) of a match.

        The DataFrame must contain: stint_number, home_player_ids,
        away_player_ids, duration_minutes, home_xg, away_xg, xg_diff,
        boundary_type.
        """
        ...

    def get_cached_stints(self, match_id: int) -> pd.DataFrame:
        """Return previously persisted stints for a match."""
        ...

    # ------------------------------------------------------------------
    # Action values (xT / VAEP)
    # ------------------------------------------------------------------

    def save_action_values(self, values_df: pd.DataFrame, match_id: int) -> None:
        """Insert or update action values (xT/VAEP) of a match.

        The DataFrame must contain: event_index, player_id, player_name,
        team, action_type, start_x, start_y, end_x, end_y, xt_value,
        vaep_value, vaep_offensive, vaep_defensive.
        """
        ...

    # ------------------------------------------------------------------
    # Pressing
    # ------------------------------------------------------------------

    def save_pressing_metrics(
        self, metrics_df: pd.DataFrame, match_id: int
    ) -> None:
        """Insert or update pressing metrics of a match.

        The DataFrame must contain: team, ppda, pressing_success_rate,
        counter_pressing_fraction, high_turnovers,
        shot_ending_high_turnovers, pressing_zone_1..6.
        """
        ...

    def get_pressing_metrics(
        self, team: str, season: str | None = None
    ) -> list:
        """Return pressing metrics of a team, optionally by season."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the repository connection/session."""
        ...
