"""Football Moneyball domain models.

Pure dataclasses with no dependency on ORM or external frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

@dataclass
class MatchInfo:
    """Metadata of a football match."""

    match_id: int
    competition: str = ""
    season: str = ""
    match_date: str = ""
    home_team: str = ""
    away_team: str = ""
    home_score: int = 0
    away_score: int = 0


# ---------------------------------------------------------------------------
# Player Match Metrics
# ---------------------------------------------------------------------------

@dataclass
class PlayerMatchMetrics:
    """Individual metrics of a player in a match."""

    match_id: int
    player_id: int
    player_name: str = ""
    team: str = ""

    # Core metrics
    minutes_played: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    shots: float = 0.0
    shots_on_target: float = 0.0
    xg: float = 0.0
    xa: float = 0.0
    passes: float = 0.0
    passes_completed: float = 0.0
    pass_pct: float = 0.0
    progressive_passes: float = 0.0
    progressive_carries: float = 0.0
    key_passes: float = 0.0
    through_balls: float = 0.0
    crosses: float = 0.0
    tackles: float = 0.0
    interceptions: float = 0.0
    blocks: float = 0.0
    clearances: float = 0.0
    aerials_won: float = 0.0
    aerials_lost: float = 0.0
    fouls_committed: float = 0.0
    fouls_won: float = 0.0
    dribbles_attempted: float = 0.0
    dribbles_completed: float = 0.0
    touches: float = 0.0
    carries: float = 0.0
    dispossessed: float = 0.0
    pressures: float = 0.0
    pressure_regains: float = 0.0

    # v0.2.0 expanded metrics
    progressive_receptions: float = 0.0
    big_chances: float = 0.0
    big_chances_missed: float = 0.0
    passes_short: float = 0.0
    passes_short_completed: float = 0.0
    passes_medium: float = 0.0
    passes_medium_completed: float = 0.0
    passes_long: float = 0.0
    passes_long_completed: float = 0.0
    passes_under_pressure: float = 0.0
    passes_under_pressure_completed: float = 0.0
    switches_of_play: float = 0.0
    ground_duels_won: float = 0.0
    ground_duels_total: float = 0.0
    tackle_success_rate: float = 0.0
    xt_generated: float = 0.0
    vaep_generated: float = 0.0
    pressing_success_rate_individual: float = 0.0


# ---------------------------------------------------------------------------
# Pass Network
# ---------------------------------------------------------------------------

@dataclass
class PassEdge:
    """Edge of the pass network between two players in a match."""

    passer_id: int
    receiver_id: int
    passer_name: str = ""
    receiver_name: str = ""
    weight: int = 0
    features: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Player Embedding
# ---------------------------------------------------------------------------

@dataclass
class PlayerEmbedding:
    """Vector embedding of a player's playing style per season."""

    player_id: int
    season: str
    player_name: str = ""
    team: str = ""
    competition: str = ""
    embedding: list[float] = field(default_factory=list)
    cluster_label: int | None = None
    archetype: str | None = None
    position_group: str | None = None


# ---------------------------------------------------------------------------
# Stint (RAPM)
# ---------------------------------------------------------------------------

@dataclass
class Stint:
    """Continuous period of play with the same set of players on the pitch."""

    match_id: int
    stint_number: int
    home_player_ids: list[int] = field(default_factory=list)
    away_player_ids: list[int] = field(default_factory=list)
    duration_minutes: float = 0.0
    home_xg: float = 0.0
    away_xg: float = 0.0
    xg_diff: float = 0.0
    boundary_type: str = "period_start"


# ---------------------------------------------------------------------------
# Action Value (xT / VAEP)
# ---------------------------------------------------------------------------

@dataclass
class ActionValue:
    """Action value (xT and VAEP) of an event in a match."""

    event_index: int
    player_id: int | None = None
    player_name: str = ""
    team: str = ""
    action_type: str = ""
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    xt_value: float | None = None
    vaep_value: float | None = None
    vaep_offensive: float | None = None
    vaep_defensive: float | None = None


# ---------------------------------------------------------------------------
# Pressing Metrics
# ---------------------------------------------------------------------------

@dataclass
class PressingProfile:
    """Pressing metrics of a team in a match."""

    team: str
    ppda: float = 0.0
    pressing_success_rate: float = 0.0
    counter_pressing_fraction: float = 0.0
    high_turnovers: int = 0
    shot_ending_high_turnovers: int = 0
    zones: list[float] = field(default_factory=lambda: [0.0] * 6)
