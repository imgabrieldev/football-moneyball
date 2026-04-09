"""Football Moneyball domain constants.

Centralizes all constants used by multiple modules of the project.
"""

from __future__ import annotations


# ===========================================================================
# Pitch dimensions (StatsBomb coordinate system)
# ===========================================================================

PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0

# Progressive action threshold: 10 yards closer to opponent goal
PROGRESSIVE_DISTANCE_THRESHOLD: float = 10.0


# ===========================================================================
# xT Grid (Expected Threat — Karun Singh 2018)
# ===========================================================================

XT_GRID_L: int = 16   # cells in x-dimension
XT_GRID_W: int = 12   # cells in y-dimension
XT_MAX_ITER: int = 50
XT_EPS: float = 1e-5


# ===========================================================================
# Position Group Mapping (StatsBomb position_id -> group)
# ===========================================================================

POSITION_GROUP_MAP: dict[int, str] = {
    1: "GK",
    2: "DEF", 3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF", 8: "DEF",
    9: "MID", 10: "MID", 11: "MID", 12: "MID", 13: "MID", 14: "MID",
    15: "MID", 16: "MID",
    17: "FWD", 18: "MID", 19: "MID", 20: "MID", 21: "FWD",
    22: "FWD", 23: "FWD", 24: "FWD", 25: "FWD",
}


# ===========================================================================
# Per-90 normalization columns (player_embeddings)
# ===========================================================================

PER90_COLUMNS: list[str] = [
    "passes_completed",
    "passes_attempted",
    "tackles",
    "interceptions",
    "shots",
    "shots_on_target",
    "goals",
    "assists",
    "key_passes",
    "dribbles_completed",
    "dribbles_attempted",
    "aerial_duels_won",
    "aerial_duels_lost",
    "fouls_committed",
    "fouls_won",
    "crosses",
    "long_balls",
    "through_balls",
    "progressive_passes",
    "progressive_carries",
    "carries",
    "touches",
    "pressures",
    "blocks",
    "clearances",
    "recoveries",
    # v0.2.0 metrics
    "progressive_receptions",
    "big_chances",
    "passes_short",
    "passes_short_completed",
    "passes_medium",
    "passes_medium_completed",
    "passes_long",
    "passes_long_completed",
    "passes_under_pressure",
    "passes_under_pressure_completed",
    "switches_of_play",
    "ground_duels_won",
    "ground_duels_total",
]


# ===========================================================================
# Archetypes by position group (player_embeddings clustering)
# ===========================================================================

GROUP_ARCHETYPES: dict[str, dict[str, str]] = {
    "DEF": {
        "progressive_passes": "Playmaking CB",
        "tackles": "Stopper",
        "carries": "Ball-Playing FB",
        "crosses": "Attacking FB",
        "interceptions": "Sweeper",
    },
    "MID": {
        "passes_completed": "Deep-Lying Playmaker",
        "tackles": "Box-to-Box",
        "interceptions": "Defensive Mid",
        "key_passes": "Creative AM",
        "progressive_carries": "Mezzala",
    },
    "FWD": {
        "aerials_won": "Target Man",
        "dribbles_completed": "Inside Forward",
        "goals": "Poacher",
        "assists": "Complete Forward",
        "xg": "Goal Threat",
    },
    "GK": {
        "passes_completed": "Sweeper Keeper",
        "passes_long_completed": "Distribution GK",
        "clearances": "Traditional GK",
    },
}


# ===========================================================================
# Pressing constants
# ===========================================================================

# 6 horizontal zones across the pitch (StatsBomb coords: 0-120)
PRESSING_ZONE_BOUNDARIES: list[tuple[int, int]] = [
    (0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120),
]

DEFENSIVE_ACTION_TYPES: frozenset[str] = frozenset({
    "Pressure", "Duel", "Interception", "Foul Committed", "Block",
})

# High turnover threshold: within 40m of opponent goal (x >= 80 in StatsBomb)
HIGH_TURNOVER_X_THRESHOLD: float = 80.0

# Time windows (seconds) for pressing analysis
COUNTERPRESS_WINDOW: float = 5.0
PRESSING_RECOVERY_WINDOW: float = 5.0
SHOT_AFTER_TURNOVER_WINDOW: float = 15.0


# ===========================================================================
# Export / Report constants
# ===========================================================================

METRIC_CATEGORIES: dict[str, list[str]] = {
    "Attack": [
        "goals", "shots", "shots_on_target", "xg",
        "dribbles_attempted", "dribbles_completed",
        "big_chances", "big_chances_missed",
    ],
    "Creation": [
        "assists", "xa", "key_passes", "through_balls", "crosses",
        "progressive_passes", "progressive_carries",
        "progressive_receptions", "switches_of_play",
    ],
    "Defense": [
        "tackles", "interceptions",
        "blocks", "clearances", "aerials_won", "aerials_lost",
        "pressures", "pressure_regains",
        "ground_duels_won", "ground_duels_total", "tackle_success_rate",
    ],
    "Possession Value": [
        "xt_generated", "vaep_generated",
    ],
    "Possession": [
        "passes", "passes_completed", "pass_pct",
        "touches", "carries", "dispossessed",
        "fouls_committed", "fouls_won",
        "passes_short", "passes_short_completed",
        "passes_medium", "passes_medium_completed",
        "passes_long", "passes_long_completed",
        "passes_under_pressure", "passes_under_pressure_completed",
    ],
}

ALL_METRIC_COLS: list[str] = [
    "minutes_played", "goals", "assists", "shots", "shots_on_target",
    "xg", "xa", "passes", "passes_completed", "pass_pct",
    "progressive_passes", "progressive_carries",
    "key_passes", "through_balls", "crosses",
    "tackles", "interceptions",
    "blocks", "clearances", "aerials_won", "aerials_lost",
    "fouls_committed", "fouls_won",
    "dribbles_attempted", "dribbles_completed",
    "touches", "carries", "dispossessed",
    "pressures", "pressure_regains",
]

METRIC_LABELS: dict[str, str] = {
    "minutes_played": "Minutes Played",
    "goals": "Goals",
    "assists": "Assists",
    "shots": "Shots",
    "shots_on_target": "Shots on Target",
    "xg": "Expected Goals (xG)",
    "xa": "Expected Assists (xA)",
    "passes": "Passes Attempted",
    "passes_completed": "Passes Completed",
    "pass_pct": "Pass Accuracy (%)",
    "progressive_passes": "Progressive Passes",
    "progressive_carries": "Progressive Carries",
    "key_passes": "Key Passes",
    "through_balls": "Through Balls",
    "crosses": "Crosses",
    "tackles": "Tackles",
    "interceptions": "Interceptions",
    "blocks": "Blocks",
    "clearances": "Clearances",
    "aerials_won": "Aerial Duels Won",
    "aerials_lost": "Aerial Duels Lost",
    "fouls_committed": "Fouls Committed",
    "fouls_won": "Fouls Won",
    "dribbles_attempted": "Dribbles Attempted",
    "dribbles_completed": "Dribbles Completed",
    "touches": "Touches",
    "carries": "Carries",
    "dispossessed": "Dispossessed",
    "pressures": "Pressures",
    "pressure_regains": "Pressure Regains",
}


# ===========================================================================
# Pass distance thresholds (meters, StatsBomb coords)
# ===========================================================================

PASS_SHORT_THRESHOLD: float = 15.0
PASS_LONG_THRESHOLD: float = 30.0
SWITCH_OF_PLAY_LATERAL_THRESHOLD: float = 30.0


# ===========================================================================
# Shot quality threshold
# ===========================================================================

BIG_CHANCE_XG_THRESHOLD: float = 0.3
