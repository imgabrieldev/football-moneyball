"""Fixtures compartilhadas para testes do Football Moneyball."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_events_df():
    """DataFrame de eventos StatsBomb fabricado para testes unitarios."""
    return pd.DataFrame([
        # Passes
        {"index": 0, "type": "Pass", "player": "Player A", "player_id": 1,
         "team": "Home", "location": [30, 40], "pass_end_location": [60, 40],
         "pass_outcome": np.nan, "possession": 1, "period": 1, "minute": 5,
         "second": 0, "timestamp": "00:05:00.000", "under_pressure": False},
        {"index": 1, "type": "Pass", "player": "Player A", "player_id": 1,
         "team": "Home", "location": [60, 40], "pass_end_location": [90, 35],
         "pass_outcome": np.nan, "possession": 1, "period": 1, "minute": 5,
         "second": 10, "timestamp": "00:05:10.000", "under_pressure": True},
        {"index": 2, "type": "Pass", "player": "Player B", "player_id": 2,
         "team": "Home", "location": [50, 20], "pass_end_location": [55, 60],
         "pass_outcome": "Incomplete", "possession": 2, "period": 1, "minute": 10,
         "second": 0, "timestamp": "00:10:00.000", "under_pressure": False},
        {"index": 3, "type": "Pass", "player": "Player B", "player_id": 2,
         "team": "Home", "location": [40, 10], "pass_end_location": [42, 15],
         "pass_outcome": np.nan, "possession": 3, "period": 1, "minute": 15,
         "second": 0, "timestamp": "00:15:00.000", "under_pressure": False},
        # Shot
        {"index": 4, "type": "Shot", "player": "Player A", "player_id": 1,
         "team": "Home", "location": [105, 40], "shot_outcome": "Goal",
         "shot_statsbomb_xg": 0.45, "possession": 1, "period": 1, "minute": 5,
         "second": 15, "timestamp": "00:05:15.000"},
        {"index": 5, "type": "Shot", "player": "Player A", "player_id": 1,
         "team": "Home", "location": [100, 35], "shot_outcome": "Saved",
         "shot_statsbomb_xg": 0.15, "possession": 4, "period": 1, "minute": 20,
         "second": 0, "timestamp": "00:20:00.000"},
        # Carry
        {"index": 6, "type": "Carry", "player": "Player B", "player_id": 2,
         "team": "Home", "location": [40, 40], "carry_end_location": [65, 38],
         "period": 1, "minute": 12, "second": 0, "timestamp": "00:12:00.000"},
        # Duel (Tackle)
        {"index": 7, "type": "Duel", "player": "Player C", "player_id": 3,
         "team": "Home", "duel_type": "Tackle", "duel_outcome": "Won",
         "location": [30, 50], "period": 1, "minute": 8, "second": 0,
         "timestamp": "00:08:00.000"},
        {"index": 8, "type": "Duel", "player": "Player C", "player_id": 3,
         "team": "Home", "duel_type": "Tackle", "duel_outcome": "Lost",
         "location": [35, 45], "period": 1, "minute": 25, "second": 0,
         "timestamp": "00:25:00.000"},
        # Pressure
        {"index": 9, "type": "Pressure", "player": "Player C", "player_id": 3,
         "team": "Home", "location": [70, 40], "counterpress": True,
         "period": 1, "minute": 11, "second": 0, "timestamp": "00:11:00.000"},
        {"index": 10, "type": "Pressure", "player": "Player C", "player_id": 3,
         "team": "Home", "location": [50, 30], "counterpress": False,
         "period": 1, "minute": 18, "second": 0, "timestamp": "00:18:00.000"},
        # Ball Recovery
        {"index": 11, "type": "Ball Recovery", "player": "Player C", "player_id": 3,
         "team": "Home", "location": [85, 40], "period": 1, "minute": 11,
         "second": 3, "timestamp": "00:11:03.000"},
        # Ball Receipt
        {"index": 12, "type": "Ball Receipt*", "player": "Player B", "player_id": 2,
         "team": "Home", "location": [60, 40], "possession": 1, "period": 1,
         "minute": 5, "second": 5, "timestamp": "00:05:05.000"},
        # Opponent pass (for PPDA)
        {"index": 13, "type": "Pass", "player": "Opp Player", "player_id": 10,
         "team": "Away", "location": [50, 40], "pass_end_location": [60, 45],
         "pass_outcome": np.nan, "possession": 5, "period": 1, "minute": 30,
         "second": 0, "timestamp": "00:30:00.000"},
        {"index": 14, "type": "Pass", "player": "Opp Player", "player_id": 10,
         "team": "Away", "location": [55, 30], "pass_end_location": [65, 35],
         "pass_outcome": np.nan, "possession": 5, "period": 1, "minute": 30,
         "second": 5, "timestamp": "00:30:05.000"},
        # Interception
        {"index": 15, "type": "Interception", "player": "Player C", "player_id": 3,
         "team": "Home", "location": [45, 40], "period": 1, "minute": 35,
         "second": 0, "timestamp": "00:35:00.000"},
    ])


@pytest.fixture
def mock_lineups():
    """Simula retorno de sb.lineups() com dados de posicao."""
    return {
        "Home": pd.DataFrame([
            {"player_id": 1, "player_name": "Player A", "jersey_number": 9,
             "positions": [{"position_id": 23, "position": "Striker",
                           "from": "00:00", "to": None, "from_period": 1,
                           "to_period": None, "start_reason": "Starting XI"}]},
            {"player_id": 2, "player_name": "Player B", "jersey_number": 8,
             "positions": [{"position_id": 14, "position": "Center Midfield",
                           "from": "00:00", "to": None, "from_period": 1,
                           "to_period": None, "start_reason": "Starting XI"}]},
            {"player_id": 3, "player_name": "Player C", "jersey_number": 4,
             "positions": [{"position_id": 4, "position": "Center Back",
                           "from": "00:00", "to": None, "from_period": 1,
                           "to_period": None, "start_reason": "Starting XI"}]},
        ]),
        "Away": pd.DataFrame([
            {"player_id": 10, "player_name": "Opp Player", "jersey_number": 7,
             "positions": [{"position_id": 17, "position": "Right Wing",
                           "from": "00:00", "to": None, "from_period": 1,
                           "to_period": None, "start_reason": "Starting XI"}]},
        ]),
    }
