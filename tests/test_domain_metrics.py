"""Testes para football_moneyball.domain.metrics — metricas via modulo hexagonal.

Diferenca chave do test_player_metrics.py: funcoes de dominio recebem
DataFrames diretamente (sem sb.events/sb.lineups), entao ZERO mocks sao
necessarios.
"""

import pandas as pd
import pytest

from football_moneyball.domain.constants import POSITION_GROUP_MAP
from football_moneyball.domain.metrics import (
    extract_match_metrics,
    extract_player_positions,
)


class TestPositionGroupMap:
    def test_goalkeeper(self):
        assert POSITION_GROUP_MAP[1] == "GK"

    def test_defenders(self):
        for pid in [2, 3, 4, 5, 6, 7, 8]:
            assert POSITION_GROUP_MAP[pid] == "DEF"

    def test_midfielders(self):
        for pid in [9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]:
            assert POSITION_GROUP_MAP[pid] == "MID"

    def test_forwards(self):
        for pid in [17, 21, 22, 23, 24, 25]:
            assert POSITION_GROUP_MAP[pid] == "FWD"

    def test_covers_all_25_positions(self):
        assert len(POSITION_GROUP_MAP) == 25


class TestExtractPlayerPositions:
    def test_extracts_positions_from_lineups(self, mock_lineups):
        positions = extract_player_positions(mock_lineups)

        assert positions[1] == "FWD"   # Striker (23)
        assert positions[2] == "MID"   # Center Midfield (14)
        assert positions[3] == "DEF"   # Center Back (4)
        assert positions[10] == "FWD"  # Right Wing (17)


class TestExtractMatchMetrics:
    def test_pass_breakdown(self, mock_events_df):
        df = extract_match_metrics(mock_events_df)

        player_a = df[df["player_name"] == "Player A"].iloc[0]
        # Player A has 2 passes: one 30y (medium) and one 30+y (long)
        assert player_a["passes_short"] + player_a["passes_medium"] + player_a["passes_long"] == 2

    def test_big_chances(self, mock_events_df):
        df = extract_match_metrics(mock_events_df)

        player_a = df[df["player_name"] == "Player A"].iloc[0]
        # Player A has 1 shot with xG=0.45 (big chance) and 1 with xG=0.15
        assert player_a["big_chances"] == 1
        # The big chance was a Goal, so big_chances_missed = 0
        assert player_a["big_chances_missed"] == 0

    def test_ground_duels(self, mock_events_df):
        df = extract_match_metrics(mock_events_df)

        player_c = df[df["player_name"] == "Player C"].iloc[0]
        assert player_c["ground_duels_total"] == 2
        assert player_c["ground_duels_won"] == 1
        assert player_c["tackle_success_rate"] == 50.0

    def test_passes_under_pressure(self, mock_events_df):
        df = extract_match_metrics(mock_events_df)

        player_a = df[df["player_name"] == "Player A"].iloc[0]
        # Player A has 1 pass under pressure (index 1)
        assert player_a["passes_under_pressure"] == 1
        assert player_a["passes_under_pressure_completed"] == 1

    def test_switches_of_play(self, mock_events_df):
        df = extract_match_metrics(mock_events_df)

        player_b = df[df["player_name"] == "Player B"].iloc[0]
        # Player B pass index 2: |60-20| = 40 > 30 -> switch
        assert player_b["switches_of_play"] == 1

    def test_all_new_columns_present(self, mock_events_df):
        df = extract_match_metrics(mock_events_df)

        new_cols = [
            "progressive_receptions", "big_chances", "big_chances_missed",
            "passes_short", "passes_short_completed",
            "passes_medium", "passes_medium_completed",
            "passes_long", "passes_long_completed",
            "passes_under_pressure", "passes_under_pressure_completed",
            "switches_of_play",
            "ground_duels_won", "ground_duels_total", "tackle_success_rate",
        ]
        for col in new_cols:
            assert col in df.columns, f"Missing column: {col}"
