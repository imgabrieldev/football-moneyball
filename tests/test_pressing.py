"""Testes para pressing.py — metricas de pressing v0.2.0."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from football_moneyball.pressing import (
    _compute_ppda,
    _compute_pressing_success,
    _compute_counter_pressing_fraction,
    _compute_high_turnovers,
    _compute_pressing_zones,
    compute_match_pressing,
)


class TestComputePPDA:
    def test_basic_ppda(self, mock_events_df):
        # Home team: 1 defensive action (Pressure at min 11, Pressure at min 18,
        # Duel x2, Interception, Block = but let's count)
        ppda = _compute_ppda(mock_events_df, "Home", "Away")
        # Away has 2 passes with x >= 40 (both at x=50, 55)
        # Home has defensive actions: 2 Pressure + 2 Duel + 1 Interception = 5
        assert ppda == round(2 / 5, 2)

    def test_ppda_no_defensive_actions(self):
        events = pd.DataFrame([
            {"type": "Pass", "team": "Away", "location": [50, 40]},
        ])
        ppda = _compute_ppda(events, "Home", "Away")
        assert ppda == float("inf")


class TestComputePressingSuccess:
    def test_basic_success_rate(self, mock_events_df):
        rate = _compute_pressing_success(mock_events_df, "Home")
        # Home has 2 Pressures (min 11:00 and min 18:00)
        # Ball Recovery at min 11:03 (3s after first pressure) → success
        # No recovery near min 18 → fail
        assert rate == 50.0

    def test_no_pressures(self):
        events = pd.DataFrame([
            {"type": "Pass", "team": "Home", "player": "A", "period": 1,
             "minute": 1, "second": 0},
        ])
        rate = _compute_pressing_success(events, "Home")
        assert rate == 0.0


class TestComputeHighTurnovers:
    def test_detects_high_turnovers(self, mock_events_df):
        high, shot_ending = _compute_high_turnovers(mock_events_df, "Home")
        # Ball Recovery at x=85 >= 80 → 1 high turnover
        assert high == 1

    def test_no_recoveries(self):
        events = pd.DataFrame([
            {"type": "Pass", "team": "Home", "location": [50, 40]},
        ])
        high, shot_ending = _compute_high_turnovers(events, "Home")
        assert high == 0
        assert shot_ending == 0


class TestComputePressingZones:
    def test_zones_sum_to_100(self, mock_events_df):
        zones = _compute_pressing_zones(mock_events_df, "Home")
        assert len(zones) == 6
        assert abs(sum(zones) - 100.0) < 1.0  # Allow rounding error

    def test_empty_returns_zeros(self):
        events = pd.DataFrame([
            {"type": "Pass", "team": "Home", "location": [50, 40]},
        ])
        zones = _compute_pressing_zones(events, "Home")
        assert zones == [0.0] * 6


class TestComputeMatchPressing:
    @patch("football_moneyball.pressing.sb")
    def test_returns_both_teams(self, mock_sb, mock_events_df):
        mock_sb.events.return_value = mock_events_df
        df = compute_match_pressing(match_id=1)
        assert len(df) == 2
        assert set(df["team"]) == {"Home", "Away"}

    @patch("football_moneyball.pressing.sb")
    def test_has_all_columns(self, mock_sb, mock_events_df):
        mock_sb.events.return_value = mock_events_df
        df = compute_match_pressing(match_id=1)
        expected_cols = [
            "team", "ppda", "pressing_success_rate", "counter_pressing_fraction",
            "high_turnovers", "shot_ending_high_turnovers",
            "pressing_zone_1", "pressing_zone_2", "pressing_zone_3",
            "pressing_zone_4", "pressing_zone_5", "pressing_zone_6",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    @patch("football_moneyball.pressing.sb")
    def test_empty_events(self, mock_sb):
        mock_sb.events.return_value = pd.DataFrame()
        df = compute_match_pressing(match_id=1)
        assert df.empty
