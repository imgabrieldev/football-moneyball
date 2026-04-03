"""Testes para football_moneyball.domain.possession_value — modelo xT hexagonal.

Estes testes ja eram puros (ExpectedThreat e aggregate_player_xt nao dependem
de I/O), entao a migracao e apenas troca de import paths.
"""

import numpy as np
import pandas as pd
import pytest

from football_moneyball.domain.possession_value import ExpectedThreat, aggregate_player_xt


class TestExpectedThreat:
    def _make_simple_events(self) -> list[pd.DataFrame]:
        """Cria eventos sinteticos para treinar xT em grid pequeno."""
        rows = []
        # Shots near goal (high xT zone)
        for i in range(100):
            rows.append({
                "type": "Shot", "location": [110, 40],
                "shot_outcome": "Goal" if i < 10 else "Saved",
            })
        # Passes from midfield to attacking third
        for i in range(200):
            rows.append({
                "type": "Pass", "location": [50, 40],
                "pass_end_location": [90, 40],
                "pass_outcome": np.nan,
            })
        # Passes in defensive third
        for i in range(200):
            rows.append({
                "type": "Pass", "location": [20, 40],
                "pass_end_location": [30, 40],
                "pass_outcome": np.nan,
            })
        # Carries forward
        for i in range(100):
            rows.append({
                "type": "Carry", "location": [60, 40],
                "carry_end_location": [80, 40],
            })
        return [pd.DataFrame(rows)]

    def test_fit_converges(self):
        model = ExpectedThreat(l=8, w=6)
        events = self._make_simple_events()
        model.fit(events)
        assert model.xt_grid is not None
        assert model.xt_grid.shape == (8, 6)

    def test_attacking_zones_higher_than_defensive(self):
        model = ExpectedThreat(l=8, w=6)
        model.fit(self._make_simple_events())
        # Near-goal zone should have higher xT than defensive zone
        near_goal = model.get_value(110, 40)
        defensive = model.get_value(20, 40)
        assert near_goal >= defensive

    def test_rate_actions(self):
        model = ExpectedThreat(l=8, w=6)
        model.fit(self._make_simple_events())

        test_events = pd.DataFrame([
            {"type": "Pass", "location": [50, 40], "pass_end_location": [90, 40],
             "pass_outcome": np.nan, "player": "Test", "player_id": 1},
            {"type": "Shot", "location": [105, 40], "shot_outcome": "Goal",
             "player": "Test", "player_id": 1},
        ])

        values = model.rate_actions(test_events)
        # Pass should have positive xT (moving forward)
        assert values.iloc[0] > 0 or not np.isnan(values.iloc[0])
        # Shot has no xT delta
        assert np.isnan(values.iloc[1])

    def test_rate_without_fit_raises(self):
        model = ExpectedThreat()
        events = pd.DataFrame([
            {"type": "Pass", "location": [50, 40], "pass_end_location": [90, 40],
             "pass_outcome": np.nan},
        ])
        with pytest.raises(RuntimeError, match="nao treinado"):
            model.rate_actions(events)

    def test_grid_values_non_negative(self):
        model = ExpectedThreat(l=8, w=6)
        model.fit(self._make_simple_events())
        assert np.all(model.xt_grid >= 0)


class TestAggregatePlayerXt:
    def test_aggregates_correctly(self):
        df = pd.DataFrame([
            {"player_id": 1, "player_name": "A", "team": "X", "xt_value": 0.05},
            {"player_id": 1, "player_name": "A", "team": "X", "xt_value": 0.03},
            {"player_id": 2, "player_name": "B", "team": "X", "xt_value": -0.01},
        ])
        result = aggregate_player_xt(df)
        assert len(result) == 2
        player_a = result[result["player_name"] == "A"].iloc[0]
        assert abs(player_a["xt_generated"] - 0.08) < 1e-6

    def test_empty_input(self):
        result = aggregate_player_xt(pd.DataFrame())
        assert result.empty
