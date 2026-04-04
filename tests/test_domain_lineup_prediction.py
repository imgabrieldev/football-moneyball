"""Testes para football_moneyball.domain.lineup_prediction.

Lógica pura, zero mocks. Inputs determinísticos.
"""

import pandas as pd

from football_moneyball.domain.lineup_prediction import (
    minutes_weight,
    probable_xi,
)


class TestMinutesWeight:
    def test_full_regular(self):
        # 5/5 jogos, 90min cada → 1.0
        assert minutes_weight(5, 450, 5) == 1.0

    def test_partial_regular(self):
        # 2/5 jogos, 90min cada → 0.4
        assert minutes_weight(2, 180, 5) == 0.4

    def test_half_minutes(self):
        # 5/5 jogos, 45min cada → 0.5
        assert abs(minutes_weight(5, 225, 5) - 0.5) < 1e-9

    def test_rare_substitute(self):
        # 1/5 jogos, 30min → 0.2 × 0.333... ≈ 0.0667
        w = minutes_weight(1, 30, 5)
        assert 0.06 < w < 0.07

    def test_zero_matches(self):
        assert minutes_weight(0, 0, 5) == 0.0

    def test_clamp_above_one(self):
        # Impossível ter >5/5 mas se acontecer, clamp pra 1.0
        assert minutes_weight(10, 900, 5) == 1.0

    def test_zero_last_n(self):
        assert minutes_weight(5, 450, 0) == 0.0


class TestProbableXI:
    def _make_aggregates(self, n_players: int) -> pd.DataFrame:
        """Cria agregados sintéticos: jogador i com minutos decrescentes."""
        rows = []
        for i in range(n_players):
            minutes = 450 - i * 20  # Player 0: 450, Player 1: 430, etc
            rows.append({
                "player_id": i + 1,
                "player_name": f"Player {i+1}",
                "matches_played": 5,
                "minutes_total": minutes,
                "xg_total": 1.0 + i * 0.1,
            })
        return pd.DataFrame(rows)

    def test_returns_top_11(self):
        aggs = self._make_aggregates(15)
        xi = probable_xi(aggs, last_n_matches=5)
        assert len(xi) == 11

    def test_fewer_than_11(self):
        aggs = self._make_aggregates(8)
        xi = probable_xi(aggs, last_n_matches=5)
        assert len(xi) == 8

    def test_ordered_by_minutes(self):
        aggs = self._make_aggregates(15)
        xi = probable_xi(aggs, last_n_matches=5)
        # Player 1 tem mais minutos → primeiro
        assert xi.iloc[0]["player_id"] == 1
        # Player 11 é o 11º → último
        assert xi.iloc[-1]["player_id"] == 11

    def test_xg_per_90_computed(self):
        aggs = self._make_aggregates(11)
        xi = probable_xi(aggs, last_n_matches=5)
        # Player 1: xG=1.0, minutos=450 → 0.2 xG/90
        p1 = xi[xi["player_id"] == 1].iloc[0]
        assert abs(p1["xg_per_90"] - 0.2) < 1e-9

    def test_weight_computed(self):
        aggs = self._make_aggregates(11)
        xi = probable_xi(aggs, last_n_matches=5)
        # Player 1: 5/5 jogos, 450min → peso 1.0
        p1 = xi[xi["player_id"] == 1].iloc[0]
        assert p1["weight"] == 1.0

    def test_empty_input(self):
        xi = probable_xi(pd.DataFrame())
        assert xi.empty

    def test_handles_zero_minutes(self):
        df = pd.DataFrame([{
            "player_id": 1, "player_name": "P1",
            "matches_played": 0, "minutes_total": 0, "xg_total": 0,
        }])
        xi = probable_xi(df)
        assert xi.iloc[0]["xg_per_90"] == 0.0
        assert xi.iloc[0]["weight"] == 0.0
