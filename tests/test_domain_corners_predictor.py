"""Testes para football_moneyball.domain.corners_predictor."""

from football_moneyball.domain.corners_predictor import predict_corners


class TestPredictCorners:
    def test_average_teams(self):
        # Todos fazem/sofrem média da liga (5) → λ = 5
        lh, la = predict_corners(5.0, 5.0, 5.0, 5.0, 5.0)
        assert abs(lh - 5.0) < 1e-9
        assert abs(la - 5.0) < 1e-9

    def test_strong_home_attack(self):
        # Home faz 8 corners, away sofre 6 (↑20%), league 5
        # λ_home = 8 × (6/5) = 9.6
        lh, la = predict_corners(8.0, 4.0, 4.0, 6.0, 5.0)
        assert abs(lh - 9.6) < 1e-9

    def test_minimum_clamp(self):
        # Zero corners → clamped to 1.0
        lh, la = predict_corners(0, 0, 0, 0, 5.0)
        assert lh == 1.0
        assert la == 1.0

    def test_zero_league_fallback(self):
        lh, la = predict_corners(5.0, 5.0, 5.0, 5.0, 0)
        # Falls back to 5.0 default
        assert abs(lh - 5.0) < 1e-9

    def test_symmetric_swap(self):
        # Trocar home com away deve produzir valores simétricos
        lh1, la1 = predict_corners(6.0, 4.0, 3.0, 5.0, 5.0)
        lh2, la2 = predict_corners(4.0, 6.0, 5.0, 3.0, 5.0)
        assert abs(lh1 - la2) < 1e-9
        assert abs(la1 - lh2) < 1e-9
