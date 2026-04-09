"""Tests for football_moneyball.domain.shots_predictor."""

from football_moneyball.domain.shots_predictor import predict_shots


class TestPredictShots:
    def test_average(self):
        lh, la = predict_shots(10.0, 10.0, 10.0, 10.0, 10.0)
        assert abs(lh - 10.0) < 1e-9
        assert abs(la - 10.0) < 1e-9

    def test_attacking_home(self):
        # Home shoots 14, away concedes 12 (league 10)
        # lambda = 14 * (12/10) = 16.8
        lh, la = predict_shots(14.0, 8.0, 8.0, 12.0, 10.0)
        assert abs(lh - 16.8) < 1e-9

    def test_minimum_clamp(self):
        lh, la = predict_shots(0, 0, 0, 0, 10.0)
        assert lh == 3.0
        assert la == 3.0
