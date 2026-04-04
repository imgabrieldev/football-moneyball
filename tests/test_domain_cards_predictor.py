"""Testes para football_moneyball.domain.cards_predictor."""

from football_moneyball.domain.cards_predictor import predict_cards


class TestPredictCards:
    def test_baseline(self):
        # 2 cards/jogo, 12 faltas média, juiz médio
        lh, la = predict_cards(2.0, 2.0, 12.0, 12.0, 1.0, 1.0)
        # base = 2 + 12*0.15*0.3 = 2 + 0.54 = 2.54
        assert abs(lh - 2.54) < 0.01
        assert abs(la - 2.54) < 0.01

    def test_strict_referee(self):
        # referee_factor=1.5 → multiplica lambda por 1.5
        lh_avg, _ = predict_cards(2.0, 2.0, 12.0, 12.0, 1.0, 1.0)
        lh_strict, _ = predict_cards(2.0, 2.0, 12.0, 12.0, 1.5, 1.0)
        assert abs(lh_strict - lh_avg * 1.5) < 0.01

    def test_derby(self):
        # derby_factor=1.2
        lh_normal, _ = predict_cards(2.0, 2.0, 12.0, 12.0, 1.0, 1.0)
        lh_derby, _ = predict_cards(2.0, 2.0, 12.0, 12.0, 1.0, 1.2)
        assert abs(lh_derby - lh_normal * 1.2) < 0.01

    def test_minimum_clamp(self):
        lh, la = predict_cards(0, 0, 0, 0, 1.0, 1.0)
        assert lh == 0.5
        assert la == 0.5

    def test_strict_ref_plus_derby(self):
        lh, _ = predict_cards(2.0, 2.0, 12.0, 12.0, 1.5, 1.2)
        expected = 2.54 * 1.5 * 1.2
        assert abs(lh - expected) < 0.02
