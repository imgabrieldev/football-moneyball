"""Tests for football_moneyball.domain.market_features."""

from football_moneyball.domain.market_features import (
    blend_with_market,
    consensus_devig,
    devig_odds,
)


class TestDevigOdds:
    def test_probabilities_sum_to_one(self):
        d = devig_odds(1.85, 3.40, 4.20)
        total = d["p_home"] + d["p_draw"] + d["p_away"]
        assert abs(total - 1.0) < 1e-9

    def test_home_favorite_has_higher_prob(self):
        d = devig_odds(1.50, 4.00, 7.00)
        assert d["p_home"] > d["p_draw"]
        assert d["p_home"] > d["p_away"]

    def test_invalid_odds_returns_uniform(self):
        d = devig_odds(1.0, 3.0, 3.0)  # odds<=1 = invalid
        assert d["p_home"] == 1 / 3
        assert d["p_draw"] == 1 / 3
        assert d["p_away"] == 1 / 3


class TestBlendWithMarket:
    def test_alpha_one_returns_model(self):
        model = {"home_win_prob": 0.7, "draw_prob": 0.2, "away_win_prob": 0.1}
        market = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
        blend = blend_with_market(model, market, alpha=1.0)
        assert abs(blend["home_win_prob"] - 0.7) < 1e-9
        assert abs(blend["draw_prob"] - 0.2) < 1e-9

    def test_alpha_zero_returns_market(self):
        model = {"home_win_prob": 0.7, "draw_prob": 0.2, "away_win_prob": 0.1}
        market = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
        blend = blend_with_market(model, market, alpha=0.0)
        assert abs(blend["home_win_prob"] - 0.5) < 1e-9
        assert abs(blend["draw_prob"] - 0.3) < 1e-9

    def test_alpha_half_averages(self):
        model = {"home_win_prob": 0.6, "draw_prob": 0.3, "away_win_prob": 0.1}
        market = {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}
        blend = blend_with_market(model, market, alpha=0.5)
        assert abs(blend["home_win_prob"] - 0.5) < 1e-9
        assert abs(blend["away_win_prob"] - 0.2) < 1e-9

    def test_result_sums_to_one(self):
        model = {"home_win_prob": 0.65, "draw_prob": 0.21, "away_win_prob": 0.14}
        market = {"p_home": 0.55, "p_draw": 0.25, "p_away": 0.20}
        blend = blend_with_market(model, market, alpha=0.6)
        total = blend["home_win_prob"] + blend["draw_prob"] + blend["away_win_prob"]
        assert abs(total - 1.0) < 1e-9


class TestConsensusDevig:
    def test_empty_returns_none(self):
        assert consensus_devig([]) is None

    def test_single_bookmaker(self):
        odds = [{"odds_home": 2.0, "odds_draw": 3.0, "odds_away": 4.0}]
        d = consensus_devig(odds)
        # p_h ~0.46, p_d ~0.31, p_a ~0.23 after devig
        assert d["p_home"] > d["p_draw"] > d["p_away"]
        assert abs(d["p_home"] + d["p_draw"] + d["p_away"] - 1.0) < 1e-9

    def test_averages_across_books(self):
        odds = [
            {"odds_home": 2.0, "odds_draw": 3.0, "odds_away": 4.0},
            {"odds_home": 2.1, "odds_draw": 3.1, "odds_away": 3.8},
        ]
        d = consensus_devig(odds)
        assert d is not None
        assert abs(d["p_home"] + d["p_draw"] + d["p_away"] - 1.0) < 1e-9
