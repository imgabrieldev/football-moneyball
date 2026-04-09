"""Tests for domain/value_detector.py — value bet detection."""

import pytest

from football_moneyball.domain.value_detector import (
    odds_to_implied_prob,
    remove_vig,
    calculate_edge,
    expected_value,
    find_value_bets,
)


class TestOddsToImpliedProb:
    def test_even_odds(self):
        assert odds_to_implied_prob(2.0) == 0.5

    def test_favorite(self):
        prob = odds_to_implied_prob(1.5)
        assert abs(prob - 0.6667) < 0.001

    def test_underdog(self):
        prob = odds_to_implied_prob(5.0)
        assert abs(prob - 0.2) < 0.001

    def test_zero_odds(self):
        assert odds_to_implied_prob(0.0) == 0.0


class TestRemoveVig:
    def test_removes_margin(self):
        # Odds: 1.80, 3.50, 4.50 -> implied: 0.556, 0.286, 0.222 -> sum=1.064
        probs = [0.556, 0.286, 0.222]
        clean = remove_vig(probs)
        assert abs(sum(clean) - 1.0) < 0.01

    def test_already_fair(self):
        probs = [0.5, 0.3, 0.2]
        clean = remove_vig(probs)
        assert abs(sum(clean) - 1.0) < 0.01


class TestCalculateEdge:
    def test_positive_edge(self):
        edge = calculate_edge(0.60, 0.55)
        assert edge == 0.05

    def test_negative_edge(self):
        edge = calculate_edge(0.40, 0.55)
        assert edge == -0.15

    def test_zero_edge(self):
        edge = calculate_edge(0.50, 0.50)
        assert edge == 0.0


class TestExpectedValue:
    def test_positive_ev(self):
        # prob=0.6, odds=2.0 -> EV = 0.6*2 - 1 = 0.2
        ev = expected_value(0.6, 2.0)
        assert abs(ev - 0.2) < 0.001

    def test_negative_ev(self):
        # prob=0.4, odds=2.0 -> EV = 0.4*2 - 1 = -0.2
        ev = expected_value(0.4, 2.0)
        assert abs(ev - (-0.2)) < 0.001


class TestFindValueBets:
    def test_finds_value(self):
        predictions = {
            "home_win_prob": 0.60,
            "draw_prob": 0.20,
            "away_win_prob": 0.20,
            "over_25": 0.55,
            "btts_prob": 0.50,
        }
        odds_data = [{
            "name": "bet365",
            "markets": [
                {"market": "h2h", "outcome": "Home", "odds": 1.80},  # implied 55.6%
                {"market": "h2h", "outcome": "Draw", "odds": 3.50},
                {"market": "h2h", "outcome": "Away", "odds": 5.00},
            ],
        }]
        vbets = find_value_bets(predictions, odds_data, min_edge=0.03)
        # Home: model 60% vs implied 55.6% -> edge 4.4% -> should be found
        assert len(vbets) >= 1
        assert any(vb["outcome"] == "Home" for vb in vbets)

    def test_no_value_with_high_threshold(self):
        predictions = {
            "home_win_prob": 0.56,
            "draw_prob": 0.22,
            "away_win_prob": 0.22,
            "over_25": 0.50,
            "btts_prob": 0.50,
        }
        odds_data = [{
            "name": "pinnacle",
            "markets": [
                {"market": "h2h", "outcome": "Home", "odds": 1.80},
            ],
        }]
        # Edge = 56% - 55.6% = 0.4% -> below 5% threshold
        vbets = find_value_bets(predictions, odds_data, min_edge=0.05)
        assert len(vbets) == 0

    def test_empty_odds(self):
        predictions = {"home_win_prob": 0.60}
        vbets = find_value_bets(predictions, [], min_edge=0.03)
        assert len(vbets) == 0
