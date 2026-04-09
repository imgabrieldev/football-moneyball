"""Tests for domain/bankroll.py — Kelly Criterion."""

import pytest

from football_moneyball.domain.bankroll import (
    kelly_criterion,
    fractional_kelly,
    calculate_stake,
    calculate_ev_per_bet,
)


class TestKellyCriterion:
    def test_positive_edge(self):
        # prob=0.6, odds=2.0 → f* = (1*0.6 - 0.4)/1 = 0.2
        f = kelly_criterion(0.6, 2.0)
        assert abs(f - 0.2) < 0.001

    def test_no_edge(self):
        # prob=0.5, odds=2.0 → f* = (1*0.5 - 0.5)/1 = 0
        f = kelly_criterion(0.5, 2.0)
        assert f == 0.0

    def test_negative_edge(self):
        # prob=0.3, odds=2.0 → f* = (1*0.3 - 0.7)/1 = -0.4 → capped at 0
        f = kelly_criterion(0.3, 2.0)
        assert f == 0.0

    def test_high_odds(self):
        # prob=0.15, odds=10.0 → f* = (9*0.15 - 0.85)/9 = 0.0556
        f = kelly_criterion(0.15, 10.0)
        assert abs(f - 0.0556) < 0.001

    def test_invalid_odds(self):
        assert kelly_criterion(0.6, 0.0) == 0.0
        assert kelly_criterion(0.6, 1.0) == 0.0


class TestFractionalKelly:
    def test_quarter_kelly(self):
        full = kelly_criterion(0.6, 2.0)
        frac = fractional_kelly(0.6, 2.0, fraction=0.25)
        assert abs(frac - full * 0.25) < 0.001

    def test_half_kelly(self):
        full = kelly_criterion(0.6, 2.0)
        frac = fractional_kelly(0.6, 2.0, fraction=0.5)
        assert abs(frac - full * 0.5) < 0.001


class TestCalculateStake:
    def test_basic_stake(self):
        stake = calculate_stake(1000.0, 0.6, 2.0, kelly_fraction=0.25)
        assert stake > 0
        assert stake <= 50.0  # max 5% of 1000

    def test_max_cap(self):
        # Kelly might suggest more than 5%
        stake = calculate_stake(1000.0, 0.9, 2.0, kelly_fraction=1.0)
        assert stake <= 50.0  # capped at 5%

    def test_zero_bankroll(self):
        assert calculate_stake(0.0, 0.6, 2.0) == 0.0

    def test_no_edge_zero_stake(self):
        stake = calculate_stake(1000.0, 0.3, 2.0)
        assert stake == 0.0


class TestCalculateEVPerBet:
    def test_positive_ev(self):
        ev = calculate_ev_per_bet(0.6, 2.0, 100.0)
        assert ev == 20.0  # 0.6*2 - 1 = 0.2, * 100 = 20

    def test_negative_ev(self):
        ev = calculate_ev_per_bet(0.4, 2.0, 100.0)
        assert ev == -20.0
