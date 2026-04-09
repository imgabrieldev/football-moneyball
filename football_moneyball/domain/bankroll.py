"""Bankroll management module via Kelly Criterion.

Computes optimal stakes for positive-expectation bets,
using fractional Kelly to reduce variance.
"""

from __future__ import annotations


def kelly_criterion(prob: float, odds: float) -> float:
    """Compute the optimal Kelly fraction.

    f* = (b * p - q) / b

    Where:
    - b = odds - 1 (net odds)
    - p = probability of winning
    - q = 1 - p

    Parameters
    ----------
    prob : float
        Estimated probability of winning (0-1).
    odds : float
        Decimal odds.

    Returns
    -------
    float
        Fraction of the bankroll to bet (0 if there is no edge).
    """
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0

    b = odds - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b

    return max(f, 0.0)  # Never bet negative


def fractional_kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly — reduces variance by betting a fraction of Kelly.

    In practice, full Kelly is too aggressive. Using 25% of Kelly
    is the industry standard to reduce risk of ruin.

    Parameters
    ----------
    prob : float
        Estimated probability.
    odds : float
        Decimal odds.
    fraction : float
        Kelly fraction to use (default 0.25 = 25%).

    Returns
    -------
    float
        Adjusted fraction of the bankroll.
    """
    return kelly_criterion(prob, odds) * fraction


def calculate_stake(
    bankroll: float,
    prob: float,
    odds: float,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """Compute the bet amount with safety caps.

    Parameters
    ----------
    bankroll : float
        Total bankroll amount.
    prob : float
        Estimated probability.
    odds : float
        Decimal odds.
    kelly_fraction : float
        Kelly fraction to use.
    max_stake_pct : float
        Maximum stake as % of the bankroll (default 5%).

    Returns
    -------
    float
        Bet amount in monetary units.
    """
    if bankroll <= 0:
        return 0.0

    frac = fractional_kelly(prob, odds, kelly_fraction)
    stake = bankroll * frac

    # Cap at max percentage of bankroll
    max_stake = bankroll * max_stake_pct
    stake = min(stake, max_stake)

    return round(stake, 2)


def calculate_ev_per_bet(prob: float, odds: float, stake: float) -> float:
    """Compute the expected value of a bet.

    Parameters
    ----------
    prob : float
        Estimated probability.
    odds : float
        Decimal odds.
    stake : float
        Amount bet.

    Returns
    -------
    float
        Expected value (positive = profitable).
    """
    ev_per_unit = prob * odds - 1.0
    return round(ev_per_unit * stake, 2)
