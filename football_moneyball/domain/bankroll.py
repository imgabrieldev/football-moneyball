"""Modulo de gestao de bankroll via Kelly Criterion.

Calcula stakes otimos para apostas com expectativa positiva,
usando Kelly fracionario para reduzir variancia.
"""

from __future__ import annotations


def kelly_criterion(prob: float, odds: float) -> float:
    """Calcula a fracao Kelly otima.

    f* = (b * p - q) / b

    Onde:
    - b = odds - 1 (net odds)
    - p = probabilidade de ganhar
    - q = 1 - p

    Parameters
    ----------
    prob : float
        Probabilidade estimada de ganhar (0-1).
    odds : float
        Odds decimais.

    Returns
    -------
    float
        Fracao do bankroll a apostar (0 se nao ha edge).
    """
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0

    b = odds - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b

    return max(f, 0.0)  # Never bet negative


def fractional_kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Kelly fracionario — reduz variancia ao apostar uma fracao do Kelly.

    Na pratica, Kelly puro e muito agressivo. Usar 25% do Kelly
    e o padrao da industria para reduzir risco de ruina.

    Parameters
    ----------
    prob : float
        Probabilidade estimada.
    odds : float
        Odds decimais.
    fraction : float
        Fracao do Kelly a usar (default 0.25 = 25%).

    Returns
    -------
    float
        Fracao ajustada do bankroll.
    """
    return kelly_criterion(prob, odds) * fraction


def calculate_stake(
    bankroll: float,
    prob: float,
    odds: float,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """Calcula o valor da aposta com limites de seguranca.

    Parameters
    ----------
    bankroll : float
        Valor total do bankroll.
    prob : float
        Probabilidade estimada.
    odds : float
        Odds decimais.
    kelly_fraction : float
        Fracao do Kelly a usar.
    max_stake_pct : float
        Stake maximo como % do bankroll (default 5%).

    Returns
    -------
    float
        Valor da aposta em unidades monetarias.
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
    """Calcula o valor esperado de uma aposta.

    Parameters
    ----------
    prob : float
        Probabilidade estimada.
    odds : float
        Odds decimais.
    stake : float
        Valor apostado.

    Returns
    -------
    float
        Valor esperado (positivo = lucrativo).
    """
    ev_per_unit = prob * odds - 1.0
    return round(ev_per_unit * stake, 2)
