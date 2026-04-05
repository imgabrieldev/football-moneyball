"""Calibração de modelos de predição.

Implementa:
1. Dixon-Coles correction (1997) — corrige Poisson independente em placares baixos
2. Platt scaling — calibra probabilidades via regressão logística

Lógica pura — zero deps de infra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


@dataclass
class PlattParams:
    """Parâmetros de uma calibração Platt: p_cal = sigmoid(a · logit(p) + b)."""
    a: float
    b: float

    def apply(self, p: np.ndarray | float) -> np.ndarray | float:
        """Aplica calibração a probabilidade(s) raw."""
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
        logit = np.log(p / (1 - p))
        return _sigmoid(self.a * logit + self.b)


# ---------------------------------------------------------------------------
# Dixon-Coles correction
# ---------------------------------------------------------------------------

def dixon_coles_tau(
    home_xg: float,
    away_xg: float,
    rho: float,
) -> np.ndarray:
    """Retorna matriz 2x2 de fatores τ para os 4 placares baixos.

    τ(0,0) = 1 - λh·λa·ρ
    τ(0,1) = 1 + λh·ρ
    τ(1,0) = 1 + λa·ρ
    τ(1,1) = 1 - ρ
    """
    tau = np.ones((2, 2), dtype=np.float64)
    tau[0, 0] = 1.0 - home_xg * away_xg * rho
    tau[0, 1] = 1.0 + home_xg * rho
    tau[1, 0] = 1.0 + away_xg * rho
    tau[1, 1] = 1.0 - rho
    # Garantir não-negatividade (τ < 0 quebra PMF)
    return np.maximum(tau, 1e-9)


def dixon_coles_score_matrix(
    home_xg: float,
    away_xg: float,
    rho: float,
    max_goals: int = 10,
) -> np.ndarray:
    """Calcula PMF conjunta P(X=x, Y=y) com correção Dixon-Coles.

    Returns
    -------
    np.ndarray
        Matriz (max_goals+1, max_goals+1) com probabilidades normalizadas.
    """
    home_pmf = poisson.pmf(np.arange(max_goals + 1), max(home_xg, 0.05))
    away_pmf = poisson.pmf(np.arange(max_goals + 1), max(away_xg, 0.05))
    joint = np.outer(home_pmf, away_pmf)

    # Aplicar τ nos 4 placares baixos
    tau = dixon_coles_tau(home_xg, away_xg, rho)
    joint[:2, :2] *= tau

    # Renormalizar (τ quebra a soma = 1)
    total = joint.sum()
    if total > 0:
        joint /= total
    return joint


def dixon_coles_log_likelihood(
    matches: list[tuple[float, float, int, int]],
    rho: float,
    max_goals: int = 10,
) -> float:
    """Log-likelihood de ρ dado um conjunto de partidas (λh, λa, goals_h, goals_a)."""
    ll = 0.0
    for home_xg, away_xg, gh, ga in matches:
        matrix = dixon_coles_score_matrix(home_xg, away_xg, rho, max_goals)
        x = min(int(gh), max_goals)
        y = min(int(ga), max_goals)
        p = matrix[x, y]
        if p > 0:
            ll += np.log(p)
        else:
            ll += -1e9
    return ll


def fit_dixon_coles_rho(
    matches: list[tuple[float, float, int, int]],
    rho_bounds: tuple[float, float] = (-0.25, 0.05),
) -> float:
    """Fitta ρ via MLE. Recebe list[(home_xg, away_xg, home_goals, away_goals)]."""
    def neg_ll(rho: float) -> float:
        return -dixon_coles_log_likelihood(matches, rho)

    result = minimize_scalar(
        neg_ll,
        bounds=rho_bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid numericamente estável."""
    z = np.asarray(z, dtype=np.float64)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def fit_platt_binary(
    raw_probs: np.ndarray,
    labels: np.ndarray,
) -> PlattParams:
    """Fitta Platt params via MLE (equivale a log-loss).

    Resolve: min Σ -[y·log(σ(a·logit(p)+b)) + (1-y)·log(1-σ(...))]
    via scipy.optimize.minimize.

    Parameters
    ----------
    raw_probs : np.ndarray
        Probabilidades raw do modelo (entre 0 e 1).
    labels : np.ndarray
        Labels binárias 0/1.
    """
    from scipy.optimize import minimize

    p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=np.float64)
    logit_p = np.log(p / (1 - p))

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        z = a * logit_p + b
        # log-sigmoid numericamente estável
        log_sig = -np.logaddexp(0, -z)
        log_one_minus_sig = -np.logaddexp(0, z)
        return -float(np.sum(y * log_sig + (1 - y) * log_one_minus_sig))

    # Start near identity: a=1, b=0
    result = minimize(
        neg_log_likelihood,
        x0=np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 1000},
    )
    return PlattParams(a=float(result.x[0]), b=float(result.x[1]))


def calibrate_1x2(
    raw_probs: np.ndarray,
    platt_home: PlattParams,
    platt_draw: PlattParams,
    platt_away: PlattParams,
) -> np.ndarray:
    """Aplica Platt scaling 3-class (one-vs-rest) e renormaliza.

    Parameters
    ----------
    raw_probs : np.ndarray
        Shape (n_samples, 3) com colunas [p_home, p_draw, p_away].
    platt_home/draw/away : PlattParams
        Parâmetros fitados.
    """
    raw = np.asarray(raw_probs, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    cal = np.zeros_like(raw)
    cal[:, 0] = platt_home.apply(raw[:, 0])
    cal[:, 1] = platt_draw.apply(raw[:, 1])
    cal[:, 2] = platt_away.apply(raw[:, 2])

    # Renormalizar para somar 1
    totals = cal.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    cal = cal / totals

    return cal if raw_probs.ndim > 1 else cal[0]


# ---------------------------------------------------------------------------
# Integração com simulate_match
# ---------------------------------------------------------------------------

def sample_scores_dixon_coles(
    home_xg: float,
    away_xg: float,
    rho: float,
    n_simulations: int,
    seed: int | None = None,
    max_goals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra (home_goals, away_goals) da distribuição Dixon-Coles corrigida.

    Returns
    -------
    (home_goals, away_goals) arrays de shape (n_simulations,).
    """
    rng = np.random.default_rng(seed)
    matrix = dixon_coles_score_matrix(home_xg, away_xg, rho, max_goals)
    flat = matrix.flatten()
    flat = flat / flat.sum()  # Safety renormalize

    indices = rng.choice(len(flat), size=n_simulations, p=flat)
    n_cols = max_goals + 1
    home_goals = indices // n_cols
    away_goals = indices % n_cols
    return home_goals.astype(np.int64), away_goals.astype(np.int64)
