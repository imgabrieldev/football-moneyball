"""Prediction model calibration.

Implementa:
1. Dixon-Coles correction (1997) — corrige Poisson independente in placares baixos
2. Platt scaling — calibrate probabilities via logistic regression (sigmoid)
3. Isotonic regression — non-parametric monotonic calibration (v1.11.0)
4. Temperature scaling — 1-parameter calibration via rescaled softmax (v1.11.0)
5. Calibration metrics: Brier multi-class, ECE

Pure logic — zero infra deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression


@dataclass
class PlattParams:
    """Parameters of a calibration Platt: p_cal = sigmoid(a · logit(p) + b)."""
    a: float
    b: float

    def apply(self, p: np.ndarray | float) -> np.ndarray | float:
        """Apply calibration to raw probability(ies)."""
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
    """Returns matriz 2x2 of fatores τ for the 4 placares baixos.

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
    # Garantir not-negatividade (τ < 0 quebra PMF)
    return np.maximum(tau, 1e-9)


def dixon_coles_score_matrix(
    home_xg: float,
    away_xg: float,
    rho: float,
    max_goals: int = 10,
) -> np.ndarray:
    """Compute joint PMF P(X=x, Y=y) with correction Dixon-Coles.

    Returns
    -------
    np.ndarray
        Matriz (max_goals+1, max_goals+1) with probabilitys normalizadas.
    """
    home_pmf = poisson.pmf(np.arange(max_goals + 1), max(home_xg, 0.05))
    away_pmf = poisson.pmf(np.arange(max_goals + 1), max(away_xg, 0.05))
    joint = np.outer(home_pmf, away_pmf)

    # Aplicar τ in the 4 placares baixos
    tau = dixon_coles_tau(home_xg, away_xg, rho)
    joint[:2, :2] *= tau

    # Renormalizar (τ quebra a sum = 1)
    total = joint.sum()
    if total > 0:
        joint /= total
    return joint


def dixon_coles_log_likelihood(
    matches: list[tuple[float, float, int, int]],
    rho: float,
    max_goals: int = 10,
) -> float:
    """Log-likelihood of ρ dado a conjunto of matches (λh, λa, goals_h, goals_a)."""
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
    """Fitta ρ via MLE. Receives list[(home_xg, away_xg, home_goals, away_goals)]."""
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
# Bivariate Poisson diagonal-inflated (Karlis & Ntzoufras, 2003)
# ---------------------------------------------------------------------------

def bivariate_poisson_score_matrix(
    home_xg: float,
    away_xg: float,
    lambda3: float = 0.10,
    max_goals: int = 10,
) -> np.ndarray:
    """PMF joint via bivariate Poisson diagonal-inflated.

    X = X1 + X3,  Y = X2 + X3  where X1~Poi(λ1), X2~Poi(λ2), X3~Poi(λ3).
    λ1 = max(home_xg - λ3, 0.05), λ2 = max(away_xg - λ3, 0.05).

    λ3 > 0 infla P(X=Y) naturalmente (empates), without distorcer o resto.

    Returns
    -------
    np.ndarray
        Matriz (max_goals+1, max_goals+1).
    """
    lam3 = max(float(lambda3), 0.0)
    lam1 = max(float(home_xg) - lam3, 0.05)
    lam2 = max(float(away_xg) - lam3, 0.05)

    n = max_goals + 1
    joint = np.zeros((n, n), dtype=np.float64)

    # P(X=x, Y=y) = Σ_{k=0}^{min(x,y)} P(X1=x-k) P(X2=y-k) P(X3=k)
    pmf1 = poisson.pmf(np.arange(n), lam1)
    pmf2 = poisson.pmf(np.arange(n), lam2)
    pmf3 = poisson.pmf(np.arange(n), lam3)

    for k in range(n):
        if pmf3[k] < 1e-15:
            break
        for x in range(k, n):
            for y in range(k, n):
                joint[x, y] += pmf1[x - k] * pmf2[y - k] * pmf3[k]

    total = joint.sum()
    if total > 0:
        joint /= total
    return joint


def sample_scores_bivariate(
    home_xg: float,
    away_xg: float,
    lambda3: float = 0.10,
    n_simulations: int = 10_000,
    seed: int | None = None,
    max_goals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (home_goals, away_goals) from the distribution bivariate Poisson."""
    rng = np.random.default_rng(seed)
    matrix = bivariate_poisson_score_matrix(home_xg, away_xg, lambda3, max_goals)
    flat = matrix.flatten()
    flat = flat / flat.sum()

    indices = rng.choice(len(flat), size=n_simulations, p=flat)
    n_cols = max_goals + 1
    home_goals = indices // n_cols
    away_goals = indices % n_cols
    return home_goals.astype(np.int64), away_goals.astype(np.int64)


def bivariate_poisson_log_likelihood(
    matches: list[tuple[float, float, int, int]],
    lambda3: float,
    max_goals: int = 10,
) -> float:
    """Log-likelihood of λ3 dado matches [(λh, λa, goals_h, goals_a)]."""
    ll = 0.0
    for home_xg, away_xg, gh, ga in matches:
        matrix = bivariate_poisson_score_matrix(home_xg, away_xg, lambda3, max_goals)
        x = min(int(gh), max_goals)
        y = min(int(ga), max_goals)
        p = matrix[x, y]
        if p > 0:
            ll += np.log(p)
        else:
            ll += -1e9
    return ll


def fit_lambda3(
    matches: list[tuple[float, float, int, int]],
    bounds: tuple[float, float] = (0.0, 0.40),
) -> float:
    """Fitta λ3 via MLE. Receives list[(home_xg, away_xg, home_goals, away_goals)]."""
    def neg_ll(lam3: float) -> float:
        return -bivariate_poisson_log_likelihood(matches, lam3)

    result = minimize_scalar(
        neg_ll,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
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
        Probabilidades raw of the model (between 0 and 1).
    labels : np.ndarray
        Binary labels 0/1.
    """
    from scipy.optimize import minimize

    p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=np.float64)
    logit_p = np.log(p / (1 - p))

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        z = a * logit_p + b
        # log-sigmoid numerically stable
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
    """Apply Platt scaling 3-class (one-vs-rest) and renormaliza.

    Parameters
    ----------
    raw_probs : np.ndarray
        Shape (n_samples, 3) with colunas [p_home, p_draw, p_away].
    platt_home/draw/away : PlattParams
        Parameters fitados.
    """
    raw = np.asarray(raw_probs, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    cal = np.zeros_like(raw)
    cal[:, 0] = platt_home.apply(raw[:, 0])
    cal[:, 1] = platt_draw.apply(raw[:, 1])
    cal[:, 2] = platt_away.apply(raw[:, 2])

    # Renormalizar for somar 1
    totals = cal.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    cal = cal / totals

    return cal if raw_probs.ndim > 1 else cal[0]


# ---------------------------------------------------------------------------
# Temperature scaling (v1.11.0)
# ---------------------------------------------------------------------------

@dataclass
class TemperatureScaler:
    """3-class temperature scaling calibration.

    Apply p_cal_k = p_k^(1/T) / Z, where Z normaliza for somar 1.
    T > 1 comprime (reduz overconfidence), T < 1 afia.
    """
    T: float = 1.0

    def apply(self, probs_3class: np.ndarray | list) -> np.ndarray:
        """Input/output: (n, 3) ou (3,). Apply p^(1/T) + renormaliza."""
        p = np.clip(np.asarray(probs_3class, dtype=np.float64), 1e-12, 1.0)
        original_ndim = p.ndim
        if p.ndim == 1:
            p = p.reshape(1, -1)
        # p^(1/T)
        scaled = np.power(p, 1.0 / max(self.T, 1e-6))
        totals = scaled.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        cal = scaled / totals
        return cal if original_ndim > 1 else cal[0]


def fit_temperature(
    raw_probs_3class: np.ndarray,
    y_3class: np.ndarray,
) -> TemperatureScaler:
    """Fit T via multi-class NLL minimization (Nelder-Mead).

    Parameters
    ----------
    raw_probs_3class : np.ndarray
        Shape (n, 3) com [p_home, p_draw, p_away].
    y_3class : np.ndarray
        Shape (n, 3) one-hot of the resultado real.
    """
    from scipy.optimize import minimize

    raw = np.clip(np.asarray(raw_probs_3class, dtype=np.float64), 1e-12, 1.0)
    y = np.asarray(y_3class, dtype=np.float64)

    def neg_log_likelihood(params):
        T = max(params[0], 1e-6)
        scaled = np.power(raw, 1.0 / T)
        totals = scaled.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        cal = scaled / totals
        cal = np.clip(cal, 1e-12, 1.0)
        # NLL = -sum(y * log(p))
        return float(-np.sum(y * np.log(cal)))

    result = minimize(
        neg_log_likelihood,
        x0=np.array([1.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200},
    )
    return TemperatureScaler(T=float(max(result.x[0], 1e-6)))


def calibrate_1x2_temperature(
    raw_probs: np.ndarray,
    temp: TemperatureScaler,
) -> np.ndarray:
    """Apply temperature scaling 3-class direto."""
    return temp.apply(raw_probs)


# ---------------------------------------------------------------------------
# Isotonic regression (v1.11.0)
# ---------------------------------------------------------------------------

@dataclass
class IsotonicCalibrator:
    """Binary isotonic calibration — linearly interpolated in the thresholds fittados.

    Extracted from sklearn.isotonic.IsotonicRegression for clean serialization.
    Monotonically non-decreasing by construction.
    """
    x_thresholds: list = field(default_factory=list)
    y_thresholds: list = field(default_factory=list)

    def apply(self, p: np.ndarray | float) -> np.ndarray | float:
        """Apply isotonic calibration to raw binary probability(ies)."""
        x = np.asarray(self.x_thresholds, dtype=np.float64)
        y = np.asarray(self.y_thresholds, dtype=np.float64)
        if len(x) == 0:
            return np.asarray(p, dtype=np.float64)
        p_arr = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        # np.interp faz linear interpolation; extrapolates flat outside the bounds
        return np.interp(p_arr, x, y)


def fit_isotonic_binary(
    raw_probs: np.ndarray,
    labels: np.ndarray,
) -> IsotonicCalibrator:
    """Fit isotonic regression binary via sklearn.

    Extract X_thresholds_/y_thresholds_ for serialization dataclass-friendly.
    """
    p = np.clip(np.asarray(raw_probs, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return IsotonicCalibrator(
        x_thresholds=iso.X_thresholds_.tolist(),
        y_thresholds=iso.y_thresholds_.tolist(),
    )


def calibrate_1x2_isotonic(
    raw_probs: np.ndarray,
    iso_home: IsotonicCalibrator,
    iso_draw: IsotonicCalibrator,
    iso_away: IsotonicCalibrator,
) -> np.ndarray:
    """Apply isotonic one-vs-rest 3-class and renormaliza."""
    raw = np.asarray(raw_probs, dtype=np.float64)
    original_ndim = raw.ndim
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    cal = np.zeros_like(raw)
    cal[:, 0] = iso_home.apply(raw[:, 0])
    cal[:, 1] = iso_draw.apply(raw[:, 1])
    cal[:, 2] = iso_away.apply(raw[:, 2])

    totals = cal.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    cal = cal / totals

    return cal if original_ndim > 1 else cal[0]


# ---------------------------------------------------------------------------
# Calibration metrics (v1.11.0)
# ---------------------------------------------------------------------------

def compute_brier_3class(
    probs: np.ndarray,
    y: np.ndarray,
) -> float:
    """Brier multi-class: mean of sum_k (p_k - y_k)^2."""
    p = np.asarray(probs, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.sum((p - yv) ** 2, axis=1)))


def compute_ece(
    probs_3class: np.ndarray,
    y_3class: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error via binning of the confidence max.

    Groups predictions by the confidence of the most probable class, compared against
    empirical accuracy within of each bin. ECE = sum_b (|B_b|/n) * |acc_b - conf_b|.
    """
    probs = np.asarray(probs_3class, dtype=np.float64)
    y = np.asarray(y_3class, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return 0.0

    conf = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    true_idx = y.argmax(axis=1)
    correct = (pred_idx == true_idx).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += (count / n) * abs(acc_bin - conf_bin)
    return float(ece)


# ---------------------------------------------------------------------------
# Integration with simulate_match
# ---------------------------------------------------------------------------

def sample_scores_dixon_coles(
    home_xg: float,
    away_xg: float,
    rho: float,
    n_simulations: int,
    seed: int | None = None,
    max_goals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (home_goals, away_goals) from the distribution Dixon-Coles corrected.

    Returns
    -------
    (home_goals, away_goals) arrays of shape (n_simulations,).
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
