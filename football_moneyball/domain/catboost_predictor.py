"""CatBoost 1x2 predictor — prediz P(H), P(D), P(A) end-to-end.

Substitui Poisson Monte Carlo pra 1x2. Usa Pi-Rating + form EMA +
xG + odds como features. Treinado com temporal CV.

Lógica pura — zero deps de infra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CATBOOST_FEATURE_NAMES = [
    "pi_rating_diff",
    "pi_home_rating",
    "pi_away_rating",
    "home_form_ema",
    "away_form_ema",
    "home_gd_ema",
    "away_gd_ema",
    "home_xg_avg",
    "away_xg_avg",
    "home_xga_avg",
    "away_xga_avg",
    "home_rest_days",
    "away_rest_days",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]

N_FEATURES = len(CATBOOST_FEATURE_NAMES)  # 16

# Label mapping: CatBoost MultiClass usa inteiros
LABEL_MAP = {"home": 0, "draw": 1, "away": 2}
LABEL_NAMES = ["home", "draw", "away"]


def compute_form_ema(
    results: list[float],
    alpha: float = 0.1,
) -> float:
    """EMA de resultados (1=win, 0.5=draw, 0=loss). Último = mais recente."""
    if not results:
        return 0.5
    ema = 0.5
    for r in results:
        ema = alpha * r + (1 - alpha) * ema
    return ema


def compute_gd_ema(
    goal_diffs: list[float],
    alpha: float = 0.15,
) -> float:
    """EMA de goal difference."""
    if not goal_diffs:
        return 0.0
    ema = 0.0
    for gd in goal_diffs:
        ema = alpha * gd + (1 - alpha) * ema
    return ema


def build_match_features(
    pi_ratings: dict,
    home_team: str,
    away_team: str,
    home_form: list[float],
    away_form: list[float],
    home_gd: list[float],
    away_gd: list[float],
    home_xg_avg: float = 1.3,
    away_xg_avg: float = 1.1,
    home_xga_avg: float = 1.1,
    away_xga_avg: float = 1.3,
    home_rest: int = 7,
    away_rest: int = 7,
    market_home: float = 0.40,
    market_draw: float = 0.28,
    market_away: float = 0.32,
) -> np.ndarray:
    """Constroi feature vector pra um match."""
    from football_moneyball.domain.pi_rating import PiRating, rating_diff

    rd = rating_diff(pi_ratings, home_team, away_team)
    rh = pi_ratings.get(home_team, PiRating())
    ra = pi_ratings.get(away_team, PiRating())

    return np.array([
        rd,
        rh.home,
        ra.away,
        compute_form_ema(home_form),
        compute_form_ema(away_form),
        compute_gd_ema(home_gd),
        compute_gd_ema(away_gd),
        home_xg_avg,
        away_xg_avg,
        home_xga_avg,
        away_xga_avg,
        float(home_rest),
        float(away_rest),
        market_home,
        market_draw,
        market_away,
    ], dtype=np.float64)


def build_training_dataset(
    all_matches: pd.DataFrame,
    pi_gamma: float = 0.04,
    min_history: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Constroi X, y a partir de matches históricos (leak-proof).

    Pra cada match M: usa apenas matches anteriores (match_id < M)
    pra computar Pi-Ratings e form EMA.

    Parameters
    ----------
    all_matches : pd.DataFrame
        Formato: match_id, team, goals, xg, is_home.
    min_history : int
        Mínimo de matches anteriores pra incluir no dataset.

    Returns
    -------
    (X, y) : (np.ndarray, np.ndarray)
        X shape (n_samples, N_FEATURES), y shape (n_samples,) com 0/1/2.
    """
    from football_moneyball.domain.pi_rating import PiRating, update_ratings

    # Pivot pra formato (match_id, home_team, away_team, home_score, away_score)
    match_ids = sorted(all_matches["match_id"].unique())

    # Pi-Ratings evoluem match a match
    ratings: dict[str, PiRating] = {}
    # Form history per team: list of (result, goal_diff, xg)
    team_form: dict[str, list[tuple[float, float, float, float]]] = {}
    # Track last match date per team
    team_last_match: dict[str, int] = {}

    X_list = []
    y_list = []

    for idx, mid in enumerate(match_ids):
        mdata = all_matches[all_matches["match_id"] == mid]
        home_row = mdata[mdata["is_home"] == True]  # noqa: E712
        away_row = mdata[mdata["is_home"] == False]  # noqa: E712

        if home_row.empty or away_row.empty:
            continue

        home_team = str(home_row.iloc[0]["team"])
        away_team = str(away_row.iloc[0]["team"])
        hg = int(home_row.iloc[0]["goals"])
        ag = int(away_row.iloc[0]["goals"])

        # Skip se pouco histórico
        if idx < min_history:
            update_ratings(ratings, home_team, away_team, hg, ag, pi_gamma)
            _update_form(team_form, home_team, hg, ag, float(home_row.iloc[0].get("xg", 1.3)), True)
            _update_form(team_form, away_team, ag, hg, float(away_row.iloc[0].get("xg", 1.1)), False)
            team_last_match[home_team] = idx
            team_last_match[away_team] = idx
            continue

        # Build features BEFORE updating ratings (leak-proof)
        home_hist = team_form.get(home_team, [])
        away_hist = team_form.get(away_team, [])

        home_results = [h[0] for h in home_hist[-20:]]
        away_results = [h[0] for h in away_hist[-20:]]
        home_gd = [h[1] for h in home_hist[-20:]]
        away_gd = [h[1] for h in away_hist[-20:]]

        home_xg = np.mean([h[2] for h in home_hist[-5:]]) if home_hist else 1.3
        away_xg = np.mean([h[2] for h in away_hist[-5:]]) if away_hist else 1.1
        home_xga = np.mean([h[3] for h in home_hist[-5:]]) if home_hist else 1.1
        away_xga = np.mean([h[3] for h in away_hist[-5:]]) if away_hist else 1.3

        home_rest = max(1, idx - team_last_match.get(home_team, idx - 7))
        away_rest = max(1, idx - team_last_match.get(away_team, idx - 7))

        features = build_match_features(
            pi_ratings=ratings,
            home_team=home_team,
            away_team=away_team,
            home_form=home_results,
            away_form=away_results,
            home_gd=home_gd,
            away_gd=away_gd,
            home_xg_avg=float(home_xg),
            away_xg_avg=float(away_xg),
            home_xga_avg=float(home_xga),
            away_xga_avg=float(away_xga),
            home_rest=home_rest,
            away_rest=away_rest,
            # No market odds in training (not available retroactively for all)
            market_home=0.40,
            market_draw=0.28,
            market_away=0.32,
        )

        # Label
        if hg > ag:
            label = LABEL_MAP["home"]
        elif hg == ag:
            label = LABEL_MAP["draw"]
        else:
            label = LABEL_MAP["away"]

        X_list.append(features)
        y_list.append(label)

        # NOW update ratings and form (after feature extraction)
        update_ratings(ratings, home_team, away_team, hg, ag, pi_gamma)
        _update_form(team_form, home_team, hg, ag, float(home_row.iloc[0].get("xg", 1.3)), True)
        _update_form(team_form, away_team, ag, hg, float(away_row.iloc[0].get("xg", 1.1)), False)
        team_last_match[home_team] = idx
        team_last_match[away_team] = idx

    return np.array(X_list), np.array(y_list)


def _update_form(
    team_form: dict[str, list],
    team: str,
    goals_for: int,
    goals_against: int,
    xg: float,
    is_home: bool,
) -> None:
    """Atualiza histórico de forma do time."""
    if team not in team_form:
        team_form[team] = []

    if goals_for > goals_against:
        result = 1.0
    elif goals_for == goals_against:
        result = 0.5
    else:
        result = 0.0

    gd = float(goals_for - goals_against)
    xga = xg * 0.85 if is_home else xg * 1.15  # proxy
    team_form[team].append((result, gd, xg, xga))


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_catboost_1x2(
    X: np.ndarray,
    y: np.ndarray,
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 3.0,
    draw_weight: float = 1.3,
    early_stopping_rounds: int = 50,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Treina CatBoost MultiClass pra 1x2.

    Returns
    -------
    (model, metrics) : tuple
        model = CatBoostClassifier treinado
        metrics = dict com RPS, accuracy, log_loss no val set
    """
    from catboost import CatBoostClassifier, Pool

    # Time-split: últimos val_fraction% como validação
    n = len(X)
    cut = int(n * (1 - val_fraction))
    X_train, X_val = X[:cut], X[cut:]
    y_train, y_val = y[:cut], y[cut:]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=seed,
        eval_metric="MultiClass",
        class_weights=[1.0, draw_weight, 1.0],
        verbose=0,
        allow_writing_files=False,
    )

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=early_stopping_rounds,
    )

    # Evaluate
    probs_val = model.predict_proba(X_val)
    metrics = _compute_metrics(probs_val, y_val)
    metrics["n_train"] = cut
    metrics["n_val"] = n - cut
    metrics["best_iteration"] = model.get_best_iteration() or iterations
    metrics["feature_importance"] = dict(
        zip(CATBOOST_FEATURE_NAMES, model.get_feature_importance().tolist())
    )

    return model, metrics


def _compute_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """Computa RPS, accuracy, Brier no validation set."""
    n = len(y)
    if n == 0:
        return {"rps": 1.0, "accuracy": 0.0, "brier": 2.0}

    # One-hot
    y_onehot = np.zeros((n, 3))
    y_onehot[np.arange(n), y.astype(int)] = 1

    # Accuracy
    preds = probs.argmax(axis=1)
    accuracy = float((preds == y).mean())

    # Brier
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # RPS (Ranked Probability Score)
    rps_total = 0.0
    for i in range(n):
        cum_pred = np.cumsum(probs[i])
        cum_actual = np.cumsum(y_onehot[i])
        rps_total += float(np.mean((cum_pred - cum_actual) ** 2))
    rps = rps_total / n

    return {
        "rps": round(rps, 4),
        "accuracy": round(accuracy * 100, 1),
        "brier": round(brier, 4),
    }


def predict_1x2(
    model,
    features: np.ndarray,
) -> dict:
    """Prediz P(H), P(D), P(A) pra um match.

    Parameters
    ----------
    model : CatBoostClassifier
        Modelo treinado.
    features : np.ndarray
        Shape (N_FEATURES,) ou (1, N_FEATURES).

    Returns
    -------
    dict
        {home_win_prob, draw_prob, away_win_prob}
    """
    X = features.reshape(1, -1) if features.ndim == 1 else features
    probs = model.predict_proba(X)[0]

    return {
        "home_win_prob": float(probs[0]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[2]),
    }
