---
tags:
  - pitch
  - catboost
  - pi-rating
  - prediction
  - 1x2
---

# Pitch — v1.14.0: CatBoost 1x2 + Pi-Rating (End-to-End Predictor)

## Problem

The Poisson Monte Carlo model **always predicts home win** (10/10 picks matchday 10, acc 42.4%, RPS ~0.24). Structural issue: Poisson distributes probabilities via goal simulation — home is the plurality class (47-50%), so argmax is ALWAYS home. Draw is never argmax (max ~32%), away rarely.

Calibration (v1.11), draw floor (v1.13), market blending (v1.13) improved Brier by 9% but **don't change the picks**. The decision engine is fundamentally wrong.

## Solution

Replace the 1x2 engine with **CatBoost MultiClass** trained end-to-end on historical results. The model learns `P(H), P(D), P(A)` directly — bypassing Poisson.

```
[Pi-Rating] ──┐
[EMA form]  ──┤
[xG/xGA]    ──┼──→ CatBoost MultiClass ──→ P(H), P(D), P(A)
[H2H]       ──┤         ↑
[Rest days] ──┤    class_weights=[1, 1.3, 1]
[Odds devig]──┘    temporal CV, no post-hoc calibration

Poisson kept for: corners, cards, correct score, HT/FT (multi-market)
```

### Why CatBoost

- **RPS 0.1925** on the Soccer Prediction Challenge (beat bookmakers 0.2012)
- Native categorical handling (team IDs without one-hot)
- Ordered boosting (reduces target leakage)
- MultiClass loss → probs via softmax → **already calibrated, no Platt**
- `class_weights=[1, 1.3, 1]` → draws get extra weight

### Pi-Rating (Constantinou & Fenton 2013)

Separate HOME/AWAY ratings per team — solves home bias at the root.

```python
error = goal_diff_capped - (R_home[team] - R_away[opp])
R_home[team] += γ * error    # γ = 0.04
R_away[opp]  -= γ * error
```

Brier 0.2065 on the EPL, superior to standalone Elo.

## Architecture

### New modules

| Module | Responsibility |
|---|---|
| `domain/pi_rating.py` | PiRating dataclass, update(), compute_all() |
| `domain/catboost_predictor.py` | CatBoostPredictor: train(), predict(), feature engineering |
| `use_cases/train_catboost.py` | Temporal CV training pipeline |

### Modified modules

| Module | Change |
|---|---|
| `use_cases/predict_all.py` | Use CatBoost for 1x2, Poisson for multi-market |
| `cli.py` | `train-catboost` command |
| `adapters/postgres_repository.py` | `get_training_dataset()` (features + temporal labels) |

### Features (13 per team-pair)

```python
CATBOOST_FEATURES = [
    # Pi-Rating (2)
    "pi_rating_diff",       # R_home[home] - R_away[away]
    "pi_rating_home",       # absolute R_home[home]
    
    # Form EMA (4)
    "home_form_ema",        # α=0.1, last ~10 matches
    "away_form_ema",
    "home_gd_ema",          # goal difference EMA, α=0.15
    "away_gd_ema",
    
    # Strength (4)
    "home_xg_avg",          # xG/90 last 5
    "away_xg_avg",
    "home_xga_avg",         # xGA/90 last 5
    "away_xga_avg",
    
    # Market (3) — if available
    "market_home_prob",     # Pinnacle devigged
    "market_draw_prob",
    "market_away_prob",
]
# + cat_features: home_team, away_team (native CatBoost)
```

### Schema

No new tables. Pi-ratings computed in-memory per season (like current Elo). Model saved in `/data/models/catboost_1x2.cbm`.

## Scope

### In Scope

- [x] Pi-Rating: dataclass, update, promoted team handling, compute_all
- [x] CatBoost training: temporal CV (expanding window), class_weights, early stopping
- [x] Feature engineering: pi-rating diff + form EMA + xG + market odds
- [x] Predict pipeline: CatBoost → P(H,D,A), Poisson → multi-market
- [x] CLI: `train-catboost` command
- [x] Tests: Pi-Rating convergence, CatBoost feature shapes, temporal CV splits

### Out of Scope

- Residual modeling (target = outcome - market_prob) — v1.15.0
- Feature selection / SHAP analysis — v1.15.0
- Automatic retrain via cron — v1.16.0
- CatBoost for multi-market (corners, cards) — v1.16.0

## Research

- [[catboost-1x2-implementation]] — implementation details
- [[advanced-prediction-models]] — comparative benchmarks
- [[prediction-error-analysis]] — home bias diagnosis

## Testing Strategy

### Unit (pure domain)
- `test_pi_rating.py`: update formula, cap ±3, promoted team init, convergence
- `test_catboost_predictor.py`: feature shape, predict_proba output, class ordering

### Integration
- Training with 2022-2024, eval on 2025: accuracy > 48%, RPS < 0.22
- Temporal CV: 3 folds expanding window, no leakage

### Manual
- Compare P(H,D,A) from CatBoost vs Poisson vs market on matchday 10 matches
- Verify that draw/away picks exist (not 10/10 home)

## Success Criteria

- [ ] CatBoost trained on 1610 matches (2022-2026)
- [ ] Temporal CV: **RPS < 0.215** (vs Poisson 0.24)
- [ ] **≥1 draw or away pick** on a typical 10-match matchday
- [ ] 1x2 accuracy > **48%** (vs current 42.4%)
- [ ] Poisson kept and functional for multi-market
- [ ] Feature importance: pi-rating in top-3, odds top-1 if present
- [ ] Pi-Rating converged in 2 seasons (stable ratings)
- [ ] 0 regression in multi-market (corners, cards, CS — Poisson pipeline untouched)
