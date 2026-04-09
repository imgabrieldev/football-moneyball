---
tags:
  - pitch
  - prediction
  - elo
  - features
  - xgboost
  - ml
---

# Pitch — Feature-Rich Predictor (v1.5.0)

## Problem

v1.3.0 ML trains XGBoost/GBR with only **12 features** (xG/xA for/against, corners, cards, league avg, is_home). Empirical research across multiple SHAP papers (2024-2025) shows that **Elo rating and form EMA are among the top-3 features** in importance ranking. Our model ignores this completely.

**Current baseline:**
- Brier 1X2 score: **0.2158**
- 1X2 accuracy: **51.1%**
- Goals MAE (GBR): **0.902** (close to baseline — 87 samples is little)

**Gap:** Starlizard/professional syndicates sit at Brier ~0.20, 55-60% accuracy. To reach them, we need richer features. Since the ML layer already exists (v1.3.0), adding features is pure **feature engineering** — no new infrastructure.

Research: [[implementation-details]], [[comprehensive-prediction-models]]

## Solution

Add **~10 new features** fed by the same pipeline (Sofascore stats + match history). Three categories:

### 1. Rating System (top-3 in SHAP)
- **Dynamic Elo rating** (FiveThirtyEight-style): each team starts at 1500, updated after each match via K-factor
- **Goal difference EMA** (last N matches, exponential decay)

### 2. xG Overperformance (corrects luck)
- **xG overperformance per game** = `(goals - xG) / matches` — how much the team is "above xG" (usually regresses)
- **xG allowed overperformance** = same for defense

### 3. Synthetic Sofascore features (we already have the data)
- **creation_index** = `xA/90 + keyPass/90 × 0.05`
- **defensive_intensity** = `(tackles + interceptions + ballRecovery) / 90` per team
- **possession_quality** = `accuratePass_rate × touches_final_third` (control proxy)
- **rest_days** = days since last match (fatigue)

Total: **12 old features + 10 new = 22 features per team**.

Feed everything into the existing XGBoost/GBR. Creates no new models — just feeds the 3 existing ones (goals, corners, cards) with more signal.

## Architecture

### Affected modules

| Module | Action | Description |
|--------|------|-----------|
| `domain/elo.py` | NEW | Elo rating computation (update after each result) |
| `domain/feature_engineering.py` | MODIFY | Extend from 12 → 22 features |
| `domain/ml_lambda.py` | NO CHANGE | Receives larger X, same code |
| `adapters/postgres_repository.py` | MODIFY | Queries for creation_index, defensive_intensity, rest_days |
| `use_cases/train_ml_models.py` | NO CHANGE | Trains with new features automatically |
| `use_cases/predict_all.py` | MODIFY | Pass new features in `_ml_predict_pair` |

### Schema

**No table changes.** Features are computed on-the-fly via JOIN queries on `matches`, `player_match_metrics`, `match_stats`.

**Optional (future optimization):** `team_form` materialized view refreshed on each ingest — avoids recomputing features every time.

### Infra (K8s)

No changes. ML models retrained via `moneyball train-models` (already exists).

## Scope

### In Scope

- [ ] `domain/elo.py`: EloRating class with `update(home, away, home_goals, away_goals)` + `get_rating(team)`
- [ ] `feature_engineering.py`: `build_rich_features()` function with 22 features
- [ ] Repository: `get_team_advanced_stats(team, last_n)` returning creation_index, defensive_intensity, etc
- [ ] Repository: `get_rest_days(team, match_date)` days since last match
- [ ] Repository: `get_elo_ratings(season)` — computes Elo of all teams up to date X (history)
- [ ] `predict_all.py`: use build_rich_features on ML path
- [ ] Tests: `test_domain_elo.py` with known scenarios
- [ ] Tests: update `test_domain_feature_engineering.py` with 22 features
- [ ] Retrain models and compare Brier before/after
- [ ] Frontend: expose Elo rating on the prediction card (optional)

### Out of Scope

- Event-level data scraping (WhoScored) — [[event-data-integration]] (v1.6.0)
- Real PPDA via SPADL (research shows high variance at match level)
- Player-level Elo (team only)
- Real xT via events — deferred to v1.6.0
- Platt/isotonic calibration — reserved for v1.7.0

## Research Needed

- [x] Elo rating for football — FiveThirtyEight method
- [x] Feature importance SHAP in football models — [[implementation-details]]
- [x] xG overperformance regression — 2024 research
- [ ] Optimal K-factor for Brasileirão 2026 (test 20, 30, 40)
- [ ] Optimal EMA window (5, 7, 10 matches)

## Testing Strategy

### Unit (domain — zero mocks)

- `test_domain_elo.py`:
  - 2 teams start at 1500, draw → both stay at 1500
  - Strong team beats weak → strong goes up a little, weak drops a little
  - Weak team beats strong → weak goes up a lot, strong drops a lot
  - K-factor=20 vs 40 produce different magnitudes
- `test_domain_feature_engineering.py`:
  - build_rich_features returns 22 features
  - Feature order is consistent
  - Fallback defaults when data is missing

### Integration (with PG)

- Feature engineering with real data: 87 matches populate all 22 dimensions
- Full training pipeline: 22 features → GBR → stable MAE
- Retroactively computed Elo ratings match expectations

### Manual

- Compare Elo ratings with https://eloratings.net (if it covers Brasileirão)
- Inspect feature importance of trained GBR — Elo should be top-3
- Verify that Brier drops after retraining

## Success Criteria

- [ ] 22 features implemented and tested
- [ ] Elo rating with manual validation (Flamengo ~1700, Remo ~1350, etc)
- [ ] XGBoost retrained with MAE_goals < 0.85 (was 0.902)
- [ ] **Brier < 0.200 in backtest** (main target — was 0.2158)
- [ ] **1X2 accuracy > 54%** in backtest (was 51.1%)
- [ ] Feature importance shows Elo/goal_diff_ema in top-5
- [ ] Tests passing (210+ total, ~15 new)
- [ ] Pure domain layer (no infra deps)
- [ ] Backward compatible: old models keep working

## Next steps after v1.5.0

If Brier < 0.200 ✓ → v1.6.0: Event data via WhoScored (PPDA, xT, pass networks)
If Brier ≥ 0.200 ✗ → debug feature importance, tune hyperparams, or reconsider approach
