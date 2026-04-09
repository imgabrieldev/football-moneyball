---
tags:
  - pitch
  - catboost
  - features
  - prediction
  - brasileirao
---

# Pitch — v1.15.0: Context Features Pipeline (xG Form + Coach Profile + Rest Days)

## Problem

The v1.14.2 model has **40.7% 1X2 accuracy** (Brier 0.2358) versus ~52% for the market (Brier ~0.20). Volatile teams like Corinthians (20%), Palmeiras (11%), Mirassol (0%) drag the average down. CatBoost already has 28 features (Pi-Rating, form EMA, xG avg, rest days, rolling match stats), but it lacks **situational context** that the market prices in and we don't:

1. **Form based on goals, not xG** — xG-based form dominates 8/10 of the best configs in benchmarks
2. **Coach as a binary flag** — we have `team_coaches` with 49 rows but don't use it as a feature. In Brasileirão 2025 there were 22 coach changes over 38 matchdays
3. **Basic rest days** — we have `home_rest_days`/`away_rest_days` but no multi-competition fixture congestion
4. **Zero standings features** — points gap, relative position, table momentum

The ~15% Brier gap to the market comes mainly from contextual features that bettors and sharp books use but our model ignores.

Ref: [[../research/volatile-teams-features|Research: Features for Volatile Teams]]

## Solution

Add **~15 new contextual features** to CatBoost, leveraging data **already available in the database** (match_stats, team_coaches, league_standings). No new scraping, no infra change.

### 3 axes:

1. **xG Form** — replace goal-based form with xG-based form (rolling xG For/Against EMA)
2. **Coach Profile** — tenure, win rate, recent change, adaptation bucket
3. **Standings + Congestion** — table position, points gap, refined rest days

## Architecture

### Affected modules

| Module | Change |
|--------|---------|
| `domain/catboost_predictor.py` | Expand `CATBOOST_FEATURE_NAMES` (+15 features), update `build_training_dataset()` |
| `domain/features.py` (**new**) | Centralized feature engineering module: `compute_xg_form()`, `compute_coach_features()`, `compute_standings_features()` |
| `use_cases/train_catboost.py` | Pass coach and standings data to `build_training_dataset()` |
| `use_cases/predict_all.py` | Extract context features at prediction time |
| `adapters/postgres_repository.py` | New queries: `get_coach_for_team()`, `get_standings_at_date()` |

### New Features (15)

#### xG Form (4 features)
```
home_xg_form_ema    — EMA of xG For over the last 10 matches (alpha=0.15)
away_xg_form_ema    — idem away
home_xg_diff_ema    — EMA of (xG For - xG Against) last 10 matches
away_xg_diff_ema    — idem away
```
Partially replaces `home_xg_avg`/`away_xg_avg` which are simple averages. EMA reacts faster to form changes.

#### Coach Profile (6 features)
```
home_coach_tenure_days   — days since home coach appointment
away_coach_tenure_days   — idem away
home_coach_win_rate      — % wins of the coach at this team
away_coach_win_rate      — idem away
home_coach_changed_30d   — flag: coach changed in the last 30 days (1/0)
away_coach_changed_30d   — idem away
```
Derived from `team_coaches` table (49 rows already in DB). Win rate computed from matches under the current coach.

#### Standings & Congestion (5 features)
```
home_league_position     — table position
away_league_position     — idem away
position_gap             — |pos_home - pos_away| (close-ranked teams draw more)
home_points_last_5       — points in the last 5 matches (momentum)
away_points_last_5       — idem away
```
Derived from `league_standings` (20 rows) + retroactive computation from matches.

### Schema

**No schema change.** All required data already exists in tables:
- `match_stats` (1616 rows) — xG per match
- `team_coaches` (49 rows) — coach per team with dates
- `league_standings` (20 rows) — current position
- `matches` — results to compute rolling points

### Infra (K8s)

**No change.** Same CronJobs, same container. Just rebuild the image after merge.

## Scope

### In Scope

- [ ] Create `domain/features.py` with pure feature engineering functions
- [ ] Implement `compute_xg_form()` — EMA of xG For/Against
- [ ] Implement `compute_coach_features()` — tenure, win rate, change flag
- [ ] Implement `compute_standings_features()` — position, gap, momentum
- [ ] Expand `CATBOOST_FEATURE_NAMES` from 28 to ~43 features
- [ ] Update `build_training_dataset()` to include new features (leak-proof)
- [ ] Update `predict_all.py` to extract context features at inference
- [ ] Add coach and standings queries in the repository
- [ ] Retrain CatBoost and compare metrics (RPS, Brier, accuracy)
- [ ] Run backtest with new features vs v1.14.2 baseline
- [ ] Unit tests for each function in features.py

### Out of Scope

- Coach tactical profile (8 Analytics FC metrics) — Tier 2, separate pitch
- Full coach career history (Transfermarkt) — Tier 3
- Key player absence score — Tier 2
- Ensemble meta-learner — Tier 3
- Edge-based optimization (custom loss) — Tier 3
- Changes to Poisson/Dixon-Coles — these use a separate pipeline
- Draw-specific features (derby flag, style matchup) — Tier 2
- Travel distance — needs a city coordinates dataset

## Research Needed

- [x] State-of-the-art features ([[../research/volatile-teams-features|volatile-teams-features]])
- [x] Coach profiling frameworks (Analytics FC, Dartmouth)
- [x] xG vs goals form comparison (beatthebookie)
- [ ] Verify whether `team_coaches.games_coached/wins/draws/losses` are populated or NULL
- [ ] Verify coverage of `league_standings` per matchday (is there only a current snapshot?)

## Testing Strategy

### Unit (`tests/test_features.py`)
- `test_compute_xg_form_ema_with_known_values` — verify EMA with known sequence
- `test_compute_xg_form_ema_empty_history` — returns default (league avg)
- `test_compute_coach_features_new_coach` — tenure < 30d, flag=1
- `test_compute_coach_features_no_coach_data` — graceful defaults
- `test_compute_standings_features_with_gap` — correct position and gap
- `test_compute_standings_features_missing` — defaults when no standings

### Integration
- Retrain CatBoost with new features and compare RPS/Brier/accuracy vs v1.14.2
- Full backtest: ROI and hit rate with new features
- Verify that `predict-all` produces valid predictions with expanded features

### Manual
- `moneyball train-catboost` — train and check feature importances
- `moneyball predict-all` — predict matchday and compare with v1.14.2
- `moneyball backtest` — compare ROI

## Success Criteria

- [ ] **Brier < 0.220** (~7% improvement vs 0.2358 current)
- [ ] **1X2 accuracy > 45%** (vs 40.7% current)
- [ ] **Volatile teams > 30%** accuracy (Corinthians, Palmeiras, Mirassol — vs ~15% current)
- [ ] **RPS < 0.205** (competitive with academic benchmarks)
- [ ] Feature importances show coach features with contribution > 0
- [ ] Zero regression in already-predictable teams (Flamengo, Fluminense)
- [ ] Betfair backtest ROI does not get worse (>= 0%)
- [ ] Unit tests passing
- [ ] No schema or infra changes
