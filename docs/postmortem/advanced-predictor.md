---
tags:
  - pitch
  - predictor
  - monte-carlo
  - xT
  - pressing
  - rapm
  - sofascore
  - brasileirao
---

# Pitch — Advanced Predictive Model (v0.5.0)

## Problem

The current prediction model (`domain/match_predictor.py`) uses only **raw xG** as input for Monte Carlo, ignoring ~80% of the data we already compute:

1. **Brier score of 0.76** — worse than random (0.25). The model is not calibrated.
2. **Ignores xT** — we have Expected Threat computed but it doesn't feed the predictor. A team that creates lots of xT but finishes poorly is different from one that finishes little but with quality.
3. **Ignores pressing** — PPDA, counter-pressing and high turnovers are already in the database but don't affect the prediction. Teams with high pressing force more errors → more xG created against opponents.
4. **Ignores RAPM** — we have individual impact of each player but don't use it to adjust xG by lineup. If the top scorer is suspended, expected xG should drop.
5. **No regression to the mean** — Palmeiras with +9.1 goals above xG will probably regress. The model should pull overperformers toward the average.
6. **No automatic ingestion** — Sofascore data is ingested manually. The predictor should ensure fresh data before predicting.
7. **Under bias** — the model predicts Under 2.5 too often because average xG at the start of the tournament is low and doesn't adjust for the season's evolution.

### Data we already have in the database (87 Brasileirão 2026 matches):

| Data | Table | Used in predictor? |
|------|--------|:---:|
| xG per player per match | player_match_metrics | ✅ (partial, only sum) |
| xA per player | player_match_metrics | ❌ |
| Actual goals vs xG | player_match_metrics | ❌ (regression to mean) |
| PPDA per team | pressing_metrics | ❌ |
| Pressing success rate | pressing_metrics | ❌ |
| Counter-pressing fraction | pressing_metrics | ❌ |
| High turnovers | pressing_metrics | ❌ |
| xT per action (shotmap) | action_values | ❌ |
| Completed passes/type | player_match_metrics | ❌ |
| Big chances created | player_match_metrics | ❌ |
| Tackles/interceptions | player_match_metrics | ❌ |
| Player position | player_embeddings | ❌ |
| Individual RAPM | (computable) | ❌ |
| Embeddings/archetypes | player_embeddings | ❌ |

Research: [[betting-value-model]], [[market-standard-analytics]]

## Solution

Rewrite `domain/match_predictor.py` with a 6-stage pipeline that integrates all available data:

### Prediction Pipeline

```
┌─────────────────────────────────────────────────┐
│ 1. INGESTION — Update Sofascore data             │
│    Fetch new matches since last ingestion        │
│    Update player_match_metrics + pressing        │
├─────────────────────────────────────────────────┤
│ 2. BASE xG — Weighted history                    │
│    Offensive xG: exponential avg last 6 matches  │
│    Defensive xG: opponent's avg xGA              │
│    Calibrated home factor (compute from dataset) │
├─────────────────────────────────────────────────┤
│ 3. TACTICAL ADJUSTMENTS                          │
│    + Pressing: low PPDA → more xG created        │
│    + xT: high-xT teams create more danger        │
│    + Big chances: big chance conversion rate     │
│      adjusts over/under                          │
├─────────────────────────────────────────────────┤
│ 4. LINEUP ADJUSTMENT                             │
│    + RAPM of confirmed starters                  │
│    + Key player absent → xG penalty              │
│    + Team top scorer suspended → impact          │
├─────────────────────────────────────────────────┤
│ 5. REGRESSION TO THE MEAN                        │
│    Actual goals vs xG over the season            │
│    If team scores well above xG → reduce         │
│    If below → increase                           │
│    Factor: pull 30-50% toward the mean           │
├─────────────────────────────────────────────────┤
│ 6. MONTE CARLO — Final simulation                │
│    λ_home = adjusted xG (stages 2-5)             │
│    λ_away = adjusted xG (stages 2-5)             │
│    Poisson(λ) × 10,000 simulations               │
│    → P(1X2), P(O/U), P(BTTS), scorelines         │
└─────────────────────────────────────────────────┘
```

### Adjustment details

#### Stage 2: Base xG

```python
def compute_base_xg(team_history, opponent_defense, is_home):
    # Exponential average (decay=0.85, last 6 matches)
    base_xg = weighted_avg(team_history.xg, decay=0.85)
    
    # Opponent defensive strength
    # If opponent concedes more xG than average → boost our xG
    league_avg_xga = mean(all_teams.xga)
    defense_factor = opponent.xga / league_avg_xga
    base_xg *= defense_factor
    
    # Brasileirão calibrated home factor
    # Compute empirically: mean(home_xg) - mean(away_xg)
    home_boost = calibrated_home_advantage  # ~0.25-0.35
    if is_home:
        base_xg += home_boost
    
    return base_xg
```

#### Stage 3: Tactical Adjustments

```python
def apply_tactical_adjustments(base_xg, team_pressing, opponent_pressing, team_xt):
    adjusted = base_xg
    
    # Pressing: teams with low PPDA (high pressure) force more turnovers
    # Correlation: low opponent PPDA → our xG decreases
    league_avg_ppda = mean(all_teams.ppda)
    pressing_factor = opponent.ppda / league_avg_ppda
    # High opponent PPDA (weak pressure) → boost our xG
    adjusted *= (0.85 + 0.15 * pressing_factor)
    
    # xT: quality of play creation
    # Teams with high xT per match create more danger
    league_avg_xt = mean(all_teams.xt_per_match)
    xt_factor = team.xt_per_match / league_avg_xt
    adjusted *= (0.90 + 0.10 * xt_factor)
    
    # Counter-pressing: teams that recover quickly create more
    if team.counter_pressing_fraction > 50:
        adjusted *= 1.03  # +3% boost
    
    return adjusted
```

#### Stage 4: Lineup Adjustment

```python
def apply_lineup_adjustment(base_xg, confirmed_lineup, team_rapm):
    # If we have confirmed lineup (Sofascore ~1h before)
    if confirmed_lineup:
        # Sum RAPM of the 11 starters
        lineup_rapm = sum(rapm[player_id] for player_id in confirmed_lineup)
        # Compare with team's average RAPM for the season
        avg_rapm = mean(all_lineups_rapm)
        rapm_delta = lineup_rapm - avg_rapm
        # Adjust xG: each 0.1 of RAPM ≈ 0.05 xG
        base_xg += rapm_delta * 0.5
    
    return max(base_xg, 0.1)
```

#### Stage 5: Regression to the Mean

```python
def apply_regression_to_mean(xg_estimate, team_season_stats):
    # Actual goals vs xG for the season
    goals = team_season_stats.total_goals
    xg_total = team_season_stats.total_xg
    overperformance = (goals - xg_total) / matches_played  # per match
    
    # Pull 40% toward the mean (research: ~50% of overperformance is luck)
    regression_factor = 0.40
    adjustment = -overperformance * regression_factor
    
    return max(xg_estimate + adjustment, 0.1)
```

### Pre-prediction auto-ingestion

Before each prediction, the use case `PredictMatch` checks whether there are new matches on Sofascore that have not yet been ingested:

```python
class PredictMatch:
    def execute(self, match_id, home, away):
        # Step 0: ensure fresh data
        self._auto_ingest_if_needed()
        
        # Steps 1-6: prediction pipeline
        ...
    
    def _auto_ingest_if_needed(self):
        last_match = repo.get_latest_match_date()
        if (today - last_match).days >= 1:
            sofascore = SofascoreProvider()
            new_matches = sofascore.get_matches(...)
            # ingest new matches
```

## Architecture

### Affected modules

| Module | Action | Description |
|--------|------|-----------|
| `domain/match_predictor.py` | **REWRITE** | 6-stage pipeline with all adjustments |
| `domain/constants.py` | MODIFY | Add calibration constants (home advantage, decay, regression factor) |
| `use_cases/predict_match.py` | MODIFY | Fetch pressing, xT, RAPM from repo + auto-ingestion |
| `use_cases/backtest.py` | MODIFY | Use new predictor, compare Brier score before/after |
| `adapters/postgres_repository.py` | MODIFY | New queries: pressing per team, xT per team, season RAPM |
| `adapters/sofascore_provider.py` | MODIFY | Method to fetch only new matches (delta ingestion) |

### Schema

No changes in PostgreSQL. All required data already exists in current tables:
- `player_match_metrics` — xG, xA, big_chances, passes, tackles
- `pressing_metrics` — PPDA, success rate, counter-pressing, high turnovers
- `action_values` — xT per action (shotmap)
- `matches` — actual results for regression to the mean

### Infra (K8s)

No changes.

## Scope

### In Scope

- [ ] Rewrite `domain/match_predictor.py` with 6-stage pipeline
- [ ] `compute_base_xg()` — exponential average + defensive strength + calibrated home factor
- [ ] `apply_tactical_adjustments()` — PPDA, xT, counter-pressing, big chances
- [ ] `apply_lineup_adjustment()` — RAPM of confirmed starters
- [ ] `apply_regression_to_mean()` — pull overperformers toward the mean
- [ ] `calibrate_home_advantage()` — compute empirically from Brasileirão dataset
- [ ] Update `use_cases/predict_match.py` — fetch pressing, xT, RAPM + auto-ingestion
- [ ] Update `use_cases/backtest.py` — use new pipeline, compare Brier before/after
- [ ] New repository queries — average pressing per team, xT per team, season RAPM
- [ ] Delta ingestion in `sofascore_provider.py` — fetch only new matches
- [ ] Unit tests for each pipeline stage in isolation
- [ ] Comparative backtesting: v0.4.0 model vs v0.5.0 with the same 87+ matches

### Out of Scope

- Machine learning (neural networks, gradient boosting) — keep Poisson + deterministic adjustments
- Tracking data (real-time positioning) — only event data from Sofascore
- Player markets (top scorer, cards) — only match markets
- New data providers — use existing Sofascore
- REST API — separate pitch (v0.6.0)

## Research Needed

- [ ] Calibrate empirical home factor for Brasileirão 2026 with our 87 matches
- [ ] Measure PPDA → xG created correlation in the dataset (validate that pressing affects xG)
- [ ] Measure xT → goals correlation (validate xT is predictive beyond xG)
- [ ] Determine optimal regression factor (30%, 40%, 50%?) via backtesting
- [ ] Test model sensitivity to each adjustment in isolation (ablation study)

## Testing Strategy

### Unit tests (domain — zero mocks)
- `compute_base_xg()` — known inputs, deterministic output
- `apply_tactical_adjustments()` — high vs low PPDA, high vs low xT
- `apply_lineup_adjustment()` — with/without lineup, positive vs negative RAPM
- `apply_regression_to_mean()` — overperformer vs underperformer
- `calibrate_home_advantage()` — synthetic dataset with known advantage

### Integration
- Full pipeline: 6 chained stages produce reasonable xG (0.5-3.0)
- Backtesting v0.4.0 vs v0.5.0: Brier score should decrease

### Manual
- Compare prediction of specific match with real odds
- Verify that auto-ingestion picks up new matches from Sofascore
- Compare Palmeiras (overperformer) before/after regression to the mean

## Success Criteria

- [ ] Brier score < 0.25 in backtesting (better than random)
- [ ] v0.5.0 Brier score < v0.4.0 Brier score (measurable improvement)
- [ ] Positive backtest ROI with real odds (The Odds API)
- [ ] Value bet hit rate > 45%
- [ ] Each pipeline stage has measurable impact (ablation test)
- [ ] Auto-ingestion works without manual intervention
- [ ] Zero regression in existing CLI commands
- [ ] All tests pass
