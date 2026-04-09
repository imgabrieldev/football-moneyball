---
tags:
  - pitch
  - prediction
  - features
  - context
  - coach
  - injuries
  - ml
---

# Pitch — Context-Aware Predictor (v1.6.0)

## Problem

Our stats-only model (v1.5.x) looks only at aggregated numbers from the last 5 matches. **It ignores context that professional bettors use heavily.**

**Real case found (Vasco × Botafogo, 04/04/2026):**
- Model predicted Vasco 77% winning
- But the real context is even more favorable to Vasco:
  - **Vasco under a new coach (Renato Gaúcho)** — 5 matches unbeaten, 73% win ratio
  - **Botafogo with an INTERIM coach** (historically -20% performance)
  - **4 Botafogo starters out** (Joaquín Correa, Chris Ramos, Marçal, Kaio Pantaleão)
  - **Worst Série A defense** (Botafogo)
  - **São Januário packed** (Rio derby, 97-42 historical wins)
- Context suggests real **Vasco 80-85%**. Model underestimates.

**The inverse case also happens:**
- Team with "good statistical form" but coach fired this week → model too optimistic
- Team with injured star (our player-aware λ still includes her) → overestimates

**Gap:** the model doesn't know about:
1. Recent coach change
2. Interim coach
3. Specific injuries in the starting XI
4. Fatigue (fixture congestion)
5. Table context (relative position)

Research: [[feature-rich-predictor]], [[comprehensive-prediction-models]]

## Solution

Add **5 contextual features** capturing non-statistical info. All available via Sofascore API + light scraping. Feeds the same XGBoost/GBR from v1.3.0/v1.5.0.

### New features (v1.6.0)

**1. Coach context**
```python
coach_change_recent: bool       # coach changed in the last 30 days?
games_since_coach_change: int   # number of matches under current coach
coach_win_rate: float           # current coach win rate
is_interim_coach: bool          # interim coach?
```

**2. Injuries/Availability**
```python
key_players_out: int            # absences among top 3 by xG/90
starter_xi_available: bool      # likely starting XI available?
xg_contribution_missing: float  # xG/90 of absentees (sum)
```

**3. Fixture congestion (fatigue)**
```python
games_last_7d: int              # matches in the last 7 days
games_next_7d: int              # matches in the next 7 days
rest_days: int                  # already have in v1.5.0 ✓
competition_count: int          # how many simultaneous competitions
```

**4. League context**
```python
position_gap: int               # position difference (home - away)
points_gap: int                 # points difference
home_relegation_pressure: bool  # Z4 zone or nearby
away_relegation_pressure: bool
```

**5. Derby/rivalry**
```python
is_derby: bool                  # state derby
h2h_home_advantage: float       # historical home win %
```

Total: 15+ new features → **FEATURE_DIM 24 → 40**.

## Architecture

### Affected modules

| Module | Action | Description |
|--------|------|-----------|
| `domain/context_features.py` | NEW | Pure logic for contextual feature engineering |
| `domain/feature_engineering.py` | MODIFY | Expand FEATURE_DIM 24→40 |
| `adapters/sofascore_provider.py` | MODIFY | `get_coach_info(team_id)`, `get_injuries(team_id)` |
| `adapters/postgres_repository.py` | MODIFY | `get_games_in_window(team, start, end)`, contextual queries |
| `adapters/orm.py` | MODIFY | New `team_coaches` table (history), `injured` column in lineups |
| `use_cases/ingest_context.py` | NEW | Ingests contextual data (coaches + injuries) |
| `use_cases/predict_all.py` | MODIFY | Feed contextual features into ML |

### Schema

```sql
-- Coach history per team (who coached when)
CREATE TABLE team_coaches (
    team VARCHAR(100),
    coach_name VARCHAR(100),
    start_date DATE,
    end_date DATE,              -- NULL = current
    is_interim BOOLEAN DEFAULT false,
    games_coached INTEGER,
    wins INTEGER,
    draws INTEGER,
    losses INTEGER,
    PRIMARY KEY (team, start_date)
);

-- Active injuries (per-player per-date status)
CREATE TABLE player_injuries (
    player_id INTEGER,
    player_name VARCHAR(100),
    team VARCHAR(100),
    injury_type VARCHAR(50),
    reported_date DATE,
    expected_return DATE,        -- NULL if undefined
    status VARCHAR(20),          -- 'out', 'doubt', 'returned'
    PRIMARY KEY (player_id, reported_date)
);

-- Standings per matchday (for position_gap, pressure zones)
CREATE TABLE league_standings (
    competition VARCHAR(100),
    season VARCHAR(20),
    round INTEGER,
    team VARCHAR(100),
    position INTEGER,
    points INTEGER,
    played INTEGER,
    wins INTEGER, draws INTEGER, losses INTEGER,
    goals_for INTEGER, goals_against INTEGER,
    PRIMARY KEY (competition, season, round, team)
);
```

### Infra (K8s)

**New CronJob:** `ingest-context` (daily, 6am)
- Fetches coaches + injuries from Sofascore
- Updates standings for the current matchday

No changes to API deployment (only uses PG data).

## Scope

### In Scope

- [ ] `team_coaches`, `player_injuries`, `league_standings` tables
- [ ] Sofascore adapter: `get_team_managers()`, `get_team_injuries()`, `get_standings()`
- [ ] `ingest_context.py` — pulls data daily
- [ ] `context_features.py` — computes the 15 features
- [ ] Extend `feature_engineering.py` FEATURE_DIM 24→40
- [ ] Historical backfill of coaches + standings per matchday
- [ ] `predict_all.py` — feeds contextual features
- [ ] Frontend: "New coach", "Interim", "X absences" badges
- [ ] CronJob `ingest-context` on K8s
- [ ] Tests: `test_domain_context_features.py`
- [ ] Retrain models with 40 features

### Out of Scope

- Event-level data (WhoScored SPADL) — deferred to v1.7.0
- Weather data (rain, heat) — marginal, ignore for now
- Player form (individual streaks) — reserved for v1.8.0
- Manager tactical style (3-5-2 vs 4-3-3) — complex, later
- Media/fan sentiment analysis — low ROI

## Research Needed

- [x] Context matters: Vasco × Botafogo real case validated
- [x] Sofascore API validated:
  - `event/{id}/managers` → home/away manager per match
  - `event/{id}/lineups.missingPlayers` → injuries/suspensions with reason code
  - `unique-tournament/.../standings/total` → standings
- [x] "New manager bounce" literature:
  - **Modest effect** (~10 matches of boost) — *mostly regression to the mean*
  - Studies: Bryson 2024 (Scottish J of Pol Econ), PMC 2021
- [x] "Interim coaches" literature:
  - **Not necessarily worse** — 2010 study shows they can perform BETTER (player motivation)
  - Implication: use real `coach_win_rate`, not a hardcoded negative flag
- [x] "Fixture congestion" literature:
  - Affects **injury risk** more than direct performance (squad rotation mitigates)
  - Meta-analysis: PMC 2021 confirms
- [x] "Player impact" literature:
  - Player Impact Metric (PIM) 2025 confirms: absent top-N players reduce expected outcome
  - Transfermarkt value is a strong predictor (we'll use xG/90 as proxy)

## Testing Strategy

### Unit (pure domain)

- `test_domain_context_features.py`:
  - `coach_change_recent`: True if < 30 days
  - `key_players_out`: correct count from top 3 by xG/90
  - `fixture_congestion`: matches in window
  - `position_gap`: numeric rank difference
  - Fallback when data is missing

### Integration (with PG)

- Ingest real coaches from Sofascore → table populated
- Contextual query returns expected values for Vasco × Botafogo
- Rich (40-dim) vs old (24-dim) features reproduce v1.5.0 when context=neutral

### Manual

- Predict Vasco × Botafogo v1.5.0 vs v1.6.0 — compare probabilities
- Verify v1.6.0 gives Vasco 80%+ (favorable context)
- Compare with Betfair odds (edge should increase if model is right)

## Success Criteria

- [ ] 15+ contextual features implemented
- [ ] Coach history backfilled for the 20 Brasileirão 2026 teams
- [ ] Injuries backfilled via Sofascore
- [ ] Standings per matchday ingested
- [ ] Retrained models with 40 features, stable or better MAE
- [ ] **Brier < 0.19** on time-split backtest (was 0.21-0.22)
- [ ] Feature importance shows ≥2 contextual features in top-10
- [ ] Frontend shows contextual badges (interim coach, absences, etc)
- [ ] Vasco × Botafogo correctly predicted: Vasco 80%+

## Next Steps after v1.6.0

If Brier < 0.19 ✓ → v1.7.0: Event data via WhoScored (real xT, PPDA)
If Brier ≥ 0.19 ✗ → investigate overfit, tune hyperparams, weighted feature groups
