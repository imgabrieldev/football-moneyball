---
tags:
  - pitch
  - markets
  - corners
  - cards
  - asian-handicap
  - correct-score
  - betfair
---

# Pitch — Multi-Market Prediction (v1.0.0)

## Problem

Our model only predicts 2 of the ~8 Betfair markets: result (1X2) and Over/Under 2.5. We're leaving on the table:

1. **Correct Score** — already computed (score_matrix) but not shown
2. **Over/Under 0.5, 1.5, 3.5** — already computed but only 2.5 exposed
3. **BTTS** — already computed but not used for bets
4. **Asian Handicap** — derivable from the data we have
5. **Corners** — Sofascore has data, needs a separate Poisson
6. **Cards** — Sofascore has data, needs Poisson + referee factor
7. **Half Time** — needs separate Monte Carlo for the 1st half
8. **Player Goal** — needs individual xG + confirmed lineup

Each additional market = more value bet opportunities. Starlizard (Tony Bloom, £600M/year) uses separate models for each market.

Research: [[multi-market-prediction]]

## Solution

3 phases: P0 (expose what we have), P1 (new models), P2 (player props).

### Phase P0 — Expose existing markets (zero new models)

From the Monte Carlo we already run, derive:

**Correct Score:**
```python
# We already have score_matrix: {"1x0": 0.15, "2x1": 0.12, ...}
# Expose top 10 scorelines with probability
```

**Full Over/Under:**
```python
# We already have over_05, over_15, over_25, over_35
# Expose all + matching under
```

**BTTS:**
```python
# We already have btts_prob
# Compute btts_no = 1 - btts_prob
```

**Asian Handicap (derived from score_matrix):**
```python
def calculate_asian_handicap(score_matrix: dict) -> dict:
    """Derive handicap probabilities from the score matrix."""
    # AH -0.5 home = P(home win) = sum where home > away
    # AH -1.5 home = P(home win by 2+) = sum where home > away + 1
    # AH +0.5 home = P(home win or draw)
    # AH -0.5 away = P(away win)
    # etc.
```

### Phase P1 — New Poisson models

**Corners (Corners Over/Under):**
```python
def predict_corners(
    home_corners_avg: float,    # average home corners (last 6 matches)
    away_corners_avg: float,    # average away corners
    home_corners_against: float, # corners conceded at home
    away_corners_against: float,
) -> dict:
    """Poisson with λ = team_avg × opponent_factor."""
    lambda_home = home_corners_avg * (away_corners_against / league_avg)
    lambda_away = away_corners_avg * (home_corners_against / league_avg)
    # Monte Carlo: simulate home_corners + away_corners
    # Over/Under 7.5, 8.5, 9.5, 10.5, 11.5
```

Required data: corners per team per match (Sofascore has).

**Cards (Cards Over/Under):**
```python
def predict_cards(
    home_fouls_avg: float,      # home team fouls per match
    away_fouls_avg: float,      # away team fouls
    referee_cards_avg: float,   # average cards for this referee
    is_derby: bool,             # derby?
) -> dict:
    """Zero-Inflated Poisson adjusted by the referee."""
    base_lambda = (home_fouls_avg + away_fouls_avg) * referee_card_rate
    if is_derby:
        base_lambda *= 1.2  # +20% in derbies
    # Monte Carlo: simulate total cards
    # Over/Under 2.5, 3.5, 4.5, 5.5
```

Required data: fouls and cards per team (already have), match referee (Sofascore has, needs ingestion).

**Half Time Result:**
```python
def predict_half_time(home_xg: float, away_xg: float) -> dict:
    """Monte Carlo with λ_HT ≈ 45% of λ_FT."""
    home_ht_xg = home_xg * 0.45
    away_ht_xg = away_xg * 0.45
    # Simulate: P(home HT), P(draw HT), P(away HT)
    # Combined HT/FT: P(home/home), P(draw/home), etc.
```

### Phase P2 — Player Props

**Player X Goal:**
```python
def predict_player_goal(player_xg_per90: float, expected_minutes: int) -> float:
    """P(goal) = 1 - e^(-xG_per90 × min/90)."""
```

Required: confirmed lineup (~1h before), individual xG (already have).

## Architecture

### Affected modules

| Module | Action | Description |
|--------|------|-----------|
| `domain/match_predictor.py` | MODIFY | Add `calculate_asian_handicap()` derived from score_matrix |
| `domain/corners_predictor.py` | NEW | Poisson for corners |
| `domain/cards_predictor.py` | NEW | ZIP for cards |
| `domain/markets.py` | NEW | Aggregate all markets into unified dict |
| `adapters/sofascore_provider.py` | MODIFY | Ingest corners, cards, fouls, referee |
| `adapters/postgres_repository.py` | MODIFY | Queries for corners/cards/referee stats |
| `adapters/orm.py` | MODIFY | New fields in player_match_metrics (or separate table) |
| `use_cases/predict_all.py` | MODIFY | Run all predictors |
| `api.py` | MODIFY | Return all markets |
| `frontend/` | MODIFY | Market tabs on each card |

### Schema

```sql
-- Extra fields in player_match_metrics (Sofascore already returns)
-- corners, cards_yellow, cards_red, fouls already partially exist

-- New table for referee stats
CREATE TABLE IF NOT EXISTS referee_stats (
    referee_name VARCHAR PRIMARY KEY,
    matches_officiated INTEGER,
    avg_cards_per_match REAL,
    avg_fouls_per_match REAL,
    avg_corners_per_match REAL,
    last_updated VARCHAR
);

-- New table for match-level stats (total corners, cards)
CREATE TABLE IF NOT EXISTS match_stats (
    match_id INTEGER PRIMARY KEY,
    home_corners INTEGER,
    away_corners INTEGER,
    home_yellow_cards INTEGER,
    away_yellow_cards INTEGER,
    home_red_cards INTEGER,
    away_red_cards INTEGER,
    home_fouls INTEGER,
    away_fouls INTEGER,
    referee_name VARCHAR,
    ht_home_score INTEGER,
    ht_away_score INTEGER
);
```

### Infra (K8s)

No changes.

## Scope

### Phase P0 — In Scope (sprint 1)

- [ ] `domain/markets.py` — aggregate score_matrix → correct score, asian handicap, all O/U, BTTS
- [ ] `api.py` — return complete `markets` dict in each prediction
- [ ] Frontend — tabs/sections: "Result", "Goals", "Correct Score", "Handicap"
- [ ] Value bets for all existing markets (not only h2h and totals)

### Phase P1 — In Scope (sprint 2)

- [ ] Ingest extra Sofascore data: corners, cards, fouls, referee per match
- [ ] `match_stats` and `referee_stats` tables
- [ ] `domain/corners_predictor.py` — Poisson for corners
- [ ] `domain/cards_predictor.py` — ZIP for cards
- [ ] Half Time prediction in Monte Carlo
- [ ] Frontend — tabs: "Corners", "Cards", "Half Time"

### Phase P2 — In Scope (sprint 3)

- [ ] Player X Goal (individual xG + lineup)
- [ ] Frontend — "Player" tab

### Out of Scope

- "Bet Builder" (custom combination) — too complex
- "Safe Substitution" — depends on coach decision
- Live/in-play predictions
- Team markets (corners asian handicap)

## Research Needed

- [x] Betfair markets and how to predict them — [[multi-market-prediction]]
- [ ] Validate that Sofascore returns corners and referee for Brasileirão
- [ ] Calibrate Brasileirão corners λ (average ~10/match?)
- [ ] Calibrate Brasileirão cards λ + variance per referee
- [ ] Test: HT goals = 45% or different in Brasileirão?

## Testing Strategy

### Unit (domain — zero mocks)
- `calculate_asian_handicap`: known score_matrix → correct handicaps
- `corners_predictor`: λ=5 home + λ=5 away → Over 9.5 ~50%
- `cards_predictor`: referee with 5 cards/game → high Over 3.5

### Manual
- Compare computed Asian Handicap with Betfair odds
- Compare corners prediction with bookmaker lines

## Success Criteria

- [ ] P0: all derivable markets exposed on the frontend
- [ ] P0: value bets identified in correct score and asian handicap (not only h2h)
- [ ] P1: corners prediction with accuracy > 50% on Over/Under 9.5
- [ ] P1: cards prediction with accuracy > 50% on Over/Under 3.5
- [ ] Each market has calibration validated against Betfair odds

---

## Update: Player-Aware Prediction (pre and post lineup)

### Concept

Two prediction rounds per match:

1. **Pre-lineup (~24h before):** Prediction based on team averages + likely lineup (most frequent starters)
2. **Post-lineup (~1h before):** Sofascore publishes confirmed lineup → recompute EVERYTHING with real data from the 11 starters

### What changes with player data

Instead of generic team λ, build λ from the 22 on the pitch:

```
λ_goals = Σ xG/90 of the 11 starters (adjusted by opposition)
λ_shots = Σ shots/90 of forwards + midfielders
λ_corners = f(crosses/90 of fullbacks, opponent's blocked shots)
λ_cards = Σ fouls/90 of defensive mids × referee_card_rate
λ_goalkeeper_saves = expected opponent shots × (1 - xG/shot)
```

### Per-player data Sofascore already has

- `expectedGoals` (xG per player per match)
- `totalShots`, `shotsOnTarget` (shots)
- `totalCross`, `accurateCross` (crosses → corners)
- `fouls`, `wasFouled` (fouls → cards)
- `totalTackle`, `wonTackle` (tackles)
- `saves` (goalkeeper saves)
- `ballRecovery` (recoveries)
- `touches`, `passes` (involvement)

### Impact per market

| Market | Without player | With player |
|---------|------------|-------------|
| Total goals | team avg λ | Σ xG/90 of the 11 |
| Goals per team | generic | Σ xG of forwards/midfielders |
| Shots on target | generic | Σ shotsOnTarget/90 of the 11 |
| Corners | team avg | fullback crosses + blocked shots |
| Cards | team avg | defensive mid fouls/90 × referee |
| Scorer X | impossible | individual xG |
| Goalkeeper saves | generic | opponent_shots × (1 - save_rate) |
| Assist X | impossible | individual xA |
| First goal | generic | xG/90 × expected minutes |

### Updated flow

```
24h before:
  CronJob predict → prediction with likely starters
  Frontend shows with "Pre-lineup" badge

1h before:
  Sofascore publishes lineup
  CronJob detect-lineup → fetches lineups
  CronJob predict-lineup → recompute with 22 real
  Frontend updates with "Lineup confirmed" badge
  Value bets recomputed with new λ
```

### Revised priority

| Sprint | What | Player-aware? |
|--------|-------|:---:|
| P0 | Expose derivable markets | No (already done) |
| P1 | Corners, cards, HT, shots | **Yes — player-based λ** |
| P2 | Scorer, assist, individual props | **Yes — mandatory** |
