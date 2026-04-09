---
tags:
  - pitch
  - track-record
  - verify
  - automation
  - frontend
---

# Pitch — Track Record (v0.9.0)

## Problem

The system predicts matches and identifies value bets, but doesn't track whether it got them right or wrong. Today:

1. **Predictions are ephemeral** — `match_predictions` is overwritten on every recompute. There's no history of "what I predicted on matchday 5".

2. **Verification is manual** — you have to run `moneyball verify` and look in the terminal. There's no automatic comparison when results arrive.

3. **No track record** — we don't know: accuracy per matchday, Brier score evolution over the tournament, which markets the model hits more (1X2 vs Over/Under), which teams the model misses systematically.

4. **Value bets without follow-up** — we identified 59 value bets on the matchday, but after the matches are played we don't know how many won and what the real (non-simulated) ROI was.

5. **No model confidence** — without a historical track record, you can't tell if the model is improving or getting worse over the season.

The system needs to be **future-proof**: every prediction is recorded, every result is compared, and the full history stays accessible.

## Solution

### Lifecycle of a prediction:

```
1. PREDICT  → Prediction saved with "pending" status and matchday/date
2. MATCH    → Match happens (result on Sofascore)
3. RESOLVE  → CronJob automatically compares prediction vs result
4. DISPLAY  → Frontend shows history with hits/misses
```

### Components:

#### A. Prediction History (immutable table)

Instead of overwriting `match_predictions`, each prediction is an immutable record:

```
prediction_history:
  id SERIAL
  match_id (hash of teams)
  home_team, away_team
  commence_time
  round (matchday)
  
  # Model prediction
  home_win_prob, draw_prob, away_win_prob
  over_25_prob, btts_prob
  home_xg_expected, away_xg_expected
  most_likely_score
  predicted_at
  
  # Actual result (filled in later)
  actual_home_goals (NULL until match happens)
  actual_away_goals
  actual_outcome (Home/Draw/Away)
  resolved_at
  status (pending → resolved)
  
  # Hit metrics
  correct_1x2 BOOLEAN
  correct_over_under BOOLEAN
  brier_score FLOAT
```

#### B. Value Bet History (immutable table)

Each identified value bet is recorded with its result:

```
value_bet_history:
  id SERIAL
  prediction_id (FK → prediction_history)
  market, outcome
  model_prob, best_odds, bookmaker
  edge, kelly_stake
  
  # Result
  won BOOLEAN (NULL until resolved)
  profit FLOAT
  resolved_at
```

#### C. Auto-Resolve (use case)

When new results are ingested from Sofascore, a `resolve_predictions` use case automatically:
1. Fetches predictions with `status = 'pending'`
2. For each, checks whether the result exists in the database (`matches` table)
3. If yes: fills in result, computes brier, marks hit/miss, changes status → `resolved`
4. Runs in the ingest CronJob (after ingesting results)

#### D. Track Record API

```
GET /api/track-record              — general summary (accuracy, brier, ROI)
GET /api/track-record/predictions  — historical list of predictions
GET /api/track-record/value-bets   — historical list of value bets with P/L
GET /api/track-record/by-round     — accuracy per matchday
GET /api/track-record/by-team      — accuracy per team
GET /api/track-record/by-market    — accuracy per market (1X2, O/U, BTTS)
```

#### E. Frontend — Track Record Page

New `/track-record` page with:
- **Summary**: overall accuracy, Brier, real ROI, total predicted/hit
- **Evolution per matchday**: line chart (accuracy and Brier over time)
- **Per market**: which bet type we hit more (table)
- **Per team**: which teams the model misses more (table)
- **Full history**: list of each prediction with result (scrollable)
- **Value bets P/L**: list of each bet with gain/loss

## Architecture

### Affected modules

| Module | Action | Description |
|--------|------|-----------|
| `adapters/orm.py` | MODIFY | New tables PredictionHistory, ValueBetHistory |
| `adapters/postgres_repository.py` | MODIFY | CRUD for history + track record queries |
| `domain/track_record.py` | NEW | Resolution logic (compare pred vs result) |
| `use_cases/resolve_predictions.py` | NEW | Auto-resolve when results arrive |
| `use_cases/predict_all.py` | MODIFY | Save to prediction_history (immutable) |
| `use_cases/find_value_bets.py` | MODIFY | Save to value_bet_history |
| `api.py` | MODIFY | 6 new track record endpoints |
| `cli.py` | MODIFY | `moneyball track-record` command |
| `frontend/` | MODIFY | New `/track-record` page |
| `k8s/cronjob-ingest.yaml` | MODIFY | Run resolve after ingest |

### Schema

```sql
CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    match_key INTEGER,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    commence_time VARCHAR,
    round INTEGER,
    
    home_win_prob REAL,
    draw_prob REAL,
    away_win_prob REAL,
    over_25_prob REAL,
    btts_prob REAL,
    home_xg_expected REAL,
    away_xg_expected REAL,
    most_likely_score VARCHAR(10),
    predicted_at VARCHAR,
    
    actual_home_goals INTEGER,
    actual_away_goals INTEGER,
    actual_outcome VARCHAR(10),
    resolved_at VARCHAR,
    status VARCHAR(20) DEFAULT 'pending',
    
    correct_1x2 BOOLEAN,
    correct_over_under BOOLEAN,
    brier_score REAL
);

CREATE TABLE IF NOT EXISTS value_bet_history (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES prediction_history(id),
    market VARCHAR(50),
    outcome VARCHAR(50),
    model_prob REAL,
    best_odds REAL,
    bookmaker VARCHAR(100),
    edge REAL,
    kelly_stake REAL,
    
    won BOOLEAN,
    profit REAL,
    resolved_at VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_pred_history_status ON prediction_history(status);
CREATE INDEX IF NOT EXISTS idx_pred_history_round ON prediction_history(round);
```

### Infra (K8s)

Modify CronJob `ingest-sofascore` to run `moneyball resolve` after ingest:
```yaml
command: ["sh", "-c", "moneyball ingest --provider sofascore && moneyball resolve"]
```

## Scope

### In Scope

- [ ] `prediction_history` table (immutable, 1 record per prediction)
- [ ] `value_bet_history` table (immutable, 1 record per value bet)
- [ ] `domain/track_record.py` — resolves predictions vs results
- [ ] `use_cases/resolve_predictions.py` — auto-resolve
- [ ] `predict_all` saves to `prediction_history` instead of overwriting `match_predictions`
- [ ] `find_value_bets` saves to `value_bet_history`
- [ ] 6 track record API endpoints
- [ ] CLI `moneyball resolve` and `moneyball track-record`
- [ ] Frontend `/track-record` page with summary, charts, tables
- [ ] CronJob updated to resolve after ingest
- [ ] Unit tests for resolution

### Out of Scope

- Alerts/notifications when results arrive
- Comparison with other models (benchmark)
- PDF report export
- Automatic model adjustment based on track record (meta-learning)

## Research Needed

- [ ] Define what counts as "matchday" (does Sofascore have round info? or infer by date?)
- [ ] Define matching between prediction (odds names without accents) and result (Sofascore names with accents) — we already have fuzzy match

## Testing Strategy

### Unit
- `domain/track_record.py`: resolve prediction with known result, brier computed correctly
- Prediction pending → resolved when result arrives
- Value bet won/lost computed correctly

### Integration
- Full flow: predict → ingest result → resolve → check track record

### Manual
- Recompute predictions, wait for matchday, ingest, verify track-record on frontend

## Success Criteria

- [ ] Predictions are never overwritten (immutable history)
- [ ] After ingesting results, predictions are resolved automatically
- [ ] `/track-record` shows accuracy, Brier, ROI per matchday
- [ ] Value bets show real P/L
- [ ] No manual intervention in the predict → resolve flow
- [ ] Frontend accessible and informative
