---
tags:
  - pitch
  - betting
  - odds
  - monte-carlo
  - kelly-criterion
  - sofascore
  - brasileirao
---

# Pitch — Betting Value Finder (v0.4.0)

## Problem

We have a complete analytics engine (xG, pressing, RAPM, embeddings) and data for 87 Brasileirão 2026 matches. But all this analysis is retroactive — "what happened". We don't answer the question that matters for the betting market: **"what IS GOING TO happen?"**

Bookmakers (Betano, Bet365, Pinnacle) price matches using odds that reflect implied probabilities. If our statistical model estimates more accurate probabilities than the market's, there are **value bets** — bets with positive expected return.

The problem is threefold:
1. **We don't have access to odds** — we need to integrate an odds API
2. **We don't have a predictive model** — our xG is retroactive, not prospective
3. **We don't know if the model works** — we need backtesting before risking real money

Research: [[betting-value-model]]

## Solution

4-layer system that turns retroactive analysis into probabilistic prediction:

### A. Odds Provider — The Odds API integration

New adapter that fetches pre-match odds from multiple bookmakers:
- **The Odds API** free tier (500 req/month — enough for Brasileirão)
- Markets: 1X2 (`h2h`), over/under (`totals`), both teams to score (`btts`), handicap (`spreads`)
- ~30 bookmakers including Bet365, Pinnacle
- Historical odds since 2020 for backtesting

### B. Match Predictor — Monte Carlo + Poisson + xG

Predictive model that estimates the probability of each outcome:

1. **Expected xG per team**: weighted average of the last 5-8 matches, adjusted by:
   - Opponent strength (xG against of the opponent)
   - Home factor (+0.25-0.35 xG in Brasileirão)
   - Recent form (exponential decay)
   - Pressing intensity (PPDA correlates with xG created)

2. **Monte Carlo simulation** (N=10,000):
   - Each team's goals drawn from a Poisson distribution: `P(k) = (λ^k × e^-λ) / k!`
   - Where `λ = adjusted expected xG`
   - Each simulation produces a scoreline

3. **Derived probabilities** from 10,000 simulations:
   - P(home win), P(draw), P(away win)
   - P(over 0.5), P(over 1.5), P(over 2.5), P(over 3.5)
   - P(BTTS yes), P(BTTS no)
   - Most likely scoreline

### C. Value Detector — Compare Model vs Odds

For each market of each match:
1. Convert odds → implied probability: `prob = 1/odds`
2. Remove margin (vig): `prob_real = prob / sum(probs)`
3. Compute edge: `edge = prob_model - prob_implied`
4. Classify: **value bet** if `edge > threshold` (default 3%)

### D. Bankroll Manager — Kelly Criterion

For identified value bets:
1. Compute optimal stake via Kelly: `f* = (b×p - q) / b`
2. Apply fractional Kelly (25%): `stake = 0.25 × f* × bankroll`
3. Limit maximum stake (5% of bankroll per bet)

### E. Backtesting Engine — Validate with Historical Data

Before using on future matches:
1. For each already-played match, simulate as if pre-match
2. Use only data available up to that moment (no lookahead)
3. Fetch historical odds via The Odds API
4. Compute ROI, hit rate, Brier score, max drawdown
5. Compare with baseline (always bet on favorite)

## Architecture

### Affected modules

| Module | Action | Layer |
|--------|------|--------|
| **`adapters/odds_provider.py`** | NEW | Adapter: fetches odds via The Odds API |
| **`domain/match_predictor.py`** | NEW | Domain: Poisson + Monte Carlo + xG adjustment |
| **`domain/value_detector.py`** | NEW | Domain: identifies value bets (edge > threshold) |
| **`domain/bankroll.py`** | NEW | Domain: Kelly criterion + stake sizing |
| **`use_cases/predict_match.py`** | NEW | Use case: single match prediction |
| **`use_cases/find_value_bets.py`** | NEW | Use case: matchday value bet scanner |
| **`use_cases/backtest.py`** | NEW | Use case: backtesting with historical data |
| `ports/odds_provider.py` | NEW | Port: interface for odds provider |
| `adapters/postgres_repository.py` | MODIFY | Historical xG queries per team |
| `cli.py` | MODIFY | New commands: predict, value-bets, backtest |
| `adapters/matplotlib_viz.py` | MODIFY | New plots: probability, ROI curve |

### Schema

```sql
-- Match odds
CREATE TABLE IF NOT EXISTS match_odds (
    match_id INTEGER,
    bookmaker VARCHAR(100),
    market VARCHAR(50),        -- h2h, totals, btts, spreads
    outcome VARCHAR(50),       -- Home, Away, Draw, Over, Under, Yes, No
    point REAL,                -- line (2.5 for over/under, etc.)
    odds REAL,                 -- decimal odds
    implied_prob REAL,         -- implied probability (1/odds)
    fetched_at TIMESTAMP,
    PRIMARY KEY (match_id, bookmaker, market, outcome, point)
);

-- Model predictions
CREATE TABLE IF NOT EXISTS match_predictions (
    match_id INTEGER PRIMARY KEY,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    home_xg_expected REAL,
    away_xg_expected REAL,
    home_win_prob REAL,
    draw_prob REAL,
    away_win_prob REAL,
    over_25_prob REAL,
    btts_prob REAL,
    most_likely_score VARCHAR(10),
    simulations INTEGER DEFAULT 10000,
    predicted_at TIMESTAMP
);

-- Identified value bets
CREATE TABLE IF NOT EXISTS value_bets (
    id SERIAL PRIMARY KEY,
    match_id INTEGER,
    market VARCHAR(50),
    outcome VARCHAR(50),
    model_prob REAL,
    best_odds REAL,
    bookmaker VARCHAR(100),
    implied_prob REAL,
    edge REAL,                  -- model_prob - implied_prob
    kelly_fraction REAL,
    recommended_stake REAL,
    actual_result VARCHAR(50),  -- filled in after the match
    profit REAL,                -- filled in after the match
    created_at TIMESTAMP
);

-- Model performance (backtesting)
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    run_date TIMESTAMP,
    matches_analyzed INTEGER,
    bets_placed INTEGER,
    total_staked REAL,
    total_return REAL,
    roi REAL,
    hit_rate REAL,
    brier_score REAL,
    max_drawdown REAL,
    config JSONB               -- backtest parameters
);
```

### Infra (K8s)

No changes. The Odds API is called from the local CLI.

## Scope

### In Scope

- [ ] Port `odds_provider.py` — interface for odds APIs
- [ ] Adapter `odds_provider.py` — The Odds API integration (free tier)
- [ ] Domain `match_predictor.py` — Poisson model + Monte Carlo (10K sims)
- [ ] Domain `value_detector.py` — model vs odds comparison, edge calculation
- [ ] Domain `bankroll.py` — fractional Kelly criterion
- [ ] Use case `predict_match.py` — single match prediction
- [ ] Use case `find_value_bets.py` — matchday value bet scanner
- [ ] Use case `backtest.py` — backtesting with 87+ historical matches
- [ ] CLI `predict <match_id>` — display model prediction
- [ ] CLI `value-bets [--round N]` — list matchday value bets
- [ ] CLI `backtest --season 2026` — run backtesting and display ROI
- [ ] Schema: match_odds, match_predictions, value_bets, backtest_results tables
- [ ] Viz: cumulative ROI chart, probability calibration
- [ ] Unit tests for Poisson, Monte Carlo, Kelly, edge detection

### Out of Scope

- Automated betting (no direct integration with bookmakers)
- Live/in-play betting (pre-match only)
- Player markets (top scorer, cards) — match markets only
- Cross-bookmaker arbitrage (focus is value, not arb)
- Advanced machine learning (Poisson + adjustments only, no neural networks)
- Managing multiple bankrolls/wallets

## Research Needed

- [x] Odds APIs for Brasileirão — [[betting-value-model]]
- [x] Monte Carlo + Poisson for match prediction — [[betting-value-model]]
- [x] Kelly Criterion — [[betting-value-model]]
- [ ] Validate that The Odds API covers Brasileirão 2026 historical odds (since Jan/2026)
- [ ] Calibrate Brasileirão home factor with our data (87 matches)
- [ ] Test model sensitivity to xG window (last 3, 5, 8 matches)

## Testing Strategy

### Unit (domain — zero mocks)
- `match_predictor.py`: Poisson PMF with known λ, Monte Carlo convergence (P(home) should stabilize with large N)
- `value_detector.py`: edge calculation with known odds, threshold filtering
- `bankroll.py`: Kelly with deterministic inputs, stake limits

### Integration
- `odds_provider.py`: mock HTTP response, verify odds parsing
- `backtest.py`: run with 5 test matches, verify ROI calculation

### Manual
- Compare specific match prediction with real Betano odds
- Verify that backtesting has no lookahead bias
- Check that Kelly never recommends > 5% of bankroll

## Success Criteria

- [ ] Backtesting on 87 matches produces positive ROI (any edge > 0 is a signal)
- [ ] Model Brier score < 0.25 (better than random)
- [ ] Value bet hit rate > 50% (if edge is real, should hit more than miss)
- [ ] Monte Carlo with 10K sims converges (deviation < 1% between runs)
- [ ] The Odds API free tier sufficient for monthly operation
- [ ] All new modules in the domain layer (no infra imports)
- [ ] Zero regression in existing commands
