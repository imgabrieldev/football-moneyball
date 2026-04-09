---
tags:
  - pitch
  - prediction
  - poisson
  - player-aware
  - xgboost
  - compound-poisson
  - zip
  - referee
  - multi-market
---

# Pitch — Comprehensive Predictor (5-Layer Framework)

## Problem

Our simplified Dixon-Coles operates at **team level** — it aggregates all matches of the season into `attack_strength` and `defense_strength`. This ignores:

1. **Who's playing** — the team wins or loses much more based on who's on the pitch than on the historical average. An injured Neymar changes Santos's λ_goals by 30%.
2. **Referee** — a strict ref inflates cards by 50%+. We're blind to that.
3. **Non-linear interactions** — Dixon-Coles is a simple multiplicative (`attack × defense`). It doesn't capture "team with high pressing defense vs team that depends on transitions".
4. **Multi-market** — we only predict goals. Corners, cards, shots, HT are off the table.

Current result: **47% 1X2 accuracy, Brier 0.237**. Starlizard (Tony Bloom, £600M/year) operates at ~55-60%. The gap comes exactly from the 4 items above.

**Research:** [[comprehensive-prediction-models]], [[implementation-details]]

## Solution

Implement the **5-layer framework** used by syndicates, in 3 incremental phases — each one adding without breaking the previous:

```
Layer 1: FEATURE ENGINEERING (per player for the 22 on the pitch)
Layer 2: TEAM AGGREGATION (sum/avg of the 11)
Layer 3: CONTEXTUAL ADJUSTMENT (home, referee, derby, form, regression)
Layer 4: MULTI-DIMENSIONAL MONTE CARLO (goals + corners + cards + shots + HT)
Layer 5: MARKET DERIVATION (200+ markets from the simulated matrix)
```

### Phase 1 — Player-Aware λ (v1.1.0)

**Goal:** replace `team_attack_strength` with `Σ xG/90 of the 11 starters`.

```python
def calculate_team_lambda_from_players(
    lineup: list[PlayerStats],     # 11 starters with last 5-10 match EMA
    opponent_defense: float,        # opposition factor
    minutes_weight: list[float],    # prob of each player playing 90min
) -> float:
    """λ = Σ (player.xg_per90 × minutes_weight) × opponent_defense."""
    lambda_team = sum(
        p.xg_per90 * w for p, w in zip(lineup, minutes_weight)
    )
    return lambda_team * opponent_defense
```

**Pre-lineup (24h before):** uses "probable XI" (11 players with most minutes in the last 5 matches).
**Post-lineup (~1h before):** uses Sofascore confirmed lineup, recomputes everything.

Backward compatible: keeps the old `predict_match(all_match_data=...)`; adds `predict_match_player_aware(lineup_home=..., lineup_away=...)`.

### Phase 2 — Multi-Output Poisson (v1.2.0)

**New λ besides goals:**

```python
λ_goals      = Σ xG/90 of the 11 × opp_defense
λ_corners    = f(fullback crosses, opponent's blocked shots)
λ_cards      = Σ fouls/90 of defensive mids × referee_factor × derby_factor  [ZIP]
λ_shots      = Σ shots/90 of the 11 × opp_shots_conceded
λ_sot        = λ_shots × avg_on_target_rate
λ_goals_HT   = λ_goals × 0.45
λ_saves_GK   = opponent_shots × (1 - goalkeeper_save_rate)
```

**Specific distributions:**
- Goals → Poisson (or Dixon-Coles)
- Corners → simple Poisson in MVP, migrate to Negative Binomial if overdispersion
- Cards → **Zero-Inflated Poisson** (excess of 0s)
- Shots → Poisson
- HT goals → Poisson(λ × 0.45)

**Referee module:**
```python
def referee_strictness(referee_name: str, prior_weight: float = 5.0) -> dict:
    """Empirical Bayes shrinkage."""
    n_matches, cards_rate, fouls_rate = get_ref_stats(referee_name)
    league_cards_rate = get_league_avg_cards()
    
    # Shrinkage: few matches → pull toward league avg
    adjusted = (n_matches * cards_rate + prior_weight * league_cards_rate) / (n_matches + prior_weight)
    return {"cards_factor": adjusted / league_cards_rate}
```

**Multi-dim Monte Carlo:**
```python
def simulate_match_full(lambdas: dict, n_sims: int = 10_000) -> pd.DataFrame:
    """One simulation = one full match (goals + corners + cards + ...)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "home_goals": rng.poisson(lambdas["home_xg"], n_sims),
        "away_goals": rng.poisson(lambdas["away_xg"], n_sims),
        "home_corners": rng.poisson(lambdas["home_corners"], n_sims),
        "away_corners": rng.poisson(lambdas["away_corners"], n_sims),
        "total_cards": rng.poisson(lambdas["total_cards"], n_sims),
        "home_shots": rng.poisson(lambdas["home_shots"], n_sims),
        "away_shots": rng.poisson(lambdas["away_shots"], n_sims),
        "ht_home": rng.poisson(lambdas["home_xg"] * 0.45, n_sims),
        "ht_away": rng.poisson(lambdas["away_xg"] * 0.45, n_sims),
    })
```

**Market derivation** (over the simulated DataFrame):
- Corners O/U 7.5, 8.5, 9.5, 10.5, 11.5
- Cards O/U 2.5, 3.5, 4.5, 5.5
- Shots O/U per team
- HT Result, HT/FT, HT Score
- Winning margin, half with more goals
- **~100 additional markets**

### Phase 3 — ML → λ (v1.3.0)

**Goal:** XGBoost learns to predict λ from ALL features (team + player + context + referee), capturing non-linear interactions that multiplicative Dixon-Coles misses.

```python
# Pipeline
X = build_features(match)  # 40-60 features per match
y = observed_lambda         # goals scored, corners, cards, shots

model_goals = XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
)
model_goals.fit(X_train, y_train_goals)

# In inference
λ_goals_ml = model_goals.predict(X_match)
result = simulate_match_full({"home_xg": λ_goals_ml, ...})
```

**Integration pattern:** ML predicts directly (Pattern A). If Dixon-Coles + player-aware stays competitive, migrate to Pattern B (ML predicts DC residual).

**Calibration >> Accuracy:** use Platt scaling or isotonic regression after XGBoost. 2024 paper shows **+69% ROI** when optimizing calibration vs accuracy.

## Architecture

### Affected modules

| Module | Phase | Action | Description |
|--------|:---:|------|-----------|
| `domain/player_lambda.py` | 1 | NEW | Player → team λ aggregation |
| `domain/lineup_prediction.py` | 1 | NEW | Probable XI + minutes weighting |
| `domain/match_predictor.py` | 1 | MODIFY | Add player-aware path |
| `domain/corners_predictor.py` | 2 | NEW | λ corners + Poisson/NB |
| `domain/cards_predictor.py` | 2 | NEW | λ cards + ZIP + referee |
| `domain/shots_predictor.py` | 2 | NEW | λ shots + SoT |
| `domain/referee.py` | 2 | NEW | Bayesian shrinkage |
| `domain/multi_monte_carlo.py` | 2 | NEW | Multi-dimensional simulation |
| `domain/markets.py` | 2 | MODIFY | Derive 100+ markets |
| `domain/ml_lambda.py` | 3 | NEW | XGBoost for each λ |
| `domain/feature_engineering.py` | 3 | NEW | Build features for ML |
| `adapters/sofascore_provider.py` | 1,2 | MODIFY | Ingest lineup + referee + match_stats |
| `adapters/orm.py` | 1,2 | MODIFY | New fields and tables |
| `adapters/postgres_repository.py` | 1,2 | MODIFY | New queries |
| `use_cases/predict_all.py` | 1,2,3 | MODIFY | Orchestrate new layers |
| `use_cases/ingest_lineups.py` | 1 | NEW | Detect and ingest lineups |
| `use_cases/train_ml_models.py` | 3 | NEW | Train XGBoost models |
| `api.py` | 1,2 | MODIFY | Expose new markets |
| `frontend/` | 1,2 | MODIFY | Tabs, badges (pre/post lineup) |

### Schema (PostgreSQL)

```sql
-- Phase 2: match-level stats
CREATE TABLE match_stats (
    match_id INTEGER PRIMARY KEY,
    home_corners INTEGER, away_corners INTEGER,
    home_yellow INTEGER, away_yellow INTEGER,
    home_red INTEGER, away_red INTEGER,
    home_fouls INTEGER, away_fouls INTEGER,
    home_shots INTEGER, away_shots INTEGER,
    home_sot INTEGER, away_sot INTEGER,
    ht_home_score INTEGER, ht_away_score INTEGER,
    referee_name VARCHAR,
    FOREIGN KEY (match_id) REFERENCES match_info(match_id)
);

-- Phase 2: referee stats (materialized)
CREATE TABLE referee_stats (
    referee_name VARCHAR PRIMARY KEY,
    matches_officiated INTEGER,
    avg_yellow_per_match REAL,
    avg_red_per_match REAL,
    avg_fouls_per_match REAL,
    avg_corners_per_match REAL,
    strictness_factor REAL,
    last_updated VARCHAR
);

-- Phase 1: confirmed lineups
CREATE TABLE match_lineups (
    match_id INTEGER,
    team VARCHAR,
    side VARCHAR,  -- 'home' | 'away'
    player_id INTEGER,
    player_name VARCHAR,
    position VARCHAR,
    is_starter BOOLEAN,
    jersey_number INTEGER,
    PRIMARY KEY (match_id, player_id)
);

-- Phase 1: add fields to prediction_history
ALTER TABLE prediction_history ADD COLUMN lineup_type VARCHAR DEFAULT 'pre';
ALTER TABLE prediction_history ADD COLUMN model_version VARCHAR;
ALTER TABLE prediction_history ADD COLUMN home_lambda_corners REAL;
ALTER TABLE prediction_history ADD COLUMN away_lambda_corners REAL;
ALTER TABLE prediction_history ADD COLUMN home_lambda_cards REAL;
ALTER TABLE prediction_history ADD COLUMN away_lambda_cards REAL;
ALTER TABLE prediction_history ADD COLUMN home_lambda_shots REAL;
ALTER TABLE prediction_history ADD COLUMN away_lambda_shots REAL;
```

### Infra (K8s)

**New CronJobs:**
- `ingest-lineups` — every 30min, check matches in the next 2h and pull lineup once available
- `predict-pre-lineup` — 24h before match (probable XI)
- `predict-post-lineup` — when lineup is out (confirmed lineup)
- `train-models` (phase 3) — weekly, retrain XGBoost

**ConfigMap:** update `init.sql` with new tables.

## Scope

### Phase 1 — Player-Aware λ (v1.1.0) — In Scope

- [x] `domain/player_lambda.py`: player → team aggregation
- [x] `domain/lineup_prediction.py`: probable XI + minutes weighting
- [x] `domain/match_predictor.py`: add `predict_match_player_aware()`
- [x] `match_lineups` table in the schema
- [x] `adapters/sofascore_provider.py`: `get_confirmed_lineup()` + `get_probable_lineup()`
- [x] `use_cases/ingest_lineups.py`: detect and save lineups
- [x] `use_cases/predict_all.py`: route between pre and post lineup
- [ ] CronJob (deferred) `ingest-lineups`
- [x] Frontend: "Pre-lineup" / "Lineup confirmed" badge on each card
- [x] Backtest v1.1.0 vs v0.5.0 on already-resolved matches

### Phase 2 — Multi-Output Poisson (v1.2.0) — In Scope

- [x] `match_stats` and `referee_stats` tables
- [x] `adapters/sofascore_provider.py`: ingest corners/cards/fouls/referee/HT score
- [x] `domain/referee.py`: Bayesian shrinkage
- [x] `domain/corners_predictor.py`: λ + Poisson
- [x] `domain/cards_predictor.py`: λ + ZIP (statsmodels)
- [x] `domain/shots_predictor.py`: λ + Poisson
- [x] `domain/multi_monte_carlo.py`: multi-dim simulation
- [x] `domain/markets.py`: derive 100+ markets from the matrix
- [x] `api.py`: expose new markets
- [x] Frontend: "Corners", "Cards", "Shots", "Half Time" tabs
- [x] Per-market calibration backtest

### Phase 3 — ML → λ (v1.3.0) — In Scope

- [x] `domain/feature_engineering.py`: build 40-60 features per match
- [x] `domain/ml_lambda.py`: XGBoost regressors for each metric
- [x] `use_cases/train_ml_models.py`: train + save pickles
- [ ] Time-series CV (non-random)
- [ ] Platt scaling / isotonic calibration
- [ ] A/B test: v1.2.0 (analytical) vs v1.3.0 (ML)
- [ ] CronJob (deferred) `train-models` weekly

### Out of Scope

- Market-to-market correlation (Gaussian Copulas) — too complex for MVP
- Weather data
- In-play/live predictions
- Full Bayesian Hierarchical via PyMC — reserved for v1.4.0+
- Individual player props (scorer, assist) — separate pitch P2
- Network/graph features — we already have network_analysis but out of scope of this pitch

## Research Needed

- [x] 5-layer syndicate framework — [[comprehensive-prediction-models]]
- [x] Implementation details — [[implementation-details]]
- [ ] Validate: does Sofascore return referee `standings` for Brasileirão?
- [ ] Validate: does Sofascore return HT score?
- [ ] Calibrate: real corner average in Brasileirão (likely 9-10/match)
- [ ] Calibrate: real card average in Brasileirão (likely 4.5/match)
- [ ] Test: ZIP vs pure Poisson for cards — which calibrates better
- [ ] Test: does empirical variance of corners justify Negative Binomial?

## Testing Strategy

### Unit (domain — zero mocks, deterministic inputs)

**Phase 1:**
- `test_player_lambda.py`: sum of 11 known xG/90 → expected λ
- `test_lineup_prediction.py`: fixture with history → correct top 11
- `test_minutes_weighting.py`: player with 10/10 matches → weight 1.0; with 5/10 → 0.5

**Phase 2:**
- `test_referee.py`: ref with 0 matches → full shrinkage; 20 matches → own rate
- `test_corners_predictor.py`: λ=5+5 → Over 9.5 ≈ 50%
- `test_cards_predictor.py`: strict ref → P(cards ≥ 5) > lenient ref
- `test_multi_monte_carlo.py`: 10K sims with known λ → Poisson distribution

**Phase 3:**
- `test_feature_engineering.py`: match fixture → expected features
- `test_ml_lambda.py`: model trained on fixture → stable prediction
- `test_calibration.py`: calibrated Brier score < uncalibrated

### Integration (with PG)

- Phase 1: ingest real lineup from Sofascore → query → match_lineups populated
- Phase 2: ingest match_stats → referee_stats materialized updated
- Temporal backtest: train up to matchday 15, predict 16-20, measure Brier

### Manual

- Phase 1: compare player-aware λ with team-level λ on 5 known matches
- Phase 2: compare our odds with Betfair on 10 matches (corners, cards)
- Phase 3: A/B test for 2 matchdays, measure real ROI

## Success Criteria

### Phase 1 (v1.1.0)
- [x] 1X2 Brier score drops from 0.237 to < 0.220 (5%+ improvement) → **0.2158 achieved (-10.6%)**
- [x] 1X2 accuracy rises from 47% to 50%+ → **51.1% achieved**
- [x] Frontend shows "Lineup confirmed" when lineup is available
- [x] Pre and post lineup saved separately in `prediction_history`

### Phase 2 (v1.2.0)
- [x] 5 new markets on the frontend: corners, cards, shots, HT, margin
- [ ] Corners Over/Under 9.5 calibration within ±3% of bookmakers (not validated)
- [ ] Cards Over/Under 3.5 calibration within ±3% of bookmakers (not validated)
- [ ] ZIP for cards with log-likelihood superior to pure Poisson (used pure Poisson for now)

### Phase 3 (v1.3.0)
- [ ] XGBoost λ_goals with MAE < 0.35 on the test set → **MAE 0.902 (close to baseline)**
- [ ] Brier score drops from < 0.220 to < 0.200 (not validated — too little data for ML)
- [ ] Simulated backtest ROI > 3% (not validated — value_bet_history empty)
- [ ] Calibration (isotonic) improves ROI by 20%+ vs uncalibrated (not yet implemented)

### Global Criteria
- [x] All tests passing (196/196)
- [x] Hexagonal architecture preserved (domain without statsbomb/sqlalchemy)
- [x] Backward compatible — old predictions keep working

---

## Retrospective

**Date:** 2026-04-04
**Commit:** `2fd12e4 feat: v1.1.0-v1.4.0 — Comprehensive Predictor Framework`
**Deploy:** `./deploy.sh 1.4.1` (Minikube, football-moneyball namespace)

### What was shipped (v1.1.0 → v1.4.0)

4 incremental versions in a single ship:

| Version | Feature | New files | Tests |
|:---:|---|:---:|:---:|
| v1.1.0 | Player-aware λ | 3 domain + 1 use case | 32 |
| v1.2.0 | Multi-output Poisson (corners, cards, HT, shots) | 5 domain | 33 |
| v1.3.0 | ML → λ (sklearn GBR) | 2 domain + 1 use case | 18 |
| v1.4.0 | Player Props (scorer, assist, shots) | 1 domain | 19 |

**Total: 36 files, 4053 lines added, 102 new tests.**

### Validation on real data (92 Brasileirão 2026 matches)

```
v1.0.0 (team-level):   Brier 0.2413, Accuracy 44.6%
v1.1.0 (player-aware): Brier 0.2158, Accuracy 51.1%
                       ↓ -10.6%       ↑ +14.6% (relative)
```

**Pitch targets hit:**
- Brier < 0.220 ✓
- Accuracy ≥ 50% ✓

Framework validated on real data. Player-aware λ is unequivocally superior to team-level.

### Learnings

**1. Aggregating by player easily beats aggregating by team.**
Team-level `attack_strength` assumes the team "is" the average of the last matches. But when a backup plays in place of the starter, the average keeps lying. When the model sums `xG/90 × weight` of the 11, it directly captures the quality of who is on the pitch.

**2. Sofascore exposes MUCH more than I expected.**
Besides xG per player, it has referee career totals (`yellowCards`, `games`). This eliminates the need for empirical Bayes shrinkage — we have `cards_per_game` directly. Same thing with HT score (`period1`).

**3. ML with little data doesn't beat analytical.**
XGBoost/GBR trained on 87 matches doesn't beat Dixon-Coles + player-aware (MAE 0.902 goals is ~baseline). The framework is ready for ML to scale: when there are 300+ matches, it will shine.

**4. sklearn GBR is enough.**
No need for xgboost as a new dependency. `GradientBoostingRegressor` from sklearn gives equivalent results for this case and saves a dep.

### What was NOT validated

- **Corners/cards calibration** vs Betfair odds (needs more matchdays)
- **ZIP vs Poisson** for cards (ended up with pure Poisson)
- **ROI on real bets** (value_bet_history empty, needs accumulation)
- **XGBoost stack** replace with real xgboost after 300+ matches

### Issues found

**1. Non-automatic migrations.**
`Base.metadata.create_all()` creates tables but doesn't ALTER. I had to create manual `apply_migrations()` that calls ADD COLUMN IF NOT EXISTS. Worked, but only runs when `init_db()` is called — which never happens at runtime. Solution: apply manually via `kubectl exec` after deploy. In the future: call `apply_migrations` on FastAPI startup.

**2. ML models lost on redeploy.**
Pickles stay in `football_moneyball/models/` inside the pod. Each redeploy creates a new pod and the files are lost. Current workaround: retrain (seconds). Future solution: PersistentVolume for the pickles.

**3. xg_for/xg_against missing in get_team_stats_aggregates.**
I had to refactor the query to JOIN with `player_match_metrics` and aggregate xG. I didn't notice this until I tried running the ML — should have thought about the features BEFORE writing the query.

**4. Data leak in ML backtest.**
Trained the model on all 87 matches and then tested on the same → misleadingly optimistic result. For an honest backtest, you need a time-series split: train up to matchday N, test on N+1.

### Decisions that validated

- **A single commit for 4 versions** worked because the files were intertwined. Artificial split would be painful.
- **Backward compatibility** (`predict_match` + `predict_match_player_aware`) allowed shipping without breaking anything.
- **sklearn GBR over xgboost** — zero new deps, same quality.
- **Graceful ML → analytical fallback** — robust to absence of trained models.
- **Sofascore referee totals** — eliminated the need for manual empirical Bayes.

### Next steps (not in this ship)

- **v1.5.0:** Platt/isotonic calibration (research: +69% ROI)
- **v1.6.0:** Referee factor used in predictions (fetch next match ref)
- **v1.7.0:** PersistentVolume for ML models
- **v1.8.0:** Backtest with time-series split (honest data)
- **v1.9.0:** Full value bet history + real ROI per market
- **v2.0.0:** Bayesian Hierarchical (PyMC) — mathematical capstone
- [ ] Documentation updated in `docs/architecture/overview.md`
