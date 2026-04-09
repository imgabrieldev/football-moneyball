---
tags:
  - research
  - implementation
  - xgboost
  - compound-poisson
  - zip
  - referee
---

# Research — Implementation Details for the 5-Layer Framework

> Research date: 2026-04-04
> Complements: [[comprehensive-prediction-models]]

## Context

The [[comprehensive-predictor]] pitch proposes 5 layers. This doc collects the specific technical details for each piece: distributions to use, hyperparameters validated in the literature, imputation patterns for pre-lineup prediction.

## Findings

### 1. Distributions by Market

Each metric has a different appropriate distribution — pure Poisson does not fit everything.

| Metric | Distribution | Why | Paper |
|--------|:---:|---|---|
| Goals | Poisson or Dixon-Coles | rare independent events, adjustment for 0×0/1×0/0×1 | Dixon-Coles 1997 |
| Corners | **Compound Poisson** (Geometric-Poisson) | arrive in batches (clusters), serial correlation between phases | arxiv 2112.13001 |
| Cards | **Zero-Inflated Poisson (ZIP)** | excess of matches with 0 yellows (early-game fouls, ref strictness) | statsmodels |
| Shots | Poisson | more independent events, more regular shape | consensus |
| Shots on target | Conditional Poisson | P(on target \| shot) × total shots | consensus |
| Fouls | Poisson | ok as baseline, overdispersion may exist | Bayesian NB |
| HT goals | Poisson(λ × 0.45) | 45% of goals happen in the 1st half (consolidated) | Dixon-Coles |

**Implementation decision:** start with simple Poisson for everything (baseline), then swap to the specific ones where calibration fails.

### 2. Compound Poisson for Corners

**Why Compound:** corners come in sequence (team presses → several corners in a row). Simple Poisson assumes independence → underestimates variance → misses Over 10.5 / Under 8.5.

**Geometric-Poisson (Bayesian):**

```
N_batches ~ Poisson(λ_match)
corners_per_batch ~ Geometric(p)
Total_corners = Σ corners_per_batch
```

**Typical parameters (Premier League 2020-22):**

- λ_match ≈ 3.5 batches
- p ≈ 0.65 (average ~1.5 corners per batch)
- Total corners ≈ 10-11/match

**Simple implementation at first:** run Poisson and compare empirical variance. If var >> mean, migrate to Negative Binomial (which is a Compound Poisson with gamma).

### 3. Zero-Inflated Poisson for Cards

**Why ZIP:** cards have "excess zeros" — many matches without cards for a given player, team, or time interval.

```python
# statsmodels
from statsmodels.discrete.count_model import ZeroInflatedPoisson

# Fit
model = ZeroInflatedPoisson(
    endog=y_cards,              # total cards
    exog=X,                      # features
    exog_infl=X_infl,           # features for the inflation model
    inflation='logit',
).fit()
```

**Features for λ (count part):**

- Fouls/90 of home + away teams (EMA 5 matches)
- Referee card rate (history of that referee)
- Is_derby (derby → +20-30% cards)
- Home advantage (fewer cards at home)

**Features for π (inflation part — prob of 0 cards):**

- Peaceful team (average < 1 card/match)
- Tolerant referee (average < 2.5 cards/match)

**Brasileirão hyperparameters (estimated):**

- λ_base ≈ 4.5 cards/match
- Referee factor: 0.7 to 1.4 (tolerant to strict)
- Derby: × 1.25

### 4. Referee Strictness — How to Compute

**Problem:** referees call very different amounts of cards. The same team gets 2.5 cards with one referee and 4.5 with another.

**Bayesian formula with shrinkage:**

```
ref_card_rate = (n_cards + prior_weight × league_avg) / (n_matches + prior_weight)
```

With `prior_weight = 5`:

- Referee with 1 match → almost entirely league_avg
- Referee with 20 matches → almost entirely own history
- Smooth interpolation in between

**Alternative: Bayesian Hierarchical**

```
μ_ref ~ Normal(μ_league, σ_league)  # prior
cards_i ~ Poisson(μ_ref × feats_i)  # likelihood
```

Fit via PyMC or a simple conjugate update with Gamma-Poisson.

**For MVP:** use simple shrinkage formula (empirical Bayes). Reserve Bayesian hierarchical for v1.3.0+.

### 5. Pre-Lineup Prediction (Lineup Imputation)

**Scenario:** 24h before the match, Sofascore has not published the lineup. We need to predict without knowing the 11.

**Strategy 1: Most Frequent XI**

```python
def probable_lineup(team: str, last_n: int = 5) -> list[int]:
    """Returns the top 11 players by minutes over the last N matches."""
    recent = get_team_matches(team, last=last_n)
    minutes = defaultdict(int)
    for match in recent:
        for player in match.starters:
            minutes[player.id] += player.minutes_played
    return sorted(minutes, key=minutes.get, reverse=True)[:11]
```

**Strategy 2: Position-Aware**

- 1 GK: max minutes at position G
- 4 DEF: top 4 in minutes at position D
- 3-4 MID: top 3-4 at M
- 2-3 FWD: top 2-3 at F

Considers the team's most frequent formation (4-3-3, 4-4-2, 3-5-2) over the last matches.

**Strategy 3: Confidence-Weighted**

Some starters are a certainty (100% of matches), others are rotation. Weight features by the "prob of being a starter":

```python
for player in probable_xi:
    weight = player.matches_started / last_n_matches  # 0.0 to 1.0
    team_lambda += player.xg_per90 * weight
```

**Decision:** start with simple Most Frequent XI. Add position-aware when we have the formation.

### 6. ML → Poisson Pipeline (XGBoost for λ)

**Architecture:**

```
Features (team + player + context) → XGBoost Regressor → λ → Poisson → score matrix → markets
```

**Team features (last 6 matches, EMA with decay=0.9):**

- Goals for/against
- xG for/against
- Shots/shots on target
- Crosses
- Corners, cards, fouls
- PPDA (pressing)
- Possession %
- Progressive/completed passes

**Derived features:**

- xG overperformance (goals - xG in the last 10)
- H2H form (last 3 meetings)
- Rest days (fatigue)
- Home/away form separately

**Contextual features:**

- Is_home
- League avg xG (baseline)
- Opponent defense rating
- Is_derby
- Referee card rate

**Target:** empirical λ_goals (average goals scored in similar matches). Alternatively, train **against observed xG** (weak supervision) or real goals (strong supervision with more noise).

**XGBoost hyperparameters (Beat the Bookie + papers):**

```python
xgb.XGBRegressor(
    n_estimators=500,
    max_depth=4,           # shallow trees, prevents overfit
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    objective='reg:squarederror',
)
```

**Pipeline:**

```python
X_train, y_train = build_features(historical_matches)
model.fit(X_train, y_train)

# Temporal cross-validation (not random)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

**Expected performance (literature):**

- XGBoost 67% accuracy on 1X2
- ML Poisson: 5.3% average ROI in simulation (Beat the Bookie)

**Calibration matters more than accuracy:**

> "calibration-optimized model generated 69.86% higher average returns"
> — ScienceDirect 2024

Use `sklearn.calibration.CalibratedClassifierCV` or Platt scaling after XGBoost.

### 7. ML + Poisson Integration: Patterns

**Pattern A: ML predicts λ directly**

```
features → XGBoost → λ → Poisson(λ) → score matrix
```

Simple. Fails if ML's λ deviates from real λ.

**Pattern B: ML predicts residual of baseline Poisson**

```
features → Dixon-Coles → λ_base
features → XGBoost → δ
λ_final = λ_base + δ
→ Poisson(λ_final) → score matrix
```

More robust. XGBoost learns "what Dixon-Coles gets wrong".

**Pattern C: Ensemble**

```
λ_final = 0.5 × λ_DC + 0.5 × λ_XGB
```

Simple, combines both worlds.

**Decision:** start with Pattern A (v1.3.0). Migrate to B if calibrated Dixon-Coles remains competitive.

### 8. Multi-Dimensional Monte Carlo

**Joint simulation of all markets:**

```python
def simulate_match_full(
    home_xg, away_xg,
    home_corners_lambda, away_corners_lambda,
    home_cards_lambda, away_cards_lambda,
    home_shots_lambda, away_shots_lambda,
    n_sims=10_000,
):
    rng = np.random.default_rng(42)
    
    # Independent for simplicity (real: correlation between goals and shots)
    home_goals = rng.poisson(home_xg, n_sims)
    away_goals = rng.poisson(away_xg, n_sims)
    home_corners = rng.poisson(home_corners_lambda, n_sims)
    away_corners = rng.poisson(away_corners_lambda, n_sims)
    cards = rng.poisson(home_cards_lambda + away_cards_lambda, n_sims)
    home_shots = rng.poisson(home_shots_lambda, n_sims)
    away_shots = rng.poisson(away_shots_lambda, n_sims)
    
    # HT goals: 45% of total (approx)
    ht_home = rng.poisson(home_xg * 0.45, n_sims)
    ht_away = rng.poisson(away_xg * 0.45, n_sims)
    
    return simulate_df  # each row = 1 complete simulated match
```

**Important correlations (ignore in MVP):**

- Goals ↔ Shots (team that scores shoots more)
- Goals ↔ Corners (pressure → corners → goals)
- Cards ↔ Goals (losing → late goals → harder fouls)

Implementation with correlation: Gaussian Copulas (complex). **For v1.0: simulate independently and see if Brier degrades.**

### 9. Market Derivation from the Simulated Matrix

From the 10K simulation DataFrame, extract EVERYTHING:

```python
# We already have 1X2, O/U, BTTS, correct score, asian handicap

# New:
corners_over_95 = (sim_df.home_corners + sim_df.away_corners > 9.5).mean()
cards_over_35 = (sim_df.cards > 3.5).mean()
home_shots_over_125 = (sim_df.home_shots > 12.5).mean()
ht_result_home = (sim_df.ht_home > sim_df.ht_away).mean()
ht_ft_h_h = ((sim_df.ht_home > sim_df.ht_away) & 
             (sim_df.home_goals > sim_df.away_goals)).mean()

# HT score matrix
ht_scores = Counter(zip(sim_df.ht_home, sim_df.ht_away))

# Winning margin
margin_home_2 = (sim_df.home_goals - sim_df.away_goals == 2).mean()
```

**Total derivable markets:** 80-100+ from the single matrix.

### 10. Database Schema — New Fields

**match_stats (new):**

- match_id PK
- home_corners, away_corners
- home_yellow, away_yellow, home_red, away_red
- home_fouls, away_fouls
- home_shots, away_shots, home_sot, away_sot
- ht_home_score, ht_away_score
- referee_name
- attendance (optional)
- weather (optional)

**referee_stats (new):**

- referee_name PK
- matches INTEGER
- avg_yellow_per_match, avg_red_per_match
- avg_fouls_per_match
- avg_corners_per_match
- strictness_factor (card_rate / league_avg)

**team_form (materialized view? or computed on-the-fly):**

- team, competition, season
- last_5_goals_for, last_5_goals_against
- last_5_xg_for, last_5_xg_against
- last_5_corners_for, last_5_corners_against
- last_5_cards
- last_updated

**prediction_history — add fields:**

- home_lambda_corners, away_lambda_corners REAL
- home_lambda_cards, away_lambda_cards REAL
- home_lambda_shots, away_lambda_shots REAL
- model_version VARCHAR (e.g. "v1.1.0-player-aware")
- lineup_type VARCHAR ("pre" | "post")

## Implications for Football Moneyball

### Required libraries

Already have:

- numpy, pandas, scikit-learn, xgboost
- sqlalchemy, psycopg2, pgvector
- requests, typer, rich

**To add:**

- `statsmodels` (ZIP, GLM) — ~5MB
- Optional: `pymc` for Bayesian hierarchical (later)

### Implementation order (matching risk/value)

1. **v1.1.0 — Player-Aware λ** (2 weeks)
   - Sofascore already has the data
   - Backward compatible
   - Big accuracy jump expected

2. **v1.2.0 — Multi-Output Poisson** (2 weeks)
   - New λ for corners, cards, shots, HT
   - Referee module
   - Multi-dim Monte Carlo

3. **v1.3.0 — ML → λ** (2 weeks)
   - XGBoost for each λ
   - Calibration + backtest
   - A/B test against v1.2.0

## Sources

- [Compound Poisson for Corners (arxiv)](https://arxiv.org/abs/2112.13001)
- [ZIP Regression (statsmodels)](https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html)
- [NumPyro ZIP Example](https://num.pyro.ai/en/stable/examples/zero_inflated_poisson.html)
- [Bayesian Dynamic Models (arxiv 2508)](https://arxiv.org/html/2508.05891v1)
- [XGBoost Football Prediction (Research Gate)](https://www.researchgate.net/publication/369469857_Expected_Goals_Prediction_in_Football_using_XGBoost)
- [Beat the Bookie — Inflated Poisson](https://beatthebookie.blog/2022/08/22/inflated-ml-poisson-model-to-predict-football-matches/)
- [Calibration vs Accuracy (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2772662224001413)
- [Dixon-Coles Time-Weighted (dashee87)](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
