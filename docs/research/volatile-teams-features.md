---
tags:
  - research
  - prediction
  - features
  - volatility
  - brasileirao
---

# Research — Features for Volatile Teams and State-of-the-Art

> Research date: 2026-04-09
> Context: Model v1.14.2 has 40.7% 1X2 accuracy (Brier 0.2358). The market delivers ~0.20. Teams like Corinthians (20%), Palmeiras (11%), Mirassol (0%) are unpredictable. We need to close the ~15% gap in Brier.

## 1. State of the Art — What Works

### 1.1 Models and Benchmarks

| Model | RPS | Accuracy | Source |
|-------|-----|----------|--------|
| CatBoost + Pi-Ratings | **0.1925** | **55.8%** | Razali et al. |
| XGBoost + Pi-Ratings | 0.2063 | 52.4% | Hubacek et al. (2017 Challenge) |
| XGBoost + Berrar Ratings | 0.2054 | 51.9% | Berrar et al. |
| Hybrid Bayesian Network | 0.2083 | — | Constantinou |
| xG Poisson (best config) | Brier 58.6 | — | beatthebookie |
| Goals Poisson (best config) | Brier 59.7 | — | beatthebookie |
| Bet365 closing odds | Brier 57.2 | — | beatthebookie |
| **Our model v1.14.2** | **Brier ~0.236** | **40.7%** | — |

**Conclusion**: CatBoost + Pi-Ratings is the state-of-the-art (RPS 0.1925). We already have CatBoost + Pi-Rating, but something is off — probably insufficient features.

### 1.2 Features That Matter Most (ranked)

Based on XGBoost feature importance from a study with 269 teams, 7 European leagues, 20 seasons:

1. **ELO/Pi-Rating difference** — contribution ~4-8x larger than contextual features
2. **Recent domestic performance** (rolling form)
3. **Travel distance** (>500 miles = -15% away win rate)
4. **Rest days between matches** (<3 days after travel = vulnerable)
5. **Manager tenure** — smaller but significant contribution
6. **Player rotations** — indirect indicator of fixture congestion

### 1.3 The Brier vs Profit Paradox

Critical finding from beatthebookie:

- The model with **worst Brier score** generated the **highest profit** in betting
- Minimizing prediction error != maximizing profit
- Betting models should optimize **gap detection** (edge vs market), not pure calibration
- Implication: our Brier focus may be secondary to the focus on real edge vs Betfair

## 2. Features Missing From Our Model

### 2.1 Rolling Form with EMA (Exponential Moving Average)

**Current status**: we have basic form EMA in CatBoost.

**What is missing**:

- **xG-based form** (not just goals): rolling xG For - xG Against over the last 5-10 matches
- **Venue-specific form**: separate home/away form (we have partial)
- **Window testing**: test windows of 5, 10, 15, 20, 35 matches
- **Key insight**: the biggest improvement comes from using **xG instead of goals**, not from the smoothing type (EMA vs SMA)

Beat the bookie tested 14 combinations of window x weighting:

- xG-based models dominated (8 of the 10 best)
- EMA showed marginal improvement vs SMA
- **The feature itself (xG) matters more than the smoothing**

### 2.2 Coach Profile — Full Coach Profile

**Brasileirão context**: 22 coaches fired in 38 matchdays (2025). It is the #1 volatility factor.
The coach is not just a binary flag — it is a multidimensional profile that impacts tactics, results and style.

#### 2.2.1 Coach Performance Rating

Based on the Analytics FC framework (2021):

**Dual ratings (ELO-like)**:

- **Results Rating**: W/D/L adjusted by pre-match expectation (a strong team beating a weak one = small gain)
- **Performance Rating**: based on xT (Expected Threat) — rewards coaches whose teams generate more threat regardless of the score

**Derived features**:

- `coach_win_rate_career`: career win %
- `coach_win_rate_current_team`: win % with this team
- `coach_win_rate_last_10`: recent coach form (EMA last 10 matches)
- `coach_avg_xg_for`: average xG produced by their teams
- `coach_avg_xg_against`: average xG conceded
- `coach_tenure_days`: time in the current role
- `coach_tenure_bucket`: <30d (honeymoon), 30-90d (adaptation), 90-180d (consolidated), >180d (established)
- `coach_changed_30d`: binary flag — recent change
- `coach_teams_count`: how many teams they have coached (experience)

#### 2.2.2 Coach Tactical Profile (8 metrics — Analytics FC)

Each coach can be profiled on 8 tactical dimensions:

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| **Long Balls** | Direct play in the defensive third | % long passes in the defensive third |
| **Deep Circulation** | Short vs direct build-up | ratio of short/long passes in the defensive third |
| **Wing Play** | Progression down the flanks | % offensive actions in the wide lanes |
| **Territory** | Territorial dominance (field tilt) | % actions in the offensive third |
| **Crossing** | Box entries via crossing | % crosses vs other entries |
| **High Press** | High press intensity | PPDA (passes per defensive action) — we already have |
| **Counters** | Counter-attacks | fast transitions after recovery |
| **Low Block** | Defensive low block | % defensive actions in the defensive third |

**Similarity via Kolmogorov-Smirnov Test**: compares full distributions (not just means) between coaches. Allows measuring coach-team "compatibility".

#### 2.2.3 Coach-Team Compatibility

Key feature: **how different is the coach's profile vs what the team played before?**

- `style_distance`: Euclidean distance between the coach's tactical profile and the team's historical style
- `similar_teams_coached`: % of the coach's previous teams with profiles similar to the current team
- `league_experience`: has the coach coached in this league/division before? (Brasileirão is very different from Série B)

**Hypothesis**: teams with high `style_distance` (new coach with a very different style from the previous one) should have more variance in early results — the model can adjust confidence on those matches.

#### 2.2.4 Regime Detection

- **Honeymoon effect**: teams usually improve in the first 5-7 matches after a coach change
- **Performance reset**: when the coach changes, reduce the weight of recent history (more aggressive decay factor)
- **Data**: `ingest-context` already brings managers from Sofascore. Need to enrich with the coach's history.

#### 2.2.5 Data Sources for Coach Profile

- **Sofascore API**: already in use, has manager per team
- **Transfermarkt**: full career history, teams coached, dates
- **Sofascore match-level**: xG, possession, PPDA per match = allows recomputing the tactical profile
- **Our DB**: match_stats already has xG, possession, shots — we can derive the team tactical profile under each coach

### 2.3 Fixture Congestion & Rest Days

**Features**:

- `rest_days_home` / `rest_days_away`: days since the last match
- `games_last_7d` / `games_last_14d`: we already have in context
- `travel_distance_km`: distance between cities (Brazil has long flights: Porto Alegre -> Manaus)
- `cup_match_midweek`: flag if there was a cup match in midweek

**Impact**: contribution 4-8x smaller than ratings, but relevant for teams with a tight calendar (Libertadores + Brasileirão + Copa do Brasil).

### 2.4 Squad Depth & Key Player Absence

**Features**:

- `key_players_missing`: count of absent starters (injury/suspension)
- `squad_rotation_rate`: % of changes in the XI vs the previous match
- `total_market_value_ratio`: home/away market value ratio (Transfermarkt)
- **Already have**: `ingest-lineups` + `ingest-context` (injuries). Missing pipeline into CatBoost.

### 2.5 Draw-Specific Features

Draws are the weak point of ALL models (F1 ~0.30 vs 0.75 for wins). Specific features:

- **Style matchup**: defensive vs defensive = more draws
- **League position proximity**: teams close in the table draw more
- **Goal expectation < 2.0**: matches with low total xG draw more
- **Derby flag**: derbies tend to have more draws
- **Handicap spread**: if spread < 0.5 goal, draw prob rises

## 3. Action Plan — Prioritized by Impact

### Tier 1: High Impact, Data Already Available (1-2 days each)

1. **xG Rolling Form**: replace goals with xG in the form strength calculation
   - We already have xG in match_stats. Just need to change the attack/defense strength computation
   - Expected: ~2-3% Brier improvement (xG models dominate 8/10 top configs)

2. **Basic Coach Profile**: tenure + win rate + change flag
   - Manager data already in DB via `ingest-context`
   - Features: `coach_tenure_days`, `coach_changed_30d`, `coach_tenure_bucket`
   - `coach_win_rate_current_team` computable from our match results
   - Expected: strong improvement on volatile teams (Corinthians, Mirassol, etc.)

3. **Rest days + fixture congestion**: days between matches per team
   - `rest_days = commence_time - last_match_time`
   - `games_last_14d` we already have partial in context
   - Expected: improvement on midweek + Libertadores weeks

### Tier 2: Medium Impact, Partial Data (3-5 days)

4. **Coach Tactical Profile (8 metrics)**: tactical profile derived from match_stats
   - We already have PPDA. Possession, shots, long balls — derive from match_stats
   - Compute `style_distance` (coach profile vs team history)
   - Needs a new pipeline but the data is already in the DB

5. **Draw-specific features**: league position gap, total xG, style matchup
   - We already have standings. Compute points gap + expected total xG
   - Derby flags, handicap spread derived from odds
   - Can significantly improve draw F1 (0.30 -> ?)

6. **Key player absence score**: lineups pipeline -> absences feature
   - Lineups ingested. Need a "key player" heuristic
   - Impact score = minutes_played * (goals+assists) / total_time

### Tier 3: High Impact, High Effort (1+ week)

7. **Full coach history**: career, previous teams, compatibility
   - Transfermarkt scraping or Sofascore enrichment
   - `similar_teams_coached`, `league_experience`, `coach_avg_xg_career`
   - Kolmogorov-Smirnov test to measure style similarity

8. **Ensemble meta-learner**: combine Poisson + CatBoost + Dixon-Coles with stacking
   - Layer 1: each model generates independent probs
   - Layer 2: meta-learner (logistic regression) combines the outputs
   - Benchmarks show 70%+ accuracy with ensembles

9. **Edge-based optimization**: instead of minimizing Brier, maximize edge vs Betfair
   - Custom loss function that penalizes errors where we had an edge
   - Can improve ROI without improving Brier (Brier vs profit paradox)

## 4. Implications for Football Moneyball

### What changes in the architecture

- `match_predictor.py`: add xG-based form (swap goals for xG in the rolling)
- `train_catboost.py`: add context features (coach, rest days, standings gap)
- `predict_all.py`: pipeline to extract new features before predicting
- `domain/features.py` (new?): centralized feature engineering module

### What does NOT change

- Infra (K8s, PG, CronJobs) — everything works
- Market blending — still important (65% market)
- Calibration — Dixon-Coles rho + draw floor 26% remain

### Suggested priority

**Next pitch**: implement Tier 1 (xG form + coach tenure + rest days) as CatBoost features. Improvement estimate: Brier from 0.236 -> ~0.215-0.220 (market gap drops from 15% to ~5-8%).

## Sources

- [Best Football Prediction Algorithms 2026](https://www.golsinyali.com/en/blog/best-football-prediction-algorithms-2026)
- [Scoring functions vs. betting profit](https://beatthebookie.blog/2022/03/29/scoring-functions-vs-betting-profit-measuring-the-performance-of-a-football-betting-model/)
- [Which ML Models Perform Best](https://thexgfootballclub.substack.com/p/which-machine-learning-models-perform)
- [The predictive power of xG](https://beatthebookie.blog/2021/06/07/the-predictive-power-of-xg/)
- [Predicting football results — Dixon-Coles](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
- [Match outcome factors — elite European football (Settembre et al.)](https://journals.sagepub.com/doi/10.3233/JSA-240745)
- [Travel distance impact](https://nerdytips.com/blog/the-hidden-influence-of-travel-distance-on-football-betting-outcomes/)
- [Brasileirão predictive model (RBFF)](https://www.rbff.com.br/index.php/rbff/article/view/1265)
- [Coaching turnover in Brazilian football](https://jornalismojunior.com.br/troca-repete-e-recomeca-o-ciclo-infinito-das-trocas-de-tecnicos-no-brasil/)
- [Fixture congestion meta-analysis (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7846542/)
- [CatBoost + Pi-Ratings (Razali et al.)](https://arxiv.org/html/2309.14807)
- [Profiling Coaches with Data — Analytics FC](https://analyticsfc.co.uk/blog/2021/03/22/profiling-coaches-with-data/) — 8 tactical metrics + dual ELO + K-S similarity
- [Predicting Success of Football Coaches — Dartmouth](https://sites.dartmouth.edu/sportsanalytics/2024/01/23/predicting-the-success-of-football-coaches/) — WPA/RSA metrics
- [Coaching Tactical Impact Serie A](https://arxiv.org/pdf/2509.22683) — fixed effects model, home advantage x coaching
- [Tactical Situations and Playing Styles as KPIs (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11130910/)
