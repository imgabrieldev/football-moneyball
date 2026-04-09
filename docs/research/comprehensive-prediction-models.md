---
tags:
  - research
  - models
  - machine-learning
  - poisson
  - bayesian
  - player-level
  - referee
  - xgboost
---

# Research — Complete Mathematical Models for Football Prediction

> Research date: 2026-04-04
> Sources: [listed at the end]

## Context

We need to evolve from "simple Poisson with team xG" to a complete system that uses ALL available data (players, coach, referee, corners, cards, shots, fouls) to predict ALL Betfair markets.

## Findings

### 1. Syndicate Architectures (Starlizard, Smartodds)

**Starlizard (Tony Bloom, £600M/year):**

- 4 specialized teams, each in a different role
- Computes the **most likely score** → derives all markets
- Focus on Asian Handicap (more liquid)
- Models consider: weather, team morale, lineup
- Bets close to match day (to include lineup info)
- Processes thousands of events per second

**Smartodds (Matthew Benham, owner of Brentford):**

- "We don't try to say what will happen, we try to say probabilities"
- If the probabilities are better than the bookmaker's → bet

**Insight:** Both generate a **probabilistic score** (score matrix) and derive all markets from it. Exactly what our Monte Carlo does — but they feed it with much richer data.

### 2. Academic Approaches (State of the Art 2024-2025)

#### A. Extended Dixon-Coles (Bivariate Poisson)

**Paper:** "Extending the Dixon and Coles model" (JRSS Series C, Jan 2025)

- Uses the Sarmanov family to model correlation between teams' goals
- Extensions include **tactical covariates** extracted from network clustering
- Time-varying parameters (attack/defense change throughout the season)

**Limitation:** Still operates at team level, not player level.

#### B. Inflated ML Poisson (Beat the Bookie, 2022)

**Model:** XGBoost/Random Forest → predicts λ (xG) → Poisson → score matrix

**Features (EMA of 5-20 matches):**

- Goals for/against
- xG for/against
- Shots and shots on target
- **Corner kicks** ← already includes corners
- Deep passes
- **PPDA** (pressing) ← already includes pressing

**Innovation:** Zero-Inflated Poisson (ZIP) to correct excess 0x0

**Result:** ~5.3% average profit in simulation (Big5 leagues, 7,178 matches)

#### C. Feedforward Neural Network + XGBoost Ensemble (2024)

**Paper:** "Data-driven prediction of soccer outcomes" (Journal of Big Data, 2024)

**Features include:**

- Out-to-in actions, interceptions, sprinting/min
- **Yellow/red cards, corners, shots on target**
- Possession percentage
- Half-time results (real-time)

**Best model:** Voting ensemble (Random Forest + XGBoost) + Feedforward NN

#### D. Bayes-xG: Player + Position Correction (2023)

**Paper:** "Bayes-xG" (arxiv 2311.13707)

**Innovation:** Even controlling for shot location, **player-level effects persist** — certain players have positive/negative xG adjustments. Uses a Bayesian Hierarchical Model.

**Implication:** Our model should adjust xG by player, not just by team.

#### E. Bayesian Hierarchical with Network Indicators (2021)

**Paper:** "The role of passing network indicators in modeling football outcomes" (Springer)

**Features:** Shots on target, **corners**, passing network metrics are the "main determinants" of results.

### 3. Recommended Unified Framework

Based on everything researched, the most robust model combines:

```
Layer 1: FEATURE ENGINEERING (per player of the 22 on the pitch)
├── Individual xG/90 (Sofascore expectedGoals)
├── Shots/90, shots on target/90 (totalShots, shotsOnTarget)
├── Crosses/90 (totalCross) → proxy for corners
├── Fouls/90 (fouls, wasFouled) → proxy for cards
├── Tackles/90 (totalTackle)
├── Passes/90 (totalPass, accuratePass)
└── Carries, progressive actions

Layer 2: TEAM AGGREGATION (sum/average of the 11 starters)
├── λ_goals = Σ xG/90 of the 11 × opponent_defense_factor
├── λ_shots = Σ shots/90 of the 11
├── λ_corners = f(crosses from the fullbacks + blocked shots)
├── λ_cards = Σ fouls/90 of the holding mids × referee_card_rate
├── λ_saves = opponent_shots × (1 - goalkeeper_save_rate)
└── λ_goals_HT = λ_goals × 0.45

Layer 3: CONTEXTUAL ADJUSTMENT
├── Home advantage (dynamic, computed from the DB)
├── Referee strictness (cards/match of this referee)
├── Derby factor (+20% cards, +10% corners)
├── Form/momentum (exponential EMA, last 5 matches weigh more)
├── Regression to the mean (dynamic k/(k+n))
└── Weather (if available)

Layer 4: MULTI-DIMENSIONAL MONTE CARLO
├── Simulate 10K matches
│   ├── Home goals ~ Poisson(λ_goals_home)
│   ├── Away goals ~ Poisson(λ_goals_away)
│   ├── Home corners ~ Poisson(λ_corners_home)
│   ├── Away corners ~ Poisson(λ_corners_away)
│   ├── Cards ~ ZIP(λ_cards × referee_factor)
│   ├── Shots ~ Poisson(λ_shots)
│   └── HT goals ~ Poisson(λ_goals × 0.45)
└── Each simulation produces a "complete match"

Layer 5: MARKET DERIVATION (from the simulated match)
├── 1X2, Correct Score, Asian Handicap
├── Over/Under goals (0.5 to 5.5)
├── BTTS, Double Chance, Draw No Bet
├── Over/Under corners (4.5 to 15.5)
├── Over/Under cards (0.5 to 8.5)
├── HT result, HT/FT, HT goals
├── Winning margin
├── Goals per team (home/away)
├── First goal
└── Player props (scorer, shots, fouls)
```

### 4. ML vs Poisson: Which to Use?

| Approach | Pros | Cons | When to use |
|----------|------|------|-------------|
| **Pure Poisson** | Interpretable, fast, score matrix | Ignores rich features | Baseline |
| **ML → Poisson** | Rich features → better λ | Requires data | **RECOMMENDED** |
| **XGBoost direct** | Better 1X2 accuracy | No score matrix, no multi-market | 1X2 only |
| **Neural Network** | Captures non-linearities | Black box, needs lots of data | If you have 10K+ matches |
| **Bayesian Hierarchical** | Quantified uncertainty, player-level | Complex, slow | If you want confidence intervals |

**Recommendation: ML → Poisson (Inflated)**

- XGBoost/Random Forest predicts λ for each metric (goals, corners, cards, shots)
- Feeds multi-dimensional Poisson
- Monte Carlo simulates the complete match
- All markets derived

### 5. What the Best Use That We Don't

| Feature | Starlizard | Academics | Us (today) | Us (v1.0.0) |
|---------|:---:|:---:|:---:|:---:|
| Team xG | Yes | Yes | Yes | Yes |
| Player xG | Yes | Yes | No | Yes |
| Lineup | Yes | No | No | Yes |
| Referee | Yes | partial | No | Yes |
| Corners | Yes | Yes | No | Yes |
| Cards | Yes | Yes | No | Yes |
| PPDA/pressing | Yes | Yes | partial | Yes |
| Weather | Yes | partial | No | No |
| Morale/momentum | Yes | No | partial | Yes |
| EMA form | Yes | Yes | Yes | Yes |
| Network analysis | No | Yes | have | evaluate |
| ML for λ | probable | Yes | No | **NEXT** |

## Implications for Football Moneyball

### Recommended evolution

1. **Now (v1.0.0 P0):** Already done — markets derived from the simple Monte Carlo
2. **Next (v1.0.0 P1):** Player-aware λ + new Poisson (corners, cards, shots)
3. **Later (v1.1.0):** ML → Poisson (XGBoost predicts λ using all features)
4. **Future (v1.2.0):** Bayesian Hierarchical with player-level effects + uncertainty

### The most impactful leap: player-aware λ

The gap between our model (47% accuracy) and the syndicates (~55-60% estimated) lies in:

1. **Lineup** — who plays changes everything
2. **Referee** — changes the cards λ by 50%+
3. **ML for λ** — XGBoost captures non-linear interactions between features

## Sources

- [Dixon-Coles Extensions — JRSS 2025](https://academic.oup.com/jrsssc/article/74/1/167/7818323)
- [Dixon-Coles Extensions — arxiv](https://arxiv.org/pdf/2307.02139)
- [Inflated ML Poisson — Beat the Bookie](https://beatthebookie.blog/2022/08/22/inflated-ml-poisson-model-to-predict-football-matches/)
- [Data-driven prediction — Journal of Big Data 2024](https://link.springer.com/article/10.1186/s40537-024-01008-2)
- [Predictive analytics framework — ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S2772662224001413)
- [Bayes-xG Player Correction — arxiv 2023](https://arxiv.org/html/2311.13707)
- [Bayesian Hierarchical — UCL](https://discovery.ucl.ac.uk/16040/1/16040.pdf)
- [Passing Network Indicators — Springer 2021](https://link.springer.com/article/10.1007/s10182-021-00411-x)
- [XGBoost + LSTM — IEEE 2024](https://ieeexplore.ieee.org/iel8/10935288/10935360/10935531.pdf)
- [Starlizard — The Dark Room](https://thedarkroom.co.uk/inside-tony-blooms-secret-betting-syndicate/)
- [Starlizard — Yahoo Finance](https://uk.finance.yahoo.com/news/inside-starlizard-story-britains-most-090759947.html)
- [Smartodds — Bleacher Report](https://bleacherreport.com/articles/2200795-mugs-and-millionaires-inside-the-murky-world-of-professional-football-gambling)
- [Soccermatics — Prediction](https://soccermatics.readthedocs.io/en/latest/lesson5/Prediction.html)
