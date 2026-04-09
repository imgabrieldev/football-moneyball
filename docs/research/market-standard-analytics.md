---
tags:
  - research
  - analytics
  - xT
  - VAEP
  - OBV
  - pressing
  - RAPM
  - embeddings
  - market-standard
---

# Research — Market Standard in Football Analytics

> Research date: 2026-04-03
> Sources: [listed at the end]

## Context

The Football Moneyball analytical engine is at ~60% of the market standard. This research maps what elite clubs (Liverpool, Man City, Brighton, Brentford) and analytics companies (StatsBomb/Hudl, Opta/Stats Perform, SciSports) actually use, in order to prioritize the most impactful gaps.

---

## 1. Possession Value Models — The Core of Modern Analytics

### What the market uses

Every serious analytics company has a **Possession State Value (PSV) model** — the metric that answers "how much is each action on the pitch worth?". There are 5 main variants:

| Model | Creator | Approach | Required data |
|-------|---------|----------|---------------|
| **xT** (Expected Threat) | Karun Singh, 2018 | 16×12 grid, Markov chain, iterative | Event stream |
| **VAEP** | KU Leuven (Tom Decroos) | ML (gradient boosting), last 3 actions | Event stream (SPADL) |
| **OBV** (On-Ball Value) | StatsBomb/Hudl | 2 separate models (GF/GA), xG-trained | Event stream + pressure |
| **EPV** (Expected Possession Value) | Javier Fernández | State-of-the-art, off-ball | **Tracking data** |
| **g+** (Goals Added) | American Soccer Analysis | ML, endpoint-based | Event stream |

### xT — Expected Threat (Karun Singh)

The most accessible and widely implemented model. It works as follows:

1. **16×12 grid** divides the pitch into 192 zones
2. For each zone it computes:
   - `s(x,y)` — probability of shooting from that zone
   - `m(x,y)` — probability of passing/carrying
   - `g(x,y)` — probability of scoring when shooting from there
   - `T(x,y→z,w)` — transition matrix (probability of moving to each other zone)
3. **Iterative equation:**

   ```
   xT(x,y) = [s(x,y) × g(x,y)] + [m(x,y) × Σ T(x,y→z,w) × xT(z,w)]
   ```

4. Converges in ~5 iterations
5. **Action value** = `xT(destination) - xT(origin)`

**Implementable with StatsBomb open data.** No tracking required. The `socceraction` library (Python, MIT license) already implements xT and VAEP with StatsBomb loaders.

### VAEP — Valuing Actions by Estimating Probabilities

More sophisticated than xT:

- Uses the **last 3 actions** as context (not just location)
- Trains 2 ML models: P(score) and P(concede) in the next N events
- **VAEP(action) = ΔP(score) - ΔP(concede)**
- Features: action type, location, bodypart, result, context
- **SPADL** format standardizes events from any provider

**Advantage over xT:** captures context (a counter-attack is worth more than sterile possession). **Disadvantage:** requires training an ML model, more complex.

### OBV — On-Ball Value (StatsBomb/Hudl)

StatsBomb's commercial standard:

- **2 separate models**: Goals For and Goals Against (not just net)
- Trained on **StatsBomb xG** (not real goals — reduces variance)
- Features: location (x, y, distance/angle to goal), context (set piece vs open play), **defensive pressure**, bodypart
- **Excludes** possession history (avoids bias from playing style/team strength)
- **Includes defensive and goalkeeper actions**
- Pass receivers **do not receive** direct credit

**Key insight:** OBV separates offensive from defensive value, allowing players to be evaluated in both dimensions independently.

### Which to implement?

For Football Moneyball, the recommendation is:

1. **xT first** — simple implementation, high impact, data already available
2. **VAEP second** — via `socceraction` lib, ML-based, more accurate
3. OBV/EPV are proprietary or require tracking data

---

## 2. Pressing Metrics — The Most Visible Gap

### What the market measures

| Metric | Definition | Who uses |
|--------|------------|----------|
| **PPDA** | Passes allowed per defensive action (lower = more intense) | Everyone |
| **Counter-pressing fraction** | % of possessions where pressure is applied within ≤5s of losing the ball | StatsBomb |
| **Pressing success rate** | % of pressures that result in a recovery | StatsBomb, clubs |
| **High turnovers** | Recoveries within ≤40m of the opponent's goal | Opta, WhoScored |
| **Shot-ending high turnovers** | High turnovers that lead to a shot | StatsBomb |
| **Pressing intensity** | # of players involved in the counter-press | Tracking-based |

### Available StatsBomb data

StatsBomb defines a "pressure event" as a player **within ≤5 yards** of an opponent with the ball (expanded to 10y for goalkeepers). The data includes:

- Players involved
- Location on the pitch
- Pressure duration
- Outcome (recovery, foul, ball exit)

### Club references

- **Liverpool** (Slot): average PPDA of 9.89 — the lowest in the Premier League 2024/25
- **Man City** (Guardiola): PPDA ~8.3 in the 2017/18 season, "6-second rule"
- Pressing vs. possession correlation: **r = 0.86**

### What our engine is missing

We only have `pressure_events` (raw count) and `pressure_regains`. Missing:

- PPDA (computable: opponent passes / defensive actions)
- Pressing success rate (pressures → recovery / total pressures)
- Counter-pressing fraction (pressure within ≤5s of losing possession)
- High turnovers (recoveries in the final third)
- Pressing zone analysis (6 horizontal zones)

---

## 3. Advanced RAPM — From Simple Ridge to MBAPPE

### Our current RAPM

Simple Ridge regression: `stint × player` matrix, target = xG differential, CV alpha.

### MBAPPE — The state of the art

**M**ulti-League, **B**ayesian, **A**djusted and **P**enalized **P**lus-Minus **E**stimate:

1. **Multi-season** (2017-2022): more data → less collinearity
2. **Splints** (not stints): segments between substitutions **or goals** — more observations
3. **Modified xG**: adjusts shot value for the finisher's skill
4. **SPM as a Bayesian prior**: uses box-score stats (goals, assists, tackles/90) as a prior to regularize RAPM
5. **Ridge with prior**: `λ = σ²/τ²` balances data vs. informative prior
6. **Offensive/defensive split**: duplicates variables to estimate impact separately
7. **Design-weighted**: weights based on touch/pass location (not binary 1/-1)
8. **Multi-league adjustment**: normalizes metrics across leagues for cross-league comparison

### CMU Soccer RAPM (academic)

- ~4,000 stints per season (380 Premier League matches)
- FIFA ratings as external prior
- xG difference/90 as target
- Stint duration as weight

### What to improve in ours

1. **Splints** instead of stints (also break on goals)
2. **SPM prior** using box-score metrics we already compute
3. **Separate offensive/defensive split**
4. **Multi-season** for stability
5. **Design weights** by action location (not binary)

---

## 4. Position-Aware Embeddings — Fair Comparison

### Current problem

Our embedding compares all players in the same vector space. A goalkeeper can be "similar" to a forward.

### Market approaches

#### Clustering by position (Standard)

- **4 clusters per positional group** (defenders, midfielders, forwards)
- Defenders: "Playmaking Defender", "Traditional CB", "Balanced Defender", "Attacking FB"
- Midfielders: "Deep-lying Playmaker", "Box-to-box", "Creative AM", "Defensive Mid"
- Forwards: "Target Man", "Inside Forward", "Complete Forward", "Poacher"

#### Football2Vec (Ofir Magdaci)

- **Doc2Vec**: actions tokenized as "words", match as "document"
- **PlayerMatch2Vec**: 32-dim vector per player×match
- **Player2Vec**: average of the vectors of all matches
- Position captured implicitly from action patterns
- UMAP visualization shows natural position clusters

#### Graph Convolutional Networks

- Player-similarity graph with cosine distance
- GCN generates embeddings that capture topological relationships
- Used for transfer recommendation

### What to implement

1. **Position filter** before computing similarity (minimum viable)
2. **Separate embeddings per positional group** (more robust)
3. **Expanded archetypes**: from 6 to 12-16 contextualized roles
4. **Explained variance** of PCA (report % of information retained)
5. **Silhouette analysis** to determine optimal K in clustering

---

## 5. Complementary Metrics — Quick Wins

### Progressive Actions (partial already)

- Progressive passes (>10 yards closer to goal) — HAVE
- Progressive carries — HAVE
- Progressive receptions (receiving a progressive pass) — MISSING
- Contextual normalization (progressive in defensive third ≠ in final third) — MISSING

### Shot Quality

- Basic xG — HAVE
- Post-shot xG (PSxG) — shot quality after the strike — MISSING
- Shot placement analysis — MISSING
- Big chances created/missed — MISSING

### Dueling Detail

- Aerials won/lost — HAVE
- Ground duel win rate — MISSING
- Tackle success rate — MISSING
- Duel context (pitch third, under pressure) — MISSING

### Pass Breakdown

- Pass completion % — HAVE
- Short/medium/long pass success — MISSING
- Pass under pressure success — MISSING
- Switches of play — MISSING
- Through ball accuracy — MISSING

---

## 6. Open Source Tools in the Ecosystem

| Tool | What it does | Use it? |
|------|--------------|---------|
| **socceraction** | SPADL + xT + VAEP with StatsBomb loader | Yes — xT and VAEP ready |
| **mplsoccer** | Football viz (already used) | Already used |
| **statsbombpy** | StatsBomb API (already used) | Already used |
| **football2vec** | Doc2Vec embeddings | Evaluate — alternative approach |
| **kloppy** | Universal event data loader | Evaluate — if we want multi-provider |

---

## Implications for Football Moneyball

### Prioritization by impact × effort

| # | Feature | Impact | Effort | Priority |
|---|---------|--------|--------|----------|
| 1 | **xT model** (via socceraction or custom) | High | Medium | P0 |
| 2 | **Pressing metrics** (PPDA, success rate, high turnovers) | High | Low | P0 |
| 3 | **Position-aware similarity** (filter + separate embeddings) | High | Medium | P0 |
| 4 | **RAPM with SPM prior** + off/def split | Medium | High | P1 |
| 5 | **VAEP** (via socceraction) | Medium | Medium | P1 |
| 6 | **Shot/pass/duel breakdowns** | Medium | Low | P1 |
| 7 | **Multi-season RAPM** | Medium | Medium | P2 |
| 8 | **Football2Vec embeddings** | Low | High | P2 |

### Suggested roadmap

**v0.2.0 — Engine Upgrade (P0)**

- xT model (custom, 16×12 grid)
- Pressing metrics suite (PPDA, success rate, counter-press, high turnovers)
- Position-aware embeddings and expanded archetypes

**v0.3.0 — Advanced Models (P1)**

- VAEP integration
- RAPM with Bayesian SPM prior + offensive/defensive split
- Detailed metrics (shot quality, pass breakdown, duel context)

**v0.4.0 — Research Grade (P2)**

- Multi-season RAPM
- Football2Vec / GCN embeddings
- Cross-league normalization

---

## Sources

- [Introducing Expected Threat (xT) — Karun Singh](https://karun.in/blog/expected-threat.html)
- [VAEP — Valuing Actions by Estimating Probabilities — KU Leuven](https://dtai.cs.kuleuven.be/sports/vaep/)
- [socceraction — SPADL + xT + VAEP library](https://github.com/ML-KULeuven/socceraction)
- [On-Ball Value (OBV) — Hudl/StatsBomb](https://www.hudl.com/blog/introducing-on-ball-value-obv)
- [OBV Explainer — Hudl](https://www.hudl.com/blog/statsbomb-on-ball-value)
- [StatsBomb Counter-Pressing Metrics](https://blogarchive.statsbomb.com/articles/soccer/how-statsbomb-data-helps-measure-counter-pressing/)
- [PPDA Explained — Coaches' Voice](https://learning.coachesvoice.com/cv/ppda-explained-passes-per-defensive-action/)
- [MBAPPE Ratings — Game Models](https://www.gamemodelsfootball.com/about/about-mbappe-ratings)
- [CMU Soccer RAPM Model](https://www.stat.cmu.edu/cmsac/sure/2022/showcase/soccer_rapm.html)
- [Possession Value Models — FiveThirtyEight](https://fivethirtyeight.com/features/possession-is-the-puzzle-of-soccer-analytics-these-models-are-trying-to-solve-it/)
- [Football2Vec — Ofir Magdaci](https://github.com/ofirmg/football2vec)
- [Dynamic Expected Threat (DxT) — MDPI](https://www.mdpi.com/2076-3417/15/8/4151)
- [Devin Pleuler Analytics Handbook](https://github.com/devinpleuler/analytics-handbook)
- [Edd Webster Football Analytics Collection](https://github.com/eddwebster/football_analytics)
- [Decoding Player Roles via Clustering](https://medium.com/@marwanehamdani/decoding-player-roles-a-data-driven-clustering-approach-in-football-764654afb45b)
- [xT vs VAEP Comparison — Tom Decroos](https://tomdecroos.github.io/reports/xt_vs_vaep.pdf)
