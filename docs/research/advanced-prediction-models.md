---
tags:
  - research
  - models
  - prediction
  - elo
  - machine-learning
  - draws
  - market-based
---

# Research — Advanced Football Prediction Models

> Research date: 2026-04-05
> Trigger: Matchday 10 — 10/10 home picks, 3 correct, -55% ROI. Structurally biased model.

## Context

Model v1.12.0 uses independent Poisson + Dixon-Coles + xG features + Platt calibration + market blending (65/35). Problem: **always predicts home win** — 10/10 picks on matchday 10, only 3 correct. Draws and away wins are never selected as favorites.

## 1. Rating Systems — Elo, Glicko-2, Pi-Rating

### Elo (FiveThirtyEight SPI / ClubElo)

```
R_new = R_old + K × G × (W - W_e)
W_e = 1 / (1 + 10^((R_away - R_home) / 400))
```

- **K=20** (league), K=30-60 (cups)
- **G** (goal diff): `(11 + goal_diff) / 8` if diff ≤ 1, log-scale afterwards
- **Home advantage**: +65 Elo (FiveThirtyEight) or +100 (ClubElo)
- **Benchmark**: ClubElo ~52-54% accuracy, Brier ~0.21

FiveThirtyEight SPI converts ratings → Poisson lambdas → bivariate simulation with correlation 0.1-0.2.

### Glicko-2 (Glickman, 2001)

Adds **rating deviation (RD)** and volatility. Advantage: promoted teams / early season have high RD → ratings move faster.

```
mu_new = mu + (phi² × Σ g(phi_j) × (s_j - E_j))
```

- **tau** (volatility constraint): 0.3-0.6 for football
- Theoretically superior to Elo for leagues with promotion/relegation

### Pi-Rating (Constantinou & Fenton, 2013)

**Designed specifically for football.** Keeps separate HOME/AWAY ratings per team.

```
R_home_new = R_home_old + γ × e × (goal_diff - e)
R_away_new = R_away_old + γ × e × (goal_diff - e)
e = (R_home_H - R_away_A) / 3
```

- **Benchmark: Brier 0.2065 on the EPL** (2001-2012), beat bookmaker odds in some studies
- Separate home/away ratings directly solve our home bias problem
- Source: JQAS 2013

### Recommendation

**Pi-Rating** as the rating backbone. Adjust Poisson lambdas by the Pi differential:

```python
lambda_home = xG_home × (1 + alpha × (pi_home_H - pi_away_A))
lambda_away = xG_away × (1 + alpha × (pi_away_A - pi_home_H))
```

When Pi-rating says the away team is stronger, lambda_home DROPS and lambda_away RISES.

---

## 2. Machine Learning — XGBoost, CatBoost, Ensemble

### Comparative benchmarks (2017 Soccer Prediction Challenge + recent papers)

| Model | Accuracy | RPS | Notes |
|---|---|---|---|
| **CatBoost + pi-ratings** | 55.82% | **0.1925** | Challenge winner, BEAT bookmakers |
| XGBoost + pi-ratings | 52.43% | 0.2063 | Strong but behind CatBoost |
| Berrar ratings + XGBoost/k-NN | 51.94% | 0.2054 | Hybrid approach |
| **Bookmaker consensus** | ~53% | 0.2012 | Practical ceiling (hard to beat) |
| Poisson (Dixon-Coles) | 48-52% | 0.21-0.22 | Baseline |
| Dolores (dynamic ratings + BN) | — | 0.2007 | 2nd in ML for Soccer |
| **Our model (v1.12)** | **42.4%** | **~0.24** | **Below standalone Elo** |

**Critical insight**: CatBoost + pi-ratings at **0.1925 RPS beat bookmakers** (0.2012) using relatively simple features. Our system has rich features (xG, pressing, network) but a weak model.

### Features that matter (papers)

1. **Pi-ratings / Elo** (highest importance)
2. **EMA form** (5-20 matches, exponential moving average)
3. **xG / xGA** (attack and defense)
4. **Shots, PPDA, possession** (pressure)
5. **Devigged odds** (market consensus as a feature!)
6. **H2H** (head to head)

### Recommended architecture

**Ensemble stacking**:

1. Poisson → score matrix → multi-market (keep, it is the base for corners/cards/correct score)
2. **CatBoost/XGBoost → 1x2 probs** (softmax, features including odds)
3. **Blend**: Poisson 1x2 × XGBoost 1x2 → calibration → final output

---

## 3. Market Approach — What Professionals Do

### Finding #1: Odds as features (not just for blending!)

BORS paper (PLOS ONE): pre-match odds contain **MORE information** than post-match results. Devigged bookmaker probs should be **input features**, not just a benchmark.

```python
# Instead of: blend(model_prob, market_prob)
# Do: model(features + [market_home, market_draw, market_away])
```

**Impact**: largest possible individual gain. Market already solves home bias.

### Finding #2: CLV as a metric

Sharp bettors measure success by **Closing Line Value** (CLV) — the Pinnacle line at closing is the market "truth". If our model does NOT consistently beat the closing line, it has no edge.

### Finding #3: Invert the architecture

Professionals (Starlizard, syndicates): start from the **market** and adjust with a model, not the other way around.

```
Odds → devig → true probs baseline → model makes marginal adjustments → bet if delta > threshold
```

Our pipeline: `model → calibration → light blend with odds`. It should be: `odds → fine-tuning with model`.

### Finding #4: Wisdom of crowds

Kaunitz et al. (arXiv): average odds from 30+ bookmakers, bet where the deviation > threshold → ~80% annual ROI (10 years sim, fractional Kelly).

### Finding #5: Starlizard

Tony Bloom (~£600M/year volume): ~100 analysts, soft data (morale, training), bets on Asian Handicap (more liquid). Uses model + soft data layering. Not purely quantitative.

---

## 4. Draw Prediction — Specific Methods

### Why draws are hard

- Independent Poisson systematically underestimates P(X=Y) by ~3-5pp
- In top leagues, draw rate = 25-28%, Poisson models give 20-24%
- Draw is never the argmax (even at 28%, home/away are usually > 30%)
- **Every ML model has a draw F1 of ~0.30** — it is the hardest class

### Methods ranked by impact

| Method | Draw improvement | Complexity | Our status |
|---|---|---|---|
| **Draw probability floor** | +3-5pp | Trivial | Not implemented |
| **Dixon-Coles rho fit** | +2-4pp | Already have | rho=0.009 (≈0) |
| **Bivariate Poisson** | +1-3pp | Already have | λ₃=0.0001 (≈0) |
| **Draw-likelihood features** | +2-3pp on "draw-prone" matches | Medium | Not implemented |
| **Copula (Frank)** | +0.5pp (marginal) | High | Skip |
| **Zero-inflated Poisson** | 0-0 only | Low | Skip |

### Draw probability floor (recommended, trivial)

If the model's mean draw_prob < empirical league rate (25-27%):

```python
empirical_draw_rate = 0.26  # historical Brasileirão
correction = empirical_draw_rate / model_mean_draw_prob
draw_prob *= correction
# renormalize
```

### Draw-likelihood features

Features correlated with draws:

- Both teams xG < 1.2
- O/U 2.5 market with under favored
- H2H with draw rate > 40%
- Both in the lower half of attack + upper half of defense

Binary flag "draw_prone" → boost draw_prob by 10-15%.

---

## 5. Diagnosis of Our Model

### Why it always predicts HOME

1. **Majority class**: home wins = 47-50% in the Brasileirão (plurality class). Without a counter-mechanism, the model defaults to home.
2. **Home advantage as a flat boost**: `home_xg += 0.3-0.5 xG` without modulating by opponent quality. Palmeiras away gets the same penalty as Remo away.
3. **Features correlate with the home label**: team_xg_for is typically > opp_xg_for for home teams (sample bias).
4. **Draw is never the argmax**: even at 32%, home is usually 40%+.
5. **No odds as input**: the market knows when a match is balanced. The model does not.

### Performance hierarchy (literature)

```
aggregated odds > wisdom of crowds > Elo + XGBoost > standalone Elo > our model
```

---

## 6. Prioritized Action Plan

### Phase 1 — Quick wins (1-2 days, high impact)

1. **Odds as features in XGBoost**: add devigged Betfair probs (home/draw/away) as 3 features in `feature_engineering.py`. The GBR will learn to weigh model vs market.
2. **Draw floor**: post-calibration, if draw_prob < 0.22, boost to min(0.22, draw_prob×1.3), renormalize.
3. **Invert blend alpha**: 35% model / 65% market (was 65/35). Market is more calibrated than our model.

### Phase 2 — Structural (1 week)

4. **Pi-Rating** with separate home/away ratings. Use the Pi differential to modulate the Poisson lambda.
5. **CatBoost 1x2**: train a 3-class model with features [Elo, Pi, xG, form EMA, odds, H2H, rest days]. Target: RPS < 0.21.
6. **RPS as the primary metric** instead of Brier. Implement in track-record and backtest.

### Phase 3 — Advanced (2+ weeks)

7. **Bayesian hierarchical** (PyMC/Stan) for team attack/defense with early-season shrinkage.
8. **Ensemble stacking**: Poisson (multi-market) + CatBoost (1x2) + ordinal regression (draw specialist).
9. **CLV tracking**: compare our probs with the Pinnacle/Betfair closing line.

---

## Sources

### Rating Systems

- [FiveThirtyEight SPI Methodology](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/)
- [ClubElo](http://clubelo.com/System)
- [Pi-Rating — Constantinou & Fenton 2013 (JQAS)](https://www.degruyter.com/document/doi/10.1515/jqas-2012-0054/html)
- [Glicko-2 — Glickman 2001](http://www.glicko.net/glicko/glicko2.pdf)

### Machine Learning

- [xG Football Club — Which ML Models](https://thexgfootballclub.substack.com/p/which-machine-learning-models-perform)
- [CatBoost 0.1925 RPS — Soccer Prediction Challenge](https://link.springer.com/article/10.1007/s10994-018-5703-7)
- [Journal of Big Data 2024 — Data-driven prediction](https://link.springer.com/article/10.1186/s40537-024-01008-2)
- [Bayesian state-space EPL — JRSS 2025](https://academic.oup.com/jrsssc/article/74/3/717/7929974)
- [Ordinal probit — Univ. St. Gallen](https://ux-tauri.unisg.ch/RePEc/usg/econwp/EWP-1811.pdf)
- [Systematic Review ML in Sports Betting — arXiv 2024](https://arxiv.org/html/2410.21484v1)

### Market-Based

- [BORS — PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
- [Kaunitz et al. — arXiv](https://arxiv.org/abs/1710.02824)
- [Pinnacle CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)
- [Starlizard — Off The Pitch](https://offthepitch.com/a/secrets-starlizard-how-tony-blooms-football-data-monolith-using-its-knowledge-protect-game)
- [Wilkens 2026 — Bundesliga](https://journals.sagepub.com/doi/10.1177/22150218261416681)
- [LSE Wisdom of Crowds](https://blogs.lse.ac.uk/europpblog/2025/05/29/football-forecasting-harnessing-the-power-of-the-crowd/)

### Draw Prediction

- [Penaltyblog — Which Model (RPS benchmarks)](https://pena.lt/y/2025/03/10/which-model-should-you-use-to-predict-football-matches/)
- [Karlis & Ntzoufras 2003 — Bivariate Poisson](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)
- [Pinnacle — Draw Inflation](https://www.pinnacle.com/betting-resources/en/soccer/inflating-or-deflating-the-chance-of-a-draw-in-soccer/cge2jp2sdkv3a9r5)
- [Wheatcroft 2021 — Match Statistics](https://journals.sagepub.com/doi/10.3233/JSA-200462)
