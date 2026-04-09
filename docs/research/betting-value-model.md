---
tags:
  - research
  - betting
  - odds
  - monte-carlo
  - kelly-criterion
  - value-betting
---

# Research — Betting Value Model

> Research date: 2026-04-03
> Sources: [listed at the end]

## Context

Investigate the feasibility of using our statistical predictions (xG, pressing, RAPM) to identify value bets in the Brasileirão by comparing against bookmaker odds.

## Findings

### 1. Available Odds APIs

**The Odds API** (the-odds-api.com) — Best option:

- **Free tier: 500 requests/month** (enough for ~50 matchdays of 10 matches)
- Covers **Brasileirão Série A** (sport key: `soccer_brazil_campeonato`)
- Available markets:
  - `h2h` — moneyline (1X2 with draw)
  - `totals` — over/under goals
  - `btts` — both teams to score (Yes/No)
  - `spreads` — handicap
  - `draw_no_bet`, `double_chance`
- ~30 bookmakers (Bet365, Betano via region, Pinnacle, 1xBet, etc.)
- **Historical odds** since Jun/2020 (snapshots every 5-10 min)
- Paid plan: $30/month for 20K requests

**API-Football** ($19/month) — Alternative:

- `/odds` endpoint with pre-match odds
- Covers Brasileirão
- Fewer markets than The Odds API

### 2. Prediction Model: Monte Carlo + Poisson + xG

**Industry standard approach:**

1. **Compute expected xG** for each team for the upcoming match (average xG/match over the last N matches, adjusted for opponent strength)
2. **Model goals as Poisson**: `P(k goals) = (λ^k × e^(-λ)) / k!` where `λ = expected xG`
3. **Simulate N=10,000+ matches** via Monte Carlo: draw goals for each team from the Poisson distribution
4. **Compute probabilities**: P(home win), P(draw), P(away win), P(over 2.5), P(BTTS), etc.

**Advanced adjustments:**

- Home factor (home advantage ~+0.3 xG in the Brasileirão)
- Opponent pressing intensity (PPDA affects expected xG)
- Recent form (last 5 matches exponentially weighted)
- Key players' RAPM (lineup matters)

### 3. Odds → Implied Probability Conversion

```
Decimal odds: implied_prob = 1 / odds
Example: Palmeiras @ 1.80 → 1/1.80 = 55.6%

Remove margin (vig/juice):
total = sum(1/odds_i for each outcome)
real_prob_i = (1/odds_i) / total
```

### 4. Value Bet Identification

**Value bet** = when our estimated probability > the odds' implied probability.

```
edge = model_prob - implied_prob
If edge > 0 → value bet

Example:
- Model says: Palmeiras wins with 60%
- Betano offers odds 1.80 → implied 55.6%
- Edge = 60% - 55.6% = +4.4% → VALUE BET
```

**Minimum threshold:** edge > 3% (safety margin against model error)

### 5. Kelly Criterion for Sizing

```
f* = (b × p - q) / b

Where:
- f* = fraction of bankroll to bet
- b = decimal odds - 1 (net odds)
- p = probability estimated by the model
- q = 1 - p

Example:
- p = 0.60, odds = 1.80, b = 0.80
- f* = (0.80 × 0.60 - 0.40) / 0.80 = 0.10 = 10% of bankroll
```

**In practice:** use **fractional Kelly (25%)** to reduce variance:

- Bet = 0.25 × f* × bankroll

### 6. Backtesting

With 87 Brasileirão 2026 matches already ingested:

1. For each match, compute expected xG based on previous matches
2. Run Monte Carlo → model probabilities
3. Compare with historical odds (The Odds API historical endpoint)
4. Identify where there would have been value bets
5. Compute simulated ROI with fractional Kelly

**Evaluation metrics:**

- **ROI** (Return on Investment): profit / total staked
- **Hit rate**: % of value bets that won
- **Brier score**: probability calibration (0=perfect, 0.25=random)
- **Max drawdown**: largest losing streak
- **Adapted Sharpe ratio**: return adjusted for volatility

## Implications for Football Moneyball

### Feasibility

- **The Odds API free tier** (500 req/month) is enough for Brasileirão (~10 matches × 4 markets × ~9 matchdays = ~360 requests)
- Our Sofascore xG data is reliable for the Poisson model
- Monte Carlo is computationally trivial (10K simulations < 1 second)
- Backtesting with 87 matches gives a reasonable initial sample

### Risks

- xG model may have bias (Sofascore vs StatsBomb compute xG differently)
- 87 matches is a small sample to validate a statistical edge (ideal: 500+)
- Opening vs closing odds: timing matters
- The Brazilian betting market has evolving regulation

## Sources

- [The Odds API](https://the-odds-api.com/)
- [The Odds API Documentation V4](https://the-odds-api.com/liveapi/guides/v4/)
- [The Odds API Betting Markets](https://the-odds-api.com/sports-odds-data/betting-markets.html)
- [Monte Carlo Football Match Sim](https://github.com/TacticsBadger/MonteCarloFootballMatchSim)
- [Kelly Criterion — Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Football Odds Monte Carlo — Medium](https://medium.com/@arit.pom/football-odds-data-analysis-using-montecarlo-simulation-in-python-part-2-43f5e951c1fc)
- [OddAlerts Football Data API](https://www.oddalerts.com/football-data-api)
- [Bankroll Management — Tradematesports](https://www.tradematesports.com/en/blog/bankroll-management-sports-betting)
