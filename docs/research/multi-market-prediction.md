---
tags:
  - research
  - betting
  - markets
  - corners
  - cards
  - poisson
  - betfair
---

# Research — Multi-Market Prediction (All Betfair Markets)

> Research date: 2026-04-04
> Sources: [listed at the end]

## Context

Betfair Exchange offers ~8 market categories for each Brasileirão match. Our model only predicts 2 (1X2 and Over/Under goals). We need to expand to cover all of them.

## Betfair Markets and How to Predict Each

### 1. Match Odds (1X2) — ALREADY HAVE

- **Model:** Bivariate Poisson (Dixon-Coles)
- **Input:** historical xG, attack/defense strength
- **Already implemented in v0.5.0**

### 2. Over/Under Goals (0.5, 1.5, 2.5, 3.5) — ALREADY HAVE

- **Model:** Derived from Monte Carlo (we already compute over_05, over_15, over_25, over_35)
- **Just needs to be exposed in the frontend**

### 3. BTTS (Both Teams To Score) — ALREADY HAVE

- **Model:** Derived from Monte Carlo (btts_prob)
- **Just needs to be exposed**

### 4. Correct Score — ALREADY HAVE

- **Model:** score_matrix from Monte Carlo (top 10 scores with probability)
- **Just needs to be exposed**

### 5. Corners (Over/Under) — NEEDS IMPLEMENTATION

- **Model:** Compound Poisson Regression (paper: Arxiv 2112.13001, published 2024)
- **Required input:** average corners per team (home/away), opponent corners
- **Average λ:** ~10 corners per match in football (5 + 5)
- **Does not follow simple Poisson** — has clustering (serial correlation between consecutive corners)
- **Best model:** Geometric-Poisson (compound) with Bayesian implementation
- **Data:** Sofascore HAS per-match corner data
- **Pragmatic approach:** simple Poisson with λ = team corners average × opponent factor. Not perfect but works for Over/Under 8.5, 9.5, 10.5

### 6. Cards (Over/Under) — NEEDS IMPLEMENTATION

- **Model:** Zero-Inflated Poisson (ZIP) — many matches with 0-1 cards, distribution is not pure Poisson
- **Required input:**
  - **Team Aggression Score:** fouls per match, team's historical cards
  - **Referee Strictness Score:** average cards per match for the specific referee
  - **Match Context:** derby/rivalry, match importance
- **Average λ:** ~4 cards per match
- **Conceptual formula:** `Expected Cards = (team_A_fouls_avg + team_B_fouls_avg) × referee_card_rate`
- **Data:** Sofascore HAS per-match data for cards and fouls. **Referee** is the differentiator — Sofascore shows the assigned referee

### 7. Asian Handicap — DERIVABLE FROM WHAT WE HAVE

- **No new model needed** — it is derived from 1X2 probabilities
- **Asian -0.5:** = win probability (without draw)
- **Asian -1.0:** = probability of winning by 2+ goals
- **Asian -1.5:** = probability of winning by 2+ goals (already in score_matrix)
- **Calculation:** sum probabilities of the relevant scores from the score_matrix

### 8. Half Time Result — NEEDS IMPLEMENTATION

- **Model:** separate Poisson with λ_HT ≈ 0.45 × λ_FT (first half has ~45% of goals)
- **Simulate:** separate Monte Carlo with xG adjusted for the 1st half
- **Data:** Sofascore HAS half-time score for every match

### 9. Player to Score — PARTIALLY POSSIBLE

- **Requires:** individual player xG + expected minutes
- **We have:** xG per player per match in player_match_metrics
- **Approach:** P(player scores) = 1 - e^(-xG_individual_per90 × minutes/90)
- **Limitation:** need to know who will play (confirmed lineup ~1h before)

## How Professionals Do It

### Starlizard (Tony Bloom — £600M/year)

- "Treats gambling like a hedge fund treats stocks"
- Teams of analysts, programmers, and mathematicians
- Statistical models to compute sharper odds than the bookmakers
- Focus on **Asian Handicap** (most liquid market, smallest margin)
- Bets close to match day (to include lineup info)
- Thousands of global events processed per second
- **Secret:** proprietary models that are never disclosed

### Pinnacle (sharp bookmaker)

- ~3% margin (vs 8-12% of regular bookmakers)
- Pinnacle's "closing line" = benchmark of true probability
- Uses wisdom of the crowd: odds adjust with betting volume
- Sharp bettors are WELCOME (unlike Bet365, which bans them)

### Academic Model (Corner Kicks — Arxiv 2112.13001)

- Compound Poisson with Geometric-Poisson distribution
- Bayesian implementation with varying shape parameter
- Uses odds from other markets to inform (cross-market information)
- Handles serial correlation between consecutive corners

### Academic Model (Cards — Bivariate ZIP)

- Zero-Inflated Poisson (ZIP) via Frank copula
- Encapsulates: fouls, historical cards, shots on target, corners
- Key variables: FoulF, FoulA, RedCA, YelCA, CornP, ShotT

## Required Data from Sofascore

| Data | Available on Sofascore? | Field |
|------|:---:|---|
| Corners per team per match | Yes | Match stats |
| Yellow cards per team | Yes | Match stats |
| Red cards | Yes | Match stats |
| Fouls per team | Yes | `fouls` in player stats |
| Match referee | Yes | Match metadata |
| Half-time score | Yes | 1st half score |
| Player xG | Yes | `expectedGoals` |
| Confirmed lineup | Yes | ~1h before |

## Prioritization (impact × effort)

| # | Market | Impact | Effort | Data? | Priority |
|---|--------|--------|--------|-------|----------|
| 1 | Asian Handicap | High | Low | Already have | P0 |
| 2 | Correct Score | High | Zero | Already have | P0 |
| 3 | Over/Under 0.5-3.5 | High | Zero | Already have | P0 |
| 4 | BTTS | Medium | Zero | Already have | P0 |
| 5 | Corners O/U | High | Medium | Needs ingestion | P1 |
| 6 | Cards O/U | High | Medium | Needs ingestion | P1 |
| 7 | Half Time | Medium | Medium | Needs ingestion | P1 |
| 8 | Player to Score | High | High | Needs lineup | P2 |

## Implications for Football Moneyball

### P0 — Expose what we already have (zero new model code)

- Correct Score: we already have `score_matrix` → show top 10 scores with probabilities
- Over/Under 0.5-3.5: already computed → expose all in the frontend
- BTTS: already computed → expose
- Asian Handicap: compute by summing scores from the score_matrix:
  ```
  AH -0.5 = P(home win) = sum(score_matrix where home > away)
  AH -1.5 = sum(score_matrix where home > away + 1)
  AH +0.5 = P(away win or draw) = 1 - P(home win)
  ```

### P1 — New models (Poisson for corners/cards)

- Ingest extra data from Sofascore (corners, cards, fouls, referee)
- Poisson model for corners: λ_corners = team_corners_avg × opponent_factor
- ZIP model for cards: λ_cards = (team_fouls_rate + opp_fouls_rate) × referee_rate
- Monte Carlo to simulate Over/Under 8.5, 9.5, 10.5 corners
- Monte Carlo for Over/Under 2.5, 3.5, 4.5 cards

### P2 — Player props

- Needs confirmed lineup (Sofascore ~1h before)
- Individual xG → P(goal) = 1 - e^(-xG/90 × minutes)

## Sources

- [Starlizard — Inside Tony Bloom's Syndicate](https://thedarkroom.co.uk/inside-tony-blooms-secret-betting-syndicate/)
- [Starlizard — Racing Post (£600M/year)](https://www.racingpost.com/news/britain/high-court-case-alleges-tony-blooms-betting-empire-makes-600m-a-year)
- [Corner Kicks Compound Poisson — Arxiv 2112.13001](https://arxiv.org/abs/2112.13001)
- [Corner Prediction — Journal of OR Society 2024](https://www.tandfonline.com/doi/abs/10.1080/01605682.2024.2306170)
- [Yellow Card Betting Guide — StatsHub](https://www.statshub.com/blog/yellow-card-betting)
- [Poisson Distribution for Football — FBetPrediction](https://football-bet-prediction.com/football-predictions/)
- [Bivariate Poisson — SAPUB](http://article.sapub.org/10.5923.j.ajms.20201003.01.html)
- [Asian Handicap — Wikipedia](https://en.wikipedia.org/wiki/Asian_handicap)
- [Pinnacle Asian Handicap](https://bookmakers.net/pinnacle/asian-handicap/)
- [Trefik Poisson Calculator (Goals, Corners, Cards)](https://www.trefik.cz/predict_poisson.aspx)
