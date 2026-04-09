---
tags:
  - research
  - prediction
  - home-advantage
  - draws
  - brasileirao-2026
---

# Research — Prediction Error Analysis (Matchday 10, 2026-04-05)

> Research date: 2026-04-05
> Trigger: 2/7 correct predictions on the matchday (29%), 4 missed away wins, 0 correct draws

## Context

Model v1.10.0 predicted 7 matches on 2026-04-05 with 29% accuracy. Error pattern:

- 4 away teams won (Palmeiras, Inter, Botafogo, Bragantino) — model had all as underdogs
- 1 draw (Chapecoense-Vitória) — model gave Chapecoense 50%
- 2 correct picks (Flamengo, Atlético-MG) — both home favorites that won

## Findings

### 1. Inflated Home Advantage

Academic research shows a post-COVID decline in HA in Brazil:

- **Pre-COVID (2019):** 57.9% home wins
- **COVID (2020):** 44.9%
- **Post-COVID (2022):** 48.6%
- **Internal data 2024:** 47% H / 26% D / 27% A (HA xG = 0.34)
- **Internal data 2025:** 50% H / 26% D / 24% A (HA xG = 0.49)
- **Internal data 2026:** 50% H / 25% D / 25% A (HA xG = 0.51)

**Problem in the code:** `max(home_advantage, 0.0)` was clamping HA ≥ 0, preventing away teams from getting a boost in extreme scenarios. The 0.30 xG fallback when no data was available was reasonable but aggressive.

**Fix applied:** clamp removed, fallback reduced 0.30 → 0.20.

Sources:

- [Nortis Journal - Home Advantage in Brazilian Football](https://nortisjournal.com/index.php/pub/article/view/6)
- [MDPI - Two Years of COVID-19 Pandemic in Serie A](https://www.mdpi.com/1660-4601/19/16/10308)

### 2. Dixon-Coles ρ Had No Practical Effect

Fitted ρ = 0.0089 (≈ 0) via MLE, but:

- Default in `simulate_match()` was already -0.10 (correct)
- The fitted value from calibration.pkl **was never injected** into the prediction pipeline
- Typical in European leagues: ρ = -0.10 to -0.15

Draws underestimated: independent Poisson gives ~20-24%, real is 25-27%.

**Fix applied:** rho from calibration.pkl is now injected in `predict_match()` and `predict_match_player_aware()`.

Sources:

- [dashee87 - Dixon-Coles and Time-Weighting](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
- [Pinnacle - Inflating/Deflating Draw Chance](https://www.pinnacle.com/betting-resources/en/soccer/inflating-or-deflating-the-chance-of-a-draw-in-soccer/cge2jp2sdkv3a9r5)
- [Karlis & Ntzoufras 2003 - Bivariate Poisson](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)

### 3. Brasileirão 2026 Is Atypical

- **2.72 goals/match** (highest since 2007, vs historical 2.4-2.6)
- 65 consecutive matches without a 0-0 (record)
- Volatile promoted teams: Coritiba better away than at home
- Palmeiras dominant (8W-1D-1L), Botafogo and Inter falling

The model does not capture:

- Early-season volatility (few data points per team)
- Promoted teams without a Brasileirão prior (weak Elo initialization)
- Trend of more goals (outdated over/under threshold)

Sources:

- [OneFootball - Brasileirão 2026 highest goals since 2007](https://onefootball.com/en/news/brasileirao-betano-2026-sees-highest-goals-per-game-since-2007-42405408)
- [FootyStats - Brazil Serie A 2026](https://footystats.org/brazil/serie-a)

## Implications for Football Moneyball

### Fixes applied (v1.11.1)

1. **Removed clamp** `max(home_advantage, 0.0)` → allows negative HA
2. **Reduced fallback** 0.30 → 0.20 xG
3. **Calibrated rho injected** into the pipeline (predict_match + predict_match_player_aware)

### Future (upcoming pitches)

4. **Bivariate Poisson** with diagonal inflation (Karlis & Ntzoufras 2003) → structurally corrects draws
5. **Team-specific away strength** → top-6 in budget should have reduced HA
6. **Early-season regularization** → stronger shrinkage during the first 5 rounds
7. **Promoted team priors** → Elo initialization based on Série B performance

## Sources

- [Nortis Journal - Home Advantage Pre/During/Post COVID](https://nortisjournal.com/index.php/pub/article/view/6)
- [MDPI - COVID-19 in Serie A](https://www.mdpi.com/1660-4601/19/16/10308)
- [Benz 2021 - Bivariate Poisson](https://pmc.ncbi.nlm.nih.gov/articles/PMC8313421/)
- [PLOS One - Crowd Effects](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289899)
- [ResearchGate - Travel Distance in Brazil](https://www.researchgate.net/publication/285650377)
- [dashee87 - Dixon-Coles](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
- [Pinnacle - Draw Inflation](https://www.pinnacle.com/betting-resources/en/soccer/inflating-or-deflating-the-chance-of-a-draw-in-soccer/cge2jp2sdkv3a9r5)
- [Karlis & Ntzoufras 2003](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)
- [PMC - Double Poisson Euro 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9119507/)
- [OneFootball - Brasileirão 2026](https://onefootball.com/en/news/brasileirao-betano-2026-sees-highest-goals-per-game-since-2007-42405408)
- [FootyStats - Brazil Serie A](https://footystats.org/brazil/serie-a)
