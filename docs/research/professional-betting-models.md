---
tags:
  - research
  - betting
  - calibration
  - market-efficiency
  - home-bias
---

# Research — How Professional Bettors Build Football Prediction Models

> Research date: 2026-04-05
> Sources: [listed at bottom]

## Context

Our model predicts home wins 10/10 in the last round. Professional bettors consistently profit, so they must handle draws and away wins better. This research investigates what production-grade betting operations do differently.

## Key Findings

### 1. Closing Line Value (CLV) as Ground Truth

Sharp bettors measure success not by win rate but by **consistently beating Pinnacle's closing line**. Pinnacle's closing odds are the most efficient price in the market because they accept sharp action and don't limit winners. A consistent +2% CLV predicts long-term profitability. **Implication:** Our model should be evaluated against closing odds, not just match outcomes.

### 2. Bookmaker Odds as Features (Not Just Targets)

The BORS paper (Hvattum & Arntzen, PLOS ONE 2018) demonstrated that **betting odds known before a match contain more information than the result known after the match**. They extract ELO-style ratings from devigged odds rather than from results. This is the single most powerful insight: market-implied probabilities should be a primary feature in any model, not just a benchmark.

### 3. The "Beat Bookies With Their Own Numbers" Strategy

Kaunitz et al. (arXiv 2017) showed that **aggregating odds from dozens of bookmakers** and betting where individual bookmaker odds deviate from the consensus creates a profitable edge. Using fractional Kelly (0.5x) with a 10% threshold, they achieved ~80% annual ROI over 11 years in simulation. The bookmakers' response: account limitations. The key: the wisdom of the crowd of bookmakers IS the model.

### 4. Dixon-Coles: Why Our Model Under-Predicts Draws

The rho parameter (typically -0.13) corrects the independent Poisson model's **systematic underestimation of draws** (especially 0-0, 1-1). Without this correction, the model over-predicts decisive results. Our Dixon-Coles implementation already has rho, but if the Platt calibration or ensemble blending overrides it, the draw correction is lost. The home advantage parameter is typically ~0.27 (log-scale), which translates to about 35-40% home win probability in an average match — not 100%.

### 5. Starlizard / Tony Bloom's Approach

~100 analysts collecting granular data: team morale, training reports, beat writer intel, weather. They were using xG models "a decade before anyone knew what it was." Key insight: **proprietary soft data** (injuries, morale, tactical changes) layered on top of statistical models. They primarily bet Asian handicaps, not 1X2, because handicap markets are more liquid and efficient.

### 6. Proper Scoring Rules: Use RPS, Not Accuracy

Brier score is proper but treats home win vs away win the same as home win vs draw. The **Ranked Probability Score (RPS)** is more appropriate for ordered outcomes (H > D > A). A model that always predicts home win gets a terrible RPS because it assigns 0% to draw/away. We should switch our evaluation metric.

### 7. Wisdom of Crowds vs Bookmakers

Aggregated crowd predictions achieve ~52% accuracy (vs ~48% for best individual). But **aggregated bookmaker odds still outperform crowd predictions**. The hierarchy: aggregated bookmaker odds > wisdom of crowds > ELO > individual models > "home team always wins." Our model is currently at the bottom of this hierarchy.

## Diagnosing Our Home Bias Problem

The 10/10 home prediction problem likely stems from:

1. **Class imbalance**: Home wins are ~45% of outcomes in most leagues. Without correction, the model's loss function rewards always predicting the plurality class.
2. **Feature correlation**: Home-team features (attack strength, form) correlate with the home label, creating a feedback loop.
3. **Missing draw mechanism**: Dixon-Coles has rho for this, but if the ensemble or calibration step flattens probabilities, the draw signal is lost.
4. **No market anchor**: Without bookmaker odds as a feature or calibration target, the model has no external reference for "this match is actually close."

## Practical Fixes (Ordered by Impact)

| Fix | Impact | Complexity | Description |
|-----|--------|------------|-------------|
| Add devigged odds as features | Very High | Low | Fetch Pinnacle/Betfair odds, devig, use H/D/A probabilities as 3 input features |
| Calibrate against market | High | Medium | Use devigged closing odds as soft labels (knowledge distillation) |
| Use RPS as loss/eval metric | High | Low | Replace accuracy with RPS in evaluation; consider RPS-based loss |
| Class-weighted training | Medium | Low | Weight draw/away samples higher in loss function |
| Fractional Kelly staking | Medium | Low | Size bets by edge over closing line, not flat stakes |
| Temperature scaling | Medium | Low | Post-hoc calibration to spread probability mass across 3 outcomes |
| Ensemble with market consensus | High | Medium | Blend model probabilities with devigged market odds (e.g., 40% model / 60% market) |

## Implications for Football Moneyball

1. **Immediate**: Add devigged market odds as features in the prediction pipeline. This alone should break the home-bias pattern.
2. **Evaluation**: Switch from accuracy to RPS. Add CLV tracking (compare our predictions to closing lines).
3. **Architecture**: The `market_blend` we already have (v1.10.0) is the right idea, but it should be the PRIMARY signal, not an adjustment. Professional bettors start from the market and adjust, not the other way around.
4. **Betting strategy**: Only bet where our model disagrees with the market by >2% (positive CLV). Use fractional Kelly sizing.

## Sources

- [Pinnacle — What is Closing Line Value](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)
- [BORS — Using Soccer Forecasts to Forecast Soccer (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
- [Kaunitz et al. — Beating the Bookies with Their Own Numbers (arXiv)](https://arxiv.org/abs/1710.02824)
- [Dixon-Coles Implementation — dashee87](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
- [Starlizard — Secrets of Tony Bloom's Operation](https://offthepitch.com/a/secrets-starlizard-how-tony-blooms-football-data-monolith-using-its-knowledge-protect-game)
- [Racing Post — Starlizard Makes GBP 600m/year](https://www.racingpost.com/news/britain/high-court-case-alleges-tony-blooms-betting-empire-makes-600m-a-year-so-what-do-we-know-about-his-starlizard-syndicate-aNlkE7t8daxQ/)
- [Evaluating Probabilistic Forecasts of Football Matches (arXiv)](https://arxiv.org/pdf/1908.08980)
- [LSE — Football Forecasting: Wisdom of Crowds](https://blogs.lse.ac.uk/europpblog/2025/05/29/football-forecasting-harnessing-the-power-of-the-crowd/)
- [Analytics FC — Wisdom of Crowds vs Experts](https://analyticsfc.co.uk/blog/2023/04/19/the-wisdom-of-crowds-can-multiple-voices-outperform-football-experts-and-big-data/)
- [Wilkens 2026 — Can Simple Models Beat the Odds? (Bundesliga)](https://journals.sagepub.com/doi/10.1177/22150218261416681)
- [Goal-line Oracles — Wisdom of Crowds for Football (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312487)
- [Systematic Review of ML in Sports Betting (arXiv)](https://arxiv.org/html/2410.21484v1)
