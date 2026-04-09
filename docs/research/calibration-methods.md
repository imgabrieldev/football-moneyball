---
tags:
  - research
  - calibration
  - probability
  - betting
  - brier
---

# Research — Probability Calibration in Football Models

> Research date: 2026-04-05
> Sources: listed at the end

## Context

The current Football Moneyball model has a Brier score of 0.2437 across 78 resolved predictions (v1.6.0 pitch target: <0.19). Reliability diagram analysis shows systematic overconfidence: in the 70%+ bin the model predicts an average of 81.1% but the empirical hit rate is only 38.9% (gap of +42 points).

Platt scaling is already implemented (v1.9.0) but the current parameters (a=0.46, b=0.04) compress insufficiently — 81% becomes only 67%. The research aims to validate which calibration methods are used by professional bookmakers and clubs, and what a realistic Brier score target should be.

## Findings

### 1. Academic benchmarks — what is good?

Bookmaker RPS (Ranked Probability Score) over 250M odds rows:

| League | Typical RPS |
|---|---|
| Eredivisie / Liga Portugal | 0.175–0.185 |
| Premier League | ~0.19 |
| General European range | 0.175–0.224 |

**Bookmakers are well calibrated** — H/D/A curves follow the diagonal, with no systematic bias. Pinnacle keeps an overround of 2.39–3.00% (the lowest in the market).

Realistic target: **Brier 0.20** (bookmaker line). Beating 0.19 would require consistently beating Pinnacle.

### 2. Calibration > Accuracy (for betting)

ScienceDirect paper (Wilkens, 2024) compared model selection criteria:

- Model selected by **calibration**: +34.69% ROI
- Model selected by **accuracy**: −35.17% ROI

Conclusion: for betting, calibration is strictly more important than hit rate. A model that says "55%" and is right 55% of the time is more useful than a model with higher accuracy but distorted probabilities.

### 3. Comparison of calibration methods

| Method | Parameters | Ideal for | Overfit risk |
|---|---|---|---|
| **Platt scaling** | 2 (a, b) | n < 1000, sigmoid distortion | Low |
| **Isotonic regression** | non-parametric | n ≥ 1000, any monotonic distortion | Medium |
| **Temperature scaling** | 1 (T) | NN logits, quick fix | Very low |
| **Beta calibration** | 3 | asymmetric tail skew | Low-medium |

**Professional recommendation (Sports AI)**: hybrid approach — temperature scaling first, then isotonic regression on the residuals. Re-fit when ECE (Expected Calibration Error) > 0.015.

Isotonic is superior to Platt when there is enough data (~1000+ samples). It can correct ANY monotonic distortion, whereas Platt assumes a sigmoid distortion.

### 4. xG model miscalibration — a known pattern

Analysis of Opta and StatsBomb models (Tony ElHabr, 2024):

- **Well calibrated** for xG < 0.25 (~90% of shots)
- **Overestimates** for xG > 0.25

This is exactly the pattern of our 1x2 model: calibrated at moderate probabilities, overconfident at high probabilities. It is not a bug unique to us — it is a characteristic bias of stat-based football models.

StatsBomb XGBoost benchmark: AUC 0.878, Brier 0.069 on held-out shots.

### 5. Market blending — academically validated

PLOS ONE paper (Betting Odds Rating System): bookmaker consensus predicts BETTER than Elo and FIFA ranking. Model+market blending is standard technique:

- Market provides strong calibration (diagonal)
- Model captures inefficiencies the market misses
- Ensemble weights both sources

Our v1.10.0 architecture (market blending post-calibration) aligns with state-of-the-art.

### 6. Dixon-Coles benchmark

Original 1997 paper, >500 citations. EPL 2018-19 (272 matches): DC + player ratings improves probabilities in 53%+ of cases vs. base DC. DC outperforms base Poisson by 15% in accuracy. Our pipeline already implements this (v1.9.0).

### 7. Professionals (Brentford, Liverpool, Pinnacle)

- **Brentford**: Medallion architecture (Bronze→Silver→Gold), Airflow pipelines, focus on xG as a recruitment predictor, ML for injury risk
- **Liverpool**: dedicated department of PhDs, spatio-temporal models with tracking data, tactical + recruitment focus
- **Pinnacle**: sport-specific frameworks, logged errors + back-tested + continuous retraining, sharp bettors as pricing input (feedback loop)

Professionals use calibration as a basic prerequisite — their focus is advanced modeling (tracking, GNN, Bayesian state-space).

## Implications for Football Moneyball

### Revised targets

| Metric | Current | v1.6.0 pitch | Research-based |
|---|---|---|---|
| 1x2 Brier | 0.2437 | <0.19 | **0.20–0.21** (bookmaker line) |
| 1x2 Acc | 42.3% | >54% | **50–54%** (practical limit) |

The original pitch target of <0.19 is more ambitious than real bookmakers. Realistic target: **Brier 0.20**.

### Prioritized actions (research-backed)

**1. Swap Platt → Isotonic regression (high priority)**

- `sklearn.isotonic.IsotonicRegression`
- Validate with Brier CV (k=5)
- n=409 is borderline — do rigorous CV
- If overfit: fall back to temperature scaling

**2. Re-fit calibration with more data (high priority)**

- Add 2022 + 2023 to training (+~760 matches)
- With n>1000, isotonic becomes safe

**3. Hybrid temperature + isotonic (medium priority)**

- Temperature scaling first (1 param, safe)
- Isotonic on the residuals
- Pattern recommended by Sports AI blog

**4. Accept market blending as primary path (validated)**

- Do not invest further in making the base model "perfect" in isolation
- Market is a strong baseline; our value is in identifying edges

**5. Monitor ECE in addition to Brier**

- New fit when ECE > 0.015
- Implement the metric in the backtest

### What NOT to do

- Blind shrinkage (pulling extremes to the mean) — loses signal
- Try to beat <0.19 without validating — may be overfitting the pitch target
- Invest in new contextual features before fixing calibration (v1.6.0 features already proved inert in feature importance)

## Sources

- [How Accurate Are Soccer Odds? — penaltyblog](https://pena.lt/y/2025/07/16/how-accurate-are-soccer-odds/)
- [Machine learning for sports betting: accuracy or calibration? — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266682702400015X)
- [AI Model Calibration for Sports Betting — Sports AI](https://www.sports-ai.dev/blog/ai-model-calibration-brier-score)
- [Opta xG Model Calibration — Tony ElHabr](https://tonyelhabr.rbind.io/posts/opta-xg-model-calibration/)
- [scikit-learn Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Platt Scaling Complete Guide — Train in Data](https://www.blog.trainindata.com/complete-guide-to-platt-scaling/)
- [Temperature Scaling for NN Calibration — Geoff Pleiss](https://geoffpleiss.com/blog/nn_calibration.html)
- [Calibration Introduction Part 2 (Platt/Isotonic/Beta) — Abzu](https://www.abzu.ai/data-science/calibration-introduction-part-2/)
- [Dixon-Coles Model — Grokipedia](https://grokipedia.com/page/DixonColes_model)
- [Betting Odds Rating System — PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
- [Soccer Analytics Review 2025 — Jan Van Haaren](https://janvanhaaren.be/posts/soccer-analytics-review-2025/index.html)
- [How sharp are bookmakers? — DataGolf](https://datagolf.com/how-sharp-are-bookmakers)
- [The Brentford FC story — Sport Performance Analysis](https://www.sportperformanceanalysis.com/article/2018/6/8/the-history-of-brentford-football-analytics)
