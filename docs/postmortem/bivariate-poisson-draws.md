---
tags:
  - pitch
  - prediction
  - poisson
  - draws
  - monte-carlo
---

# Pitch — Bivariate Poisson + Diagonal Inflation (Draws Fix)

## Problem

The current model uses independent Poisson with Dixon-Coles τ correction on low scorelines. Analysis of 91 2026 predictions shows:

- **Under-predicted draws**: real draw rate in Brasileirão = 25-27%, model gives max 30% and typically 20-24%
- **Fitted Dixon-Coles ρ = 0.009** (≈ 0) — the τ correction barely does anything
- **Matchday 10 (2026-04-05)**: 1 real draw (Chapecoense 1-1 Vitória), model predicted none as favorite
- **Under-predicted away wins**: model had 4/4 away winners as underdogs

Independent Poisson assumes `P(X,Y) = P(X) × P(Y)` — ignoring tactical correlation (both "close down" the match, parking the bus at 0-0, etc.). Dixon-Coles corrects only the 4 low scorelines (0-0, 0-1, 1-0, 1-1) with a multiplicative factor, but the literature shows that **bivariate Poisson with diagonal inflation** is superior.

**Research**: [[prediction-error-analysis]], [[calibration-methods]]

## Solution

Replace the score sampling engine with **diagonal-inflated bivariate Poisson** (Karlis & Ntzoufras, 2003):

```
X = X₁ + X₃
Y = X₂ + X₃

X₁ ~ Poisson(λ₁)   # "pure" home goals
X₂ ~ Poisson(λ₂)   # "pure" away goals
X₃ ~ Poisson(λ₃)   # shared component (diagonal inflation)
```

- `λ₃ > 0` naturally inflates P(X=Y) (draws) without distorting the rest
- Reduces to `λ₁ = λ_home - λ₃` and `λ₂ = λ_away - λ₃`, keeping expected means intact
- `λ₃ ≈ 0.10-0.15` in the literature for football

**Approach:**
1. New `bivariate_poisson_score_matrix()` in `calibration.py`
2. New `sample_scores_bivariate()` 
3. `simulate_match()` gains flag `method="bivariate"` (default) vs `"dixon-coles"` (legacy)
4. `fit_calibration` fits `λ₃` via MLE together with ρ (or replacing ρ)

## Architecture

### Affected modules

| Module | Change |
|---|---|
| `domain/calibration.py` | +`bivariate_poisson_score_matrix()`, +`sample_scores_bivariate()`, +`fit_lambda3()` |
| `domain/match_predictor.py` | `simulate_match()` gains `method` param, dispatches to bivariate or DC |
| `use_cases/fit_calibration.py` | Fits λ₃ via MLE on the leak-proof dataset |
| `use_cases/predict_all.py` | Passes `method` from calibration.pkl |
| `cli.py` | No changes (transparent) |

### Schema

No schema change. `calibration.pkl` gains `lambda3: float` field.

### Infra (K8s)

No changes.

## Scope

### In Scope

- [x] `bivariate_poisson_score_matrix(λ₁, λ₂, λ₃, max_goals)` — joint PMF
- [x] `sample_scores_bivariate()` — sampling via flat PMF (like current DC)
- [x] `fit_lambda3()` — MLE over historical (λ_home, λ_away, goals_home, goals_away)
- [x] `simulate_match()` dispatch by method
- [x] Unit tests for all functions
- [x] Keep Dixon-Coles as fallback (`method="dixon-coles"`)
- [x] Auto-select in fit_calibration (bivariate vs DC, by val Brier)

### Out of Scope

- Copula models (Frank, Gaussian) — unnecessary complexity
- Bivariate Poisson with full covariance matrix (only diagonal inflation)
- Feature changes in the model — this affects only the sampling engine

## Research Needed

- [x] Karlis & Ntzoufras (2003) — original paper, implementation → [[prediction-error-analysis]]
- [x] Pinnacle draw inflation article → confirmed that bookmakers adjust draws
- [x] Brasileirão draw rate 25-27% → validated on internal data
- [ ] Benchmark: typical λ₃ for South American football (literature cites 0.10-0.15 for Europe)

## Testing Strategy

- **Unit:**
  - `bivariate_poisson_score_matrix`: sum = 1, draw prob > independent Poisson
  - `sample_scores_bivariate`: shape, mean ≈ λ, seed reproducibility
  - `fit_lambda3`: recovers λ₃ from synthetic data
  - `simulate_match(method="bivariate")`: draw_prob > simulate_match(method="dixon-coles") with same λ
- **Integration:**
  - Re-fit calibration with bivariate: Brier val < DC val
  - Retro backtest: draw accuracy increases
- **Manual:**
  - Compare Bahia-Palmeiras with bivariate vs DC: draw prob should rise from 30% → 33-35%

## Success Criteria

- [ ] `bivariate_poisson_score_matrix` passes tests (sum=1, draw inflation)
- [ ] `fit_lambda3` recovers λ₃=0.12 from synthetic data (±0.03)
- [ ] Average draw prob rises 2-4pp vs Dixon-Coles on 91 predictions
- [ ] Brier val with bivariate ≤ Brier val with DC (auto-select confirms)
- [ ] 0 regression in 1x2 accuracy (draws don't steal correct picks)
- [ ] λ₃ fitted on Brasileirão falls between 0.05 and 0.20 (sanity check)
