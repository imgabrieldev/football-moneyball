---
tags:
  - pitch
  - calibration
  - monte-carlo
  - dixon-coles
  - platt
---

# Pitch — Calibration (Dixon-Coles + Platt Scaling)

## Problem

Leak-proof re-prediction of 59 2026 matches exposed two structural flaws:

1. **Zero predicted draws** — across 59 matches, the model produced 0 Draw picks. Independent Poisson structurally underestimates low scorelines 0-0, 1-1, etc.
2. **Overconfident** — picks with ≥60% confidence only hit 48% (near random). The model isn't calibrated.

Overall 1x2 acc: **47.5%** (below the 2024→2026 baseline of 51%). Max-confidence picks bombing (e.g., RB Bragantino 92% vs Botafogo → lost 1x2).

## Solution

Two corrections from the literature, applied in series:

### 1. Dixon-Coles correction (1997)

Apply factor τ(x,y) to the 4 low scorelines Poisson gets wrong:

```
τ(0,0) = 1 - λh·λa·ρ
τ(0,1) = 1 + λh·ρ
τ(1,0) = 1 + λa·ρ
τ(1,1) = 1 - ρ
τ(x,y) = 1 otherwise
```

ρ ∈ [-0.2, 0] tuned to match historical draw rate. Fit via MLE on 2024+2026 data (~470 matches).

### 2. Platt Scaling

Post-processes ensemble probs. Trains 3 binary logistic regressions (Home/Draw/Away one-vs-rest):

```
p_cal = sigmoid(a · logit(p_raw) + b)
```

Parameters (a, b) fitted on 2024 predictions (leak-proof train) vs actual outcomes.

## Architecture

### Affected modules

- `football_moneyball/domain/match_predictor.py` — new `simulate_match_dixon_coles()`
- `football_moneyball/domain/calibration.py` (NEW) — `fit_platt`, `apply_platt`, `fit_dixon_coles_rho`
- `football_moneyball/use_cases/train_ml_models.py` — adds Platt + ρ fit, saves to `/data/models/calibration.pkl`
- `football_moneyball/use_cases/predict_all.py` — applies calibration after ensemble

### Schema

No changes in PostgreSQL. Parameters go in `/data/models/calibration.pkl`:

```python
{
    "dixon_coles_rho": -0.12,
    "platt_home": {"a": ..., "b": ...},
    "platt_draw": {"a": ..., "b": ...},
    "platt_away": {"a": ..., "b": ...},
}
```

## Scope

### In Scope

- [ ] `simulate_match_dixon_coles()` (replaces independent Poisson)
- [ ] `fit_dixon_coles_rho()` — MLE on 2024+2026
- [ ] 3-class Platt scaling (Home/Draw/Away one-vs-rest)
- [ ] `train-models` saves calibration alongside ML models
- [ ] `predict_all` applies calibration
- [ ] Re-backtest to validate gains

### Out of Scope

- Isotonic regression (more data needed)
- Multi-market calibration (over/under, BTTS) — goes in v1.10
- Bayesian hierarchical

## Success Criteria

- [ ] Draw picks: 0 → ≥15% of picks (target: 25-30%)
- [ ] Brier 1x2: 0.21 → <0.19
- [ ] High-confidence 1x2 accuracy (≥60%): 48% → ≥55%
- [ ] Fitted ρ ∈ [-0.25, 0] (sanity check)
