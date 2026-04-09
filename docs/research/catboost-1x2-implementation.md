---
tags:
  - research
  - catboost
  - pi-rating
  - odds-features
  - implementation
---

# Research — CatBoost 1x2 + Pi-Rating Implementation Details

> Research date: 2026-04-05
> Trigger: Poisson model always picks home (10/10). CatBoost + Pi-rating reached RPS 0.1925 on benchmark.

## 1. CatBoost Hyperparameters (winning solutions)

Hubáček et al. (IJCAI 2019, Soccer Prediction Challenge):

```python
CatBoostClassifier(
    loss_function='MultiClass',      # softmax → calibrated probs
    iterations=1000,
    depth=6,                         # shallow, sweet spot 4-7
    learning_rate=0.03,              # low LR + many iterations
    l2_leaf_reg=3,                   # light regularization
    early_stopping_rounds=50,
    cat_features=['home_team', 'away_team'],  # native categorical
    class_weights=[1, 1.3, 1],       # upweight draws!
)
```

CatBoost > XGBoost > LightGBM for football (native categorical, ordered boosting). No one-hot needed for teams.

**Calibration**: CatBoost MultiClass is already well calibrated via softmax. **Skip Platt/isotonic** — it typically makes things worse.

## 2. Features (by importance)

| # | Feature | Computation | Importance |
|---|---|---|---|
| 1 | **Market odds** | Pinnacle devigged (power method) | Largest single feature |
| 2 | **Pi-Rating diff** | R_home_i - R_away_j | Replaces Elo |
| 3 | **EMA form** | `form_t = 0.1 * result + 0.9 * form_{t-1}` (~10 matches) | High |
| 4 | **Goal diff EMA** | `gd_t = 0.15 * gd + 0.85 * gd_{t-1}` | High |
| 5 | **Attack/Defense rating** | Iterative: `att_i = avg(scored / def_opp)`, 5 iter | Medium |
| 6 | **xG / xGA** | From existing data | Medium |
| 7 | **H2H last 5** | Win%, avg goals | Medium-low |
| 8 | **Rest days** | Days since last match | Low |
| 9 | **Home win % (season)** | Rolling, min 3 matches | Low |

**Without odds**: features 2-9 carry ~80% of the predictive power.
**With odds**: feature 1 dominates. Risk of model collapse → mitigate with residual modeling.

## 3. Pi-Rating (Constantinou & Fenton 2013)

### Update formula

```python
goal_diff = min(max(home_goals - away_goals, -3), 3)  # cap ±3
expected_diff = R_home[home_team] - R_away[away_team]
error = goal_diff - expected_diff

R_home[home_team] += γ * error
R_away[away_team] -= γ * error
```

### Parameters

- **γ (learning rate)**: 0.035 (original EPL). Brasileirão: 0.04-0.05 (more parity)
- **Initial rating**: 0.0 (all teams)
- **Promoted teams**: receive the average of the relegated teams from the previous season
- **Goal diff cap**: ±3 (a 5-0 counts as 3-0)
- **Convergence**: ~2 seasons of historical data

### Ratings → Probabilities

```python
λ_home = 1.36 * exp(α * rating_diff)
λ_away = 1.07 * exp(-α * rating_diff)
# Then Poisson/Dixon-Coles for P(H,D,A)
```

Or use directly as a feature in CatBoost (better).

### Benchmark

- **Brier 0.2065 on EPL** (2001-2012)
- Outperformed standalone Elo and basic Poisson

## 4. Odds as Features

### Devig (power method)

```python
def devig_power(probs):
    """Power method: solves sum(p_i^k) = 1."""
    from scipy.optimize import brentq
    def f(k): return sum(p**k for p in probs) - 1
    k = brentq(f, 0.5, 2.0)
    fair = [p**k for p in probs]
    return [p/sum(fair) for p in fair]
```

### Sharpness hierarchy

Pinnacle > Betfair Exchange > SBO > bet365 > the rest

### Preventing model collapse

1. **Residual modeling**: target = `outcome - market_prob` (forces edge finding)
2. **Feature ablation**: train with and without odds, measure lift
3. **L2 regularization** in CatBoost
4. **Two-stage**: odds as baseline, model corrects

### Temporal: use **pre-match** odds (24h before), not closing line.

## 5. Temporal Cross-Validation

```python
# Expanding window — the ONLY valid approach
splits = []
for season in [2023, 2024, 2025]:
    train = df[df['season'] < season]
    test = df[df['season'] == season]
    if len(train) >= 200:
        splits.append((train.index, test.index))
```

**Never** use random k-fold (leaks future form into training).

## 6. Expected Performance (our dataset)

| Scenario | RPS | Accuracy |
|---|---|---|
| Always home (baseline) | ~0.26 | ~47% |
| **CatBoost without odds** | 0.210-0.220 | 48-52% |
| **CatBoost with odds** | 0.195-0.205 | 50-54% |
| Bookmaker consensus | ~0.195 | ~53% |
| **Target** | **<0.205** | **>50%** |

## 7. Proposed Architecture

```
[Pi-Rating] ──┐
[EMA form]  ──┤
[xG/xGA]    ──┼──→ CatBoost MultiClass ──→ P(H), P(D), P(A)
[H2H]       ──┤         ↑
[Rest days] ──┤    class_weights=[1, 1.3, 1]
[Odds devig]──┘    temporal CV, no post-hoc calibration
```

Poisson retained for multi-market (corners, cards, correct score, HT/FT).

## Sources

- [Hubáček et al. 2019 — IJCAI](https://arxiv.org/abs/1710.02824)
- [Constantinou & Fenton 2013 — Pi-Rating (JQAS)](https://www.degruyter.com/document/doi/10.1515/jqas-2012-0054/html)
- [BORS — Hvattum & Arley (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
- [CatBoost docs — MultiClass](https://catboost.ai/en/docs/concepts/loss-functions-multiclassification)
- [Power method devig — Pinnacle](https://www.pinnacle.com/betting-resources/en/educational/removing-the-vig-from-betting-odds)
