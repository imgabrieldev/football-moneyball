---
tags:
  - research
  - catboost
  - optuna
  - shap
  - hyperparameter-tuning
  - feature-selection
---

# Research — CatBoost Hyperparameter Optimization with Optuna + SHAP Feature Pruning

> Research date: 2026-04-09
> Sources: listed at the end

## Context

The v1.14.0 CatBoost 1x2 ([[../postmortem/catboost-1x2-predictor|catboost-1x2-predictor]]) uses fixed hyperparameters defined in code ([catboost_predictor.py:466](football_moneyball/domain/catboost_predictor.py#L466)):

```python
iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
draw_weight=1.3, early_stopping_rounds=50
```

They were chosen heuristically + small manual search during the original pitch. With the current feature set of **43 features** (v1.15.0 added coach profile + xG form + standings), there are very likely redundancies (multiple form features, two xG rolling features + two xG form EMA features, etc).

Goal: (1) systematic hyperparameter tuning via Optuna TPE; (2) identify low-contribution features via SHAP and prune the feature set.

## Findings

### 1. Optuna as the market standard

Optuna is the preferred framework for HPO in tree-based models. Advantages over grid search and Hyperopt ([Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/), [CatBoost tutorials](https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb)):

- **TPE sampler** (Tree-structured Parzen Estimator): probabilistic model of `P(params | score)`, far superior to random/grid in continuous problems
- **MedianPruner / HyperbandPruner**: kills bad trials early (early stopping of trials, not just epochs) — saves hours
- **Native callbacks + visualization**
- **Conditional search space support**: e.g. `leaf_estimation_iterations` only matters if `leaf_estimation_method='Newton'`

### 2. Recommended search space for CatBoost

Consensus from [CatBoost docs](https://catboost.ai/docs/en/concepts/parameter-tuning), [Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/) and [Kaggle: Saurabh Shahane](https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna):

| Parameter | Range | Prior |
|---|---|---|
| `learning_rate` | [0.01, 0.3] | log-uniform |
| `depth` | [4, 10] | int uniform |
| `l2_leaf_reg` | [1, 10] | log-uniform |
| `bagging_temperature` | [0, 1] | uniform |
| `random_strength` | [0, 10] | uniform |
| `min_data_in_leaf` | [1, 100] | log-uniform |
| `iterations` | 2000 (fixed, with early stopping) | — |
| `border_count` | [32, 255] | int uniform |

Don't tune them all at once — the first 4 (`learning_rate`, `depth`, `l2_leaf_reg`, `bagging_temperature`) cover ~90% of the gain. Start with those.

### 3. Class weights as a hyperparameter

The current `class_weights=[1, 1.3, 1]` was chosen to fight "draw bias" (the model avoided predicting draws). It can be treated as a hyperparameter:

```python
draw_weight = trial.suggest_float("draw_weight", 0.8, 2.0)
class_weights = [1.0, draw_weight, 1.0]
```

Direct effect on RPS and draw accuracy — worth putting in the search space.

### 4. Objective: RPS > Accuracy > LogLoss

**Never** tune for accuracy. The model will collapse to argmax of the majority class.

**Preferred** is RPS (Ranked Probability Score) — penalizes "far-from-truth" errors (missing home for away is worse than missing home for draw). It is the standard metric in soccer prediction and is already implemented in [catboost_predictor.py:529](football_moneyball/domain/catboost_predictor.py#L529).

**Acceptable alternative**: multi-class Brier. Already implemented.

Optuna objective: `minimize(RPS_val)` via 3-fold temporal CV.

### 5. Temporal CV, not K-Fold

Given the temporal nature of the data, standard k-fold leaks information. The correct practice ([Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/)) is:

- **Expanding window time-split**: train[0:T₁] val[T₁:T₂], train[0:T₂] val[T₂:T₃], train[0:T₃] val[T₃:T₄]
- Each fold trains on the cumulative and evaluates on the next block
- Mean RPS across folds = objective

There is already a single-split version in [train_catboost.py](football_moneyball/use_cases/train_catboost.py) — needs to be generalized to 3 folds.

### 6. Trial pruning

`MedianPruner(n_startup_trials=10, n_warmup_steps=200)`:

- Lets the first 10 trials run to completion (calibrates the distribution)
- For subsequent ones, aborts if after 200 iterations the `RPS_intermediate` is worse than the median of previous trials at the same point

Typical gain: 30-50% reduction in total time.

### 7. Trial budget

[Zeupark](https://zeupark.github.io/2025/05/27/optuna-tuning-ml-lesson.html) and others recommend:

- **30-50 trials** is the sweet spot for 4-6 parameters
- **100+ trials** is only worth it if the search space has 10+ params or a very large dataset
- **Fix the sampler seed** (`TPESampler(seed=42)`) for reproducibility

For Moneyball: 50 trials, ~5 min each with pruning → ~4h total, acceptable to run manually or in a K8s pod.

### 8. SHAP for feature selection

Two paths to rank features in CatBoost:

**A. Built-in feature importance** ([CatBoost docs](https://catboost.ai/docs/en/concepts/python-reference_catboost_get_feature_importance)):

- `FeatureImportance`: counts how many times the feature is used in splits, weighted by gain
- Fast (ms)
- Biased toward high-cardinality features

**B. SHAP values** ([SHAP CatBoost tutorial](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html), [CatBoost ShapValues API](https://catboost.ai/docs/en/concepts/shap-values)):

- `model.get_feature_importance(type='ShapValues')` returns shape `(n_samples, n_features + 1)`
- Mean `|SHAP|` per feature is the real importance (impact on predictions)
- For MultiClass, average across classes
- More faithful but more expensive (~10-100× built-in)

**Decision** ([Springer](https://link.springer.com/article/10.1186/s40537-024-00905-w)): with 43 features and 1600 samples, SHAP is feasible and more faithful. CatBoost built-in is OK as a double-check.

### 9. Pruning strategy

Use two passes:

**Pass 1** — tuning without pruning:

1. Run Optuna with 43 features, 50 trials
2. Save the best model

**Pass 2** — SHAP pruning:

1. Compute SHAP of the best model
2. Sort features by `mean(|SHAP|)` desc
3. Threshold: features with `mean(|SHAP|) < 0.5% of the total` → candidates for removal
4. Ablation: remove candidates in blocks of 5, re-train (with best params), measure ΔRPS
5. Accept removals that keep RPS within 1% of the baseline or improve it
6. Stop when removal worsens by > 1%

**Do not remove** without ablation: built-in importance + SHAP disagree on ~20% of rankings.

### 10. Redundant correlation

Complementary to importance: compute correlation matrix of the numerical features. Features with `|corr| > 0.85` are redundant — keep only the one with the highest SHAP. Features expected to be redundant in the current feature set:

- `home_form_ema` vs `home_gd_ema` (both recent form)
- `home_xg_avg` vs `home_xg_form_ema` (recent xG in different windows)
- `home_points_last_5` vs `home_form_ema` (both form in points/win)
- `home_league_position` vs `home_points_last_5` (standings ~ form)

### 11. Expected results

Baseline v1.15.0: RPS ~ 0.200-0.210, 43 features.

Realistic gain from pure HPO: **1-3% RPS** ([Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/), general boosting literature consensus).
Gain from pruning (clean feature set, less overfit, better generalization): **0.5-1% additional RPS**.
Estimated total: baseline RPS → baseline × 0.96-0.98.

If the gain is < 1%, the baseline is already well tuned and the effort is not worth it — accept it.

## Conclusions

1. **Optuna TPE + MedianPruner** is the standard setup, no major choices to make
2. **Search space**: start with 4 core params + `draw_weight`, don't try to tune 10+ at once
3. **Objective: RPS** (or Brier), never accuracy
4. **Temporal CV 3-fold expanding**, not K-Fold
5. **50 trials** is enough
6. **SHAP + ablation** for pruning, not a direct importance threshold
7. **Check redundancy by correlation** as a prior filter
8. **Expected gain**: 1.5-4% RPS total — accept even if modest, it is worth it for the cleaner feature set

## Sources

- [CatBoost tutorials: hyperparameter tuning with Optuna and Hyperopt](https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb)
- [CatBoost Hyperparameter Tuning Guide with Optuna — Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/)
- [CatBoost parameter tuning docs](https://catboost.ai/docs/en/concepts/parameter-tuning)
- [CatBoost ShapValues API](https://catboost.ai/docs/en/concepts/shap-values)
- [CatBoost + SHAP tutorial (shap docs)](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html)
- [Feature selection strategies: a comparative analysis of SHAP-value and importance-based methods — Springer 2024](https://link.springer.com/article/10.1186/s40537-024-00905-w)
- [CatBoost HyperParameter Tuning with Optuna — Kaggle (Saurabh Shahane)](https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna)
- [Hyperparameter Tuning with Optuna: What I Learned — Zeupark](https://zeupark.github.io/2025/05/27/optuna-tuning-ml-lesson.html)
- [[catboost-1x2-implementation]] — research for the base model
- [[prediction-error-analysis]] — context for home bias and RPS target
