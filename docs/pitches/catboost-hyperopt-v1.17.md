---
tags:
  - pitch
  - catboost
  - optuna
  - shap
  - hyperparameter-tuning
  - feature-pruning
---

# Pitch — v1.17.0: CatBoost Hyperopt + SHAP Feature Pruning

## Problem

The v1.14.0 CatBoost 1x2 uses fixed hyperparameters chosen heuristically:

```python
# football_moneyball/domain/catboost_predictor.py:466
iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
draw_weight=1.3, early_stopping_rounds=50
```

And it has **43 features** ([CATBOOST_FEATURE_NAMES](football_moneyball/domain/catboost_predictor.py#L22)) accumulated across v1.14→v1.15:

1. Pi-Rating (3)
2. Form EMA (4)
3. xG rolling (4)
4. Rest days (2)
5. Market proxy (3)
6. Match stats rolling (12) — v1.14.1
7. xG form EMA (4) — v1.15.0
8. Coach profile (6) — v1.15.0
9. Standings (5) — v1.15.0

Several are likely redundant (two EMA forms, two xG rollings over different windows, standings × points_last_5 × form_ema all measuring form), without having gone through ablation.

Result: model is possibly under-tuned AND carrying latent overfit from feature bloat.

## Solution

**Phase A — Hyperopt via Optuna**

TPE sampler with 50 trials, 3-fold expanding temporal CV, RPS objective. Search space over the 5 highest-impact parameters:

```python
search_space = {
    "learning_rate": log_uniform(0.01, 0.3),
    "depth": int(4, 10),
    "l2_leaf_reg": log_uniform(1, 10),
    "bagging_temperature": uniform(0, 1),
    "draw_weight": uniform(0.8, 2.0),
}
```

With `MedianPruner(n_startup_trials=10, n_warmup_steps=200)` to kill bad trials early. Best params saved in `/data/models/catboost_best_params.json`.

**Phase B — SHAP pruning**

1. Compute SHAP values for the best-tuned model
2. Rank features by `mean(|SHAP|)` (averaged across the 3 classes)
3. Identify removal candidates: features with importance < 0.5% of the total
4. Complement with correlation: if `|corr(f_i, f_j)| > 0.85`, remove the lower-SHAP one
5. **Ablation**: remove candidates in blocks of 5, retrain with best params, measure ΔRPS
6. Accept removal if ΔRPS ≤ +1% (model holds or improves)
7. Stop when the next block worsens by > 1%

Final feature set saved as `CATBOOST_FEATURE_NAMES_V2` alongside the original for backward compat.

## Architecture

### New modules

| Module | Responsibility |
|---|---|
| `domain/feature_pruning.py` | `rank_features_by_shap(model, X) → list[tuple[name, importance]]`, `find_redundant_pairs(X, names, threshold) → list[tuple]`, `ablation_step(model_fn, X, y, features_to_drop) → ΔRPS`. Pure numpy/pandas. |
| `use_cases/tune_catboost.py` | Orchestrates the Optuna study: loads dataset via repo, defines the objective with temporal CV, saves best_params.json |
| `use_cases/prune_catboost_features.py` | Loads best model, runs SHAP, ablation loop, saves V2 feature set + retrained model |

### Modified modules

| Module | Change |
|---|---|
| `domain/catboost_predictor.py` | `train_catboost_1x2` accepts a `params` dict instead of individual kwargs; add `CATBOOST_FEATURE_NAMES_V2` |
| `domain/catboost_predictor.py` | New function `temporal_cv_rps(X, y, params, n_folds=3) → float` (expanding window) |
| `use_cases/train_catboost.py` | Refactor to use params dict + reuse `temporal_cv_rps` |
| `cli.py` | `tune-catboost [--trials 50]`, `prune-features [--ablation-tolerance 0.01]` |

### Schema

No new tables. Artifacts on the filesystem:

```
/data/models/
├── catboost_1x2.cbm              # current best model (production)
├── catboost_1x2_v2.cbm           # new model after hyperopt + pruning
├── catboost_best_params.json     # params coming from Optuna
├── catboost_feature_importance.json   # SHAP + correlation report
└── history/
    └── catboost_1x2_<timestamp>.cbm   # old versions
```

### New dependencies

- `optuna >= 3.5`
- `shap >= 0.44`

Add to `pyproject.toml`.

## Scope

### In Scope

- [ ] `tune-catboost` CLI: 50 Optuna trials, 3-fold temporal CV, saves best_params.json
- [ ] Temporal CV helper in `domain/catboost_predictor.py` (3-fold expanding)
- [ ] `prune-features` CLI: SHAP ranking + correlation + ablation loop
- [ ] `CATBOOST_FEATURE_NAMES_V2` in the code (do not delete V1)
- [ ] V2 model saved in parallel, with no auto-promotion
- [ ] Text report in `/data/models/catboost_feature_importance.json`: SHAP ranking, correlated pairs, ablation log
- [ ] Tests: `temporal_cv_rps` correct splits, `rank_features_by_shap` ordering, `find_redundant_pairs` threshold
- [ ] CLI integration test: `tune-catboost --trials 3` smoke test (3 trials only)

### Out of Scope

- Auto-promotion of V2 to production (requires explicit A/B or backtest — another iteration)
- Tuning `border_count`, `random_strength`, `min_data_in_leaf` (only the 5 core ones)
- Hyperopt of Pi-Rating γ (other module)
- SHAP interactions (main effects only)
- Frontend dashboard with SHAP plots
- Integration with v1.16 (monitored calibration) — orthogonal

## Research

See [[../research/catboost-hyperopt-shap|catboost-hyperopt-shap]]:

- **Optuna TPE** is the default sampler for tree-based HPO
- **5 core params** cover ~90% of the gain — do not tune 10+ at once
- **RPS** as the objective (never accuracy)
- **Expanding temporal CV**, not K-Fold (temporal data)
- **MedianPruner** with 10-trial warmup saves 30-50% of the time
- **SHAP + ablation** > direct threshold on feature importance (built-in and SHAP disagree on ~20% of rankings)
- **Correlation |r| > 0.85** as a pre-filter for redundancy
- **Expected gain**: 1.5-4% RPS — acceptable even if modest

## Testing

### Unit (pure domain)

```python
# tests/test_feature_pruning.py
def test_rank_features_by_shap_ordering()
def test_rank_features_by_shap_multiclass_averaging()
def test_find_redundant_pairs_high_correlation()
def test_find_redundant_pairs_respects_shap_ranking()
def test_ablation_step_returns_rps_delta()

# tests/test_catboost_temporal_cv.py
def test_temporal_cv_expanding_window_splits()
def test_temporal_cv_no_leak_across_folds()
def test_temporal_cv_rps_shape_matches_n_folds()
```

### Integration

```python
# tests/test_tune_catboost_usecase.py
def test_tune_catboost_smoke_3_trials()  # verifies it runs without crashing
def test_tune_catboost_saves_best_params_json()
def test_tune_catboost_best_params_in_search_space()

# tests/test_prune_features_usecase.py
def test_prune_features_saves_v2_model()
def test_prune_features_respects_ablation_tolerance()
```

### Manual

- `moneyball tune-catboost --trials 50` on the full dataset (1610+ matches) → check best RPS < baseline
- `moneyball prune-features --ablation-tolerance 0.01` → check that V2 feature set holds or improves RPS
- Inspect `/data/models/catboost_feature_importance.json`: check whether Pi-Rating and market_probs are in the top-5 (expected)

## Success Criteria

- [ ] `tune-catboost --trials 50` completes in < 6h on the current dataset
- [ ] Best RPS in 3-fold temporal CV ≤ baseline × 0.98 (≥ 2% improvement) **OR** a documented decision to accept the baseline if the gain is < 2%
- [ ] `prune-features` produces V2 with ≤ 35 features (pruning ≥ 8) keeping RPS within +1% of the best tuned
- [ ] SHAP report shows Pi-Rating, market probs, xG rolling in the top-10 (sanity check)
- [ ] New tests passing, zero regression in existing tests
- [ ] V1 feature names preserved in the code (trivial rollback)
- [ ] pyproject.toml updated with `optuna`, `shap`
- [ ] Pitch README documents how to run it and how to decide on promoting V2 → production

## Related upcoming pitches

- v1.16.0 — Monitored calibration (orthogonal: monitors calibrated probs, not tuning of the base)
- v1.18.0 (future) — A/B backtesting framework to promote V2 → production with confidence
- v1.19.0 (future) — Expand HPO to Pi-Rating γ and EMA α (currently hardcoded)
