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

## Problema

O CatBoost 1x2 de v1.14.0 usa hiperparâmetros fixos escolhidos por heurística:

```python
# football_moneyball/domain/catboost_predictor.py:466
iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
draw_weight=1.3, early_stopping_rounds=50
```

E tem **43 features** ([CATBOOST_FEATURE_NAMES](football_moneyball/domain/catboost_predictor.py#L22)) acumuladas em v1.14→v1.15:

1. Pi-Rating (3)
2. Form EMA (4)
3. xG rolling (4)
4. Rest days (2)
5. Market proxy (3)
6. Match stats rolling (12) — v1.14.1
7. xG form EMA (4) — v1.15.0
8. Coach profile (6) — v1.15.0
9. Standings (5) — v1.15.0

Várias são provavelmente redundantes (duas formas EMA, dois xG rollings em janelas diferentes, standings × points_last_5 × form_ema medindo forma), sem terem passado por ablation.

Resultado: modelo possivelmente sub-tunado E com overfit latente por feature bloat.

## Solução

**Fase A — Hyperopt via Optuna**

TPE sampler com 50 trials, temporal CV 3-fold expanding, objetivo RPS. Search space nos 5 parâmetros de maior impacto:

```python
search_space = {
    "learning_rate": log_uniform(0.01, 0.3),
    "depth": int(4, 10),
    "l2_leaf_reg": log_uniform(1, 10),
    "bagging_temperature": uniform(0, 1),
    "draw_weight": uniform(0.8, 2.0),
}
```

Com `MedianPruner(n_startup_trials=10, n_warmup_steps=200)` pra matar trials ruins cedo. Best params salvos em `/data/models/catboost_best_params.json`.

**Fase B — SHAP pruning**

1. Computar SHAP values do modelo best-tuned
2. Ranquear features por `mean(|SHAP|)` (média através das 3 classes)
3. Identificar candidatos a remoção: features com importância < 0.5% do total
4. Complementar com correlação: se `|corr(f_i, f_j)| > 0.85`, remover o de menor SHAP
5. **Ablation**: remover candidatos em blocos de 5, re-treinar com best params, medir ΔRPS
6. Aceitar remoção se ΔRPS ≤ +1% (modelo mantém ou melhora)
7. Parar quando próximo bloco piora > 1%

Feature set final salvo em `CATBOOST_FEATURE_NAMES_V2` ao lado do original pra backward compat.

## Arquitetura

### Novos módulos

| Módulo | Responsabilidade |
|---|---|
| `domain/feature_pruning.py` | `rank_features_by_shap(model, X) → list[tuple[name, importance]]`, `find_redundant_pairs(X, names, threshold) → list[tuple]`, `ablation_step(model_fn, X, y, features_to_drop) → ΔRPS`. Pure numpy/pandas. |
| `use_cases/tune_catboost.py` | Orquestra Optuna study: carrega dataset via repo, define objective com temporal CV, salva best_params.json |
| `use_cases/prune_catboost_features.py` | Carrega best model, roda SHAP, ablation loop, salva feature set V2 + model re-treinado |

### Módulos modificados

| Módulo | Mudança |
|---|---|
| `domain/catboost_predictor.py` | `train_catboost_1x2` aceita dict `params` em vez de kwargs individuais; adicionar `CATBOOST_FEATURE_NAMES_V2` |
| `domain/catboost_predictor.py` | Nova função `temporal_cv_rps(X, y, params, n_folds=3) → float` (expanding window) |
| `use_cases/train_catboost.py` | Refactor pra usar params dict + reutilizar `temporal_cv_rps` |
| `cli.py` | `tune-catboost [--trials 50]`, `prune-features [--ablation-tolerance 0.01]` |

### Schema

Nenhuma tabela nova. Artefatos em filesystem:

```
/data/models/
├── catboost_1x2.cbm              # best model atual (produção)
├── catboost_1x2_v2.cbm           # novo modelo após hyperopt + pruning
├── catboost_best_params.json     # params vindos do Optuna
├── catboost_feature_importance.json   # SHAP + correlação report
└── history/
    └── catboost_1x2_<timestamp>.cbm   # versões antigas
```

### Dependências novas

- `optuna >= 3.5`
- `shap >= 0.44`

Adicionar ao `pyproject.toml`.

## Scope

### Dentro do Escopo

- [ ] `tune-catboost` CLI: 50 trials Optuna, temporal CV 3-fold, salva best_params.json
- [ ] Temporal CV helper em `domain/catboost_predictor.py` (3-fold expanding)
- [ ] `prune-features` CLI: SHAP ranking + correlação + ablation loop
- [ ] `CATBOOST_FEATURE_NAMES_V2` no code (não deletar V1)
- [ ] Modelo v2 salvo em paralelo, sem auto-promoção
- [ ] Report texto em `/data/models/catboost_feature_importance.json`: ranking SHAP, pares correlacionados, ablation log
- [ ] Testes: `temporal_cv_rps` splits corretos, `rank_features_by_shap` ordenação, `find_redundant_pairs` threshold
- [ ] CLI integration test: `tune-catboost --trials 3` smoke test (3 trials só)

### Fora do Escopo

- Auto-promoção do v2 para produção (requer A/B ou backtest explícito — outra iteração)
- Tuning de `border_count`, `random_strength`, `min_data_in_leaf` (só os 5 core)
- Hyperopt do Pi-Rating γ (outro módulo)
- SHAP interactions (só main effects)
- Dashboard frontend com SHAP plots
- Integração com v1.16 (monitored calibration) — ortogonal

## Research

Ver [[../research/catboost-hyperopt-shap|catboost-hyperopt-shap]]:

- **Optuna TPE** é o sampler padrão pra tree-based HPO
- **5 params core** cobrem ~90% do ganho — não tunar 10+ de uma vez
- **RPS** como objetivo (nunca accuracy)
- **Temporal CV expanding**, não K-Fold (dados temporais)
- **MedianPruner** com warmup de 10 trials economiza 30-50% do tempo
- **SHAP + ablation** > threshold direto no feature importance (built-in e SHAP discordam em ~20% dos rankings)
- **Correlação |r| > 0.85** como filtro prévio pra redundância
- **Ganho esperado**: 1.5-4% RPS — aceitar mesmo se modesto

## Testing

### Unit (domain puro)

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
def test_tune_catboost_smoke_3_trials()  # verifica que roda sem crash
def test_tune_catboost_saves_best_params_json()
def test_tune_catboost_best_params_in_search_space()

# tests/test_prune_features_usecase.py
def test_prune_features_saves_v2_model()
def test_prune_features_respects_ablation_tolerance()
```

### Manual

- `moneyball tune-catboost --trials 50` no dataset completo (1610+ matches) → checar best RPS < baseline
- `moneyball prune-features --ablation-tolerance 0.01` → checar que feature set V2 mantém ou melhora RPS
- Inspecionar `/data/models/catboost_feature_importance.json`: ver se Pi-Rating e market_probs estão no top-5 (esperado)

## Success Criteria

- [ ] `tune-catboost --trials 50` completa em < 6h no dataset atual
- [ ] Best RPS em temporal CV 3-fold ≤ baseline × 0.98 (≥ 2% de melhora) **OU** decisão documentada de aceitar baseline se ganho < 2%
- [ ] `prune-features` produz V2 com ≤ 35 features (pruning de ≥ 8) mantendo RPS dentro de +1% do best tuned
- [ ] SHAP report mostra Pi-Rating, market probs, xG rolling no top-10 (sanity check)
- [ ] Testes novos passando, zero regressão em testes existentes
- [ ] Feature names V1 preservados no código (rollback trivial)
- [ ] pyproject.toml atualizado com `optuna`, `shap`
- [ ] README do pitch documenta como rodar e como decidir promover V2 → produção

## Próximos pitches ligados

- v1.16.0 — Calibração monitorada (ortogonal: monitora probs calibradas, não tuning do base)
- v1.18.0 (futuro) — A/B backtesting framework pra promover V2 → produção com confiança
- v1.19.0 (futuro) — Expandir HPO pro Pi-Rating γ e EMA α (atualmente hardcoded)
