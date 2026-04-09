---
tags:
  - research
  - catboost
  - optuna
  - shap
  - hyperparameter-tuning
  - feature-selection
---

# Research — CatBoost Hyperparameter Optimization com Optuna + SHAP Feature Pruning

> Research date: 2026-04-09
> Sources: listadas ao final

## Context

O CatBoost 1x2 de v1.14.0 ([[../postmortem/catboost-1x2-predictor|catboost-1x2-predictor]]) usa hiperparâmetros fixos definidos no código ([catboost_predictor.py:466](football_moneyball/domain/catboost_predictor.py#L466)):

```python
iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
draw_weight=1.3, early_stopping_rounds=50
```

Foram escolhidos por heurística + pequena busca manual durante o pitch original. Com o feature set atual de **43 features** (v1.15.0 adicionou coach profile + xG form + standings), é muito provável que existam redundâncias (múltiplas features de forma, duas de xG rolling + duas de xG form EMA, etc).

Objetivo: (1) tuning sistemático dos hiperparâmetros via Optuna TPE; (2) identificar features de baixa contribuição via SHAP e podar o feature set.

## Findings

### 1. Optuna como padrão de mercado

Optuna é o framework preferido pra HPO em tree-based models. Vantagens sobre grid search e Hyperopt ([Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/), [CatBoost tutorials](https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb)):

- **TPE sampler** (Tree-structured Parzen Estimator): modelo probabilístico de `P(params | score)`, bem superior a random/grid em problemas contínuos
- **MedianPruner / HyperbandPruner**: mata trials ruins cedo (early stopping de trials, não só de épocas) — economiza horas
- **Callbacks + visualization** nativos
- **Suporte a search space condicional**: ex. `leaf_estimation_iterations` só é relevante se `leaf_estimation_method='Newton'`

### 2. Search space recomendado pra CatBoost

Consenso de [CatBoost docs](https://catboost.ai/docs/en/concepts/parameter-tuning), [Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/) e [Kaggle: Saurabh Shahane](https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna):

| Parâmetro | Range | Prior |
|---|---|---|
| `learning_rate` | [0.01, 0.3] | log-uniform |
| `depth` | [4, 10] | int uniform |
| `l2_leaf_reg` | [1, 10] | log-uniform |
| `bagging_temperature` | [0, 1] | uniform |
| `random_strength` | [0, 10] | uniform |
| `min_data_in_leaf` | [1, 100] | log-uniform |
| `iterations` | 2000 (fixo, com early stopping) | — |
| `border_count` | [32, 255] | int uniform |

Não tunar todos de uma vez — os 4 primeiros (`learning_rate`, `depth`, `l2_leaf_reg`, `bagging_temperature`) cobrem ~90% do ganho. Começar por eles.

### 3. Class weights como hiperparâmetro

O `class_weights=[1, 1.3, 1]` atual foi escolhido pra combater "draw bias" (modelo evitava prever empate). Pode ser tratado como hiperparâmetro:

```python
draw_weight = trial.suggest_float("draw_weight", 0.8, 2.0)
class_weights = [1.0, draw_weight, 1.0]
```

Efeito direto no RPS e accuracy de draws — vale pôr no search space.

### 4. Objetivo: RPS > Accuracy > LogLoss

**Nunca** tunar por accuracy. O modelo vai colapsar pra argmax da classe majoritária.

**Preferencial** é RPS (Ranked Probability Score) — penaliza erros "longe" da verdade (errar home por away é pior que errar home por draw). É a métrica padrão em soccer prediction e já está implementada em [catboost_predictor.py:529](football_moneyball/domain/catboost_predictor.py#L529).

**Alternativa aceitável**: Brier multi-class. Já implementado.

Objetivo Optuna: `minimize(RPS_val)` via temporal CV 3-fold.

### 5. Temporal CV, não K-Fold

Dado o caráter temporal dos dados, k-fold padrão vaza informação. A prática correta ([Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/)) é:

- **Expanding window time-split**: train[0:T₁] val[T₁:T₂], train[0:T₂] val[T₂:T₃], train[0:T₃] val[T₃:T₄]
- Cada fold treina no cumulative e avalia no próximo bloco
- Média de RPS dos folds = objective

Já existe em [train_catboost.py](football_moneyball/use_cases/train_catboost.py) uma versão single-split — precisa generalizar pra 3 folds.

### 6. Pruning de trials

`MedianPruner(n_startup_trials=10, n_warmup_steps=200)`:

- Deixa os primeiros 10 trials rodarem até o fim (calibra a distribuição)
- Nos seguintes, aborta se após 200 iterations o `RPS_intermediate` estiver pior que mediana dos trials anteriores no mesmo ponto

Ganho típico: 30-50% de redução em tempo total.

### 7. Orçamento de trials

[Zeupark](https://zeupark.github.io/2025/05/27/optuna-tuning-ml-lesson.html) e outros recomendam:

- **30-50 trials** é o sweet spot pra 4-6 parâmetros
- **100+ trials** só compensa se search space tem 10+ params ou dataset muito grande
- **Fixar seed do sampler** (`TPESampler(seed=42)`) pra reprodutibilidade

Pro Moneyball: 50 trials, ~5 min cada com pruning → ~4h total, aceitável pra rodar manualmente ou em um pod K8s.

### 8. SHAP pra feature selection

Dois caminhos pra ranquear features no CatBoost:

**A. Built-in feature importance** ([CatBoost docs](https://catboost.ai/docs/en/concepts/python-reference_catboost_get_feature_importance)):
- `FeatureImportance`: conta quantas vezes a feature é usada em splits, ponderada por ganho
- Rápido (ms)
- Enviesado pra features de alta cardinalidade

**B. SHAP values** ([SHAP CatBoost tutorial](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html), [CatBoost ShapValues API](https://catboost.ai/docs/en/concepts/shap-values)):
- `model.get_feature_importance(type='ShapValues')` retorna shape `(n_samples, n_features + 1)`
- Média do `|SHAP|` por feature é a importância real (impacto nas predições)
- Pra MultiClass, pega média através das classes
- Mais fiel mas mais caro (~10-100× built-in)

**Decisão** ([Springer](https://link.springer.com/article/10.1186/s40537-024-00905-w)): com 43 features e 1600 samples, SHAP é viável e mais fiel. CatBoost built-in é OK como double-check.

### 9. Estratégia de pruning

Usar duas passes:

**Pass 1** — tuning sem pruning:
1. Rodar Optuna com 43 features, 50 trials
2. Salvar best model

**Pass 2** — SHAP pruning:
1. Computar SHAP do best model
2. Ordenar features por `mean(|SHAP|)` desc
3. Threshold: features com `mean(|SHAP|) < 0.5% do total` → candidatos a remover
4. Ablation: remover candidatos em blocos de 5, re-treinar (com best params), medir ΔRPS
5. Aceitar remoções que mantêm RPS dentro de 1% do baseline ou melhoram
6. Stop quando remoção piora > 1%

**Não remover** sem ablation: built-in importance + SHAP discordam em ~20% dos rankings.

### 10. Correlação redundante

Complementar à importância: calcular matriz de correlação das features numéricas. Features com `|corr| > 0.85` são redundantes — manter só a de maior SHAP. Features esperadas como redundantes no feature set atual:

- `home_form_ema` vs `home_gd_ema` (ambos forma recente)
- `home_xg_avg` vs `home_xg_form_ema` (xG recente em janelas diferentes)
- `home_points_last_5` vs `home_form_ema` (ambos forma em pontos/win)
- `home_league_position` vs `home_points_last_5` (standings ~ forma)

### 11. Resultados esperados

Baseline v1.15.0: RPS ~ 0.200-0.210, 43 features.

Ganho realista de HPO puro: **1-3% RPS** ([Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/), consenso geral de literatura boosting).
Ganho de pruning (feature set limpo, menos overfit, melhor generalização): **0.5-1% RPS adicional**.
Total estimado: RPS baseline → baseline × 0.96-0.98.

Se ganho < 1%, o baseline já está bem tunado e o esforço não compensa — aceitar.

## Conclusões

1. **Optuna TPE + MedianPruner** é o setup padrão, sem grandes escolhas a fazer
2. **Search space**: começar com 4 params core + `draw_weight`, não tentar tunar 10+ de uma vez
3. **Objetivo: RPS** (ou Brier), nunca accuracy
4. **Temporal CV 3-fold expanding**, não K-Fold
5. **50 trials** é suficiente
6. **SHAP + ablation** pra pruning, não threshold direto no importance
7. **Verificar redundância por correlação** como filtro prévio
8. **Ganho esperado**: 1.5-4% RPS total — aceitar mesmo se modesto, vale pelo feature set mais limpo

## Sources

- [CatBoost tutorials: hyperparameter tuning with Optuna and Hyperopt](https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb)
- [CatBoost Hyperparameter Tuning Guide with Optuna — Forecastegy](https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/)
- [CatBoost parameter tuning docs](https://catboost.ai/docs/en/concepts/parameter-tuning)
- [CatBoost ShapValues API](https://catboost.ai/docs/en/concepts/shap-values)
- [CatBoost + SHAP tutorial (shap docs)](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html)
- [Feature selection strategies: a comparative analysis of SHAP-value and importance-based methods — Springer 2024](https://link.springer.com/article/10.1186/s40537-024-00905-w)
- [CatBoost HyperParameter Tuning with Optuna — Kaggle (Saurabh Shahane)](https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna)
- [Hyperparameter Tuning with Optuna: What I Learned — Zeupark](https://zeupark.github.io/2025/05/27/optuna-tuning-ml-lesson.html)
- [[catboost-1x2-implementation]] — research do modelo base
- [[prediction-error-analysis]] — contexto do home bias e RPS target
