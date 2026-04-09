---
tags:
  - pitch
  - catboost
  - pi-rating
  - prediction
  - 1x2
---

# Pitch — v1.14.0: CatBoost 1x2 + Pi-Rating (End-to-End Predictor)

## Problema

Modelo Poisson Monte Carlo **sempre prevê home win** (10/10 picks rodada 10, acc 42.4%, RPS ~0.24). Problema estrutural: Poisson distribui probabilidades via simulação de gols — home é plurality class (47-50%), então argmax SEMPRE é home. Draw nunca é argmax (max ~32%), away raramente.

Calibração (v1.11), draw floor (v1.13), market blending (v1.13) melhoraram Brier em 9% mas **não mudam os picks**. O motor de decisão está fundamentalmente errado.

## Solução

Substituir o motor de 1x2 por **CatBoost MultiClass** treinado end-to-end em resultados históricos. O modelo aprende `P(H), P(D), P(A)` diretamente — não passa por Poisson.

```
[Pi-Rating] ──┐
[EMA form]  ──┤
[xG/xGA]    ──┼──→ CatBoost MultiClass ──→ P(H), P(D), P(A)
[H2H]       ──┤         ↑
[Rest days] ──┤    class_weights=[1, 1.3, 1]
[Odds devig]──┘    temporal CV, no post-hoc calibration

Poisson mantido pra: corners, cards, correct score, HT/FT (multi-market)
```

### Por que CatBoost

- **RPS 0.1925** no Soccer Prediction Challenge (bateu bookmakers 0.2012)
- Native categorical handling (time IDs sem one-hot)
- Ordered boosting (reduz target leakage)
- MultiClass loss → probs via softmax → **já calibrado, sem Platt**
- `class_weights=[1, 1.3, 1]` → draws recebem peso extra

### Pi-Rating (Constantinou & Fenton 2013)

Ratings separados HOME/AWAY por time — resolve home bias na raiz.

```python
error = goal_diff_capped - (R_home[team] - R_away[opp])
R_home[team] += γ * error    # γ = 0.04
R_away[opp]  -= γ * error
```

Brier 0.2065 na EPL, superior a Elo standalone.

## Arquitetura

### Novos módulos

| Módulo | Responsabilidade |
|---|---|
| `domain/pi_rating.py` | PiRating dataclass, update(), compute_all() |
| `domain/catboost_predictor.py` | CatBoostPredictor: train(), predict(), feature engineering |
| `use_cases/train_catboost.py` | Pipeline de treinamento temporal CV |

### Módulos modificados

| Módulo | Mudança |
|---|---|
| `use_cases/predict_all.py` | Usar CatBoost pra 1x2, Poisson pra multi-market |
| `cli.py` | Comando `train-catboost` |
| `adapters/postgres_repository.py` | `get_training_dataset()` (features + labels temporais) |

### Features (13 por time-pair)

```python
CATBOOST_FEATURES = [
    # Pi-Rating (2)
    "pi_rating_diff",       # R_home[home] - R_away[away]
    "pi_rating_home",       # R_home[home] absoluto
    
    # Form EMA (4)
    "home_form_ema",        # α=0.1, últimos ~10 jogos
    "away_form_ema",
    "home_gd_ema",          # goal difference EMA, α=0.15
    "away_gd_ema",
    
    # Strength (4)
    "home_xg_avg",          # xG/90 últimos 5
    "away_xg_avg",
    "home_xga_avg",         # xGA/90 últimos 5
    "away_xga_avg",
    
    # Market (3) — se disponível
    "market_home_prob",     # Pinnacle devigged
    "market_draw_prob",
    "market_away_prob",
]
# + cat_features: home_team, away_team (native CatBoost)
```

### Schema

Nenhuma tabela nova. Pi-ratings computados in-memory por season (como Elo atual). Modelo salvo em `/data/models/catboost_1x2.cbm`.

## Escopo

### Dentro do Escopo

- [x] Pi-Rating: dataclass, update, promoted team handling, compute_all
- [x] CatBoost training: temporal CV (expanding window), class_weights, early stopping
- [x] Feature engineering: pi-rating diff + form EMA + xG + market odds
- [x] Predict pipeline: CatBoost → P(H,D,A), Poisson → multi-market
- [x] CLI: `train-catboost` command
- [x] Testes: Pi-Rating convergência, CatBoost feature shapes, temporal CV splits

### Fora do Escopo

- Residual modeling (target = outcome - market_prob) — v1.15.0
- Feature selection / SHAP analysis — v1.15.0
- Retrain automático via cron — v1.16.0
- CatBoost pra multi-market (corners, cards) — v1.16.0

## Research

- [[catboost-1x2-implementation]] — detalhes de implementação
- [[advanced-prediction-models]] — benchmarks comparativos
- [[prediction-error-analysis]] — diagnóstico do home bias

## Estratégia de Testes

### Unitários (domain puro)
- `test_pi_rating.py`: update formula, cap ±3, promoted team init, convergência
- `test_catboost_predictor.py`: feature shape, predict_proba output, class ordering

### Integração
- Training com 2022-2024, eval em 2025: accuracy > 48%, RPS < 0.22
- Temporal CV: 3 folds expanding window, nenhum leak

### Manual
- Comparar P(H,D,A) do CatBoost vs Poisson vs mercado nos jogos da rodada 10
- Verificar que draw/away picks existem (não 10/10 home)

## Critérios de Sucesso

- [ ] CatBoost treinado em 1610 matches (2022-2026)
- [ ] Temporal CV: **RPS < 0.215** (vs Poisson 0.24)
- [ ] **≥1 draw ou away pick** em rodada típica de 10 jogos
- [ ] Accuracy 1x2 > **48%** (vs atual 42.4%)
- [ ] Poisson mantido e funcional pra multi-market
- [ ] Feature importance: pi-rating no top-3, odds no top-1 se presente
- [ ] Pi-Rating convergido em 2 seasons (ratings estáveis)
- [ ] 0 regressão em multi-market (corners, cards, CS — pipeline Poisson intocado)
