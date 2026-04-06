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
> Trigger: Modelo Poisson sempre pick home (10/10). CatBoost + Pi-rating atingiu RPS 0.1925 em benchmark.

## 1. CatBoost Hyperparameters (winning solutions)

Hubáček et al. (IJCAI 2019, Soccer Prediction Challenge):

```python
CatBoostClassifier(
    loss_function='MultiClass',      # softmax → probs calibradas
    iterations=1000,
    depth=6,                         # shallow, sweet spot 4-7
    learning_rate=0.03,              # low LR + many iterations
    l2_leaf_reg=3,                   # regularização leve
    early_stopping_rounds=50,
    cat_features=['home_team', 'away_team'],  # native categorical
    class_weights=[1, 1.3, 1],       # upweight draws!
)
```

CatBoost > XGBoost > LightGBM pra futebol (native categorical, ordered boosting). Não precisa one-hot pra times.

**Calibração**: CatBoost MultiClass já bem calibrado via softmax. **Pular Platt/isotonic** — tipicamente piora.

## 2. Features (por importância)

| # | Feature | Computação | Importância |
|---|---|---|---|
| 1 | **Market odds** | Pinnacle devigged (power method) | Maior feature individual |
| 2 | **Pi-Rating diff** | R_home_i - R_away_j | Substitui Elo |
| 3 | **EMA form** | `form_t = 0.1 * result + 0.9 * form_{t-1}` (~10 matches) | Alta |
| 4 | **Goal diff EMA** | `gd_t = 0.15 * gd + 0.85 * gd_{t-1}` | Alta |
| 5 | **Attack/Defense rating** | Iterativo: `att_i = avg(scored / def_opp)`, 5 iter | Média |
| 6 | **xG / xGA** | Dos dados existentes | Média |
| 7 | **H2H last 5** | Win%, avg goals | Média-baixa |
| 8 | **Rest days** | Dias desde último jogo | Baixa |
| 9 | **Home win % (season)** | Rolling, min 3 matches | Baixa |

**Sem odds**: features 2-9 carregam ~80% do poder preditivo.
**Com odds**: feature 1 domina. Risco de model collapse → mitigar com residual modeling.

## 3. Pi-Rating (Constantinou & Fenton 2013)

### Formula de update

```python
goal_diff = min(max(home_goals - away_goals, -3), 3)  # cap ±3
expected_diff = R_home[home_team] - R_away[away_team]
error = goal_diff - expected_diff

R_home[home_team] += γ * error
R_away[away_team] -= γ * error
```

### Parâmetros
- **γ (learning rate)**: 0.035 (original EPL). Brasileirão: 0.04-0.05 (mais paridade)
- **Rating inicial**: 0.0 (todos os times)
- **Times promovidos**: recebem média dos rebaixados da temporada anterior
- **Goal diff cap**: ±3 (5-0 conta como 3-0)
- **Convergência**: ~2 temporadas de dados históricos

### Ratings → Probabilidades
```python
λ_home = 1.36 * exp(α * rating_diff)
λ_away = 1.07 * exp(-α * rating_diff)
# Depois Poisson/Dixon-Coles pra P(H,D,A)
```

Ou usar diretamente como feature no CatBoost (melhor).

### Benchmark
- **Brier 0.2065 na EPL** (2001-2012)
- Superou Elo standalone e Poisson básico

## 4. Odds como Features

### Devig (power method)
```python
def devig_power(probs):
    """Power method: resolve sum(p_i^k) = 1."""
    from scipy.optimize import brentq
    def f(k): return sum(p**k for p in probs) - 1
    k = brentq(f, 0.5, 2.0)
    fair = [p**k for p in probs]
    return [p/sum(fair) for p in fair]
```

### Hierarquia de sharpness
Pinnacle > Betfair Exchange > SBO > bet365 > resto

### Prevenir model collapse
1. **Residual modeling**: target = `outcome - market_prob` (forçar edge finding)
2. **Feature ablation**: treinar com e sem odds, medir lift
3. **L2 regularization** no CatBoost
4. **Two-stage**: odds como baseline, modelo corrige

### Temporal: usar odds **pré-match** (24h antes), não closing line.

## 5. Cross-Validation Temporal

```python
# Expanding window — ÚNICA abordagem válida
splits = []
for season in [2023, 2024, 2025]:
    train = df[df['season'] < season]
    test = df[df['season'] == season]
    if len(train) >= 200:
        splits.append((train.index, test.index))
```

**Nunca** random k-fold (leak de forma futura pro training).

## 6. Performance Esperada (nosso dataset)

| Cenário | RPS | Accuracy |
|---|---|---|
| Sempre home (baseline) | ~0.26 | ~47% |
| **CatBoost sem odds** | 0.210-0.220 | 48-52% |
| **CatBoost com odds** | 0.195-0.205 | 50-54% |
| Bookmaker consensus | ~0.195 | ~53% |
| **Target** | **<0.205** | **>50%** |

## 7. Arquitetura Proposta

```
[Pi-Rating] ──┐
[EMA form]  ──┤
[xG/xGA]    ──┼──→ CatBoost MultiClass ──→ P(H), P(D), P(A)
[H2H]       ──┤         ↑
[Rest days] ──┤    class_weights=[1, 1.3, 1]
[Odds devig]──┘    temporal CV, no post-hoc calibration
```

Poisson mantido pra multi-market (corners, cards, correct score, HT/FT).

## Sources

- [Hubáček et al. 2019 — IJCAI](https://arxiv.org/abs/1710.02824)
- [Constantinou & Fenton 2013 — Pi-Rating (JQAS)](https://www.degruyter.com/document/doi/10.1515/jqas-2012-0054/html)
- [BORS — Hvattum & Arley (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
- [CatBoost docs — MultiClass](https://catboost.ai/en/docs/concepts/loss-functions-multiclassification)
- [Power method devig — Pinnacle](https://www.pinnacle.com/betting-resources/en/educational/removing-the-vig-from-betting-odds)
