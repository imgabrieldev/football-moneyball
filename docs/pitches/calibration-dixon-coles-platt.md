---
tags:
  - pitch
  - calibration
  - monte-carlo
  - dixon-coles
  - platt
---

# Pitch — Calibração (Dixon-Coles + Platt Scaling)

## Problema

Re-predição leak-proof de 59 jogos 2026 expôs duas falhas estruturais:

1. **Zero empates previstos** — em 59 jogos, modelo deu 0 picks de Draw. Poisson independente estrutural­mente subestima placares 0-0, 1-1, etc.
2. **Overconfident** — picks com ≥60% de confiança acertam só 48% (quase random). Modelo não está calibrado.

Acc 1x2 geral: **47.5%** (abaixo do baseline 2024→2026 de 51%). Max confidence picks bombando (ex: RB Bragantino 92% vs Botafogo → perdeu 1x2).

## Solução

Duas correções da literatura, aplicadas em série:

### 1. Dixon-Coles correction (1997)

Aplica fator τ(x,y) aos 4 placares baixos que Poisson erra:

```
τ(0,0) = 1 - λh·λa·ρ
τ(0,1) = 1 + λh·ρ
τ(1,0) = 1 + λa·ρ
τ(1,1) = 1 - ρ
τ(x,y) = 1 caso contrário
```

ρ ∈ [-0.2, 0] tunado pra bater draw rate histórica. Fit via MLE em 2024+2026 data (~470 jogos).

### 2. Platt Scaling

Pós-processa probs do ensemble. Treina 3 regressões logísticas binárias (Home/Draw/Away one-vs-rest):

```
p_cal = sigmoid(a · logit(p_raw) + b)
```

Parâmetros (a, b) fitados nos 2024 predictions (leak-proof train) vs outcomes reais.

## Arquitetura

### Módulos afetados

- `football_moneyball/domain/match_predictor.py` — novo `simulate_match_dixon_coles()`
- `football_moneyball/domain/calibration.py` (NOVO) — `fit_platt`, `apply_platt`, `fit_dixon_coles_rho`
- `football_moneyball/use_cases/train_ml_models.py` — adiciona fit de Platt + ρ, salva em `/data/models/calibration.pkl`
- `football_moneyball/use_cases/predict_all.py` — aplica calibração após ensemble

### Schema

Nenhuma mudança no PostgreSQL. Parâmetros vão em `/data/models/calibration.pkl`:

```python
{
    "dixon_coles_rho": -0.12,
    "platt_home": {"a": ..., "b": ...},
    "platt_draw": {"a": ..., "b": ...},
    "platt_away": {"a": ..., "b": ...},
}
```

## Escopo

### Dentro do Escopo

- [ ] `simulate_match_dixon_coles()` (substitui Poisson independente)
- [ ] `fit_dixon_coles_rho()` — MLE em 2024+2026
- [ ] Platt scaling 3-class (Home/Draw/Away one-vs-rest)
- [ ] `train-models` salva calibração junto com ML models
- [ ] `predict_all` aplica calibração
- [ ] Re-backtest validar ganhos

### Fora do Escopo

- Isotonic regression (mais dados necessários)
- Calibração de multi-markets (over/under, BTTS) — fica v1.10
- Bayesian hierarchical

## Critérios de Sucesso

- [ ] Draw picks: 0 → ≥15% dos picks (alvo: 25-30%)
- [ ] Brier 1x2: 0.21 → <0.19
- [ ] Accuracy 1x2 high-confidence (≥60%): 48% → ≥55%
- [ ] ρ fitado ∈ [-0.25, 0] (sanity check)
