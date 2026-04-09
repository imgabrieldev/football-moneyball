---
tags:
  - pitch
  - prediction
  - poisson
  - draws
  - monte-carlo
---

# Pitch — Bivariate Poisson + Diagonal Inflation (Draws Fix)

## Problema

Modelo atual usa Poisson independente com correção Dixon-Coles τ em placares baixos. Análise de 91 predições 2026 mostra:

- **Draws sub-preditos**: taxa real de empates no Brasileirão = 25-27%, modelo dá max 30% e tipicamente 20-24%
- **Dixon-Coles ρ fittado = 0.009** (≈ 0) — a correção τ praticamente não atua
- **Rodada 10 (2026-04-05)**: 1 empate real (Chapecoense 1-1 Vitória), modelo não previu nenhum como favorito
- **Away wins sub-preditos**: modelo deu 4/4 away winners como underdog

Poisson independente assume `P(X,Y) = P(X) × P(Y)` — ignora correlação tática (ambos "fecham" o jogo, parking the bus em 0-0, etc.). Dixon-Coles corrige apenas os 4 placares baixos (0-0, 0-1, 1-0, 1-1) com um fator multiplicativo, mas a literatura mostra que **bivariate Poisson com diagonal inflation** é superior.

**Research**: [[prediction-error-analysis]], [[calibration-methods]]

## Solução

Substituir o motor de score sampling por **bivariate Poisson diagonal-inflated** (Karlis & Ntzoufras, 2003):

```
X = X₁ + X₃
Y = X₂ + X₃

X₁ ~ Poisson(λ₁)   # gols "puros" do mandante
X₂ ~ Poisson(λ₂)   # gols "puros" do visitante
X₃ ~ Poisson(λ₃)   # componente compartilhado (diagonal inflation)
```

- `λ₃ > 0` naturalmente infla P(X=Y) (empates) sem distorcer o resto
- Reduz a `λ₁ = λ_home - λ₃` e `λ₂ = λ_away - λ₃`, mantendo médias esperadas intactas
- `λ₃ ≈ 0.10-0.15` na literatura para futebol

**Abordagem:**
1. Novo `bivariate_poisson_score_matrix()` em `calibration.py`
2. Novo `sample_scores_bivariate()` 
3. `simulate_match()` ganha flag `method="bivariate"` (default) vs `"dixon-coles"` (legacy)
4. `fit_calibration` fitta `λ₃` via MLE junto com ρ (ou substituindo ρ)

## Arquitetura

### Módulos afetados

| Módulo | Mudança |
|---|---|
| `domain/calibration.py` | +`bivariate_poisson_score_matrix()`, +`sample_scores_bivariate()`, +`fit_lambda3()` |
| `domain/match_predictor.py` | `simulate_match()` ganha param `method`, dispatch pra bivariate ou DC |
| `use_cases/fit_calibration.py` | Fittar λ₃ via MLE no dataset leak-proof |
| `use_cases/predict_all.py` | Passar `method` do calibration.pkl |
| `cli.py` | Nenhuma mudança (transparente) |

### Schema

Nenhuma mudança de schema. `calibration.pkl` ganha campo `lambda3: float`.

### Infra (K8s)

Nenhuma mudança.

## Escopo

### Dentro do Escopo

- [x] `bivariate_poisson_score_matrix(λ₁, λ₂, λ₃, max_goals)` — PMF conjunta
- [x] `sample_scores_bivariate()` — amostragem via PMF flat (como DC atual)
- [x] `fit_lambda3()` — MLE sobre (λ_home, λ_away, goals_home, goals_away) histório
- [x] `simulate_match()` dispatch por method
- [x] Testes unitários para todas as funções
- [x] Manter Dixon-Coles como fallback (`method="dixon-coles"`)
- [x] Auto-select no fit_calibration (bivariate vs DC, por Brier val)

### Fora do Escopo

- Copula models (Frank, Gaussian) — complexidade desnecessária
- Bivariate Poisson com covariance matrix full (só diagonal inflation)
- Mudança de features no modelo — isto afeta só o motor de sampling

## Research Necessária

- [x] Karlis & Ntzoufras (2003) — paper original, implementação → [[prediction-error-analysis]]
- [x] Pinnacle draw inflation article → confirmado que bookmakers ajustam draws
- [x] Brasileirão draw rate 25-27% → validado nos dados internos
- [ ] Benchmark: λ₃ típico para futebol sul-americano (literatura cita 0.10-0.15 para Europa)

## Estratégia de Testes

- **Unitários:**
  - `bivariate_poisson_score_matrix`: soma = 1, draw prob > Poisson independente
  - `sample_scores_bivariate`: shape, mean ≈ λ, reprodutibilidade com seed
  - `fit_lambda3`: recupera λ₃ de dados sintéticos
  - `simulate_match(method="bivariate")`: draw_prob > simulate_match(method="dixon-coles") com mesmos λ
- **Integração:**
  - Re-fit calibração com bivariate: Brier val < DC val
  - Retro backtest: draw accuracy sobe
- **Manual:**
  - Comparar Bahia-Palmeiras com bivariate vs DC: draw prob deve subir de 30% → 33-35%

## Critérios de Sucesso

- [ ] `bivariate_poisson_score_matrix` passa testes (soma=1, draw inflation)
- [ ] `fit_lambda3` recupera λ₃=0.12 de dados sintéticos (±0.03)
- [ ] Draw prob média sobe 2-4pp vs Dixon-Coles em 91 predições
- [ ] Brier val com bivariate ≤ Brier val com DC (auto-select confirma)
- [ ] 0 regressão em accuracy 1x2 (empates não roubam acertos corretos)
- [ ] λ₃ fittado no Brasileirão fica entre 0.05 e 0.20 (sanity check)
