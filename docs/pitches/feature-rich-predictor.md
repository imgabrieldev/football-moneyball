---
tags:
  - pitch
  - prediction
  - elo
  - features
  - xgboost
  - ml
---

# Pitch — Feature-Rich Predictor (v1.5.0)

## Problema

v1.3.0 ML treina XGBoost/GBR com apenas **12 features** (xG/xA for/against, corners, cards, league avg, is_home). Research empírico em múltiplos papers SHAP (2024-2025) mostra que **Elo rating e form EMA estão entre as top-3 features** em importance ranking. Nosso modelo ignora isso completamente.

**Baseline atual:**
- Brier score 1X2: **0.2158**
- Accuracy 1X2: **51.1%**
- MAE goals (GBR): **0.902** (próximo do baseline — 87 samples é pouco)

**Gap:** Starlizard/syndicates profissionais ficam em Brier ~0.20, accuracy 55-60%. Para chegar lá, precisamos de features mais ricas. Como a camada ML já existe (v1.3.0), adicionar features é puramente **feature engineering** — sem nova infraestrutura.

Research: [[implementation-details]], [[comprehensive-prediction-models]]

## Solução

Adicionar **~10 novas features** alimentadas pelo mesmo pipeline (Sofascore stats + histórico de resultados). Três categorias:

### 1. Rating System (top-3 em SHAP)
- **Elo rating dinâmico** (FiveThirtyEight-style): cada time começa em 1500, atualizado após cada jogo via K-factor
- **Goal difference EMA** (últimos N jogos, decaimento exponencial)

### 2. xG Overperformance (corrige sorte)
- **xG overperformance per game** = `(goals - xG) / matches` — quanto time tá "acima do xG" (geralmente regride)
- **xG allowed overperformance** = mesmo pra defesa

### 3. Features Sintéticas do Sofascore (já temos dados)
- **creation_index** = `xA/90 + keyPass/90 × 0.05`
- **defensive_intensity** = `(tackles + interceptions + ballRecovery) / 90` por time
- **possession_quality** = `accuratePass_rate × touches_final_third` (proxy de controle)
- **rest_days** = dias desde último jogo (fadiga)

Total: **12 features antigas + 10 novas = 22 features por time**.

Feed tudo no XGBoost/GBR existente. Não cria modelos novos — só alimenta os 3 existentes (goals, corners, cards) com mais sinal.

## Arquitetura

### Módulos afetados

| Módulo | Ação | Descrição |
|--------|------|-----------|
| `domain/elo.py` | NOVO | Cálculo de Elo rating (update após cada resultado) |
| `domain/feature_engineering.py` | MODIFICAR | Estender de 12 → 22 features |
| `domain/ml_lambda.py` | SEM MUDANÇA | Recebe X maior, mesmo código |
| `adapters/postgres_repository.py` | MODIFICAR | Queries pra creation_index, defensive_intensity, rest_days |
| `use_cases/train_ml_models.py` | SEM MUDANÇA | Treina com features novas automaticamente |
| `use_cases/predict_all.py` | MODIFICAR | Passar novas features no `_ml_predict_pair` |

### Schema

**Nenhuma mudança de tabela.** Features são calculadas on-the-fly via queries JOIN em `matches`, `player_match_metrics`, `match_stats`.

**Opcional (otimização futura):** materialized view `team_form` atualizada a cada ingest — evita recalcular features toda vez.

### Infra (K8s)

Sem mudanças. Modelos ML retreinados via `moneyball train-models` (já existe).

## Escopo

### Dentro do Escopo

- [ ] `domain/elo.py`: classe EloRating com `update(home, away, home_goals, away_goals)` + `get_rating(team)`
- [ ] `feature_engineering.py`: função `build_rich_features()` com 22 features
- [ ] Repository: `get_team_advanced_stats(team, last_n)` retornando creation_index, defensive_intensity, etc
- [ ] Repository: `get_rest_days(team, match_date)` dias desde último jogo
- [ ] Repository: `get_elo_ratings(season)` — computa Elo de todos os times até data X (histórico)
- [ ] `predict_all.py`: usar build_rich_features no path ML
- [ ] Tests: `test_domain_elo.py` com cenários conhecidos
- [ ] Tests: atualizar `test_domain_feature_engineering.py` com 22 features
- [ ] Retreinar modelos e comparar Brier antes/depois
- [ ] Frontend: expor Elo rating no card de prediction (opcional)

### Fora do Escopo

- Event-level data scraping (WhoScored) — [[event-data-integration]] (v1.6.0)
- PPDA real via SPADL (research mostra variância alta em match-level)
- Player-level Elo (só time)
- xT real via events — fica pra v1.6.0
- Calibração Platt/isotonic — reservado pra v1.7.0

## Research Necessária

- [x] Elo rating pra futebol — FiveThirtyEight method
- [x] Feature importance SHAP em football models — [[implementation-details]]
- [x] xG overperformance regression — research 2024
- [ ] K-factor ideal para Brasileirão 2026 (testar 20, 30, 40)
- [ ] Janela ideal de EMA (5, 7, 10 jogos)

## Estratégia de Testes

### Unitários (domain — zero mocks)

- `test_domain_elo.py`:
  - 2 times começam em 1500, empate → ambos continuam em 1500
  - Time forte bate fraco → forte sobe pouco, fraco cai pouco
  - Time fraco bate forte → fraco sobe muito, forte cai muito
  - K-factor=20 vs 40 produz magnitudes diferentes
- `test_domain_feature_engineering.py`:
  - build_rich_features retorna 22 features
  - Ordem das features é consistente
  - Fallback defaults quando dados missing

### Integração (com PG)

- Feature engineering com dados reais: 87 matches populam todas as 22 dimensões
- Training pipeline completo: 22 features → GBR → MAE estável
- Elo ratings calculados retroativamente batem com esperado

### Manual

- Comparar Elo ratings com https://eloratings.net (se cobre Brasileirão)
- Inspecionar feature importance do GBR treinado — Elo deveria estar top-3
- Verificar que Brier cai após retreinar

## Critérios de Sucesso

- [ ] 22 features implementadas e testadas
- [ ] Elo rating com validação manual (Flamengo ~1700, Remo ~1350, etc)
- [ ] XGBoost retreinado com MAE_goals < 0.85 (era 0.902)
- [ ] **Brier < 0.200 em backtest** (target principal — era 0.2158)
- [ ] **Accuracy 1X2 > 54%** em backtest (era 51.1%)
- [ ] Feature importance mostra Elo/goal_diff_ema no top-5
- [ ] Testes passando (210+ total, ~15 novos)
- [ ] Domain layer puro (sem deps de infra)
- [ ] Backward compatible: modelos antigos continuam funcionando

## Próximos passos após v1.5.0

Se Brier < 0.200 ✓ → v1.6.0: Event data via WhoScored (PPDA, xT, pass networks)
Se Brier ≥ 0.200 ✗ → debug feature importance, ajustar hyperparams, ou reconsiderar approach
