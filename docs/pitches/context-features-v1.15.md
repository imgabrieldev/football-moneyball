---
tags:
  - pitch
  - catboost
  - features
  - prediction
  - brasileirao
---

# Pitch — v1.15.0: Context Features Pipeline (xG Form + Coach Profile + Rest Days)

## Problema

O modelo v1.14.2 tem **40.7% accuracy 1X2** (Brier 0.2358) contra ~52% do mercado (Brier ~0.20). Times volateis como Corinthians (20%), Palmeiras (11%), Mirassol (0%) puxam a media pra baixo. O CatBoost ja tem 28 features (Pi-Rating, form EMA, xG avg, rest days, match stats rolling), mas falta **contexto situacional** que o mercado precifica e nos nao:

1. **Form baseado em gols, nao xG** — xG-based form domina 8/10 melhores configs em benchmarks
2. **Tecnico como flag binario** — temos `team_coaches` com 49 rows mas nao usamos como feature. No Brasileirao 2025 houve 22 trocas de tecnico em 38 rodadas
3. **Rest days basico** — temos `home_rest_days`/`away_rest_days` mas sem fixture congestion multi-competicao
4. **Zero features de standings** — gap de pontos, posicao relativa, momento na tabela

O gap de ~15% no Brier pro mercado vem principalmente de features contextuais que os apostadores e casas sharp usam mas nosso modelo ignora.

Ref: [[../research/volatile-teams-features|Research: Features para Times Volateis]]

## Solucao

Adicionar **~15 novas features contextuais** ao CatBoost, aproveitando dados **ja disponiveis no banco** (match_stats, team_coaches, league_standings). Sem scraping novo, sem mudanca de infra.

### 3 eixos:

1. **xG Form** — substituir goal-based form por xG-based form (rolling xG For/Against EMA)
2. **Coach Profile** — tenure, win rate, troca recente, bucket de adaptacao
3. **Standings + Congestion** — posicao na tabela, gap de pontos, rest days refinado

## Arquitetura

### Modulos afetados

| Modulo | Mudanca |
|--------|---------|
| `domain/catboost_predictor.py` | Expandir `CATBOOST_FEATURE_NAMES` (+15 features), atualizar `build_training_dataset()` |
| `domain/features.py` (**novo**) | Modulo centralizado de feature engineering: `compute_xg_form()`, `compute_coach_features()`, `compute_standings_features()` |
| `use_cases/train_catboost.py` | Passar dados de coach e standings pro `build_training_dataset()` |
| `use_cases/predict_all.py` | Extrair features de context na hora da predicao |
| `adapters/postgres_repository.py` | Novos queries: `get_coach_for_team()`, `get_standings_at_date()` |

### Novas Features (15)

#### xG Form (4 features)
```
home_xg_form_ema    — EMA de xG For nos ultimos 10 jogos (alpha=0.15)
away_xg_form_ema    — idem away
home_xg_diff_ema    — EMA de (xG For - xG Against) ultimos 10 jogos
away_xg_diff_ema    — idem away
```
Substitui parcialmente `home_xg_avg`/`away_xg_avg` que sao medias simples. O EMA reage mais rapido a mudancas de forma.

#### Coach Profile (6 features)
```
home_coach_tenure_days   — dias desde nomeacao do tecnico home
away_coach_tenure_days   — idem away
home_coach_win_rate      — % vitorias do tecnico neste time
away_coach_win_rate      — idem away
home_coach_changed_30d   — flag: tecnico trocou nos ultimos 30 dias (1/0)
away_coach_changed_30d   — idem away
```
Derivados da tabela `team_coaches` (49 rows ja no DB). Win rate calculado das matches com o tecnico atual.

#### Standings & Congestion (5 features)
```
home_league_position     — posicao na tabela
away_league_position     — idem away
position_gap             — |pos_home - pos_away| (times proximos empatam mais)
home_points_last_5       — pontos nos ultimos 5 jogos (momentum)
away_points_last_5       — idem away
```
Derivados de `league_standings` (20 rows) + calculo retroativo dos matches.

### Schema

**Nenhuma mudanca de schema.** Todos os dados necessarios ja existem nas tabelas:
- `match_stats` (1616 rows) — xG por partida
- `team_coaches` (49 rows) — tecnico por time com datas
- `league_standings` (20 rows) — posicao atual
- `matches` — resultados pra calcular pontos rolling

### Infra (K8s)

**Nenhuma mudanca.** Mesmos CronJobs, mesmo container. So rebuild da imagem apos merge.

## Escopo

### Dentro do Escopo

- [ ] Criar `domain/features.py` com funcoes puras de feature engineering
- [ ] Implementar `compute_xg_form()` — EMA de xG For/Against
- [ ] Implementar `compute_coach_features()` — tenure, win rate, flag troca
- [ ] Implementar `compute_standings_features()` — posicao, gap, momentum
- [ ] Expandir `CATBOOST_FEATURE_NAMES` de 28 pra ~43 features
- [ ] Atualizar `build_training_dataset()` pra incluir novas features (leak-proof)
- [ ] Atualizar `predict_all.py` pra extrair context features na inferencia
- [ ] Adicionar queries de coach e standings no repository
- [ ] Retreinar CatBoost e comparar metricas (RPS, Brier, accuracy)
- [ ] Rodar backtest com novas features vs baseline v1.14.2
- [ ] Testes unitarios pra cada funcao de features.py

### Fora do Escopo

- Perfil tatico do tecnico (8 metricas Analytics FC) — Tier 2, pitch separado
- Historico de carreira completo do tecnico (Transfermarkt) — Tier 3
- Key player absence score — Tier 2
- Ensemble meta-learner — Tier 3
- Edge-based optimization (custom loss) — Tier 3
- Mudancas no Poisson/Dixon-Coles — estes usam pipeline separado
- Draw-specific features (derby flag, style matchup) — Tier 2
- Travel distance — precisa de dataset de coordenadas das cidades

## Research Necessaria

- [x] State-of-the-art features ([[../research/volatile-teams-features|volatile-teams-features]])
- [x] Coach profiling frameworks (Analytics FC, Dartmouth)
- [x] xG vs goals form comparison (beatthebookie)
- [ ] Verificar se `team_coaches.games_coached/wins/draws/losses` estao populados ou NULL
- [ ] Verificar cobertura de `league_standings` por rodada (so tem snapshot atual?)

## Estrategia de Testes

### Unitarios (`tests/test_features.py`)
- `test_compute_xg_form_ema_with_known_values` — verifica EMA com sequencia conhecida
- `test_compute_xg_form_ema_empty_history` — retorna default (league avg)
- `test_compute_coach_features_new_coach` — tenure < 30d, flag=1
- `test_compute_coach_features_no_coach_data` — defaults graceful
- `test_compute_standings_features_with_gap` — posicao e gap corretos
- `test_compute_standings_features_missing` — defaults quando nao tem standings

### Integracao
- Retreinar CatBoost com features novas e comparar RPS/Brier/accuracy vs v1.14.2
- Backtest completo: ROI e hit rate com novas features
- Verificar que `predict-all` produz predicoes validas com features expandidas

### Manual
- `moneyball train-catboost` — treinar e verificar feature importances
- `moneyball predict-all` — prever rodada e comparar com v1.14.2
- `moneyball backtest` — comparar ROI

## Criterios de Sucesso

- [ ] **Brier < 0.220** (melhoria de ~7% vs 0.2358 atual)
- [ ] **1X2 accuracy > 45%** (vs 40.7% atual)
- [ ] **Times volateis > 30%** accuracy (Corinthians, Palmeiras, Mirassol — vs ~15% atual)
- [ ] **RPS < 0.205** (competitivo com benchmarks academicos)
- [ ] Feature importances mostram coach features com contribuicao > 0
- [ ] Zero regressao nos times ja previsiveis (Flamengo, Fluminense)
- [ ] Backtest ROI Betfair nao piora (>= 0%)
- [ ] Testes unitarios passando
- [ ] Nenhuma mudanca de schema ou infra
