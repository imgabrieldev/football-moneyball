---
tags:
  - research
  - prediction
  - features
  - volatility
  - brasileirao
---

# Research — Features para Times Volateis e State-of-the-Art

> Research date: 2026-04-09
> Context: Modelo v1.14.2 tem 40.7% accuracy 1X2 (Brier 0.2358). Mercado faz ~0.20. Times como Corinthians (20%), Palmeiras (11%), Mirassol (0%) sao imprevisiveis. Precisa fechar o gap de ~15% no Brier.

## 1. Estado da Arte — O que Funciona

### 1.1 Modelos e Benchmarks

| Modelo | RPS | Accuracy | Fonte |
|--------|-----|----------|-------|
| CatBoost + Pi-Ratings | **0.1925** | **55.8%** | Razali et al. |
| XGBoost + Pi-Ratings | 0.2063 | 52.4% | Hubacek et al. (2017 Challenge) |
| XGBoost + Berrar Ratings | 0.2054 | 51.9% | Berrar et al. |
| Hybrid Bayesian Network | 0.2083 | — | Constantinou |
| xG Poisson (best config) | Brier 58.6 | — | beatthebookie |
| Goals Poisson (best config) | Brier 59.7 | — | beatthebookie |
| Bet365 closing odds | Brier 57.2 | — | beatthebookie |
| **Nosso modelo v1.14.2** | **Brier ~0.236** | **40.7%** | — |

**Conclusao**: CatBoost + Pi-Ratings e o state-of-the-art (RPS 0.1925). Ja temos CatBoost + Pi-Rating, mas algo ta errado — provavelmente features insuficientes.

### 1.2 Features que Mais Importam (ranked)

Baseado em XGBoost feature importance de estudo com 269 times, 7 ligas europeias, 20 temporadas:

1. **ELO/Pi-Rating difference** — contribuicao ~4-8x maior que features contextuais
2. **Recent domestic performance** (form rolling)
3. **Travel distance** (>500 miles = -15% away win rate)
4. **Rest days between matches** (<3 dias pos viagem = vulneravel)
5. **Manager tenure** — contribuicao menor mas significativa
6. **Player rotations** — indicador indireto de fixture congestion

### 1.3 O Paradoxo Brier vs Profit

Descoberta critica do beatthebookie:
- O modelo com **pior Brier score** gerou o **maior lucro** nas apostas
- Minimizar erro de predicao != maximizar profit
- Modelos de aposta devem otimizar **gap detection** (edge vs mercado), nao calibracao pura
- Implicacao: nosso foco em Brier pode ser secundario ao foco em edge real vs Betfair

## 2. Features que Faltam no Nosso Modelo

### 2.1 Rolling Form com EMA (Exponential Moving Average)

**Status atual**: temos form EMA basico no CatBoost.

**O que falta**:
- **xG-based form** (nao so gols): rolling xG For - xG Against nos ultimos 5-10 jogos
- **Venue-specific form**: form separado mandante/visitante (ja temos parcial)
- **Window testing**: testar janelas de 5, 10, 15, 20, 35 jogos
- **Key insight**: a melhoria maior vem de usar **xG ao inves de gols**, nao do tipo de media (EMA vs SMA)

Beat the bookie testou 14 combinacoes de window x weighting:
- xG-based models dominaram (8 dos 10 melhores)
- EMA teve melhoria marginal vs SMA
- **A feature em si (xG) importa mais que o smoothing**

### 2.2 Coach Profile — Perfil Completo do Tecnico

**Contexto Brasileirao**: 22 tecnicos demitidos em 38 rodadas (2025). E o fator de volatilidade #1.
O tecnico nao e so um flag binario — e um perfil multidimensional que impacta tatica, resultado e estilo.

#### 2.2.1 Performance Rating do Tecnico

Baseado no framework da Analytics FC (2021):

**Ratings duais (ELO-like)**:
- **Results Rating**: W/D/L ajustado por expectativa pre-jogo (time forte ganha de fraco = pouco ganho)
- **Performance Rating**: baseado em xT (Expected Threat) — recompensa tecnicos cujos times criam mais perigo independente do placar

**Features derivadas**:
- `coach_win_rate_career`: % vitorias na carreira
- `coach_win_rate_current_team`: % vitorias neste time
- `coach_win_rate_last_10`: forma recente do tecnico (EMA ultimos 10 jogos)
- `coach_avg_xg_for`: media de xG produzido por seus times
- `coach_avg_xg_against`: media de xG sofrido
- `coach_tenure_days`: tempo no cargo atual
- `coach_tenure_bucket`: <30d (lua de mel), 30-90d (adaptacao), 90-180d (consolidado), >180d (estabelecido)
- `coach_changed_30d`: flag binaria — troca recente
- `coach_teams_count`: quantos times ja treinou (experiencia)

#### 2.2.2 Perfil Tatico do Tecnico (8 metricas — Analytics FC)

Cada tecnico pode ser perfilado em 8 dimensoes taticas:

| Metrica | O que Mede | Como Calcular |
|---------|-----------|---------------|
| **Long Balls** | Jogo direto no terco defensivo | % passes longos no terco defensivo |
| **Deep Circulation** | Saida de bola curta vs direta | ratio passes curtos/longos no terco defensivo |
| **Wing Play** | Progressao pelas pontas | % acoes ofensivas nas faixas laterais |
| **Territory** | Dominio territorial (field tilt) | % acoes no terco ofensivo |
| **Crossing** | Entrada na area por cruzamento | % cruzamentos vs outras entradas |
| **High Press** | Intensidade de pressao alta | PPDA (passes por acao defensiva) — ja temos |
| **Counters** | Contra-ataques | transicoes rapidas pos recuperacao |
| **Low Block** | Bloco baixo defensivo | % acoes defensivas no terco defensivo |

**Similarity via Kolmogorov-Smirnov Test**: compara distribuicoes completas (nao so medias) entre tecnicos. Permite medir "compatibilidade" tecnico-time.

#### 2.2.3 Compatibilidade Tecnico-Time

Feature chave: **quao diferente e o perfil do tecnico vs o que o time jogava antes?**

- `style_distance`: distancia euclidiana entre perfil tatico do tecnico e estilo historico do time
- `similar_teams_coached`: % de times anteriores do tecnico com perfil similar ao time atual
- `league_experience`: ja treinou nesta liga/divisao? (Brasileirao e muito diferente de Serie B)

**Hipotese**: times com alta `style_distance` (tecnico novo com estilo muito diferente do anterior) devem ter mais variancia nos resultados iniciais — o modelo pode ajustar a confianca nesses jogos.

#### 2.2.4 Regime Detection

- **Efeito lua de mel**: times geralmente melhoram nos primeiros 5-7 jogos pos-troca
- **Performance reset**: quando tecnico troca, reduzir peso do historico recente (decay factor mais agressivo)
- **Dados**: `ingest-context` ja traz managers do Sofascore. Falta enriquecer com historico do tecnico.

#### 2.2.5 Fonte de Dados para Coach Profile

- **Sofascore API**: ja usamos, tem dados de manager por time
- **Transfermarkt**: historico completo de carreira, times treinados, datas
- **Sofascore match-level**: xG, posse, PPDA por partida = permite recalcular perfil tatico
- **Nosso DB**: match_stats ja tem xG, posse, chutes — podemos derivar perfil tatico do time sob cada tecnico

### 2.3 Fixture Congestion & Rest Days

**Features**:
- `rest_days_home` / `rest_days_away`: dias desde ultimo jogo
- `games_last_7d` / `games_last_14d`: ja temos no context
- `travel_distance_km`: distancia entre cidades (BR tem voos longos: Porto Alegre->Manaus)
- `cup_match_midweek`: flag se jogou copa no meio da semana

**Impacto**: contribuicao 4-8x menor que ratings, mas relevante para times com calendario apertado (Libertadores + Brasileirao + Copa do Brasil).

### 2.4 Squad Depth & Key Player Absence

**Features**:
- `key_players_missing`: contagem de titulares ausentes (lesao/suspensao)
- `squad_rotation_rate`: % de mudancas no XI vs jogo anterior
- `total_market_value_ratio`: ratio de valor de mercado home/away (Transfermarkt)
- **Ja temos**: `ingest-lineups` + `ingest-context` (lesoes). Falta pipeline pro CatBoost.

### 2.5 Draw-Specific Features

Draws sao o ponto fraco de TODOS os modelos (F1 ~0.30 vs 0.75 pra wins). Features especificas:

- **Style matchup**: times defensivos vs defensivos = mais draws
- **League position proximity**: times proximos na tabela empatam mais
- **Goal expectation < 2.0**: jogos com xG total baixo empatam mais
- **Derby flag**: classicos tendem a ter mais draws
- **Handicap spread**: se spread < 0.5 gol, draw prob sobe

## 3. Plano de Ataque — Priorizado por Impacto

### Tier 1: Alto Impacto, Dados ja Disponiveis (1-2 dias cada)

1. **xG Rolling Form**: substituir gols por xG no calculo de form strength
   - Ja temos xG nos match_stats. So precisa mudar o calculo de attack/defense strength
   - Esperado: ~2-3% melhoria no Brier (xG models dominam 8/10 top configs)

2. **Coach Profile basico**: tenure + win rate + flag de troca
   - Dados de manager ja no DB via `ingest-context`
   - Features: `coach_tenure_days`, `coach_changed_30d`, `coach_tenure_bucket`
   - `coach_win_rate_current_team` calculavel dos nossos match results
   - Esperado: melhoria forte nos times volateis (Corinthians, Mirassol, etc.)

3. **Rest days + fixture congestion**: dias entre jogos por time
   - `rest_days = commence_time - last_match_time`
   - `games_last_14d` ja temos parcial no context
   - Esperado: melhoria em semanas com midweek + Libertadores

### Tier 2: Medio Impacto, Dados Parciais (3-5 dias)

4. **Coach Tactical Profile (8 metricas)**: perfil tatico derivado dos match_stats
   - PPDA ja temos. Posse, chutes, passes longos — derivar das match_stats
   - Calcular `style_distance` (perfil tecnico vs historico do time)
   - Precisa de pipeline novo mas dados ja estao no DB

5. **Draw-specific features**: league position gap, xG total, style matchup
   - Standings ja temos. Calcular gap de pontos + xG expected total
   - Derby flags, handicap spread derivado das odds
   - Pode melhorar significativamente o F1 de draws (0.30 -> ?)

6. **Key player absence score**: pipeline lineups -> feature de ausencias
   - Lineups ingeridas. Precisa de heuristica de "key player"
   - Impact score = minutos_jogados * (gols+assists) / total_time

### Tier 3: Alto Impacto, Alto Esforco (1+ semana)

7. **Coach historico completo**: carreira, times anteriores, compatibilidade
   - Scraping Transfermarkt ou enriquecimento Sofascore
   - `similar_teams_coached`, `league_experience`, `coach_avg_xg_career`
   - Kolmogorov-Smirnov test pra medir similaridade de estilo

8. **Ensemble meta-learner**: combinar Poisson + CatBoost + Dixon-Coles com stacking
   - Layer 1: cada modelo gera probs independentes
   - Layer 2: meta-learner (logistic regression) combina os outputs
   - Benchmarks mostram 70%+ accuracy com ensembles

9. **Edge-based optimization**: ao inves de minimizar Brier, maximizar edge vs Betfair
   - Custom loss function que penaliza erros onde tinhamos edge
   - Pode melhorar ROI sem melhorar Brier (paradoxo Brier vs profit)

## 4. Implicacoes para Football Moneyball

### O que muda na arquitetura

- `match_predictor.py`: adicionar xG-based form (trocar goals por xG no rolling)
- `train_catboost.py`: adicionar features de context (coach, rest days, standings gap)
- `predict_all.py`: pipeline pra extrair novas features antes de prever
- `domain/features.py` (novo?): modulo de feature engineering centralizado

### O que NAO muda

- Infra (K8s, PG, CronJobs) — tudo funciona
- Market blending — continua importante (65% mercado)
- Calibracao — Dixon-Coles rho + draw floor 26% continuam

### Prioridade sugerida

**Proximo pitch**: implementar Tier 1 (xG form + coach tenure + rest days) como features do CatBoost. Estimativa de melhoria: Brier de 0.236 -> ~0.215-0.220 (gap vs mercado cai de 15% pra ~5-8%).

## Sources

- [Best Football Prediction Algorithms 2026](https://www.golsinyali.com/en/blog/best-football-prediction-algorithms-2026)
- [Scoring functions vs. betting profit](https://beatthebookie.blog/2022/03/29/scoring-functions-vs-betting-profit-measuring-the-performance-of-a-football-betting-model/)
- [Which ML Models Perform Best](https://thexgfootballclub.substack.com/p/which-machine-learning-models-perform)
- [The predictive power of xG](https://beatthebookie.blog/2021/06/07/the-predictive-power-of-xg/)
- [Predicting football results — Dixon-Coles](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
- [Match outcome factors — elite European football (Settembre et al.)](https://journals.sagepub.com/doi/10.3233/JSA-240745)
- [Travel distance impact](https://nerdytips.com/blog/the-hidden-influence-of-travel-distance-on-football-betting-outcomes/)
- [Modelo preditivo Brasileirao (RBFF)](https://www.rbff.com.br/index.php/rbff/article/view/1265)
- [Troca de tecnicos no futebol brasileiro](https://jornalismojunior.com.br/troca-repete-e-recomeca-o-ciclo-infinito-das-trocas-de-tecnicos-no-brasil/)
- [Fixture congestion meta-analysis (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7846542/)
- [CatBoost + Pi-Ratings (Razali et al.)](https://arxiv.org/html/2309.14807)
- [Profiling Coaches with Data — Analytics FC](https://analyticsfc.co.uk/blog/2021/03/22/profiling-coaches-with-data/) — 8 metricas taticas + ELO dual + K-S similarity
- [Predicting Success of Football Coaches — Dartmouth](https://sites.dartmouth.edu/sportsanalytics/2024/01/23/predicting-the-success-of-football-coaches/) — WPA/RSA metrics
- [Coaching Tactical Impact Serie A](https://arxiv.org/pdf/2509.22683) — fixed effects model, home advantage x coaching
- [Tactical Situations and Playing Styles as KPIs (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11130910/)
