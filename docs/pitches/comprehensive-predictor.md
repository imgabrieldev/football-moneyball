---
tags:
  - pitch
  - prediction
  - poisson
  - player-aware
  - xgboost
  - compound-poisson
  - zip
  - referee
  - multi-market
---

# Pitch — Comprehensive Predictor (Framework 5-Camadas)

## Problema

Nosso Dixon-Coles simplificado opera a **nível de time** — agrega todos os jogos da temporada em `attack_strength` e `defense_strength`. Isso ignora:

1. **Quem joga** — o time vence ou não vence muito mais por quem tá em campo do que pela média histórica. Neymar lesionado muda λ_gols do Santos em 30%.
2. **Árbitro** — juiz rigoroso infla cartões em 50%+. Estamos cegos pra isso.
3. **Interações não-lineares** — Dixon-Coles é multiplicativo simples (`attack × defense`). Não captura "time com defesa alta pressa contra time que depende de transição".
4. **Multi-mercado** — só prevemos gols. Escanteios, cartões, chutes, HT ficam fora.

Resultado atual: **47% accuracy 1X2, Brier 0.237**. A Starlizard (Tony Bloom, £600M/ano) opera em ~55-60%. O gap vem exatamente dos 4 itens acima.

**Research:** [[comprehensive-prediction-models]], [[implementation-details]]

## Solução

Implementar o framework de **5 camadas** usado pelos syndicates, em 3 fases incrementais — cada uma agregando sem quebrar a anterior:

```
Camada 1: FEATURE ENGINEERING (por jogador dos 22 em campo)
Camada 2: TEAM AGGREGATION (soma/média dos 11)
Camada 3: CONTEXTUAL ADJUSTMENT (casa, árbitro, derby, form, regression)
Camada 4: MONTE CARLO MULTI-DIMENSIONAL (gols + corners + cards + shots + HT)
Camada 5: MARKET DERIVATION (200+ mercados da matriz simulada)
```

### Fase 1 — Player-Aware λ (v1.1.0)

**Objetivo:** trocar `team_attack_strength` por `Σ xG/90 dos 11 titulares`.

```python
def calculate_team_lambda_from_players(
    lineup: list[PlayerStats],     # 11 titulares com EMA últimos 5-10 jogos
    opponent_defense: float,        # fator de oposição
    minutes_weight: list[float],    # prob de cada player jogar 90min
) -> float:
    """λ = Σ (player.xg_per90 × minutes_weight) × opponent_defense."""
    lambda_team = sum(
        p.xg_per90 * w for p, w in zip(lineup, minutes_weight)
    )
    return lambda_team * opponent_defense
```

**Pré-escalação (24h antes):** usa "probable XI" (11 jogadores com mais minutos nos últimos 5 jogos).
**Pós-escalação (~1h antes):** usa lineup confirmada do Sofascore, recalcula tudo.

Backward compatible: mantém `predict_match(all_match_data=...)` antigo; adiciona `predict_match_player_aware(lineup_home=..., lineup_away=...)`.

### Fase 2 — Multi-Output Poisson (v1.2.0)

**Novos λ além de gols:**

```python
λ_gols       = Σ xG/90 dos 11 × opp_defense
λ_corners    = f(cruzamentos dos laterais, chutes bloqueados adversário)
λ_cards      = Σ faltas/90 dos volantes × referee_factor × derby_factor  [ZIP]
λ_shots      = Σ chutes/90 dos 11 × opp_shots_conceded
λ_sot        = λ_shots × avg_on_target_rate
λ_gols_HT    = λ_gols × 0.45
λ_saves_GK   = chutes_adversário × (1 - save_rate_goleiro)
```

**Distribuições específicas:**
- Gols → Poisson (ou Dixon-Coles)
- Escanteios → Poisson simples no MVP, migrar pra Negative Binomial se overdispersion
- Cartões → **Zero-Inflated Poisson** (excesso de 0s)
- Chutes → Poisson
- HT gols → Poisson(λ × 0.45)

**Referee module:**
```python
def referee_strictness(referee_name: str, prior_weight: float = 5.0) -> dict:
    """Empirical Bayes shrinkage."""
    n_matches, cards_rate, fouls_rate = get_ref_stats(referee_name)
    league_cards_rate = get_league_avg_cards()
    
    # Shrinkage: poucos jogos → puxa pra league avg
    adjusted = (n_matches * cards_rate + prior_weight * league_cards_rate) / (n_matches + prior_weight)
    return {"cards_factor": adjusted / league_cards_rate}
```

**Monte Carlo multi-dim:**
```python
def simulate_match_full(lambdas: dict, n_sims: int = 10_000) -> pd.DataFrame:
    """Uma simulação = um jogo completo (gols + corners + cards + ...)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "home_goals": rng.poisson(lambdas["home_xg"], n_sims),
        "away_goals": rng.poisson(lambdas["away_xg"], n_sims),
        "home_corners": rng.poisson(lambdas["home_corners"], n_sims),
        "away_corners": rng.poisson(lambdas["away_corners"], n_sims),
        "total_cards": rng.poisson(lambdas["total_cards"], n_sims),
        "home_shots": rng.poisson(lambdas["home_shots"], n_sims),
        "away_shots": rng.poisson(lambdas["away_shots"], n_sims),
        "ht_home": rng.poisson(lambdas["home_xg"] * 0.45, n_sims),
        "ht_away": rng.poisson(lambdas["away_xg"] * 0.45, n_sims),
    })
```

**Derivação de mercados** (sobre o DataFrame simulado):
- Escanteios O/U 7.5, 8.5, 9.5, 10.5, 11.5
- Cartões O/U 2.5, 3.5, 4.5, 5.5
- Chutes O/U por time
- HT Result, HT/FT, HT Score
- Margem de vitória, primeiro tempo com mais gols
- **~100 mercados adicionais**

### Fase 3 — ML → λ (v1.3.0)

**Objetivo:** XGBoost aprende a prever λ a partir de TODAS as features (team + player + context + referee), capturando interações não-lineares que Dixon-Coles multiplicativo não captura.

```python
# Pipeline
X = build_features(match)  # 40-60 features por partida
y = observed_lambda         # gols marcados, corners, cards, shots

model_goals = XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
)
model_goals.fit(X_train, y_train_goals)

# Em inference
λ_goals_ml = model_goals.predict(X_match)
result = simulate_match_full({"home_xg": λ_goals_ml, ...})
```

**Padrão de integração:** ML prediz diretamente (Padrão A). Se Dixon-Coles + player-aware continuar competitivo, migrar pra Padrão B (ML prediz resíduo do DC).

**Calibração >> Accuracy:** usar Platt scaling ou isotonic regression após XGBoost. Paper 2024 mostra **+69% ROI** quando otimiza calibração vs accuracy.

## Arquitetura

### Módulos afetados

| Módulo | Fase | Ação | Descrição |
|--------|:---:|------|-----------|
| `domain/player_lambda.py` | 1 | NOVO | Agregação jogador → time λ |
| `domain/lineup_prediction.py` | 1 | NOVO | Probable XI + minutes weighting |
| `domain/match_predictor.py` | 1 | MODIFICAR | Adicionar path player-aware |
| `domain/corners_predictor.py` | 2 | NOVO | λ corners + Poisson/NB |
| `domain/cards_predictor.py` | 2 | NOVO | λ cards + ZIP + referee |
| `domain/shots_predictor.py` | 2 | NOVO | λ shots + SoT |
| `domain/referee.py` | 2 | NOVO | Bayesian shrinkage |
| `domain/multi_monte_carlo.py` | 2 | NOVO | Simulação multi-dimensional |
| `domain/markets.py` | 2 | MODIFICAR | Derivar 100+ mercados |
| `domain/ml_lambda.py` | 3 | NOVO | XGBoost pra cada λ |
| `domain/feature_engineering.py` | 3 | NOVO | Build features pro ML |
| `adapters/sofascore_provider.py` | 1,2 | MODIFICAR | Ingerir lineup + referee + match_stats |
| `adapters/orm.py` | 1,2 | MODIFICAR | Novos campos e tabelas |
| `adapters/postgres_repository.py` | 1,2 | MODIFICAR | Queries novas |
| `use_cases/predict_all.py` | 1,2,3 | MODIFICAR | Orquestrar novas camadas |
| `use_cases/ingest_lineups.py` | 1 | NOVO | Detectar e ingerir lineups |
| `use_cases/train_ml_models.py` | 3 | NOVO | Treinar XGBoost models |
| `api.py` | 1,2 | MODIFICAR | Expor novos mercados |
| `frontend/` | 1,2 | MODIFICAR | Tabs, badges (pre/post lineup) |

### Schema (PostgreSQL)

```sql
-- Fase 2: estatísticas match-level
CREATE TABLE match_stats (
    match_id INTEGER PRIMARY KEY,
    home_corners INTEGER, away_corners INTEGER,
    home_yellow INTEGER, away_yellow INTEGER,
    home_red INTEGER, away_red INTEGER,
    home_fouls INTEGER, away_fouls INTEGER,
    home_shots INTEGER, away_shots INTEGER,
    home_sot INTEGER, away_sot INTEGER,
    ht_home_score INTEGER, ht_away_score INTEGER,
    referee_name VARCHAR,
    FOREIGN KEY (match_id) REFERENCES match_info(match_id)
);

-- Fase 2: estatísticas de árbitro (materialized)
CREATE TABLE referee_stats (
    referee_name VARCHAR PRIMARY KEY,
    matches_officiated INTEGER,
    avg_yellow_per_match REAL,
    avg_red_per_match REAL,
    avg_fouls_per_match REAL,
    avg_corners_per_match REAL,
    strictness_factor REAL,
    last_updated VARCHAR
);

-- Fase 1: lineups confirmadas
CREATE TABLE match_lineups (
    match_id INTEGER,
    team VARCHAR,
    side VARCHAR,  -- 'home' | 'away'
    player_id INTEGER,
    player_name VARCHAR,
    position VARCHAR,
    is_starter BOOLEAN,
    jersey_number INTEGER,
    PRIMARY KEY (match_id, player_id)
);

-- Fase 1: adicionar campos em prediction_history
ALTER TABLE prediction_history ADD COLUMN lineup_type VARCHAR DEFAULT 'pre';
ALTER TABLE prediction_history ADD COLUMN model_version VARCHAR;
ALTER TABLE prediction_history ADD COLUMN home_lambda_corners REAL;
ALTER TABLE prediction_history ADD COLUMN away_lambda_corners REAL;
ALTER TABLE prediction_history ADD COLUMN home_lambda_cards REAL;
ALTER TABLE prediction_history ADD COLUMN away_lambda_cards REAL;
ALTER TABLE prediction_history ADD COLUMN home_lambda_shots REAL;
ALTER TABLE prediction_history ADD COLUMN away_lambda_shots REAL;
```

### Infra (K8s)

**CronJobs novos:**
- `ingest-lineups` — a cada 30min, checar jogos dos próximos 2h e puxar lineup quando sair
- `predict-pre-lineup` — 24h antes do jogo (probable XI)
- `predict-post-lineup` — quando lineup sair (lineup confirmada)
- `train-models` (fase 3) — semanal, retreina XGBoost

**ConfigMap:** atualizar `init.sql` com novas tabelas.

## Escopo

### Fase 1 — Player-Aware λ (v1.1.0) — Dentro do Escopo

- [ ] `domain/player_lambda.py`: agregação jogador → time
- [ ] `domain/lineup_prediction.py`: probable XI + minutes weighting
- [ ] `domain/match_predictor.py`: adicionar `predict_match_player_aware()`
- [ ] Tabela `match_lineups` no schema
- [ ] `adapters/sofascore_provider.py`: `get_confirmed_lineup()` + `get_probable_lineup()`
- [ ] `use_cases/ingest_lineups.py`: detectar e salvar lineups
- [ ] `use_cases/predict_all.py`: rotear entre pré e pós-escalação
- [ ] CronJob `ingest-lineups`
- [ ] Frontend: badge "Pré-escalação" / "Escalação confirmada" em cada card
- [ ] Backtest v1.1.0 vs v0.5.0 em jogos já resolvidos

### Fase 2 — Multi-Output Poisson (v1.2.0) — Dentro do Escopo

- [ ] Tabelas `match_stats` e `referee_stats`
- [ ] `adapters/sofascore_provider.py`: ingerir corners/cards/fouls/referee/HT score
- [ ] `domain/referee.py`: Bayesian shrinkage
- [ ] `domain/corners_predictor.py`: λ + Poisson
- [ ] `domain/cards_predictor.py`: λ + ZIP (statsmodels)
- [ ] `domain/shots_predictor.py`: λ + Poisson
- [ ] `domain/multi_monte_carlo.py`: simulação multi-dim
- [ ] `domain/markets.py`: derivar 100+ mercados da matriz
- [ ] `api.py`: expor novos mercados
- [ ] Frontend: tabs "Escanteios", "Cartões", "Chutes", "Intervalo"
- [ ] Backtest de calibração por mercado

### Fase 3 — ML → λ (v1.3.0) — Dentro do Escopo

- [ ] `domain/feature_engineering.py`: build 40-60 features por partida
- [ ] `domain/ml_lambda.py`: XGBoost regressors pra cada métrica
- [ ] `use_cases/train_ml_models.py`: treinar + salvar pickles
- [ ] Time-series CV (não aleatório)
- [ ] Platt scaling / isotonic calibration
- [ ] A/B test: v1.2.0 (analytical) vs v1.3.0 (ML)
- [ ] CronJob `train-models` semanal

### Fora do Escopo

- Correlação entre mercados (Copulas Gaussianas) — complexo demais pro MVP
- Weather data
- In-play/live predictions
- Bayesian Hierarchical completo via PyMC — reservar pra v1.4.0+
- Player props individuais (marcador, assistência) — pitch separado P2
- Network/graph features — já temos network_analysis mas fora do scope desse pitch

## Research Necessária

- [x] Framework 5-camadas dos syndicates — [[comprehensive-prediction-models]]
- [x] Detalhes de implementação — [[implementation-details]]
- [ ] Validar: Sofascore retorna `standings` de árbitros pro Brasileirão?
- [ ] Validar: Sofascore retorna HT score?
- [ ] Calibrar: média real de escanteios no Brasileirão (provável 9-10/jogo)
- [ ] Calibrar: média real de cartões no Brasileirão (provável 4.5/jogo)
- [ ] Testar: ZIP vs Poisson puro pra cartões — qual calibra melhor
- [ ] Testar: variância empírica de corners justifica Negative Binomial?

## Estratégia de Testes

### Unitários (domain — zero mocks, inputs determinísticos)

**Fase 1:**
- `test_player_lambda.py`: soma de 11 xG/90 conhecidos → λ esperado
- `test_lineup_prediction.py`: fixture com histórico → top 11 correto
- `test_minutes_weighting.py`: player com 10/10 jogos → weight 1.0; com 5/10 → 0.5

**Fase 2:**
- `test_referee.py`: juiz com 0 jogos → shrinkage full; 20 jogos → próprio rate
- `test_corners_predictor.py`: λ=5+5 → Over 9.5 ≈ 50%
- `test_cards_predictor.py`: juiz rigoroso → P(cards ≥ 5) > juiz tolerante
- `test_multi_monte_carlo.py`: 10K sims com λ conhecido → distribuição Poisson

**Fase 3:**
- `test_feature_engineering.py`: fixture de partidas → features esperadas
- `test_ml_lambda.py`: modelo treinado em fixture → predição estável
- `test_calibration.py`: Brier score calibrated < uncalibrated

### Integração (com PG)

- Fase 1: ingest lineup real do Sofascore → query → match_lineups populada
- Fase 2: ingest match_stats → referee_stats materialized atualizada
- Backtest temporal: treinar até rodada 15, prever 16-20, medir Brier

### Manual

- Fase 1: comparar λ player-aware com λ team-level em 5 jogos conhecidos
- Fase 2: comparar nossas odds com Betfair em 10 jogos (escanteios, cartões)
- Fase 3: A/B test por 2 rodadas, medir ROI real

## Critérios de Sucesso

### Fase 1 (v1.1.0)
- [ ] Brier score 1X2 cai de 0.237 para < 0.220 (5%+ melhora)
- [ ] Accuracy 1X2 sobe de 47% para 50%+
- [ ] Frontend mostra "Escalação confirmada" quando lineup disponível
- [ ] Pré e pós-lineup salvos separadamente em `prediction_history`

### Fase 2 (v1.2.0)
- [ ] 5 novos mercados no frontend: escanteios, cartões, chutes, HT, margem
- [ ] Calibração corners Over/Under 9.5 dentro de ±3% das casas
- [ ] Calibração cards Over/Under 3.5 dentro de ±3% das casas
- [ ] ZIP pra cards com log-likelihood superior a Poisson puro

### Fase 3 (v1.3.0)
- [ ] XGBoost λ_gols com MAE < 0.35 no test set
- [ ] Brier score cai de < 0.220 para < 0.200 (+ 10% melhora)
- [ ] ROI simulado em backtest > 3% (benchmark Beat the Bookie: 5.3%)
- [ ] Calibração (isotonic) melhora ROI em 20%+ vs uncalibrated

### Criterios Globais
- [ ] Todos testes passando
- [ ] Arquitetura hexagonal preservada (domain sem statsbomb/sqlalchemy)
- [ ] Backward compatible — predictions antigas continuam funcionando
- [ ] Documentação atualizada em `docs/architecture/overview.md`
