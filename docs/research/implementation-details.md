---
tags:
  - research
  - implementation
  - xgboost
  - compound-poisson
  - zip
  - referee
---

# Research — Detalhes de Implementação do Framework 5-Camadas

> Research date: 2026-04-04
> Complementa: [[comprehensive-prediction-models]]

## Context

O pitch [[comprehensive-predictor]] propõe 5 camadas. Esse doc coleta os detalhes técnicos específicos de cada peça: distribuições a usar, hiperparâmetros validados na literatura, padrões de imputação pra pré-escalação.

## Findings

### 1. Distribuições por Mercado

Cada métrica tem distribuição adequada diferente — Poisson puro não serve pra tudo.

| Métrica | Distribuição | Por quê | Paper |
|---------|:---:|---|---|
| Gols | Poisson ou Dixon-Coles | eventos raros independentes, ajuste pra 0×0/1×0/0×1 | Dixon-Coles 1997 |
| Escanteios | **Compound Poisson** (Geometric-Poisson) | chegam em batch (cluster), serial correlation entre partidas | arxiv 2112.13001 |
| Cartões | **Zero-Inflated Poisson (ZIP)** | excesso de jogos com 0 amarelos (early game fouls, ref rigor) | statsmodels |
| Chutes | Poisson | eventos mais independentes, shape mais regular | consenso |
| Chutes no gol | Poisson condicional | P(on target \| chute) × total chutes | consenso |
| Faltas | Poisson | ok como baseline, overdispersion pode existir | Bayesian NB |
| HT gols | Poisson(λ × 0.45) | 45% dos gols saem no 1T (consolidado) | Dixon-Coles |

**Decisão de implementação:** começar com Poisson simples pra tudo (baseline), depois trocar pros específicos onde calibração falhar.

### 2. Compound Poisson pra Escanteios

**Por quê Compound:** escanteios vêm em sequência (time pressiona → vários escanteios seguidos). Poisson simples assume independência → subestima variância → errar Over 10.5 / Under 8.5.

**Geometric-Poisson (Bayesian):**
```
N_batches ~ Poisson(λ_match)
corners_per_batch ~ Geometric(p)
Total_corners = Σ corners_per_batch
```

**Parâmetros típicos (Premier League 2020-22):**
- λ_match ≈ 3.5 batches
- p ≈ 0.65 (média ~1.5 corners por batch)
- Total corners ≈ 10-11/jogo

**Implementação simples no início:** rodar Poisson e comparar variância empírica. Se var >> mean, migrar pra Negative Binomial (que é Compound Poisson com gamma).

### 3. Zero-Inflated Poisson pra Cartões

**Por quê ZIP:** cartões têm "excesso de zeros" — muitos jogos sem cartão certo jogador, certo time, certo tempo.

```python
# statsmodels
from statsmodels.discrete.count_model import ZeroInflatedPoisson

# Fit
model = ZeroInflatedPoisson(
    endog=y_cards,              # cartões totais
    exog=X,                      # features
    exog_infl=X_infl,           # features pro inflation model
    inflation='logit',
).fit()
```

**Features pra λ (parte count):**
- Faltas/90 do mandante + visitante (EMA 5 jogos)
- Referee card rate (histórico desse juiz)
- Is_derby (clássico → +20-30% cartões)
- Home advantage (menos cartões em casa)

**Features pra π (parte inflation — prob de 0 cartões):**
- Time pacífico (média < 1 cartão/jogo)
- Juiz tolerante (média < 2.5 cartões/jogo)

**Hiperparâmetros Brasileirão (estimados):**
- λ_base ≈ 4.5 cartões/jogo
- Referee factor: 0.7 a 1.4 (tolerante até rigoroso)
- Derby: × 1.25

### 4. Referee Strictness — Como Calcular

**Problema:** juízes apitam quantidades muito diferentes de cartões. O mesmo time recebe 2.5 cards com um juiz e 4.5 com outro.

**Fórmula Bayesiana com shrinkage:**
```
ref_card_rate = (n_cards + prior_weight × league_avg) / (n_matches + prior_weight)
```

Com `prior_weight = 5`:
- Juiz com 1 jogo → quase só league_avg
- Juiz com 20 jogos → quase todo próprio histórico
- Interpolação suave em entre

**Alternativa: Bayesian Hierarchical**
```
μ_ref ~ Normal(μ_league, σ_league)  # prior
cards_i ~ Poisson(μ_ref × feats_i)  # likelihood
```

Fit via PyMC ou simples conjugate update com Gamma-Poisson.

**Pra MVP:** usar fórmula de shrinkage simples (empirical Bayes). Reservar Bayesian hierarchical pra v1.3.0+.

### 5. Pre-Lineup Prediction (Imputação de Escalação)

**Cenário:** 24h antes do jogo, Sofascore não publicou lineup. Precisamos prever sem saber os 11.

**Estratégia 1: Most Frequent XI**
```python
def probable_lineup(team: str, last_n: int = 5) -> list[int]:
    """Retorna top 11 jogadores por minutos nos últimos N jogos."""
    recent = get_team_matches(team, last=last_n)
    minutes = defaultdict(int)
    for match in recent:
        for player in match.starters:
            minutes[player.id] += player.minutes_played
    return sorted(minutes, key=minutes.get, reverse=True)[:11]
```

**Estratégia 2: Position-Aware**
- 1 GK: máx minutos na posição G
- 4 DEF: top 4 em minutos em posição D
- 3-4 MID: top 3-4 em M
- 2-3 FWD: top 2-3 em F

Considera formação mais frequente do time (4-3-3, 4-4-2, 3-5-2) dos últimos jogos.

**Estratégia 3: Confidence-Weighted**
Alguns titulares são certeza (100% jogos), outros são rotação. Ponderar features pelo "prob de ser titular":

```python
for player in probable_xi:
    weight = player.matches_started / last_n_matches  # 0.0 a 1.0
    team_lambda += player.xg_per90 * weight
```

**Decisão:** começar com Most Frequent XI simples. Adicionar position-aware quando tivermos a formação.

### 6. ML → Poisson Pipeline (XGBoost pra λ)

**Arquitetura:**
```
Features (team + player + context) → XGBoost Regressor → λ → Poisson → score matrix → mercados
```

**Features por time (últimos 6 jogos, EMA com decay=0.9):**
- Gols marcados/sofridos
- xG for/against
- Chutes/chutes no alvo
- Cruzamentos
- Escanteios, cartões, faltas
- PPDA (pressing)
- Possession %
- Passes progressivos/completos

**Features derivadas:**
- xG overperformance (gols - xG nos últimos 10)
- Forma H2H (3 últimos confrontos)
- Rest days (fadiga)
- Home/away form separado

**Features contextuais:**
- Is_home
- League avg xG (baseline)
- Opponent defense rating
- Is_derby
- Referee card rate

**Target:** λ_gols empírico (média de gols marcados em jogos similares). Alternativamente, treinar **contra xG observado** (supervisão fraca) ou gols reais (supervisão forte com mais ruído).

**Hiperparâmetros XGBoost (Beat the Bookie + papers):**
```python
xgb.XGBRegressor(
    n_estimators=500,
    max_depth=4,           # shallow trees, previne overfit
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    objective='reg:squarederror',
)
```

**Pipeline:**
```python
X_train, y_train = build_features(historical_matches)
model.fit(X_train, y_train)

# Cross-validation temporal (não aleatório)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

**Performance esperada (literatura):**
- XGBoost 67% accuracy em 1X2
- ML Poisson: 5.3% ROI médio em simulação (Beat the Bookie)

**Calibração é mais importante que accuracy:**
> "calibration-optimized model generated 69.86% higher average returns"
> — ScienceDirect 2024

Usar `sklearn.calibration.CalibratedClassifierCV` ou Platt scaling após XGBoost.

### 7. Integração ML + Poisson: Padrões

**Padrão A: ML prediz λ diretamente**
```
features → XGBoost → λ → Poisson(λ) → score matrix
```
Simples. Erra se λ ML desvia de λ real.

**Padrão B: ML prediz resíduo do Poisson baseline**
```
features → Dixon-Coles → λ_base
features → XGBoost → δ
λ_final = λ_base + δ
→ Poisson(λ_final) → score matrix
```
Mais robusto. XGBoost aprende "o que Dixon-Coles erra".

**Padrão C: Ensemble**
```
λ_final = 0.5 × λ_DC + 0.5 × λ_XGB
```
Simples, combina os dois mundos.

**Decisão:** começar com Padrão A (v1.3.0). Migrar pra B se Dixon-Coles calibrado continuar competitivo.

### 8. Monte Carlo Multi-Dimensional

**Simulação conjunta de todos mercados:**
```python
def simulate_match_full(
    home_xg, away_xg,
    home_corners_lambda, away_corners_lambda,
    home_cards_lambda, away_cards_lambda,
    home_shots_lambda, away_shots_lambda,
    n_sims=10_000,
):
    rng = np.random.default_rng(42)
    
    # Independentes por simplicidade (real: correlação entre gols e chutes)
    home_goals = rng.poisson(home_xg, n_sims)
    away_goals = rng.poisson(away_xg, n_sims)
    home_corners = rng.poisson(home_corners_lambda, n_sims)
    away_corners = rng.poisson(away_corners_lambda, n_sims)
    cards = rng.poisson(home_cards_lambda + away_cards_lambda, n_sims)
    home_shots = rng.poisson(home_shots_lambda, n_sims)
    away_shots = rng.poisson(away_shots_lambda, n_sims)
    
    # HT goals: 45% do total (aprox)
    ht_home = rng.poisson(home_xg * 0.45, n_sims)
    ht_away = rng.poisson(away_xg * 0.45, n_sims)
    
    return simulate_df  # cada linha = 1 jogo simulado completo
```

**Correlações importantes (ignorar no MVP):**
- Gols ↔ Chutes (time que faz gol chuta mais)
- Gols ↔ Escanteios (pressão → escanteios → gols)
- Cartões ↔ Gols (perder → gols tardios → faltas duras)

Implementação com correlação: Copulas Gaussianas (complexo). **Pra v1.0: simular independente e ver se Brier degrada.**

### 9. Derivação de Mercados da Matriz Simulada

Do DataFrame de 10K simulações, extrair TUDO:

```python
# Ja temos 1X2, O/U, BTTS, correct score, asian handicap

# Novos:
escanteios_over_95 = (sim_df.home_corners + sim_df.away_corners > 9.5).mean()
cartoes_over_35 = (sim_df.cards > 3.5).mean()
chutes_home_over_125 = (sim_df.home_shots > 12.5).mean()
ht_result_home = (sim_df.ht_home > sim_df.ht_away).mean()
ht_ft_h_h = ((sim_df.ht_home > sim_df.ht_away) & 
             (sim_df.home_goals > sim_df.away_goals)).mean()

# Score matrix HT
ht_scores = Counter(zip(sim_df.ht_home, sim_df.ht_away))

# Margem de vitória
margem_home_2 = (sim_df.home_goals - sim_df.away_goals == 2).mean()
```

**Total de mercados deriváveis:** 80-100+ da matriz única.

### 10. Schema de Banco — Campos Novos

**match_stats (novo):**
- match_id PK
- home_corners, away_corners
- home_yellow, away_yellow, home_red, away_red
- home_fouls, away_fouls
- home_shots, away_shots, home_sot, away_sot
- ht_home_score, ht_away_score
- referee_name
- attendance (opcional)
- weather (opcional)

**referee_stats (novo):**
- referee_name PK
- matches INTEGER
- avg_yellow_per_match, avg_red_per_match
- avg_fouls_per_match
- avg_corners_per_match
- strictness_factor (card_rate / league_avg)

**team_form (materialized view? ou calculado on-the-fly):**
- team, competition, season
- last_5_goals_for, last_5_goals_against
- last_5_xg_for, last_5_xg_against
- last_5_corners_for, last_5_corners_against
- last_5_cards
- last_updated

**prediction_history — adicionar campos:**
- home_lambda_corners, away_lambda_corners REAL
- home_lambda_cards, away_lambda_cards REAL
- home_lambda_shots, away_lambda_shots REAL
- model_version VARCHAR (ex: "v1.1.0-player-aware")
- lineup_type VARCHAR ("pre" | "post")

## Implications for Football Moneyball

### Stack de libs necessárias

Já temos:
- numpy, pandas, scikit-learn, xgboost
- sqlalchemy, psycopg2, pgvector
- requests, typer, rich

**Adicionar:**
- `statsmodels` (ZIP, GLM) — ~5MB
- Opcional: `pymc` pra Bayesian hierarchical (depois)

### Ordem de implementação (matching risco/valor)

1. **v1.1.0 — Player-Aware λ** (2 weeks)
   - Sofascore já tem os dados
   - Backward compatible
   - Salto grande de accuracy esperado

2. **v1.2.0 — Multi-Output Poisson** (2 weeks)
   - Novos λ pra corners, cards, shots, HT
   - Referee module
   - Monte Carlo multi-dim

3. **v1.3.0 — ML → λ** (2 weeks)
   - XGBoost pra cada λ
   - Calibração + backtest
   - A/B test contra v1.2.0

## Sources

- [Compound Poisson for Corners (arxiv)](https://arxiv.org/abs/2112.13001)
- [ZIP Regression (statsmodels)](https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html)
- [NumPyro ZIP Example](https://num.pyro.ai/en/stable/examples/zero_inflated_poisson.html)
- [Bayesian Dynamic Models (arxiv 2508)](https://arxiv.org/html/2508.05891v1)
- [XGBoost Football Prediction (Research Gate)](https://www.researchgate.net/publication/369469857_Expected_Goals_Prediction_in_Football_using_XGBoost)
- [Beat the Bookie — Inflated Poisson](https://beatthebookie.blog/2022/08/22/inflated-ml-poisson-model-to-predict-football-matches/)
- [Calibration vs Accuracy (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2772662224001413)
- [Dixon-Coles Time-Weighted (dashee87)](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
