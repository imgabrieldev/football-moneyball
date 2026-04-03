---
tags:
  - pitch
  - betting
  - odds
  - monte-carlo
  - kelly-criterion
  - sofascore
  - brasileirao
---

# Pitch — Betting Value Finder (v0.4.0)

## Problema

Temos um motor analítico completo (xG, pressing, RAPM, embeddings) e dados de 87 partidas do Brasileirão 2026. Mas toda essa análise é retroativa — "o que aconteceu". Não respondemos a pergunta que interessa pro mercado de apostas: **"o que VAI acontecer?"**

Casas de apostas (Betano, Bet365, Pinnacle) precificam partidas usando odds que refletem probabilidades implícitas. Se nosso modelo estatístico estima probabilidades mais precisas que as do mercado, existem **value bets** — apostas com expectativa positiva de retorno.

O problema é triplo:
1. **Não temos acesso a odds** — precisamos integrar uma API de odds
2. **Não temos modelo preditivo** — nosso xG é retroativo, não prospectivo
3. **Não sabemos se o modelo funciona** — precisamos backtesting antes de arriscar dinheiro real

Research: [[betting-value-model]]

## Solução

Sistema de 4 camadas que transforma análise retroativa em previsão probabilística:

### A. Odds Provider — Integração com The Odds API

Novo adapter que busca odds pré-jogo de múltiplas casas de apostas:
- **The Odds API** free tier (500 req/mês — suficiente pro Brasileirão)
- Mercados: 1X2 (`h2h`), over/under (`totals`), ambas marcam (`btts`), handicap (`spreads`)
- ~30 bookmakers incluindo Bet365, Pinnacle
- Odds históricas desde 2020 para backtesting

### B. Match Predictor — Monte Carlo + Poisson + xG

Modelo preditivo que estima probabilidades de cada resultado:

1. **xG esperado por time**: média ponderada dos últimos 5-8 jogos, ajustada por:
   - Força do adversário (xG against do oponente)
   - Fator casa (+0.25-0.35 xG no Brasileirão)
   - Forma recente (decaimento exponencial)
   - Pressing intensity (PPDA correlaciona com xG criado)

2. **Simulação Monte Carlo** (N=10.000):
   - Gols de cada time sorteados via distribuição Poisson: `P(k) = (λ^k × e^-λ) / k!`
   - Onde `λ = xG esperado ajustado`
   - Cada simulação produz um placar

3. **Probabilidades derivadas** de 10.000 simulações:
   - P(home win), P(draw), P(away win)
   - P(over 0.5), P(over 1.5), P(over 2.5), P(over 3.5)
   - P(BTTS yes), P(BTTS no)
   - Placar mais provável

### C. Value Detector — Comparar Modelo vs Odds

Para cada mercado de cada partida:
1. Converter odds → probabilidade implícita: `prob = 1/odds`
2. Remover margem (vig): `prob_real = prob / sum(probs)`
3. Calcular edge: `edge = prob_modelo - prob_implícita`
4. Classificar: **value bet** se `edge > threshold` (default 3%)

### D. Bankroll Manager — Kelly Criterion

Para value bets identificadas:
1. Calcular stake ótimo via Kelly: `f* = (b×p - q) / b`
2. Aplicar Kelly fracionário (25%): `stake = 0.25 × f* × bankroll`
3. Limitar stake máximo (5% do bankroll por aposta)

### E. Backtesting Engine — Validar com Dados Históricos

Antes de usar em jogos futuros:
1. Para cada partida já jogada, simular como se fosse pré-jogo
2. Usar apenas dados disponíveis até aquele momento (sem lookahead)
3. Buscar odds históricas via The Odds API
4. Calcular ROI, hit rate, Brier score, max drawdown
5. Comparar com baseline (apostar sempre no favorito)

## Arquitetura

### Módulos afetados

| Módulo | Ação | Camada |
|--------|------|--------|
| **`adapters/odds_provider.py`** | NOVO | Adapter: busca odds via The Odds API |
| **`domain/match_predictor.py`** | NOVO | Domain: Poisson + Monte Carlo + xG adjustment |
| **`domain/value_detector.py`** | NOVO | Domain: identifica value bets (edge > threshold) |
| **`domain/bankroll.py`** | NOVO | Domain: Kelly criterion + stake sizing |
| **`use_cases/predict_match.py`** | NOVO | Use case: previsão de uma partida |
| **`use_cases/find_value_bets.py`** | NOVO | Use case: scanner de value bets na rodada |
| **`use_cases/backtest.py`** | NOVO | Use case: backtesting com dados históricos |
| `ports/odds_provider.py` | NOVO | Port: interface para provider de odds |
| `adapters/postgres_repository.py` | MODIFICAR | Queries de xG histórico por time |
| `cli.py` | MODIFICAR | Novos comandos: predict, value-bets, backtest |
| `adapters/matplotlib_viz.py` | MODIFICAR | Novos plots: probabilidade, ROI curve |

### Schema

```sql
-- Odds de partidas
CREATE TABLE IF NOT EXISTS match_odds (
    match_id INTEGER,
    bookmaker VARCHAR(100),
    market VARCHAR(50),        -- h2h, totals, btts, spreads
    outcome VARCHAR(50),       -- Home, Away, Draw, Over, Under, Yes, No
    point REAL,                -- linha (2.5 para over/under, etc.)
    odds REAL,                 -- odds decimais
    implied_prob REAL,         -- probabilidade implícita (1/odds)
    fetched_at TIMESTAMP,
    PRIMARY KEY (match_id, bookmaker, market, outcome, point)
);

-- Previsões do modelo
CREATE TABLE IF NOT EXISTS match_predictions (
    match_id INTEGER PRIMARY KEY,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    home_xg_expected REAL,
    away_xg_expected REAL,
    home_win_prob REAL,
    draw_prob REAL,
    away_win_prob REAL,
    over_25_prob REAL,
    btts_prob REAL,
    most_likely_score VARCHAR(10),
    simulations INTEGER DEFAULT 10000,
    predicted_at TIMESTAMP
);

-- Value bets identificadas
CREATE TABLE IF NOT EXISTS value_bets (
    id SERIAL PRIMARY KEY,
    match_id INTEGER,
    market VARCHAR(50),
    outcome VARCHAR(50),
    model_prob REAL,
    best_odds REAL,
    bookmaker VARCHAR(100),
    implied_prob REAL,
    edge REAL,                  -- model_prob - implied_prob
    kelly_fraction REAL,
    recommended_stake REAL,
    actual_result VARCHAR(50),  -- preenchido após o jogo
    profit REAL,                -- preenchido após o jogo
    created_at TIMESTAMP
);

-- Performance do modelo (backtesting)
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    run_date TIMESTAMP,
    matches_analyzed INTEGER,
    bets_placed INTEGER,
    total_staked REAL,
    total_return REAL,
    roi REAL,
    hit_rate REAL,
    brier_score REAL,
    max_drawdown REAL,
    config JSONB               -- parâmetros do backtest
);
```

### Infra (K8s)

Sem mudanças. The Odds API é chamada do CLI local.

## Escopo

### Dentro do Escopo

- [ ] Port `odds_provider.py` — interface para APIs de odds
- [ ] Adapter `odds_provider.py` — The Odds API integration (free tier)
- [ ] Domain `match_predictor.py` — Poisson model + Monte Carlo (10K sims)
- [ ] Domain `value_detector.py` — comparação modelo vs odds, edge calculation
- [ ] Domain `bankroll.py` — Kelly criterion fracionário
- [ ] Use case `predict_match.py` — previsão de uma partida
- [ ] Use case `find_value_bets.py` — scanner de value bets na rodada
- [ ] Use case `backtest.py` — backtesting com 87+ partidas históricas
- [ ] CLI `predict <match_id>` — exibir previsão do modelo
- [ ] CLI `value-bets [--round N]` — listar value bets da rodada
- [ ] CLI `backtest --season 2026` — rodar backtesting e exibir ROI
- [ ] Schema: tabelas match_odds, match_predictions, value_bets, backtest_results
- [ ] Viz: gráfico de ROI acumulado, calibração de probabilidades
- [ ] Testes unitários para Poisson, Monte Carlo, Kelly, edge detection

### Fora do Escopo

- Apostas automatizadas (sem integração direta com casas de apostas)
- Live/in-play betting (apenas pré-jogo)
- Mercados de jogador (artilheiro, cartões) — apenas mercados de partida
- Arbitragem entre casas (foco é value, não arb)
- Machine learning avançado (apenas Poisson + ajustes, sem neural networks)
- Gestão de múltiplas bankrolls/carteiras

## Research Necessária

- [x] APIs de odds para Brasileirão — [[betting-value-model]]
- [x] Monte Carlo + Poisson para previsão de partidas — [[betting-value-model]]
- [x] Kelly Criterion — [[betting-value-model]]
- [ ] Validar que The Odds API cobre odds históricas do Brasileirão 2026 (desde Jan/2026)
- [ ] Calibrar fator casa do Brasileirão 2026 com nossos dados (87 partidas)
- [ ] Testar sensibilidade do modelo a janela de xG (últimos 3, 5, 8 jogos)

## Estratégia de Testes

### Unitários (domain — zero mocks)
- `match_predictor.py`: Poisson PMF com λ conhecido, Monte Carlo convergência (P(home) deve estabilizar com N grande)
- `value_detector.py`: edge calculation com odds conhecidas, threshold filtering
- `bankroll.py`: Kelly com inputs determinísticos, limites de stake

### Integração
- `odds_provider.py`: mock HTTP response, verificar parsing de odds
- `backtest.py`: rodar com 5 partidas de teste, verificar ROI calculation

### Manual
- Comparar previsão de partida específica com odds reais do Betano
- Verificar que backtesting não tem lookahead bias
- Conferir que Kelly nunca recomenda > 5% do bankroll

## Critérios de Sucesso

- [ ] Backtesting com 87 partidas produz ROI positivo (qualquer edge > 0 é sinal)
- [ ] Brier score do modelo < 0.25 (melhor que aleatório)
- [ ] Hit rate em value bets > 50% (se edge é real, deve acertar mais que erra)
- [ ] Monte Carlo com 10K sims converge (desvio < 1% entre runs)
- [ ] Free tier do The Odds API suficiente para operação mensal
- [ ] Todos os novos módulos no domain layer (sem imports de infra)
- [ ] Zero regressão nos comandos existentes
