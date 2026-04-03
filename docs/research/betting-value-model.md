---
tags:
  - research
  - betting
  - odds
  - monte-carlo
  - kelly-criterion
  - value-betting
---

# Research — Betting Value Model

> Research date: 2026-04-03
> Sources: [listadas ao final]

## Context

Investigar viabilidade de usar nossas previsões estatísticas (xG, pressing, RAPM) para identificar value bets no Brasileirão comparando com odds de casas de apostas.

## Findings

### 1. APIs de Odds Disponíveis

**The Odds API** (the-odds-api.com) — Melhor opção:
- **Free tier: 500 requests/mês** (suficiente pra ~50 rodadas de 10 jogos)
- Cobre **Brasileirão Série A** (sport key: `soccer_brazil_campeonato`)
- Mercados disponíveis:
  - `h2h` — moneyline (1X2 com draw)
  - `totals` — over/under gols
  - `btts` — ambas marcam (Yes/No)
  - `spreads` — handicap
  - `draw_no_bet`, `double_chance`
- ~30 bookmakers (Bet365, Betano via região, Pinnacle, 1xBet, etc.)
- **Odds históricas** desde Jun/2020 (snapshots a cada 5-10 min)
- Plano pago: $30/mês pra 20K requests

**API-Football** ($19/mês) — Alternativa:
- Endpoint `/odds` com pre-match odds
- Cobre Brasileirão
- Menos mercados que The Odds API

### 2. Modelo de Previsão: Monte Carlo + Poisson + xG

**Abordagem padrão da indústria:**

1. **Calcular xG esperado** de cada time pra partida futura (média xG/jogo dos últimos N jogos, ajustado por força do adversário)
2. **Modelar gols como Poisson**: `P(k gols) = (λ^k × e^(-λ)) / k!` onde `λ = xG esperado`
3. **Simular N=10.000+ partidas** via Monte Carlo: sortear gols de cada time da distribuição Poisson
4. **Calcular probabilidades**: P(home win), P(draw), P(away win), P(over 2.5), P(BTTS), etc.

**Ajustes avançados:**
- Fator casa (home advantage ~+0.3 xG no Brasileirão)
- Pressing intensity do adversário (PPDA afeta xG esperado)
- Forma recente (últimos 5 jogos ponderados exponencialmente)
- RAPM dos jogadores chave (escalação importa)

### 3. Conversão de Odds → Probabilidade Implícita

```
Odds decimais: prob_implícita = 1 / odds
Exemplo: Palmeiras @ 1.80 → 1/1.80 = 55.6%

Remover margem (vig/juice):
total = sum(1/odds_i para cada outcome)
prob_real_i = (1/odds_i) / total
```

### 4. Identificação de Value Bets

**Value bet** = quando nossa probabilidade estimada > probabilidade implícita das odds.

```
edge = prob_modelo - prob_implícita
Se edge > 0 → value bet

Exemplo:
- Modelo diz: Palmeiras ganha com 60%
- Betano oferece odds 1.80 → implícita 55.6%
- Edge = 60% - 55.6% = +4.4% → VALUE BET
```

**Threshold mínimo:** edge > 3% (margem de segurança contra erro do modelo)

### 5. Kelly Criterion para Sizing

```
f* = (b × p - q) / b

Onde:
- f* = fração do bankroll a apostar
- b = odds decimais - 1 (net odds)
- p = probabilidade estimada pelo modelo
- q = 1 - p

Exemplo:
- p = 0.60, odds = 1.80, b = 0.80
- f* = (0.80 × 0.60 - 0.40) / 0.80 = 0.10 = 10% do bankroll
```

**Na prática:** usar **Kelly fracionário (25%)** pra reduzir variância:
- Aposta = 0.25 × f* × bankroll

### 6. Backtesting

Com 87 partidas do Brasileirão 2026 já ingeridas:
1. Para cada partida, calcular xG esperado baseado nos jogos anteriores
2. Simular Monte Carlo → probabilidades do modelo
3. Comparar com odds históricas (The Odds API historical endpoint)
4. Identificar onde teria havido value bets
5. Calcular ROI simulado com Kelly fracionário

**Métricas de avaliação:**
- **ROI** (Return on Investment): lucro / total apostado
- **Hit rate**: % de value bets que acertaram
- **Brier score**: calibração das probabilidades (0=perfeito, 0.25=aleatório)
- **Max drawdown**: maior sequência de perdas
- **Sharpe ratio** adaptado: retorno ajustado por volatilidade

## Implications for Football Moneyball

### Viabilidade
- **The Odds API free tier** (500 req/mês) é suficiente para Brasileirão (~10 jogos × 4 mercados × ~9 rodadas = ~360 requests)
- Nossos dados de xG do Sofascore são confiáveis para o modelo Poisson
- Monte Carlo é computacionalmente trivial (10K simulações < 1 segundo)
- Backtesting com 87 partidas dá amostra inicial razoável

### Riscos
- Modelo de xG pode ter bias (Sofascore vs StatsBomb calculam xG diferente)
- 87 partidas é amostra pequena pra validar edge estatístico (ideal: 500+)
- Odds de abertura vs fechamento: timing importa
- Mercado de apostas brasileiro tem regulação em evolução

## Sources

- [The Odds API](https://the-odds-api.com/)
- [The Odds API Documentation V4](https://the-odds-api.com/liveapi/guides/v4/)
- [The Odds API Betting Markets](https://the-odds-api.com/sports-odds-data/betting-markets.html)
- [Monte Carlo Football Match Sim](https://github.com/TacticsBadger/MonteCarloFootballMatchSim)
- [Kelly Criterion — Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Football Odds Monte Carlo — Medium](https://medium.com/@arit.pom/football-odds-data-analysis-using-montecarlo-simulation-in-python-part-2-43f5e951c1fc)
- [OddAlerts Football Data API](https://www.oddalerts.com/football-data-api)
- [Bankroll Management — Tradematesports](https://www.tradematesports.com/en/blog/bankroll-management-sports-betting)
