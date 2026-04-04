---
tags:
  - research
  - betting
  - markets
  - corners
  - cards
  - poisson
  - betfair
---

# Research — Previsão Multi-Mercado (Todos os Mercados Betfair)

> Research date: 2026-04-04
> Sources: [listadas ao final]

## Context

A Betfair Exchange oferece ~8 categorias de mercados pra cada jogo do Brasileirão. Nosso modelo só prevê 2 (1X2 e Over/Under gols). Precisamos expandir pra cobrir todos.

## Mercados da Betfair e Como Prever Cada Um

### 1. Match Odds (1X2) — JÁ TEMOS ✅
- **Modelo:** Poisson bivariado (Dixon-Coles)
- **Input:** xG histórico, attack/defense strength
- **Já implementado no v0.5.0**

### 2. Over/Under Gols (0.5, 1.5, 2.5, 3.5) — JÁ TEMOS ✅
- **Modelo:** Derivado do Monte Carlo (já calculamos over_05, over_15, over_25, over_35)
- **Só precisa expor no frontend**

### 3. BTTS (Both Teams To Score) — JÁ TEMOS ✅
- **Modelo:** Derivado do Monte Carlo (btts_prob)
- **Só precisa expor**

### 4. Placar Exato (Correct Score) — JÁ TEMOS ✅
- **Modelo:** score_matrix do Monte Carlo (top 10 placares com probabilidade)
- **Só precisa expor**

### 5. Escanteios (Over/Under) — PRECISA IMPLEMENTAR 🔨
- **Modelo:** Compound Poisson Regression (paper: Arxiv 2112.13001, publicado 2024)
- **Input necessário:** média de escanteios por time (home/away), escanteios do adversário
- **λ médio:** ~10 escanteios por jogo no futebol (5 + 5)
- **Não segue Poisson simples** — tem clustering (serial correlation entre escanteios consecutivos)
- **Melhor modelo:** Geometric-Poisson (compound) com Bayesian implementation
- **Dados:** Sofascore TEM dados de escanteios por jogo
- **Approach pragmático:** Poisson simples com λ = média de corners do time × fator adversário. Não é perfeito mas funciona pra Over/Under 8.5, 9.5, 10.5

### 6. Cartões (Over/Under) — PRECISA IMPLEMENTAR 🔨
- **Modelo:** Zero-Inflated Poisson (ZIP) — muitos jogos com 0-1 cartões, distribuição não é Poisson puro
- **Input necessário:**
  - **Team Aggression Score:** faltas por jogo, cartões históricos do time
  - **Referee Strictness Score:** cartões médios por jogo do árbitro específico
  - **Match Context:** derby/rivalidade, importância do jogo
- **λ médio:** ~4 cartões por jogo
- **Formula conceitual:** `Expected Cards = (team_A_fouls_avg + team_B_fouls_avg) × referee_card_rate`
- **Dados:** Sofascore TEM dados de cartões e faltas por jogo. **Árbitro** é o diferencial — o Sofascore mostra o árbitro designado

### 7. Asian Handicap — DERIVÁVEL DO QUE TEMOS 🔧
- **Não precisa de modelo novo** — é derivado das probabilidades 1X2
- **Asian -0.5:** = probabilidade de vitória (sem empate)
- **Asian -1.0:** = probabilidade de vitória por 2+ gols
- **Asian -1.5:** = probabilidade de vitória por 2+ gols (já temos no score_matrix)
- **Calcular:** somar probabilidades dos placares relevantes da score_matrix

### 8. Half Time Result — PRECISA IMPLEMENTAR 🔨
- **Modelo:** Poisson separado com λ_HT ≈ 0.45 × λ_FT (first half tem ~45% dos gols)
- **Simular:** Monte Carlo separado com xG ajustado pro 1T
- **Dados:** Sofascore TEM placar do intervalo em cada jogo

### 9. Gol do Jogador — PARCIALMENTE POSSÍVEL ⚠️
- **Precisa:** xG individual do jogador + minutos esperados
- **Temos:** xG por jogador por partida no player_match_metrics
- **Approach:** P(jogador marca) = 1 - e^(-xG_individual_per90 × minutos/90)
- **Limitação:** precisa saber quem vai jogar (lineup confirmada ~1h antes)

## Como os Profissionais Fazem

### Starlizard (Tony Bloom — £600M/ano)
- "Trata gambling como hedge fund trata ações"
- Times de analistas, programadores e matemáticos
- Modelos estatísticos pra calcular odds mais sharp que os bookmakers
- Foco em **Asian Handicap** (mercado mais líquido, menos margem)
- Aposta perto do match day (pra incluir info de escalação)
- Milhares de eventos globais processados por segundo
- **Segredo:** modelos proprietários nunca divulgados

### Pinnacle (bookmaker sharp)
- Margem de ~3% (vs 8-12% dos bookmakers normais)
- "Closing line" da Pinnacle = benchmark de probabilidade real
- Usa wisdom of the crowd: odds se ajustam com volume de apostas
- Sharp bettors são BEM-VINDOS (ao contrário da Bet365 que bane)

### Modelo Acadêmico (Corner Kicks — Arxiv 2112.13001)
- Compound Poisson com Geometric-Poisson distribution
- Bayesian implementation com varying shape parameter
- Usa odds de outros mercados pra informar (cross-market information)
- Handles serial correlation entre corners consecutivos

### Modelo Acadêmico (Cards — Bivariate ZIP)
- Zero-Inflated Poisson (ZIP) via Frank copula
- Encapsula: faltas, cartões históricos, chutes no alvo, escanteios
- Variáveis-chave: FoulF, FoulA, RedCA, YelCA, CornP, ShotT

## Dados Necessários do Sofascore

| Dado | Disponível no Sofascore? | Campo |
|------|:---:|---|
| Escanteios por time por jogo | ✅ | Stats da partida |
| Cartões amarelos por time | ✅ | Stats da partida |
| Cartões vermelhos | ✅ | Stats da partida |
| Faltas por time | ✅ | `fouls` no player stats |
| Árbitro do jogo | ✅ | Metadata da partida |
| Placar do intervalo | ✅ | Score 1T |
| xG por jogador | ✅ | `expectedGoals` |
| Escalação confirmada | ✅ | ~1h antes |

## Priorização (impacto × esforço)

| # | Mercado | Impacto | Esforço | Dados? | Prioridade |
|---|---------|---------|---------|--------|------------|
| 1 | Asian Handicap | Alto | Baixo | ✅ Já temos | P0 |
| 2 | Correct Score | Alto | Zero | ✅ Já temos | P0 |
| 3 | Over/Under 0.5-3.5 | Alto | Zero | ✅ Já temos | P0 |
| 4 | BTTS | Médio | Zero | ✅ Já temos | P0 |
| 5 | Escanteios O/U | Alto | Médio | ✅ Precisa ingerir | P1 |
| 6 | Cartões O/U | Alto | Médio | ✅ Precisa ingerir | P1 |
| 7 | Half Time | Médio | Médio | ✅ Precisa ingerir | P1 |
| 8 | Gol do Jogador | Alto | Alto | ⚠️ Precisa lineup | P2 |

## Implications for Football Moneyball

### P0 — Expor o que já temos (zero código de modelo novo)
- Correct Score: já temos `score_matrix` → mostrar top 10 placares com probabilidade
- Over/Under 0.5-3.5: já calculamos → expor todos no frontend
- BTTS: já calculamos → expor
- Asian Handicap: calcular somando placares da score_matrix:
  ```
  AH -0.5 = P(home win) = sum(score_matrix where home > away)
  AH -1.5 = sum(score_matrix where home > away + 1)
  AH +0.5 = P(away win or draw) = 1 - P(home win)
  ```

### P1 — Novos modelos (Poisson pra escanteios/cartões)
- Ingerir dados extras do Sofascore (escanteios, cartões, faltas, árbitro)
- Modelo Poisson pra escanteios: λ_corners = team_corners_avg × opponent_factor
- Modelo ZIP pra cartões: λ_cards = (team_fouls_rate + opp_fouls_rate) × referee_rate
- Monte Carlo pra simular Over/Under 8.5, 9.5, 10.5 corners
- Monte Carlo pra Over/Under 2.5, 3.5, 4.5 cards

### P2 — Player props
- Precisa lineup confirmada (Sofascore ~1h antes)
- xG individual → P(gol) = 1 - e^(-xG/90 × minutos)

## Sources

- [Starlizard — Inside Tony Bloom's Syndicate](https://thedarkroom.co.uk/inside-tony-blooms-secret-betting-syndicate/)
- [Starlizard — Racing Post (£600M/year)](https://www.racingpost.com/news/britain/high-court-case-alleges-tony-blooms-betting-empire-makes-600m-a-year)
- [Corner Kicks Compound Poisson — Arxiv 2112.13001](https://arxiv.org/abs/2112.13001)
- [Corner Prediction — Journal of OR Society 2024](https://www.tandfonline.com/doi/abs/10.1080/01605682.2024.2306170)
- [Yellow Card Betting Guide — StatsHub](https://www.statshub.com/blog/yellow-card-betting)
- [Poisson Distribution for Football — FBetPrediction](https://football-bet-prediction.com/football-predictions/)
- [Bivariate Poisson — SAPUB](http://article.sapub.org/10.5923.j.ajms.20201003.01.html)
- [Asian Handicap — Wikipedia](https://en.wikipedia.org/wiki/Asian_handicap)
- [Pinnacle Asian Handicap](https://bookmakers.net/pinnacle/asian-handicap/)
- [Trefik Poisson Calculator (Goals, Corners, Cards)](https://www.trefik.cz/predict_poisson.aspx)
