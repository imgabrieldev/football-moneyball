---
tags:
  - research
  - prediction
  - home-advantage
  - draws
  - brasileirao-2026
---

# Research — Análise de Erros Preditivos (Rodada 10, 2026-04-05)

> Research date: 2026-04-05
> Trigger: 2/7 acertos na rodada (29%), 4 away wins perdidos, 0 draws acertados

## Context

Modelo v1.10.0 previu 7 jogos em 2026-04-05 com 29% accuracy. Padrão de erro:
- 4 visitantes venceram (Palmeiras, Inter, Botafogo, Bragantino) — modelo deu todos como underdog
- 1 empate (Chapecoense-Vitória) — modelo deu Chapecoense 50%
- 2 acertos (Flamengo, Atlético-MG) — ambos favoritos da casa que venceram

## Findings

### 1. Home Advantage Inflada

Pesquisa acadêmica mostra declínio pós-COVID do HA no Brasil:
- **Pré-COVID (2019):** 57.9% home wins
- **COVID (2020):** 44.9%
- **Pós-COVID (2022):** 48.6%
- **Dados internos 2024:** 47% H / 26% D / 27% A (HA xG = 0.34)
- **Dados internos 2025:** 50% H / 26% D / 24% A (HA xG = 0.49)
- **Dados internos 2026:** 50% H / 25% D / 25% A (HA xG = 0.51)

**Problema no código:** `max(home_advantage, 0.0)` clampeava HA ≥ 0, impedindo que away teams tivessem boost em cenários extremos. Fallback de 0.30 xG quando sem dados era razoável mas agressivo.

**Fix aplicado:** removido clamp, fallback reduzido 0.30 → 0.20.

Sources:
- [Nortis Journal - Home Advantage in Brazilian Football](https://nortisjournal.com/index.php/pub/article/view/6)
- [MDPI - Two Years of COVID-19 Pandemic in Serie A](https://www.mdpi.com/1660-4601/19/16/10308)

### 2. Dixon-Coles ρ Sem Efeito Prático

Fitted ρ = 0.0089 (≈ 0) pelo MLE, mas:
- Default em `simulate_match()` já era -0.10 (correto)
- O fitted value do calibration.pkl **nunca era injetado** no pipeline de predição
- Típico em ligas europeias: ρ = -0.10 a -0.15

Draws sub-estimados: Poisson independente dá ~20-24%, real é 25-27%.

**Fix aplicado:** rho do calibration.pkl agora é injetado em `predict_match()` e `predict_match_player_aware()`.

Sources:
- [dashee87 - Dixon-Coles and Time-Weighting](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
- [Pinnacle - Inflating/Deflating Draw Chance](https://www.pinnacle.com/betting-resources/en/soccer/inflating-or-deflating-the-chance-of-a-draw-in-soccer/cge2jp2sdkv3a9r5)
- [Karlis & Ntzoufras 2003 - Bivariate Poisson](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)

### 3. Brasileirão 2026 É Atípico

- **2.72 gols/jogo** (maior desde 2007, vs histórico 2.4-2.6)
- 65 jogos consecutivos sem 0-0 (recorde)
- Times promovidos voláteis: Coritiba melhor fora que em casa
- Palmeiras dominante (8V-1E-1D), Botafogo e Inter caindo

Modelo não captura:
- Volatilidade early-season (poucos dados por time)
- Times promovidos sem prior de Brasileirão (Elo initialization fraca)
- Tendência de mais gols (over/under threshold desatualizado)

Sources:
- [OneFootball - Brasileirão 2026 highest goals since 2007](https://onefootball.com/en/news/brasileirao-betano-2026-sees-highest-goals-per-game-since-2007-42405408)
- [FootyStats - Brazil Serie A 2026](https://footystats.org/brazil/serie-a)

## Implications for Football Moneyball

### Fixes aplicados (v1.11.1)

1. **Removido clamp** `max(home_advantage, 0.0)` → permite HA negativo
2. **Fallback reduzido** 0.30 → 0.20 xG
3. **Rho calibrado injetado** no pipeline (predict_match + predict_match_player_aware)

### Futuro (próximos pitches)

4. **Bivariate Poisson** com diagonal inflation (Karlis & Ntzoufras 2003) → corrige draws estruturalmente
5. **Team-specific away strength** → top-6 em budget deveria ter HA reduzido
6. **Early-season regularization** → shrinkage mais forte nos primeiros 5 rounds
7. **Promoted team priors** → Elo initialization baseada em performance na Série B

## Sources

- [Nortis Journal - Home Advantage Pre/During/Post COVID](https://nortisjournal.com/index.php/pub/article/view/6)
- [MDPI - COVID-19 in Serie A](https://www.mdpi.com/1660-4601/19/16/10308)
- [Benz 2021 - Bivariate Poisson](https://pmc.ncbi.nlm.nih.gov/articles/PMC8313421/)
- [PLOS One - Crowd Effects](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289899)
- [ResearchGate - Travel Distance in Brazil](https://www.researchgate.net/publication/285650377)
- [dashee87 - Dixon-Coles](https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/)
- [Pinnacle - Draw Inflation](https://www.pinnacle.com/betting-resources/en/soccer/inflating-or-deflating-the-chance-of-a-draw-in-soccer/cge2jp2sdkv3a9r5)
- [Karlis & Ntzoufras 2003](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)
- [PMC - Double Poisson Euro 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9119507/)
- [OneFootball - Brasileirão 2026](https://onefootball.com/en/news/brasileirao-betano-2026-sees-highest-goals-per-game-since-2007-42405408)
- [FootyStats - Brazil Serie A](https://footystats.org/brazil/serie-a)
