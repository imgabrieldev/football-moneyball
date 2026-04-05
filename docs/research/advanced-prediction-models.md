---
tags:
  - research
  - models
  - prediction
  - elo
  - machine-learning
  - draws
  - market-based
---

# Research — Modelos Avançados de Predição de Futebol

> Research date: 2026-04-05
> Trigger: Rodada 10 — 10/10 picks home, 3 acertos, -55% ROI. Modelo estruturalmente enviesado.

## Context

Modelo v1.12.0 usa Poisson independente + Dixon-Coles + xG features + calibração Platt + market blending (65/35). Problema: **sempre prevê home win** — 10/10 picks na rodada 10, apenas 3 corretos. Draws e away wins nunca são selecionados como favorito.

## 1. Rating Systems — Elo, Glicko-2, Pi-Rating

### Elo (FiveThirtyEight SPI / ClubElo)

```
R_new = R_old + K × G × (W - W_e)
W_e = 1 / (1 + 10^((R_away - R_home) / 400))
```

- **K=20** (liga), K=30-60 (copas)
- **G** (goal diff): `(11 + goal_diff) / 8` se diff ≤ 1, log-scale depois
- **Home advantage**: +65 Elo (FiveThirtyEight) ou +100 (ClubElo)
- **Benchmark**: ClubElo ~52-54% accuracy, Brier ~0.21

FiveThirtyEight SPI converte ratings → Poisson lambdas → bivariate simulation com correlação 0.1-0.2.

### Glicko-2 (Glickman, 2001)

Adiciona **desvio de rating (RD)** e volatilidade. Vantagem: times promovidos / início de temporada têm RD alto → ratings movem mais rápido.

```
mu_new = mu + (phi² × Σ g(phi_j) × (s_j - E_j))
```

- **tau** (constraint de volatilidade): 0.3-0.6 pra futebol
- Teoricamente superior ao Elo pra ligas com promoção/rebaixamento

### Pi-Rating (Constantinou & Fenton, 2013) ⭐

**Projetado especificamente pra futebol.** Mantém ratings separados HOME/AWAY por time.

```
R_home_new = R_home_old + γ × e × (goal_diff - e)
R_away_new = R_away_old + γ × e × (goal_diff - e)
e = (R_home_H - R_away_A) / 3
```

- **Benchmark: Brier 0.2065 na EPL** (2001-2012), superou odds de bookmakers em alguns estudos
- Ratings separados home/away resolvem diretamente nosso problema de home bias
- Source: JQAS 2013

### Recomendação

**Pi-Rating** como backbone de rating. Ajustar lambdas do Poisson pelo diferencial Pi:

```python
lambda_home = xG_home × (1 + alpha × (pi_home_H - pi_away_A))
lambda_away = xG_away × (1 + alpha × (pi_away_A - pi_home_H))
```

Quando Pi-rating diz que away é mais forte, lambda_home CAI e lambda_away SOBE.

---

## 2. Machine Learning — XGBoost, CatBoost, Ensemble

### Benchmarks comparativos (2017 Soccer Prediction Challenge + papers recentes)

| Modelo | Accuracy | RPS | Notas |
|---|---|---|---|
| **CatBoost + pi-ratings** | 55.82% | **0.1925** | Melhor do challenge, BATEU bookmakers |
| XGBoost + pi-ratings | 52.43% | 0.2063 | Forte mas atrás do CatBoost |
| Berrar ratings + XGBoost/k-NN | 51.94% | 0.2054 | Abordagem hybrid |
| **Bookmaker consensus** | ~53% | 0.2012 | Teto prático (difícil superar) |
| Poisson (Dixon-Coles) | 48-52% | 0.21-0.22 | Baseline |
| Dolores (dynamic ratings + BN) | — | 0.2007 | 2° em ML for Soccer |
| **Nosso modelo (v1.12)** | **42.4%** | **~0.24** | **Abaixo de Elo standalone** |

**Insight crítico**: CatBoost + pi-ratings a **0.1925 RPS bateu bookmakers** (0.2012) usando features relativamente simples. Nosso sistema tem features ricas (xG, pressing, network) mas modelo fraco.

### Features que importam (papers)

1. **Pi-ratings / Elo** (maior importância)
2. **EMA form** (5-20 jogos, exponential moving average)
3. **xG / xGA** (ataque e defesa)
4. **Shots, PPDA, possession** (pressão)
5. **Devigged odds** (consenso de mercado como feature!)
6. **H2H** (confronto direto)

### Arquitetura recomendada

**Ensemble stacking**:
1. Poisson → score matrix → multi-market (manter, é a base pra corners/cards/correct score)
2. **CatBoost/XGBoost → 1x2 probs** (softmax, features incluindo odds)
3. **Blend**: Poisson 1x2 × XGBoost 1x2 → calibração → output final

---

## 3. Abordagem de Mercado — O Que Profissionais Fazem

### Achado #1: Odds como features (não só pra blending!)

Paper BORS (PLOS ONE): odds pré-jogo contêm **MAIS informação** que resultados pós-jogo. Devigged bookmaker probs devem ser **features de input**, não apenas benchmark.

```python
# Em vez de: blend(model_prob, market_prob)
# Fazer: model(features + [market_home, market_draw, market_away])
```

**Impacto**: maior ganho individual possível. Mercado já resolve home bias.

### Achado #2: CLV como métrica

Sharp bettors medem sucesso pelo **Closing Line Value** (CLV) — a linha de Pinnacle no fechamento é a "verdade" do mercado. Se nosso modelo consistentemente NÃO bate a closing line, ele não tem edge.

### Achado #3: Inverter a arquitetura

Profissionais (Starlizard, sindicatos): começam do **mercado** e ajustam com modelo, não o contrário.

```
Odds → devig → true probs baseline → modelo faz ajustes marginais → bet se delta > threshold
```

Nosso pipeline: `modelo → calibração → blend leve com odds`. Deveria ser: `odds → ajuste fino com modelo`.

### Achado #4: Wisdom of crowds

Kaunitz et al. (arXiv): média de odds de 30+ bookmakers, apostar onde desvio > threshold → ~80% ROI anual (10 anos sim, fractional Kelly).

### Achado #5: Starlizard

Tony Bloom (~£600M/ano volume): ~100 analistas, dados soft (moral, treinos), aposta em Asian Handicap (mais líquido). Usa modelo + soft data layering. Não é puramente quantitativo.

---

## 4. Draw Prediction — Métodos Específicos

### Por que draws são difíceis

- Poisson independente subestima sistematicamente P(X=Y) em ~3-5pp
- Em ligas top, draw rate = 25-28%, modelos Poisson dão 20-24%
- Draw nunca é argmax (mesmo a 28%, home/away geralmente > 30%)
- **Todo modelo ML tem F1 de draw ~0.30** — é a classe mais difícil

### Métodos ranqueados por impacto

| Método | Melhoria draw | Complexidade | Status nosso |
|---|---|---|---|
| **Draw probability floor** | +3-5pp | Trivial | ❌ não implementado |
| **Dixon-Coles rho fit** | +2-4pp | Já temos | ⚠️ rho=0.009 (≈0) |
| **Bivariate Poisson** | +1-3pp | Já temos | ⚠️ λ₃=0.0001 (≈0) |
| **Draw-likelihood features** | +2-3pp em matches "draw-prone" | Médio | ❌ não implementado |
| **Copula (Frank)** | +0.5pp (marginal) | Alto | ❌ skip |
| **Zero-inflated Poisson** | Só 0-0 | Baixo | ❌ skip |

### Draw probability floor (recomendado, trivial)

Se mean draw_prob do modelo < taxa empírica da liga (25-27%):

```python
empirical_draw_rate = 0.26  # Brasileirão histórico
correction = empirical_draw_rate / model_mean_draw_prob
draw_prob *= correction
# renormalizar
```

### Draw-likelihood features

Features correlacionadas com empates:
- Ambos times xG < 1.2
- Mercado O/U 2.5 com under favorito
- H2H com draw rate > 40%
- Ambos na metade inferior de ataque + metade superior de defesa

Flag binária "draw_prone" → boost draw_prob em 10-15%.

---

## 5. Diagnóstico do Nosso Modelo

### Por que sempre prevê HOME

1. **Classe majoritária**: home wins = 47-50% no Brasileirão (plurality class). Sem counter-mechanism, modelo defaulta pro home.
2. **Home advantage como flat boost**: `home_xg += 0.3-0.5 xG` sem modular por qualidade do adversário. Palmeiras fora leva mesmo penalty que Remo fora.
3. **Features correlacionam com home label**: team_xg_for tipicamente > opp_xg_for pra times da casa (viés no sample).
4. **Draw nunca é argmax**: mesmo a 32%, home geralmente é 40%+.
5. **Sem odds como input**: mercado sabe quando um jogo é equilibrado. Modelo não sabe.

### Hierarquia de performance (literature)

```
odds agregadas > wisdom of crowds > Elo + XGBoost > Elo standalone > nosso modelo
```

---

## 6. Plano de Ação Priorizado

### Fase 1 — Quick wins (1-2 dias, alto impacto)

1. **Odds como features no XGBoost**: adicionar devigged Betfair probs (home/draw/away) como 3 features em `feature_engineering.py`. O GBR vai aprender a ponderar modelo vs mercado.
2. **Draw floor**: pós-calibração, se draw_prob < 0.22, boost pra min(0.22, draw_prob×1.3), renormalizar.
3. **Inverter blend alpha**: 35% modelo / 65% mercado (era 65/35). Mercado é mais calibrado que nosso modelo.

### Fase 2 — Structural (1 semana)

4. **Pi-Rating** com ratings separados home/away. Usar diferencial Pi pra modular lambda do Poisson.
5. **CatBoost 1x2**: treinar modelo de 3-class com features [Elo, Pi, xG, form EMA, odds, H2H, rest days]. Target: RPS < 0.21.
6. **RPS como métrica primária** em vez de Brier. Implementar no track-record e backtest.

### Fase 3 — Advanced (2+ semanas)

7. **Bayesian hierarchical** (PyMC/Stan) pra attack/defense por time com shrinkage early-season.
8. **Ensemble stacking**: Poisson (multi-market) + CatBoost (1x2) + ordinal regression (draw specialist).
9. **CLV tracking**: comparar nossas probs com closing line Pinnacle/Betfair.

---

## Sources

### Rating Systems
- [FiveThirtyEight SPI Methodology](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/)
- [ClubElo](http://clubelo.com/System)
- [Pi-Rating — Constantinou & Fenton 2013 (JQAS)](https://www.degruyter.com/document/doi/10.1515/jqas-2012-0054/html)
- [Glicko-2 — Glickman 2001](http://www.glicko.net/glicko/glicko2.pdf)

### Machine Learning
- [xG Football Club — Which ML Models](https://thexgfootballclub.substack.com/p/which-machine-learning-models-perform)
- [CatBoost 0.1925 RPS — Soccer Prediction Challenge](https://link.springer.com/article/10.1007/s10994-018-5703-7)
- [Journal of Big Data 2024 — Data-driven prediction](https://link.springer.com/article/10.1186/s40537-024-01008-2)
- [Bayesian state-space EPL — JRSS 2025](https://academic.oup.com/jrsssc/article/74/3/717/7929974)
- [Ordinal probit — Univ. St. Gallen](https://ux-tauri.unisg.ch/RePEc/usg/econwp/EWP-1811.pdf)
- [Systematic Review ML in Sports Betting — arXiv 2024](https://arxiv.org/html/2410.21484v1)

### Market-Based
- [BORS — PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
- [Kaunitz et al. — arXiv](https://arxiv.org/abs/1710.02824)
- [Pinnacle CLV](https://www.pinnacle.com/betting-resources/en/educational/what-is-closing-line-value-clv-in-sports-betting)
- [Starlizard — Off The Pitch](https://offthepitch.com/a/secrets-starlizard-how-tony-blooms-football-data-monolith-using-its-knowledge-protect-game)
- [Wilkens 2026 — Bundesliga](https://journals.sagepub.com/doi/10.1177/22150218261416681)
- [LSE Wisdom of Crowds](https://blogs.lse.ac.uk/europpblog/2025/05/29/football-forecasting-harnessing-the-power-of-the-crowd/)

### Draw Prediction
- [Penaltyblog — Which Model (RPS benchmarks)](https://pena.lt/y/2025/03/10/which-model-should-you-use-to-predict-football-matches/)
- [Karlis & Ntzoufras 2003 — Bivariate Poisson](http://www2.stat-athens.aueb.gr/~jbn/papers2/08_Karlis_Ntzoufras_2003_RSSD.pdf)
- [Pinnacle — Draw Inflation](https://www.pinnacle.com/betting-resources/en/soccer/inflating-or-deflating-the-chance-of-a-draw-in-soccer/cge2jp2sdkv3a9r5)
- [Wheatcroft 2021 — Match Statistics](https://journals.sagepub.com/doi/10.3233/JSA-200462)
