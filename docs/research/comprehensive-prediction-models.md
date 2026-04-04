---
tags:
  - research
  - models
  - machine-learning
  - poisson
  - bayesian
  - player-level
  - referee
  - xgboost
---

# Research — Modelos Matemáticos Completos pra Prever Futebol

> Research date: 2026-04-04
> Sources: [listadas ao final]

## Context

Precisamos evoluir de "Poisson simples com xG do time" pra um sistema completo que use TODOS os dados disponíveis (jogadores, técnico, árbitro, escanteios, cartões, chutes, faltas) pra prever TODOS os mercados da Betfair.

## Findings

### 1. Arquitetura dos Syndicates (Starlizard, Smartodds)

**Starlizard (Tony Bloom, £600M/ano):**
- 4 equipes especializadas, cada uma num papel diferente
- Calcula o **placar mais provável** → deriva todos os mercados
- Foco em Asian Handicap (mais líquido)
- Modelos consideram: clima, moral do time, escalação
- Aposta perto do match day (pra incluir info de escalação)
- Processa milhares de eventos por segundo

**Smartodds (Matthew Benham, dono do Brentford):**
- "Não tentamos dizer o que vai acontecer, tentamos dizer probabilidades"
- Se as probabilidades são melhores que as do bookmaker → bet

**Insight:** Ambos geram **um placar probabilístico** (score matrix) e derivam todos os mercados dele. Exatamente o que nosso Monte Carlo faz — mas eles alimentam com dados muito mais ricos.

### 2. Abordagens Acadêmicas (Estado da Arte 2024-2025)

#### A. Dixon-Coles Estendido (Poisson Bivariado)

**Paper:** "Extending the Dixon and Coles model" (JRSS Series C, Jan 2025)
- Usa família Sarmanov pra modelar correlação entre gols dos times
- Extensões incluem **covariáveis táticas** extraídas de network clustering
- Parâmetros time-varying (attack/defense mudam ao longo da temporada)

**Limitação:** Ainda opera a nível de time, não de jogador.

#### B. ML Poisson Inflado (Beat the Bookie, 2022)

**Modelo:** XGBoost/Random Forest → prediz λ (xG) → Poisson → score matrix

**Features (EMA de 5-20 jogos):**
- Goals for/against
- xG for/against
- Shots e shots on target
- **Corner kicks** ← já inclui escanteios
- Deep passes
- **PPDA** (pressing) ← já inclui pressing

**Inovação:** Zero-Inflated Poisson (ZIP) pra corrigir excesso de 0x0

**Resultado:** ~5.3% profit médio em simulação (Big5 leagues, 7.178 jogos)

#### C. Feedforward Neural Network + XGBoost Ensemble (2024)

**Paper:** "Data-driven prediction of soccer outcomes" (Journal of Big Data, 2024)

**Features incluem:**
- Out-to-in actions, interceptions, sprinting/min
- **Yellow/red cards, corners, shots on target**
- Possession percentage
- Half-time results (real-time)

**Melhor modelo:** Voting ensemble (Random Forest + XGBoost) + Feedforward NN

#### D. Bayes-xG: Player + Position Correction (2023)

**Paper:** "Bayes-xG" (arxiv 2311.13707)

**Inovação:** Mesmo controlando por localização do chute, **player-level effects persistem** — certos jogadores têm ajuste positivo/negativo de xG. Usa Bayesian Hierarchical Model.

**Implicação:** Nosso modelo deveria ajustar xG por jogador, não só por time.

#### E. Bayesian Hierarchical com Network Indicators (2021)

**Paper:** "The role of passing network indicators in modeling football outcomes" (Springer)

**Features:** Shots on target, **corners**, passing network metrics são os "main determinants" de resultados.

### 3. Framework Unificado Recomendado

Baseado em tudo que pesquisei, o modelo mais robusto combina:

```
Camada 1: FEATURE ENGINEERING (por jogador dos 22 em campo)
├── xG/90 individual (Sofascore expectedGoals)
├── Chutes/90, chutes no alvo/90 (totalShots, shotsOnTarget)
├── Cruzamentos/90 (totalCross) → proxy pra escanteios
├── Faltas/90 (fouls, wasFouled) → proxy pra cartões
├── Tackles/90 (totalTackle)
├── Passes/90 (totalPass, accuratePass)
└── Carries, progressive actions

Camada 2: TEAM AGGREGATION (somar/média dos 11 titulares)
├── λ_gols = Σ xG/90 dos 11 × opponent_defense_factor
├── λ_chutes = Σ shots/90 dos 11
├── λ_escanteios = f(cruzamentos dos laterais + chutes bloqueados)
├── λ_cartões = Σ faltas/90 dos volantes × referee_card_rate
├── λ_defesas = chutes_adversário × (1 - goalkeeper_save_rate)
└── λ_gols_HT = λ_gols × 0.45

Camada 3: CONTEXTUAL ADJUSTMENT
├── Home advantage (dinâmico, calculado do banco)
├── Referee strictness (cartões/jogo deste árbitro)
├── Derby factor (+20% cartões, +10% escanteios)
├── Form/momentum (EMA exponencial, últimos 5 jogos pesam mais)
├── Regressão à média (k/(k+n) dinâmico)
└── Weather (se disponível)

Camada 4: MONTE CARLO MULTI-DIMENSIONAL
├── Simular 10K jogos
│   ├── Gols home ~ Poisson(λ_gols_home)
│   ├── Gols away ~ Poisson(λ_gols_away)
│   ├── Corners home ~ Poisson(λ_corners_home)
│   ├── Corners away ~ Poisson(λ_corners_away)
│   ├── Cards ~ ZIP(λ_cards × referee_factor)
│   ├── Shots ~ Poisson(λ_shots)
│   └── HT goals ~ Poisson(λ_gols × 0.45)
└── Cada simulação produz um "jogo completo"

Camada 5: MARKET DERIVATION (do jogo simulado)
├── 1X2, Correct Score, Asian Handicap
├── Over/Under gols (0.5 a 5.5)
├── BTTS, Chance Dupla, Empate sem aposta
├── Over/Under escanteios (4.5 a 15.5)
├── Over/Under cartões (0.5 a 8.5)
├── HT result, HT/FT, HT goals
├── Margem de vitória
├── Gols por time (casa/fora)
├── Primeiro gol
└── Player props (marcador, chutes, faltas)
```

### 4. ML vs Poisson: Qual usar?

| Approach | Prós | Contras | Quando usar |
|----------|------|---------|-------------|
| **Poisson puro** | Interpretável, rápido, score matrix | Ignora features ricas | Baseline |
| **ML → Poisson** | Features ricas → λ melhor | Precisa de dados | **RECOMENDADO** |
| **XGBoost direto** | Melhor accuracy em 1X2 | Sem score matrix, sem multi-market | Só 1X2 |
| **Neural Network** | Captura não-linearidades | Caixa preta, precisa muitos dados | Se tiver 10K+ jogos |
| **Bayesian Hierarchical** | Incerteza quantificada, player-level | Complexo, lento | Se quiser intervalos de confiança |

**Recomendação: ML → Poisson (Inflated)**
- XGBoost/Random Forest prediz λ pra cada métrica (gols, corners, cards, shots)
- Alimenta Poisson multidimensional
- Monte Carlo simula jogo completo
- Todos os mercados derivados

### 5. O que os melhores usam que nós não usamos

| Feature | Starlizard | Academics | Nós (hoje) | Nós (v1.0.0) |
|---------|:---:|:---:|:---:|:---:|
| xG time | ✅ | ✅ | ✅ | ✅ |
| xG jogador | ✅ | ✅ | ❌ | ✅ |
| Escalação | ✅ | ❌ | ❌ | ✅ |
| Árbitro | ✅ | parcial | ❌ | ✅ |
| Escanteios | ✅ | ✅ | ❌ | ✅ |
| Cartões | ✅ | ✅ | ❌ | ✅ |
| PPDA/pressing | ✅ | ✅ | parcial | ✅ |
| Clima | ✅ | parcial | ❌ | ❌ |
| Moral/momentum | ✅ | ❌ | parcial | ✅ |
| EMA form | ✅ | ✅ | ✅ | ✅ |
| Network analysis | ❌ | ✅ | temos | avaliar |
| ML pra λ | provável | ✅ | ❌ | **NEXT** |

## Implications for Football Moneyball

### Evolução recomendada

1. **Agora (v1.0.0 P0):** Já feito — mercados derivados do Monte Carlo simples
2. **Próximo (v1.0.0 P1):** Player-aware λ + novos Poisson (corners, cards, shots)
3. **Depois (v1.1.0):** ML → Poisson (XGBoost prediz λ usando todas as features)
4. **Futuro (v1.2.0):** Bayesian Hierarchical com player-level effects + incerteza

### O salto mais impactante: player-aware λ

O gap entre nosso modelo (47% accuracy) e os syndicates (~55-60% estimado) está em:
1. **Escalação** — quem joga muda tudo
2. **Árbitro** — muda λ de cartões em 50%+
3. **ML pra λ** — XGBoost captura interações não-lineares entre features

## Sources

- [Dixon-Coles Extensions — JRSS 2025](https://academic.oup.com/jrsssc/article/74/1/167/7818323)
- [Dixon-Coles Extensions — arxiv](https://arxiv.org/pdf/2307.02139)
- [Inflated ML Poisson — Beat the Bookie](https://beatthebookie.blog/2022/08/22/inflated-ml-poisson-model-to-predict-football-matches/)
- [Data-driven prediction — Journal of Big Data 2024](https://link.springer.com/article/10.1186/s40537-024-01008-2)
- [Predictive analytics framework — ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S2772662224001413)
- [Bayes-xG Player Correction — arxiv 2023](https://arxiv.org/html/2311.13707)
- [Bayesian Hierarchical — UCL](https://discovery.ucl.ac.uk/16040/1/16040.pdf)
- [Passing Network Indicators — Springer 2021](https://link.springer.com/article/10.1007/s10182-021-00411-x)
- [XGBoost + LSTM — IEEE 2024](https://ieeexplore.ieee.org/iel8/10935288/10935360/10935531.pdf)
- [Starlizard — The Dark Room](https://thedarkroom.co.uk/inside-tony-blooms-secret-betting-syndicate/)
- [Starlizard — Yahoo Finance](https://uk.finance.yahoo.com/news/inside-starlizard-story-britains-most-090759947.html)
- [Smartodds — Bleacher Report](https://bleacherreport.com/articles/2200795-mugs-and-millionaires-inside-the-murky-world-of-professional-football-gambling)
- [Soccermatics — Prediction](https://soccermatics.readthedocs.io/en/latest/lesson5/Prediction.html)
