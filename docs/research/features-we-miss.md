---
tags:
  - research
  - features
  - playing-style
  - finishing
  - gaps
---

# Research — Features que Ignoramos no Modelo

> Research date: 2026-04-04
> Motivado por: análise SPFC 4-1 Cruzeiro (modelo previu 55% mas com xG subestimado)

## Context

Nosso modelo v1.6.0 tem 40 features mas ainda subestima/erra casos onde o **estilo de jogo** e **eficiência de finalização** importam. Exemplo:

**SPFC 4-1 Cruzeiro (04/04/2026):**
- SPFC: 37% posse, 11 chutes, 1 SoT → **4 gols**
- Cruzeiro: 63% posse, 13 chutes, 6 SoT → **1 gol**
- SPFC xG 2.01 vs Cruzeiro xG 1.45 (diferença pequena mas placar 4-1)
- **Goleiro SPFC salvou +0.83 gols**, goleiro Cruzeiro deixou -1.29

Nosso modelo não capturou essa eficiência brutal de SPFC no contra-ataque.

## Findings — Features faltando

### Tier 1: Disponíveis no Sofascore (sem scraping novo)

| Feature | O que mede | Por quê importa |
|---|---|---|
| **Posse de bola histórica** | % médio de posse nos últimos jogos | Identifica estilo (direto vs possession) |
| **Big chances scored/missed** | Finalização de oportunidades claras | Eficiência > volume de chutes |
| **Touches in penalty area** | Presença no terço final | Ameaça real vs posse estéril |
| **Final third entries** | Frequência de ataques | Capacidade de criação |
| **Long balls %** | Proporção de bolas longas | Direto vs elaborado |
| **Aerial duels won %** | Dominância aérea | Decisivo em set pieces |
| **Pass accuracy %** | Precisão técnica | Controle de jogo |
| **Dispossessed count** | Perdas de posse sob pressão | Resistência sob pressing |
| **Goals prevented (PSxG)** | Performance do goleiro | Goalkeeper impact real |

**Onde buscar:** `event/{id}/statistics` (já validado — retorna todos esses campos)

### Tier 2: Derivados do que já temos

| Feature | Como calcular |
|---|---|
| **Shot conversion rate** | `goals / shots` (eficiência de finalização) |
| **SoT conversion** | `goals / shots_on_target` (precisão) |
| **Big chance efficiency** | `big_chances_scored / big_chances` |
| **Possession-adjusted xG** | `xG / possession_%` (ameaça por unidade de posse) |
| **Goal diff vs xG diff** | `(GF-GA) - (xGF-xGA)` (sorte/forma) |
| **Counter-attack indicator** | `low_possession + high_big_chances + wins` |

### Tier 3: Playing Style Archetypes

Baseado em research Opta/StatsBomb:

- **Possession-based**: >55% posse, >85% pass accuracy, <30% long balls
- **Direct/Counter**: <45% posse, >35% long balls, rápido em transição
- **Balanced**: 45-55% posse, mixed metrics
- **Press-heavy**: muitas recuperações no campo adversário

**Classificação**: pode ser feita via K-Means clustering nas features Tier 1.

## Research Papers

### Playing Style & Win Probability (PMC 2024)
- **Possession style** tem win probability **maior** que direct style em build-up
- **Counter-attacks** têm **40% mais sucesso** que positional attacks em transição
- Sequências de counter-attack: ~10/jogo, 7.49% converte gol

### Post-Shot xG (PSxG)
- **xG** mede qualidade do chute ANTES (posição, ângulo)
- **PSxG** mede qualidade DEPOIS (trajetória, força, canto)
- **Goals prevented** = `PSxG_allowed - goals_allowed` → performance do goleiro
- Sofascore expõe **goals_prevented** direto (nome do campo: `goalsPrevented`)

### Corner Kick Efficiency
- Corners convertem gol em apenas **2.2%**
- MAS esses gols **decidem 76% dos jogos** onde acontecem
- Out-swinging corners mais efetivos (7.1% gol)

### Counter-attack Detection (arxiv 2024)
- ~10 counter-attacks por jogo
- 5 eventos por sequência
- 40% dos counter-attacks terminam em chute
- **Contra-ataques iniciados por defensores**: success rate 7.49%

## Implications for Football Moneyball

### v1.7.0 — Playing Style Features (shipping now)

Expandir match_stats com 12 novos campos (v1.7.0 feito):
- home/away_xg (Sofascore calcula)
- big_chances + big_chances_scored
- touches_box, final_third_entries
- long_balls_pct, aerial_won_pct
- goals_prevented, passes, pass_accuracy, dispossessed

### v1.8.0 — Derived Style Features (planejado)

Calcular em `context_features.py`:
- team_possession_avg (últimos N)
- team_directness (% long balls)
- team_finishing_efficiency (goals/big_chances)
- team_gk_quality (goals_prevented médio)
- team_counter_attack_style (flag baseado em posse < 45% + aerial > 55%)

### v1.9.0 — Playing Style Clustering

K-Means com 6 features de estilo → cada time recebe label:
- "possession-dominant"
- "counter-attacker"
- "press-heavy"
- "direct-play"

Feature: **style_matchup** (ex: possession vs counter = favor ao counter).

## Sources

- [Counterattack Detection — Journal of Big Data 2025](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01128-3)
- [Tactical Styles PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11130910/)
- [Playing Styles Opta 2024-25](https://theanalyst.com/articles/analysing-premier-league-playing-styles-2024-25)
- [Post-Shot xG (PSxG)](https://the-footballanalyst.com/beyond-xg-how-post-shot-xg-changes-player-evaluation/)
- [Corner Kicks Efficiency](https://www.sciencedirect.com/science/article/pii/S3050544525000337)
- [FIFA World Cup Features PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12708546/)
- [Counter-attack GNN arxiv](https://arxiv.org/html/2411.17450v2)
- [Possession vs Counter Frontiers 2023](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1197039/full)
