---
tags:
  - research
  - features
  - playing-style
  - finishing
  - gaps
---

# Research — Features We Ignore in the Model

> Research date: 2026-04-04
> Motivated by: SPFC 4-1 Cruzeiro analysis (model predicted 55% but with underestimated xG)

## Context

Our v1.6.0 model has 40 features but still underestimates/misses cases where **playing style** and **finishing efficiency** matter. Example:

**SPFC 4-1 Cruzeiro (04/04/2026):**

- SPFC: 37% possession, 11 shots, 1 SoT → **4 goals**
- Cruzeiro: 63% possession, 13 shots, 6 SoT → **1 goal**
- SPFC xG 2.01 vs Cruzeiro xG 1.45 (small difference but final score 4-1)
- **SPFC keeper saved +0.83 goals**, Cruzeiro keeper let in -1.29
- Our model did not capture SPFC's brutal counter-attack efficiency.

## Findings — Missing features

### Tier 1: Available on Sofascore (no new scraping needed)

| Feature | What it measures | Why it matters |
|---|---|---|
| **Historical possession** | Average possession % over the last matches | Identifies style (direct vs possession) |
| **Big chances scored/missed** | Finishing of clear-cut chances | Efficiency > shot volume |
| **Touches in penalty area** | Presence in the final third | Real threat vs sterile possession |
| **Final third entries** | Attack frequency | Creative capability |
| **Long balls %** | Proportion of long balls | Direct vs elaborated |
| **Aerial duels won %** | Aerial dominance | Decisive on set pieces |
| **Pass accuracy %** | Technical precision | Game control |
| **Dispossessed count** | Possession losses under pressure | Resilience under pressing |
| **Goals prevented (PSxG)** | Goalkeeper performance | Real goalkeeper impact |

**Where to fetch:** `event/{id}/statistics` (already validated — returns all these fields)

### Tier 2: Derived from what we already have

| Feature | How to compute |
|---|---|
| **Shot conversion rate** | `goals / shots` (finishing efficiency) |
| **SoT conversion** | `goals / shots_on_target` (precision) |
| **Big chance efficiency** | `big_chances_scored / big_chances` |
| **Possession-adjusted xG** | `xG / possession_%` (threat per unit of possession) |
| **Goal diff vs xG diff** | `(GF-GA) - (xGF-xGA)` (luck/form) |
| **Counter-attack indicator** | `low_possession + high_big_chances + wins` |

### Tier 3: Playing Style Archetypes

Based on Opta/StatsBomb research:

- **Possession-based**: >55% possession, >85% pass accuracy, <30% long balls
- **Direct/Counter**: <45% possession, >35% long balls, fast in transition
- **Balanced**: 45-55% possession, mixed metrics
- **Press-heavy**: many recoveries in the opposing half

**Classification**: can be done via K-Means clustering on the Tier 1 features.

## Research Papers

### Playing Style & Win Probability (PMC 2024)

- **Possession style** has **higher** win probability than direct style in build-up
- **Counter-attacks** have **40% more success** than positional attacks in transition
- Counter-attack sequences: ~10/match, 7.49% convert to a goal

### Post-Shot xG (PSxG)

- **xG** measures shot quality BEFORE (position, angle)
- **PSxG** measures shot quality AFTER (trajectory, force, corner)
- **Goals prevented** = `PSxG_allowed - goals_allowed` → goalkeeper performance
- Sofascore exposes **goals_prevented** directly (field name: `goalsPrevented`)

### Corner Kick Efficiency

- Corners convert to goals only **2.2%** of the time
- BUT those goals **decide 76% of matches** in which they happen
- Out-swinging corners are more effective (7.1% goal rate)

### Counter-attack Detection (arxiv 2024)

- ~10 counter-attacks per match
- 5 events per sequence
- 40% of counter-attacks end in a shot
- **Counter-attacks initiated by defenders**: success rate 7.49%

## Implications for Football Moneyball

### v1.7.0 — Playing Style Features (shipping now)

Expand match_stats with 12 new fields (v1.7.0 done):

- home/away_xg (computed by Sofascore)
- big_chances + big_chances_scored
- touches_box, final_third_entries
- long_balls_pct, aerial_won_pct
- goals_prevented, passes, pass_accuracy, dispossessed

### v1.8.0 — Derived Style Features (planned)

Compute in `context_features.py`:

- team_possession_avg (last N)
- team_directness (% long balls)
- team_finishing_efficiency (goals/big_chances)
- team_gk_quality (average goals_prevented)
- team_counter_attack_style (flag based on possession < 45% + aerial > 55%)

### v1.9.0 — Playing Style Clustering

K-Means with 6 style features → each team gets a label:

- "possession-dominant"
- "counter-attacker"
- "press-heavy"
- "direct-play"

Feature: **style_matchup** (e.g. possession vs counter = favors the counter).

## Sources

- [Counterattack Detection — Journal of Big Data 2025](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-025-01128-3)
- [Tactical Styles PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11130910/)
- [Playing Styles Opta 2024-25](https://theanalyst.com/articles/analysing-premier-league-playing-styles-2024-25)
- [Post-Shot xG (PSxG)](https://the-footballanalyst.com/beyond-xg-how-post-shot-xg-changes-player-evaluation/)
- [Corner Kicks Efficiency](https://www.sciencedirect.com/science/article/pii/S3050544525000337)
- [FIFA World Cup Features PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12708546/)
- [Counter-attack GNN arxiv](https://arxiv.org/html/2411.17450v2)
- [Possession vs Counter Frontiers 2023](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1197039/full)
