---
tags:
  - research
  - brasileirao
  - data-sources
  - api
---

# Research — Data Sources for the Brasileirão

> Research date: 2026-04-03
> Sources: [listed at the end]

## Context

StatsBomb open data does not cover the Brasileirão. We need an alternative source with current-season data and granularity compatible with our schema (per-player per-match metrics, xG, passes, duels, pressure, positions).

## Findings

### API comparison

| API | Brasileirão | Current season | xG | Per-player/per-match data | Individual events | Price/month |
|-----|-------------|----------------|-----|---------------------------|-------------------|-------------|
| **API-Football** | Yes | Yes | Partial (team, not per shot) | Yes (shots, passes, tackles, dribbles, rating) | No (aggregated per player) | Free (100 req/day) / $19+ |
| **Sportmonks** | Yes | Yes | Yes (add-on, per team and lineup) | Yes | Partial | 14-day trial / paid |
| **Sofascore** (scraping) | Yes | Yes | Yes (shotmap with xG per shot) | Yes (heatmap, stats) | Yes (shotmap, heatmap coords) | Free (unofficial) |
| **FootyStats** | Yes | Yes | Yes (per team) | No (aggregated) | No | $36/month |
| **Understat** | **No** | - | - | - | - | - |
| **FBref** (scraping) | Yes | Yes | Yes (per player/season) | Partial | No | Free (scraping) |

### 1. API-Football (api-sports.io) — Best value for money

**Strengths:**

- 1200+ competitions including Brasileirão Série A/B
- `/fixtures/players` endpoint returns per player per match:
  - `shots.total`, `shots.on` (shots)
  - `goals.total`, `goals.assists`
  - `passes.total`, `passes.key`, `passes.accuracy`
  - `tackles.total`, `tackles.blocks`, `tackles.interceptions`
  - `duels.total`, `duels.won`
  - `dribbles.attempts`, `dribbles.success`
  - `fouls.drawn`, `fouls.committed`
  - `cards.yellow`, `cards.red`
  - `rating` (match rating)
  - `minutes`, `position`
- Updated every minute during live matches
- Lineups with positions

**Limitations:**

- **No per-shot xG** — only team-level aggregated in a separate endpoint
- **No sequential events** (no individual passes/carries)
- **No pressures/carries** like StatsBomb has
- No heatmap/action coordinates

**Pricing:**

- Free: 100 requests/day, current season only
- Pro: $19/month, 7,500 req/day
- Ultra: $49/month, 25,000 req/day
- Mega: $99/month, 75,000 req/day

**Compatibility with our schema:** ~70%. Covers most of the metrics in `player_match_metrics` but lacks per-shot xG, pressures, carries, progressive passes/carries. Would need to adapt the pipeline.

### 2. Sofascore (via ScraperFC / unofficial API) — More data, less stable

**Strengths:**

- **Shotmap with xG per shot** (x,y coordinates + xG value)
- **Heatmap** with positioning coordinates
- Detailed player stats per match
- Covers Brasileirão with complete data
- Python libraries `ScraperFC` (pip install) or `soccerdata`
- Free

**Limitations:**

- **Unofficial API** — can break at any time
- Aggressive rate limiting
- No stability guarantee
- No complete sequential events (individual passes, carries)
- Scraping may violate ToS

**Compatibility with our schema:** ~60%. Shotmap is excellent for xG, but lacks granularity of individual passes, pressures, and carries.

### 3. Sportmonks — More complete, more expensive

**Strengths:**

- xG per team and per lineup (more granular than API-Football)
- Brasileirão Série A covered
- Stable and well-documented API
- 14-day trial

**Limitations:**

- Price not publicly disclosed (enterprise)
- No StatsBomb-style individual events
- Requires trial evaluation to confirm granularity

### 4. FBref (scraping via soccerdata) — Free advanced data

**Strengths:**

- xG, xA, progressive passes/carries per player
- Brasileirão covered
- Python library `soccerdata` (pip install)
- **Data powered by StatsBomb/Opta** — high quality

**Limitations:**

- Data **per season**, not per individual match
- Scraping — subject to blocking
- No individual events
- Delayed updates

## Recommendation

### Best fit: **API-Football + Sofascore shotmap**

The combination gives the best coverage:

1. **API-Football** ($19/month) as the main source:
   - Per-player per-match metrics (shots, passes, tackles, dribbles, duels)
   - Lineups with positions
   - Real-time updates
   - Stable and documented API

2. **Sofascore shotmap** (free, via ScraperFC) as a complement:
   - Per-shot xG (which API-Football does not have)
   - Coordinates for heatmaps

### API-Football → our schema mapping

| Our field | API-Football field | Status |
|-----------|--------------------|--------|
| goals | goals.total | OK |
| assists | goals.assists | OK |
| shots | shots.total | OK |
| shots_on_target | shots.on | OK |
| xg | **Not available per player** | Requires Sofascore |
| passes | passes.total | OK |
| passes_completed | passes.total * passes.accuracy/100 | Derivable |
| key_passes | passes.key | OK |
| tackles | tackles.total | OK |
| interceptions | tackles.interceptions | OK |
| blocks | tackles.blocks | OK |
| dribbles_attempted | dribbles.attempts | OK |
| dribbles_completed | dribbles.success | OK |
| fouls_committed | fouls.committed | OK |
| fouls_won | fouls.drawn | OK |
| minutes_played | games.minutes | OK |
| progressive_passes | **Not available** | Requires StatsBomb/FBref |
| progressive_carries | **Not available** | Requires StatsBomb/FBref |
| carries | **Not available** | Requires StatsBomb/FBref |
| pressures | **Not available** | Requires StatsBomb/FBref |
| touches | **Not available** | Partial (FBref) |
| aerials_won/lost | duels.won / duels.total | Partial |
| position | games.position | OK |

**Coverage: ~18 out of 30 base metrics directly mappable.** Advanced metrics (pressures, carries, progressive actions) do not exist outside of StatsBomb/Opta.

## Implications for Football Moneyball

### What would need to change

1. **New `data_providers/` module** with an abstract interface — StatsBomb and API-Football as implementations
2. **Adapter pattern**: convert API-Football responses to our `PlayerMatchMetrics` schema
3. **Degraded metrics**: features like pressing, carries and progressive actions would be NULL for Brasileirão data
4. **xT not computable**: without sequential events with coordinates, xT becomes unavailable
5. **RAPM works**: stints can be reconstructed via lineups + API-Football events

### Estimated cost

- API-Football Pro: $19/month (enough for ~25 matches/day)
- For a full Brasileirão season (~380 matches): ~3,800 requests = 1 day of quota

## Sources

- [API-Football](https://www.api-football.com/)
- [API-Football Documentation](https://api-sports.io/documentation/football/v3)
- [API-Football Pricing](https://www.api-football.com/pricing)
- [Sportmonks Brazilian Serie A API](https://www.sportmonks.com/football-api/serie-a-api-brazil/)
- [Sportmonks xG Data](https://www.sportmonks.com/football-api/xg-data/)
- [FootyStats Brazil Serie A xG](https://footystats.org/brazil/serie-a/xg)
- [FootyStats API](https://footystats.org/api)
- [ScraperFC — Python scraper](https://github.com/oseymour/ScraperFC)
- [soccerdata — Python scraper](https://github.com/probberechts/soccerdata)
- [Sofascore Corporate](https://corporate.sofascore.com/widgets)
- [FBref Serie A Stats](https://fbref.com/en/comps/24/Serie-A-Stats)
