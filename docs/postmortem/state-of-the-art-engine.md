---
tags:
  - pitch
  - statsbomb
  - pgvector
  - rapm
  - embeddings
  - xT
  - VAEP
  - pressing
  - viz
  - cli
  - k8s
---

# Pitch — State of the Art Analytics Engine (v0.2.0)

## Problem

The current analytics engine operates at ~60% of the market standard. It lacks the metrics and models that elite clubs (Liverpool, Man City, Brighton) and leading companies (StatsBomb/Hudl, SciSports, Opta) consider essential:

1. **No Possession Value Model** — We have neither xT nor VAEP. Every action on the pitch is evaluated only by raw count (passes, shots) or direct xG. There's no way to measure the value of a progressive midfield pass that didn't result in a shot. Clubs and companies have treated a PSV model as a minimum requirement since 2019.

2. **Pressing is a raw count** — We have `pressures` and `pressure_regains`, but no PPDA, pressing success rate, counter-pressing fraction, or high turnovers. Liverpool defines its tactical identity by pressing (PPDA 9.89); our engine can't even detect it.

3. **Similarity ignores position** — Embeddings compare goalkeepers with strikers in the same vector space. Clustering has only 6 fixed archetypes without statistical validation. The market uses embeddings separated by positional group with 12-16 roles and silhouette analysis.

4. **Simplified RAPM** — Plain Ridge with binary +1/-1 indicators. No Bayesian prior (SPM), no real offensive/defensive split, no design weights, no multi-season. State of the art (MBAPPE) solves each of these problems.

5. **Incomplete metrics** — Missing progressive receptions, shot quality (PSxG), pass breakdown by type/distance, duel win rates with context, tackle success rate.

Research: [[market-standard-analytics]]

## Solution

Complete upgrade of the analytics engine on 6 fronts, each raising the project to market standard:

### A. Expected Threat (xT) — Possession Value Model

Custom implementation of Karun Singh's model:
- **16×12** grid (192 zones) over StatsBomb pitch (120×80)
- Iterative Markov chain: `xT(x,y) = s(x,y)×g(x,y) + m(x,y)×Σ T(x,y→z,w)×xT(z,w)`
- Convergence in ~5 iterations
- Action value = `xT(destination) - xT(origin)`
- Trained on all StatsBomb open data matches for robustness
- **New module: `possession_value.py`** — separate from player_metrics to keep architecture clean

### B. VAEP — ML-Based Action Valuation

Integration via `socceraction` library (MIT license):
- Convert StatsBomb events → **SPADL** format (Standardized Player Action Description Language)
- Train 2 models (gradient boosting): P(score) and P(concede) in the next 10 events
- **VAEP(action) = ΔP(score) - ΔP(concede)**
- Features: action type, location, bodypart, result + last 3 actions as context
- Complements xT with context awareness (counter-attack > sterile possession)

### C. Pressing Metrics Suite

Complete extraction of pressing metrics from StatsBomb events:
- **PPDA** (Passes Per Defensive Action): opponent passes in the defensive/middle third ÷ defensive actions
- **Pressing success rate**: pressures resulting in recovery within ≤5s ÷ total
- **Counter-pressing fraction**: % of ball losses followed by pressure within ≤5s
- **High turnovers**: recoveries ≤40m from the opponent's goal
- **Shot-ending high turnovers**: high turnovers that generate a shot within ≤15s
- **Pressing zones**: pressure distribution across 6 horizontal pitch zones

### D. Position-Aware Embeddings & Archetypes

Rewrite of the embeddings system:
- **Positional groups**: GK, DEF (CB, FB), MID (DM, CM, AM), FWD (W, ST)
- **Separate embeddings per group**: PCA and clustering run within each group
- **12-16 context-aware archetypes** per group (no longer 6 generic):
  - DEF: Playmaking CB, Stopper, Ball-Playing FB, Attacking FB
  - MID: Deep-Lying Playmaker, Box-to-Box, Defensive Mid, Creative AM
  - FWD: Target Man, Inside Forward, Complete Forward, Poacher
- **Silhouette analysis** to determine optimal K per group
- **Explained variance** reported on PCA (% information retained)
- **Primary position** stored in the database (new `position_group` column)
- pgvector similarity automatically filters by positional group

### E. Advanced RAPM (Bayesian + Off/Def Split)

RAPM model upgrade inspired by MBAPPE:
- **Splints** instead of stints: break also at goals scored (more observations)
- **SPM Bayesian prior**: use already-computed box-score metrics (goals, assists, tackles/90) as Ridge prior `β ~ N(β_SPM, τ²)`
- **Offensive/defensive split**: duplicate player variables — separate columns for offensive and defensive impact
- **Design weights**: weights based on the player's fraction of touches/actions in the stint (not binary +1/-1)
- **Multi-season**: combine stints from multiple seasons for stability (when available)
- **Cross-validation** for λ = σ²/τ² (balance data vs. prior)

### F. Detailed Metrics

Expansion of `player_metrics.py`:
- **Progressive receptions**: receive a pass advancing ≥10 yards
- **Shot quality**: xG per shot, big chances (xG ≥ 0.3), big chances missed
- **Pass breakdown**: short (<15y), medium (15-30y), long (>30y) with success rate each
- **Pass under pressure**: passes attempted/completed under opponent pressure
- **Switches of play**: side passes >30 yards
- **Ground duel win rate**: ground duels won / total
- **Tackle success rate**: successful tackles / total
- **Duel zones**: duel distribution by pitch third

## Architecture

### Affected modules

| Module | Action | Description |
|--------|------|-----------|
| **`possession_value.py`** | **NEW** | xT (custom) + VAEP (via socceraction). Trains models, values actions, persists to DB |
| **`pressing.py`** | **NEW** | PPDA, pressing success, counter-pressing, high turnovers, zones. Operates on raw events |
| `player_metrics.py` | MODIFY | Add ~15 metrics (progressive receptions, shot quality, pass breakdown, duel detail) |
| `player_embeddings.py` | REWRITE | Position-aware embeddings, positional groups, expanded archetypes, silhouette analysis |
| `rapm.py` | REWRITE | Splints, SPM prior, off/def split, design weights, multi-season |
| `db.py` | MODIFY | New ORM models (ActionValue, PressingMetrics, PlayerPosition), new columns in PlayerMatchMetrics and PlayerEmbedding |
| `viz.py` | MODIFY | New plots: xT heatmap, pressing zones, xT flow, detailed shot map |
| `export.py` | MODIFY | New scout report sections: possession value, pressing profile, position-aware percentiles |
| `cli.py` | MODIFY | New subcommands and flags; integrate new data into existing outputs |

### Dependency graph (updated)

```
cli.py (orchestrator)
  ├── db.py (data layer — imports nothing from project)
  ├── player_metrics.py (extraction — statsbombpy only)
  ├── possession_value.py  ← NEW (xT + VAEP — statsbombpy + socceraction + db)
  ├── pressing.py           ← NEW (pressing metrics — statsbombpy + db)
  ├── network_analysis.py (graphs — statsbombpy + networkx + db)
  ├── player_embeddings.py (ML — sklearn + db)
  ├── rapm.py (stats — statsbombpy + sklearn + db)
  ├── viz.py (visualization — matplotlib + mplsoccer + networkx)
  └── export.py (reports — db + networkx)
```

### Schema

#### New tables

```sql
-- Action values (xT and VAEP per event)
CREATE TABLE IF NOT EXISTS action_values (
    match_id INTEGER REFERENCES matches(match_id),
    event_index INTEGER,
    player_id INTEGER,
    player_name VARCHAR(100),
    team VARCHAR(100),
    action_type VARCHAR(50),       -- Pass, Carry, Shot, Dribble, etc.
    start_x REAL, start_y REAL,
    end_x REAL, end_y REAL,
    xt_value REAL,                 -- xT delta (end - start)
    vaep_value REAL,               -- VAEP delta (ΔP_score - ΔP_concede)
    vaep_offensive REAL,           -- ΔP_score only
    vaep_defensive REAL,           -- ΔP_concede only
    PRIMARY KEY (match_id, event_index)
);

-- Pressing metrics per team per match
CREATE TABLE IF NOT EXISTS pressing_metrics (
    match_id INTEGER REFERENCES matches(match_id),
    team VARCHAR(100),
    ppda REAL,
    pressing_success_rate REAL,
    counter_pressing_fraction REAL,
    high_turnovers INTEGER,
    shot_ending_high_turnovers INTEGER,
    pressing_zone_1 REAL,          -- % of pressures per zone (6 zones)
    pressing_zone_2 REAL,
    pressing_zone_3 REAL,
    pressing_zone_4 REAL,
    pressing_zone_5 REAL,
    pressing_zone_6 REAL,
    PRIMARY KEY (match_id, team)
);

-- Primary position of each player per season (for embeddings)
-- Added as a column in player_embeddings
```

#### New columns in existing tables

```sql
-- player_match_metrics: ~15 new columns
ALTER TABLE player_match_metrics ADD COLUMN IF NOT EXISTS
    progressive_receptions REAL,
    big_chances REAL,             -- shots with xG >= 0.3
    big_chances_missed REAL,
    passes_short REAL,            -- < 15y
    passes_short_completed REAL,
    passes_medium REAL,           -- 15-30y
    passes_medium_completed REAL,
    passes_long REAL,             -- > 30y
    passes_long_completed REAL,
    passes_under_pressure REAL,
    passes_under_pressure_completed REAL,
    switches_of_play REAL,
    ground_duels_won REAL,
    ground_duels_total REAL,
    tackle_success_rate REAL,
    xt_generated REAL,            -- sum of xT of player's actions
    vaep_generated REAL,          -- sum of VAEP of player's actions
    pressing_success_rate REAL;   -- individual

-- player_embeddings: position and expanded dimension
ALTER TABLE player_embeddings ADD COLUMN IF NOT EXISTS
    position_group VARCHAR(10);   -- GK, DEF, MID, FWD
-- Embedding dimension: 16 → 16 (kept, but separated by group)

-- stints: rename concept to splints (also break on goals)
-- Keep table and add type column
ALTER TABLE stints ADD COLUMN IF NOT EXISTS
    boundary_type VARCHAR(20);    -- substitution, goal, period_start
```

#### New indexes

```sql
CREATE INDEX IF NOT EXISTS idx_action_values_match ON action_values(match_id);
CREATE INDEX IF NOT EXISTS idx_action_values_player ON action_values(player_id);
CREATE INDEX IF NOT EXISTS idx_pressing_metrics_match ON pressing_metrics(match_id);
```

### Infra (K8s)

- Update `k8s/configmap.yaml` with new tables/columns in `init.sql`
- No changes to deployment, PVC or service
- Consider increasing PVC if data volume grows significantly with action_values (1 row per event)

### New dependencies

```toml
# pyproject.toml
[project.dependencies]
# ... existing ...
socceraction = ">=1.5"     # SPADL + xT + VAEP
xgboost = ">=2.0"          # VAEP model backend (gradient boosting)
```

## Scope

### In Scope

- [ ] **Module `possession_value.py`**: custom xT (16×12 grid, Markov chain) + VAEP (via socceraction, gradient boosting)
- [ ] **Module `pressing.py`**: PPDA, pressing success rate, counter-pressing fraction, high turnovers, shot-ending HT, zones
- [ ] **Rewrite `player_embeddings.py`**: positional groups, separate embeddings, 12-16 archetypes, silhouette analysis, explained variance
- [ ] **Rewrite `rapm.py`**: splints, SPM Bayesian prior, off/def split, design weights
- [ ] **Expand `player_metrics.py`**: +15 metrics (progressive receptions, shot quality, pass breakdown, duel detail, switches)
- [ ] **Schema update `db.py`**: action_values + pressing_metrics tables, new columns in player_match_metrics and player_embeddings
- [ ] **Schema update `k8s/configmap.yaml`**: init.sql synced with db.py
- [ ] **New viz in `viz.py`**: xT heatmap, pressing zones, detailed shot map
- [ ] **Export update `export.py`**: possession value, pressing profile, percentiles by position sections
- [ ] **CLI integration `cli.py`**: new data integrated into existing outputs
- [ ] **Unit tests** for each new/modified module
- [ ] **Integration tests** with PostgreSQL for new tables and queries

### Out of Scope

- EPV (Expected Possession Value) — requires tracking data we don't have
- OBV (On-Ball Value) — proprietary StatsBomb/Hudl model
- Football2Vec / GCN embeddings — high complexity, incremental impact
- Cross-league normalization — requires processed multi-league data
- Frontend/dashboard — will be a separate pitch
- Video annotation linking
- Market value / contract integration

## Research Needed

- [x] Market standard in football analytics — [[market-standard-analytics]]
- [ ] xT validation: compare our implementation with published reference values (Euro 2024, Premier League)
- [ ] VAEP: test socceraction with StatsBomb open data, verify version compatibility
- [ ] Pressing: map all StatsBomb pressure event fields to confirm feasibility of each metric
- [ ] Positions: define StatsBomb position → positional group mapping (GK/DEF/MID/FWD)
- [ ] RAPM priors: test SPM prior stability with open data (smaller sample than full leagues)

## Testing Strategy

### Unit (pytest + pytest-mock)

- **possession_value.py**: test xT convergence with synthetic data (4×3 grid), known values; test VAEP pipeline with mock events
- **pressing.py**: test PPDA with pass/defensive DataFrames with known counts; test counter-pressing detection with mock timestamps
- **player_metrics.py**: test each new metric in isolation with fabricated events; edge cases (0 minutes, no duels, etc.)
- **player_embeddings.py**: test clustering by positional group; silhouette score with synthetic data; verify GK doesn't appear in FWD cluster
- **rapm.py**: test splint boundary on goals; test SPM prior vs. no prior; test design weights vs. binary

### Integration (pytest + test PostgreSQL)

- New tables: INSERT/SELECT in action_values and pressing_metrics
- pgvector: similarity filtered by position_group
- Full pipeline: extract → possession_value → persist → query
- Schema sync: verify that db.py ORM and init.sql produce identical schemas

### Manual

- Run `moneyball analyze-match <match_id>` and verify xT/VAEP/pressing in outputs
- Compare xT values of a known match (e.g., Champions League final) with published values
- Verify that `find-similar` now groups by position
- Compare RAPM rankings before/after upgrade

## Success Criteria

- [ ] xT heatmap converges and produces threat surface consistent with literature (central zone near the box = highest xT)
- [ ] VAEP successfully trains on open data and values defensive actions (not just offensive)
- [ ] PPDA of known teams (e.g., Barcelona La Liga 2018/19) is within ±1.0 of published values
- [ ] Player similarity returns only players of the same position (or opt-in for cross-position)
- [ ] RAPM with SPM prior produces more stable ranking than plain RAPM (lower variance across seasons)
- [ ] All new metrics persisted in PostgreSQL and queryable via CLI
- [ ] Zero regression in existing commands (analyze-match, compare-players, etc.)
- [ ] Unit test coverage ≥ 80% in new modules
- [ ] `python3 -m py_compile football_moneyball/*.py` passes without errors
