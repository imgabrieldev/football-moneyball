---
tags:
  - pitch
  - architecture
  - clean-code
  - refactor
  - hexagonal
---

# Pitch — Hexagonal Architecture & Clean Code (v0.3.0)

## Problem

The codebase grew from 7 modules (v0.1.0) to 11 modules + scripts + tests (v0.2.0), and has structural problems that will scale poorly:

1. **Coupling with external providers** — `player_metrics.py`, `pressing.py`, `network_analysis.py`, `rapm.py` and `possession_value.py` all import `statsbombpy` directly. `scripts/ingest_sofascore.py` uses `requests` + Sofascore API. Adding a new provider (API-Football, Opta) requires duplicating domain logic.

2. **Coupling with infrastructure** — `player_embeddings.py`, `rapm.py` and `export.py` import SQLAlchemy directly and make raw SQL queries. Domain logic (PCA, clustering, Ridge regression) is mixed with persistence logic (session, upsert, text queries).

3. **Modules with multiple responsibilities** — `player_metrics.py` does StatsBomb data extraction + metric computation + position extraction. `cli.py` does orchestration + formatting + business logic (xG contribution, aggregation).

4. **No dependency inversion** — High-level modules (analysis, embeddings) depend directly on low-level modules (db, statsbombpy). There are no interfaces/protocols between layers.

5. **Limited testability** — Current tests mock `sb.events()` because there's no abstraction over the provider. Integration tests require real PostgreSQL with no in-memory alternative.

6. **Loose Sofascore script** — `scripts/ingest_sofascore.py` duplicates conversion and persistence logic that already exists in the main modules, without sharing interfaces.

Research: [[market-standard-analytics]], [[socceraction-statsbomb-api]], [[brasileirao-data-sources]]

## Solution

Refactor to **Hexagonal Architecture (Ports & Adapters)** with 3 clear layers:

```
┌─────────────────────────────────────────────────┐
│                  Adapters (IN)                   │
│  CLI (typer)  │  Scripts  │  API (future)        │
├─────────────────────────────────────────────────┤
│                  Use Cases                       │
│  analyze_match │ analyze_season │ compare │ ...  │
├─────────────────────────────────────────────────┤
│                   Domain                         │
│  metrics │ pressing │ xT │ embeddings │ rapm     │
│  models (Player, Match, Team, Action)            │
├─────────────────────────────────────────────────┤
│                  Ports (interfaces)               │
│  DataProvider │ Repository │ Visualizer           │
├─────────────────────────────────────────────────┤
│                 Adapters (OUT)                    │
│  StatsBomb │ Sofascore │ PostgreSQL │ Matplotlib │
└─────────────────────────────────────────────────┘
```

**Principles:**
- Domain layer doesn't import anything external (no pandas, no sqlalchemy, no statsbombpy)
- Domain receives and returns pure dataclasses/dicts
- Ports define interfaces (Protocol classes) for data access and persistence
- Adapters implement the ports (StatsBomb, Sofascore, PostgreSQL, matplotlib)
- Use cases orchestrate domain + ports
- Dependency Injection via constructor in use cases

## Architecture

### Proposed directory structure

```
football_moneyball/
├── domain/                    # Pure business logic — ZERO external dependencies
│   ├── models.py              # Dataclasses: Player, Match, MatchMetrics, ActionValue, etc.
│   ├── metrics.py             # Metric computation from events (receives lists/dicts)
│   ├── pressing.py            # PPDA, success rate, counter-pressing (pure logic)
│   ├── possession_value.py    # xT model (numpy only — it's pure math)
│   ├── embeddings.py          # PCA, clustering, archetypes (sklearn)
│   ├── rapm.py                # Ridge regression, splints, SPM prior (sklearn + numpy)
│   ├── network.py             # Pass graph, centrality (networkx)
│   └── constants.py           # POSITION_GROUP_MAP, grid sizes, thresholds
│
├── ports/                     # Interfaces (Protocol classes)
│   ├── data_provider.py       # Protocol: get_events(), get_lineups(), get_competitions()
│   ├── repository.py          # Protocol: save_match(), get_player_metrics(), find_similar()
│   └── visualizer.py          # Protocol: plot_pass_network(), plot_radar(), etc.
│
├── adapters/                  # Port implementations
│   ├── statsbomb_provider.py  # Implements DataProvider via statsbombpy
│   ├── sofascore_provider.py  # Implements DataProvider via Sofascore API
│   ├── postgres_repository.py # Implements Repository via SQLAlchemy + pgvector
│   ├── orm.py                 # ORM models (SQLAlchemy) — isolated from domain
│   └── matplotlib_viz.py     # Implements Visualizer via matplotlib + mplsoccer
│
├── use_cases/                 # Orchestration — connects domain + ports
│   ├── analyze_match.py       # UseCase: extract, compute, persist, display
│   ├── analyze_season.py      # UseCase: process full season
│   ├── compare_players.py     # UseCase: compare two players
│   ├── find_similar.py        # UseCase: similarity search
│   ├── generate_report.py     # UseCase: scout report
│   └── ingest_external.py     # UseCase: external source ingestion (Sofascore)
│
├── cli.py                     # Adapter IN: Typer CLI (thin layer → use cases)
└── config.py                  # DATABASE_URL, feature flags, provider selection
```

### Affected modules

**ALL modules will be refactored.** old → new mapping:

| Current module | Destination | What changes |
|---|---|---|
| `db.py` (ORM + queries + session) | `adapters/orm.py` + `adapters/postgres_repository.py` | Split: ORM models separated from query logic |
| `player_metrics.py` (StatsBomb + computations) | `domain/metrics.py` + `adapters/statsbomb_provider.py` | Split: data extraction vs metric computation |
| `pressing.py` (StatsBomb + computations) | `domain/pressing.py` + provider | Remove statsbombpy import |
| `possession_value.py` (StatsBomb + xT) | `domain/possession_value.py` + provider | xT becomes pure math, data comes via port |
| `network_analysis.py` (StatsBomb + networkx) | `domain/network.py` + provider | Graph receives data, doesn't fetch |
| `player_embeddings.py` (sklearn + SQLAlchemy) | `domain/embeddings.py` + repository | Remove SQLAlchemy, receive data via port |
| `rapm.py` (StatsBomb + sklearn + SQLAlchemy) | `domain/rapm.py` + provider + repository | Remove both, receive data via port |
| `viz.py` | `adapters/matplotlib_viz.py` | Rename, implement port |
| `export.py` (SQLAlchemy + formatting) | `use_cases/generate_report.py` + repository | Report logic via use case |
| `cli.py` (everything) | `cli.py` (thin) + `use_cases/*.py` | CLI becomes thin: parses args → calls use case |
| `scripts/ingest_sofascore.py` | `adapters/sofascore_provider.py` + `use_cases/ingest_external.py` | Provider implements interface, use case orchestrates |

### Domain Models (`domain/models.py`)

```python
from dataclasses import dataclass, field

@dataclass
class Player:
    player_id: int
    name: str
    team: str
    position_group: str = "MID"  # GK, DEF, MID, FWD

@dataclass
class MatchInfo:
    match_id: int
    competition: str
    season: str
    match_date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int

@dataclass
class PlayerMatchMetrics:
    player: Player
    match_id: int
    minutes_played: float = 0.0
    goals: int = 0
    assists: int = 0
    xg: float = 0.0
    xa: float = 0.0
    # ... all ~45 fields as typed attributes
    
@dataclass
class ActionValue:
    event_index: int
    player: Player
    action_type: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    xt_value: float | None = None
    vaep_value: float | None = None

@dataclass
class PressingProfile:
    team: str
    ppda: float
    pressing_success_rate: float
    counter_pressing_fraction: float
    high_turnovers: int
    shot_ending_high_turnovers: int
    zones: list[float] = field(default_factory=lambda: [0.0] * 6)
```

### Ports (`ports/data_provider.py`)

```python
from typing import Protocol

class DataProvider(Protocol):
    """Interface for football data sources."""
    
    def get_match_events(self, match_id: int) -> list[dict]:
        """Return events of a match in normalized format."""
        ...
    
    def get_lineups(self, match_id: int) -> dict[str, list[dict]]:
        """Return lineups with positions."""
        ...
    
    def get_competitions(self) -> list[dict]:
        """List available competitions."""
        ...
    
    def get_matches(self, competition_id: int, season_id: int) -> list[dict]:
        """List matches of a competition/season."""
        ...
```

```python
class Repository(Protocol):
    """Interface for data persistence."""
    
    def save_match(self, match: MatchInfo) -> None: ...
    def save_player_metrics(self, metrics: list[PlayerMatchMetrics], match_id: int) -> None: ...
    def get_player_metrics(self, player_name: str, season: str | None) -> list[PlayerMatchMetrics]: ...
    def find_similar_players(self, player_name: str, season: str, limit: int) -> list[dict]: ...
    # ...
```

### Schema

**No changes in PostgreSQL.** Tables and columns stay identical — only the code that interacts with them moves (from `db.py` to `adapters/postgres_repository.py`).

### Infra (K8s)

No changes. Kubernetes manifests stay the same.

## Scope

### In Scope

- [ ] Create `domain/models.py` with dataclasses for all entities
- [ ] Create `domain/constants.py` with POSITION_GROUP_MAP, thresholds, grid sizes
- [ ] Create `ports/data_provider.py` with DataProvider Protocol
- [ ] Create `ports/repository.py` with Repository Protocol
- [ ] Create `ports/visualizer.py` with Visualizer Protocol
- [ ] Create `adapters/statsbomb_provider.py` — implements DataProvider
- [ ] Create `adapters/sofascore_provider.py` — implements DataProvider (migrate from scripts/)
- [ ] Create `adapters/orm.py` — move ORM models from db.py
- [ ] Create `adapters/postgres_repository.py` — implements Repository
- [ ] Create `adapters/matplotlib_viz.py` — implements Visualizer (migrate from viz.py)
- [ ] Refactor `domain/metrics.py` — remove statsbombpy, receive events as list[dict]
- [ ] Refactor `domain/pressing.py` — remove statsbombpy
- [ ] Refactor `domain/possession_value.py` — remove statsbombpy
- [ ] Refactor `domain/network.py` — remove statsbombpy
- [ ] Refactor `domain/embeddings.py` — remove SQLAlchemy
- [ ] Refactor `domain/rapm.py` — remove statsbombpy and SQLAlchemy
- [ ] Create use cases: analyze_match, analyze_season, compare_players, find_similar, generate_report, ingest_external
- [ ] Refactor `cli.py` — thin layer that parses args and calls use cases
- [ ] Create `config.py` — centralized configuration with DI
- [ ] Migrate tests to new structure
- [ ] Keep 100% backward compatibility in CLI commands
- [ ] `pyproject.toml` — update package discovery

### Out of Scope

- New features or metrics (this is pure refactor)
- PostgreSQL schema changes
- K8s manifest changes
- New CLI commands
- Performance optimization
- Async/await
- REST API (will be a separate pitch when there's a frontend)

## Research Needed

- [ ] Validate that Python Protocol classes work as interfaces for DI without a framework
- [ ] Confirm that sklearn/numpy are acceptable in the domain layer (they are math libraries, not infra)

## Testing Strategy

### Unit (domain layer)
- `domain/metrics.py` tests with pure dicts (no StatsBomb mock)
- `domain/pressing.py` tests with fabricated events (no mock)
- `domain/possession_value.py` tests — xT with numpy arrays
- `domain/embeddings.py` tests — PCA/clustering with synthetic DataFrames
- `domain/rapm.py` tests — Ridge with numpy matrices

### Unit (adapters)
- `adapters/statsbomb_provider.py` — mock statsbombpy, verify conversion
- `adapters/sofascore_provider.py` — mock requests, verify conversion
- `adapters/postgres_repository.py` — mock session, verify queries

### Integration
- Use cases with mocked ports
- End-to-end: CLI → use case → domain → adapters (with test PG)

### Regression
- All 42 existing tests must keep passing (adapted)
- CLI commands produce the same outputs

## Success Criteria

- [ ] `domain/` does not import statsbombpy, sqlalchemy, requests, typer, rich, matplotlib
- [ ] `domain/` can be tested without a database and without an external API
- [ ] Adding a new DataProvider (e.g., API-Football) requires only 1 new file in `adapters/`
- [ ] All 42+ tests pass
- [ ] All 8 CLI commands work identically
- [ ] `python3 -m py_compile` passes on all modules
- [ ] No changes to PostgreSQL schema
- [ ] No changes to K8s manifests
