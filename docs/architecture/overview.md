---
tags:
  - architecture
  - overview
  - hexagonal
---

# Architecture — Football Moneyball

## Overview

CLI + REST API for football analytics and betting value finding (focused on Brasileirão). Combines StatsBomb Open Data + Sofascore + The Odds API, persistence in PostgreSQL + pgvector, probabilistic models (Dixon-Coles, Bivariate Poisson, CatBoost 1x2, Pi-Rating), and value bet detection via Kelly.

Hexagonal architecture (ports & adapters) — pure domain with no infra dependencies.

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12+ |
| CLI | Typer + Rich |
| REST API | FastAPI + uvicorn |
| Frontend | React + Vite + Tailwind (`frontend/` directory) |
| Database | PostgreSQL 16 + pgvector |
| ORM | SQLAlchemy 2.x |
| ML | scikit-learn, CatBoost (lazy import), NumPy, pandas |
| Graphs | networkx |
| Viz | matplotlib + mplsoccer |
| Data — events | statsbombpy (StatsBomb Open Data) |
| Data — BR leagues | Sofascore API |
| Data — odds | The Odds API + Betfair Exchange (betfairlightweight) |
| Infra | Minikube + Kustomize (no Helm) |
| Observability | `print`/Rich (no OTel) |

## Layers

```
┌─────────────────────────────────────────────────┐
│  Adapters IN                                    │
│  ├─ cli.py          (Typer, 22+ commands)       │
│  └─ api.py          (FastAPI, REST endpoints)   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Use Cases (18 modules in use_cases/)           │
│  Orchestrate domain + ports                     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Domain (37 modules in domain/)                 │
│  Pure logic — numpy/pandas/sklearn/scipy        │
│  ZERO infra deps                                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Ports (Protocol classes in ports/)             │
│  ├─ DataProvider      (events, lineups)         │
│  ├─ OddsProvider      (bookmaker odds)          │
│  ├─ MatchRepository   (persistence)             │
│  └─ Visualizer        (plots)                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Adapters OUT (adapters/)                       │
│  ├─ statsbomb_provider.py    (StatsBomb)        │
│  ├─ sofascore_provider.py    (Sofascore)        │
│  ├─ odds_provider.py         (The Odds API)     │
│  ├─ postgres_repository.py   (PG + pgvector)    │
│  ├─ orm.py                   (SQLAlchemy)       │
│  └─ matplotlib_viz.py        (mpl + mplsoccer)  │
└─────────────────────────────────────────────────┘
```

`config.py` implements DI: `get_provider()`, `get_repository()`, `get_odds_provider()` — instantiates concrete adapters and injects them into the use cases.

## Modules

### `domain/` — Pure logic

Grouped by domain:

| Group | Modules | Responsibility |
|---|---|---|
| **Base models** | `models.py`, `constants.py` | Dataclasses (MatchInfo, PlayerMatchMetrics), thresholds, grid sizes |
| **StatsBomb metrics** | `metrics.py`, `pressing.py`, `possession_value.py`, `network.py`, `embeddings.py`, `rapm.py` | ~45 metrics, PPDA, xT (Markov), pass graph, PCA+cluster, Ridge RAPM |
| **1x2 Prediction** | `match_predictor.py`, `catboost_predictor.py`, `pi_rating.py`, `elo.py`, `ml_lambda.py`, `player_lambda.py` | Poisson/Dixon-Coles, CatBoost MultiClass, Pi-Rating (Constantinou), Elo, lambda adjustment |
| **Multi-market** | `markets.py`, `multi_monte_carlo.py`, `corners_predictor.py`, `cards_predictor.py`, `shots_predictor.py`, `player_props.py` | 1x2, correct score, Asian handicap, O/U, BTTS, corners, cards, shots, player props |
| **Contextual features** | `feature_engineering.py`, `features.py`, `context_features.py`, `h2h_features.py`, `market_features.py`, `referee_features.py`, `referee.py` | EMA form, coach profile, rest days, H2H, odds features, referee |
| **Calibration** | `calibration.py` | Dixon-Coles τ, Platt, Isotonic, Temperature, Brier, ECE |
| **Betting** | `value_detector.py`, `bankroll.py` | Edge detection, Kelly sizing |
| **Lineups** | `lineup_prediction.py` | Lineup prediction |
| **Track record** | `track_record.py` | Immutable history of resolved predictions |

**Invariant rule:** domain does not import `statsbombpy`, `sqlalchemy`, `requests`, `matplotlib`, `fastapi`, `typer`. Only `numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx`. CatBoost is imported inside functions (lazy) to keep the top level clean.

### `ports/` — Interfaces

Protocol classes (PEP 544) — zero implementation:

- `DataProvider` — `get_events`, `get_lineups`, `list_competitions`, `list_matches`
- `OddsProvider` — `get_upcoming_odds`, `get_match_odds`, `get_historical_odds`
- `MatchRepository` — save/query matches, metrics, embeddings, stints, predictions, value bets
- `Visualizer` — `plot_*` (radar, heatmap, network, RAPM)

### `adapters/` — Implementations

| File | Port implemented | Notes |
|---|---|---|
| `statsbomb_provider.py` | DataProvider | Free open data via `statsbombpy` |
| `sofascore_provider.py` | DataProvider | Brasileirão (light scraping) |
| `odds_provider.py` | OddsProvider | The Odds API (free tier) + PG cache |
| `postgres_repository.py` | MatchRepository | SQLAlchemy + pgvector, upsert helpers, similarity queries |
| `orm.py` | — | 20 SQLAlchemy Base classes (see [#database](#database)) |
| `matplotlib_viz.py` | Visualizer | Dark theme, mplsoccer pitch |

### `use_cases/` — Orchestration

| File | CLI command | Responsibility |
|---|---|---|
| `analyze_match.py` | `analyze-match` | Extracts metrics for a match |
| `analyze_season.py` | `analyze-season` | Season + embeddings + RAPM |
| `compare_players.py` | `compare-players` | Side-by-side comparison |
| `find_similar.py` | `find-similar`, `recommend` | pgvector vector search |
| `generate_report.py` | `scout-report` | Markdown scout report |
| `ingest_matches.py` | `ingest` | Multi-provider batch ingest |
| `ingest_context.py` | `ingest-context` | Coach, injuries, standings |
| `ingest_lineups.py` | `ingest-lineups` | Lineups |
| `train_ml_models.py` | `train-models` | Legacy sklearn training |
| `train_catboost.py` | `train-catboost` | CatBoost 1x2 end-to-end |
| `fit_calibration.py` | `fit-calibration` | Platt/Isotonic/Temperature auto-select |
| `predict_match.py` | `predict` | Single match prediction |
| `predict_all.py` | `predict-all` | Predict + save to PG |
| `find_value_bets.py` | `value-bets` | Edge detection + Kelly |
| `snapshot_odds.py` | `snapshot-odds` | Odds snapshot for backtest |
| `backtest.py` | `backtest` | Walk-forward over history |
| `verify_predictions.py` | `verify` | Prediction vs. result comparison |
| `resolve_predictions.py` | `resolve`, `track-record` | Marks predictions as resolved, immutable track record |

### `cli.py` + `api.py` — Adapters IN

- [cli.py](football_moneyball/cli.py) — thin Typer layer, each command delegates to a use case. 1.6k lines.
- [api.py](football_moneyball/api.py) — read-only FastAPI for the frontend: `/api/matches`, `/api/predictions`, `/api/value-bets`, `/api/players`, `/health`. Repo injected via `Depends`.

## Data Flow

```
StatsBomb ─┐
Sofascore ─┼─→ use_cases/ingest*.py ──→ PG (matches, events, context)
Odds API  ─┘                                 │
                                             ▼
                            use_cases/train_catboost.py
                            use_cases/fit_calibration.py
                                             │
                                   /data/models/*.cbm (local)
                                             │
                                             ▼
                            use_cases/predict_all.py
                                             │
                                             ▼
                                PG (match_predictions)
                                             │
                        ┌────────────────────┼────────────────────┐
                        ▼                    ▼                    ▼
          use_cases/find_value_bets    resolve_predictions    api.py
                        │                    │                    │
                        ▼                    ▼                    ▼
                   PG (value_bets)   PG (prediction_history)  frontend/
```

## Database

20 tables. Schema duplicated in two places that **must stay in sync**:

1. [football_moneyball/adapters/orm.py](football_moneyball/adapters/orm.py) — SQLAlchemy Base
2. [k8s/configmap.yaml](k8s/configmap.yaml) — init.sql executed on container startup

### Tables

| Group | Tables |
|---|---|
| **Core analytics** | `matches`, `player_match_metrics`, `pass_networks`, `player_embeddings`, `stints`, `action_values`, `pressing_metrics` |
| **Prediction** | `match_predictions`, `prediction_history`, `match_stats` |
| **Betting** | `match_odds`, `value_bets`, `value_bet_history`, `backtest_results` |
| **Context** | `referee_stats`, `team_coaches`, `player_injuries`, `league_standings`, `match_lineups` |

### pgvector

- `player_embeddings.embedding vector(16)` — archetypes by position
- HNSW index with `vector_cosine_ops`
- Operators used: `<=>` (cosine), `<->` (L2)
- Similarity search delegated to PG (not computed in Python)

## Infra (Kubernetes)

Namespace `football-moneyball` via pure Kustomize:

```
k8s/
├── namespace.yaml              # football-moneyball
├── configmap.yaml              # init.sql (schema + pgvector extension)
├── configmap-app.yaml          # app config
├── secret.yaml                 # POSTGRES_USER/PASSWORD/DB
├── pvc.yaml                    # 1Gi storage
├── deployment.yaml             # PostgreSQL (pgvector/pgvector:pg16)
├── service.yaml                # postgres ClusterIP:5432
├── app-deployment.yaml         # FastAPI (api.py) pod
├── app-service.yaml            # app ClusterIP:8000
├── frontend-deployment.yaml    # React + nginx
├── frontend-service.yaml       # frontend ClusterIP:80
├── cronjob-ingest.yaml         # daily ingest (StatsBomb + Sofascore)
├── cronjob-odds.yaml           # odds snapshot (6h)
├── cronjob-predict.yaml        # predict-all before matchdays
└── kustomization.yaml          # ties everything together
```

The CLI runs locally and points at PG via `kubectl port-forward`. Default `DATABASE_URL`: `postgresql://moneyball:moneyball@localhost:5432/moneyball`.

## Prediction Flow (end-to-end)

```
1. cronjob-ingest
   └─ ingest_matches → StatsBomb + Sofascore → PG
   └─ ingest_context → coach, injuries, standings → PG
   └─ ingest_lineups → PG

2. train-catboost (manual / periodic)
   └─ feature engineering (Pi-Rating + EMA + xG + odds + context)
   └─ temporal CV (expanding window)
   └─ CatBoost MultiClass → catboost_1x2.cbm

3. fit-calibration (manual / after train)
   └─ Platt / Isotonic / Temperature auto-select via CV
   └─ pickle with calibration bundle

4. cronjob-predict
   └─ predict_all → CatBoost 1x2 + Poisson multi-market
   └─ Dixon-Coles correction on low scores
   └─ Calibration applied
   └─ PG (match_predictions)

5. cronjob-odds + find-value-bets
   └─ odds snapshot (The Odds API / Betfair)
   └─ compares predicted_prob vs. fair_odds
   └─ edge filter + Kelly sizing → value_bets

6. frontend + api.py
   └─ React reads via /api/matches, /api/predictions, /api/value-bets
```

## Conventions

- **Code:** English
- **Docstrings:** Portuguese (Brazilian)
- **CLI output:** Portuguese (Brazilian)
- **Commits:** Portuguese or English, descriptive
- **Metrics:** normalized per 90 minutes where applicable
- **StatsBomb:** 120×80 coordinates, goal at x=120
- **Pure domain:** zero imports of statsbombpy/sqlalchemy/matplotlib/typer/fastapi
- **Schema sync:** ORM and init.sql must change together
- **Tests:** domain unit tests = zero mocks, use cases mock adapters through the port
