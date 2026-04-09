<div align="center">

# Football Moneyball

### Analytics engine, probabilistic predictor, and value bet finder for Brazilian football

<br>

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![CatBoost](https://img.shields.io/badge/CatBoost-MultiClass-FFB800?style=for-the-badge&logo=catboost&logoColor=black)](https://catboost.ai)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![Status](https://img.shields.io/badge/status-archived-red?style=for-the-badge)]()

<br>

**CLI + REST API** that combines StatsBomb Open Data, Sofascore, and market odds (The Odds API / Betfair Exchange) to extract ~45 player metrics, predict Brasileirão match outcomes via **CatBoost + Pi-Rating + Dixon-Coles**, and identify bets with positive edge using **Kelly sizing**.

[Architecture](#architecture) · [Stack](#stack) · [Quickstart](#quickstart) · [CLI](#cli) · [Status](#status)

</div>

---

## About

`football-moneyball` is an end-to-end football analytics project built with **hexagonal architecture** (ports & adapters). The domain layer is 100% pure — zero infrastructure dependencies — and testable without mocks. It was developed iteratively across 15+ versions, covering everything from multi-provider ingestion to Bayesian probability calibration and value bet detection across every Betfair market (1x2, Asian Handicap, Correct Score, BTTS, Over/Under, Corners, Cards).

The project blends two philosophies:

- **Analytics** — StatsBomb metrics, xT (Expected Threat), PPDA, pass networks, RAPM, player embeddings via PCA + pgvector
- **Prediction & Betting** — CatBoost 1x2, Dixon-Coles, Bivariate Poisson, Pi-Rating, Platt/Isotonic/Temperature calibration, fractional Kelly

---

## Highlights

| Layer | Highlights |
|---|---|
| **Prediction model** | End-to-end CatBoost MultiClass with 43 features (Pi-Rating diff, EMA form, rolling xG, coach profile, standings, devigged market) and temporal CV |
| **Multi-market** | Monte Carlo Dixon-Coles simulating 10k scorelines per match → extracts 1x2, Correct Score, Asian Handicap, O/U, BTTS, Corners, Cards, Player Props |
| **Calibration** | Auto-selection between Platt, Isotonic, and Temperature scaling via time-split CV — ECE dropped from 0.060 to 0.028 |
| **Value detection** | Edge filter + fractional Kelly over Betfair odds with symmetric dedup and 26% draw floor |
| **Analytics** | xT (Markov), VAEP, PPDA, counter-pressing, pass graph with centrality, Ridge RAPM, PCA embeddings → position-wise archetypes |
| **Vector search** | HNSW index on pgvector (`vector(16)` cosine distance) for "find players similar to X" queries |
| **Architecture** | 37 pure-domain modules, 4 port protocols, 6 adapters, 18 use cases — domain doesn't import `statsbombpy`, `sqlalchemy`, `matplotlib`, or `fastapi` |
| **Observability** | Immutable track record with automatic prediction resolution, walk-forward backtest, Brier + RPS + ECE metrics |

---

## Architecture

Classic hexagonal architecture. Dependencies always point inward — the domain has no knowledge of databases, external APIs, or frontends.

```
┌──────────────────────────────────────────────────────────┐
│  Adapters IN                                             │
│  ├─ cli.py      Typer + Rich — 22 commands               │
│  └─ api.py      FastAPI — read-only endpoints            │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Use Cases (18)                                          │
│  ingest, train, predict, value-bets, backtest, resolve,  │
│  fit-calibration, find-similar, scout-report, ...        │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Domain (37 modules) — Pure logic                        │
│                                                          │
│  Prediction        Calibration       Features           │
│  ├ match_predictor ├ calibration     ├ context_features │
│  ├ catboost_pred   ├ dixon_coles     ├ h2h_features     │
│  ├ pi_rating       └ temperature     └ market_features  │
│  ├ elo                                                   │
│  └ ml_lambda       Multi-market      Metrics            │
│                    ├ markets         ├ metrics          │
│  Betting           ├ multi_mc        ├ pressing         │
│  ├ value_detector  ├ corners_pred    ├ possession_value │
│  └ bankroll        ├ cards_pred      ├ network          │
│                    ├ shots_pred      ├ embeddings       │
│                    └ player_props    └ rapm             │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Ports (Protocols)                                       │
│  ├─ DataProvider    ├─ OddsProvider                      │
│  └─ MatchRepository └─ Visualizer                        │
└────────────────────────────┬─────────────────────────────┘
                             ▲
┌──────────────────────────────────────────────────────────┐
│  Adapters OUT                                            │
│  ├─ statsbomb_provider   (StatsBomb Open Data)           │
│  ├─ sofascore_provider   (Brasileirão scraping)          │
│  ├─ odds_provider        (The Odds API + PG cache)       │
│  ├─ postgres_repository  (SQLAlchemy + pgvector)         │
│  ├─ orm                  (20 tables)                     │
│  └─ matplotlib_viz       (mpl + mplsoccer, dark theme)   │
└──────────────────────────────────────────────────────────┘
```

See [`docs/architecture/overview.md`](docs/architecture/overview.md) for the complete map (data flow, PG schema, k8s).

---

## Stack

<table>
<tr>
<td width="33%" valign="top">

**Runtime**
- Python 3.12+
- Typer + Rich (CLI)
- FastAPI + uvicorn (API)
- React + Vite + Tailwind (frontend)

</td>
<td width="33%" valign="top">

**Data & ML**
- NumPy / pandas / SciPy
- scikit-learn (PCA, KMeans, Ridge)
- CatBoost MultiClass
- networkx (graphs)
- mplsoccer (pitch viz)

</td>
<td width="33%" valign="top">

**Infra**
- PostgreSQL 16 + pgvector
- SQLAlchemy 2.x
- Kubernetes + Kustomize
- Minikube (dev)
- Docker + `./deploy.sh`

</td>
</tr>
</table>

**Data providers:** StatsBomb Open Data (`statsbombpy`), Sofascore (Brasileirão), The Odds API, Betfair Exchange (`betfairlightweight`).

---

## Layout

```
moneyball/
├── football_moneyball/
│   ├── domain/          37 modules   Pure logic, zero infra deps
│   ├── ports/            4 protocols DataProvider, OddsProvider, Repository, Visualizer
│   ├── adapters/         6 adapters  StatsBomb, Sofascore, Odds, PG, ORM, matplotlib
│   ├── use_cases/       18 use cases Orchestration
│   ├── cli.py           22 commands  Typer + Rich
│   ├── api.py                        FastAPI read-only
│   └── config.py                     DI: get_provider, get_repository, get_odds_provider
│
├── frontend/                         React + Vite dashboard
├── k8s/                              Kustomize (postgres + app + frontend + 3 cronjobs)
├── tests/                            pytest — domain tests = zero mocks
├── docs/
│   ├── architecture/                 System view
│   ├── pitches/                      Active feature proposals
│   ├── postmortem/                   19 shipped features with retrospectives
│   ├── research/                     16 research docs (methods, benchmarks, datasets)
│   └── roadmap/central.md
│
└── pyproject.toml
```

---

## Database

**20 tables** on PostgreSQL 16, schema defined in parallel in the SQLAlchemy ORM ([`orm.py`](football_moneyball/adapters/orm.py)) and in the init SQL ([`k8s/configmap.yaml`](k8s/configmap.yaml)):

<table>
<tr>
<td valign="top">

**Core analytics**
- `matches`
- `player_match_metrics`
- `pass_networks`
- `player_embeddings` *(vector(16))*
- `stints`
- `action_values`
- `pressing_metrics`

</td>
<td valign="top">

**Prediction**
- `match_predictions`
- `prediction_history`
- `match_stats`

**Betting**
- `match_odds`
- `value_bets`
- `value_bet_history`
- `backtest_results`

</td>
<td valign="top">

**Context**
- `referee_stats`
- `team_coaches`
- `player_injuries`
- `league_standings`
- `match_lineups`

</td>
</tr>
</table>

**pgvector HNSW** on `player_embeddings.embedding` with `vector_cosine_ops` — similarity search is delegated to PG, not computed in Python.

---

## Quickstart

```bash
# 1. Clone + install in editable mode
git clone <repo> moneyball && cd moneyball
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Bring up PostgreSQL + app on Minikube
kubectl apply -k k8s/
kubectl port-forward -n football-moneyball svc/postgres 5432:5432 &

# 3. Ingest the season data
moneyball ingest --competition "Brasileirão Série A" --season 2026
moneyball ingest-context
moneyball ingest-lineups
moneyball snapshot-odds

# 4. Train the prediction model + calibration
moneyball train-catboost
moneyball fit-calibration --method auto

# 5. Generate predictions and detect value bets
moneyball predict-all
moneyball value-bets --min-edge 0.05

# 6. Run the React dashboard
cd frontend && npm install && npm run dev
# In another terminal:
uvicorn football_moneyball.api:app --reload
```

---

## CLI

`football-moneyball` exposes 22 commands grouped by domain:

<table>
<tr>
<td valign="top">

**Analytics**
```
list-competitions
list-matches
analyze-match
analyze-season
compare-players
find-similar
recommend
scout-report
```

</td>
<td valign="top">

**Ingest**
```
ingest
ingest-context
ingest-lineups
snapshot-odds
```

**Training**
```
train-models
train-catboost
fit-calibration
```

</td>
<td valign="top">

**Prediction**
```
predict
predict-all
value-bets
backtest
verify
resolve
track-record
```

</td>
</tr>
</table>

Examples:

```bash
# Vector similarity search
moneyball find-similar "Pedro" --season 2025 --limit 10

# Full scout report as markdown
moneyball scout-report "Endrick" --output endrick.md

# Single-match prediction
moneyball predict "Palmeiras" "Flamengo"

# Round value bets with minimum 5% edge
moneyball value-bets --min-edge 0.05

# Walk-forward backtest for the season
moneyball backtest --season 2025
```

---

## Metrics & Results

Progressive backtest results across versions (`docs/postmortem/`):

| Version | Contribution | Metric |
|---|---|---|
| v1.9.0 — Dixon-Coles + Platt | Low-score correction | Brier **0.2437** |
| v1.11.0 — Isotonic/Temperature auto-select | Non-parametric calibration | ECE **0.060 → 0.028** |
| v1.12.0 — Bivariate Poisson + 2022-2025 backfill | 1610 matches trained | n = 409 → 1610 |
| v1.13.0 — Market blend + draw floor | Pinnacle × model blending | Brier improves 9% |
| v1.14.0 — CatBoost 1x2 + Pi-Rating | 1x2 engine replacement | RPS **< 0.22** |
| v1.15.0 — Context features | xG form EMA, coach profile, standings | FEATURE_DIM 24 → 43 |

See [`docs/roadmap/central.md`](docs/roadmap/central.md) and each postmortem under [`docs/postmortem/`](docs/postmortem/) for detailed metrics and reliability diagrams.

---

## Development

```bash
# Domain purity check (must return empty)
grep -r "from statsbombpy\|from sqlalchemy\|import matplotlib\|from fastapi\|from typer" football_moneyball/domain/

# Lint
python3 -m ruff check football_moneyball/

# Tests
pytest tests/ -v

# Deploy on Minikube
./deploy.sh 1.15.0
./connect.sh
```

### Contribution workflow

Every new feature follows:

1. **Pitch** in `docs/pitches/<feature>.md` — problem, solution, architecture, scope, testing, success criteria
2. **Research** in `docs/research/` when needed — papers, benchmarks, datasets
3. **Plan mode** — implementation plan
4. **Implement** following the plan, with domain-level unit tests
5. **Postmortem** in `docs/postmortem/` after shipping — metrics, decisions, next steps

Inspired by Basecamp's [Shape Up](https://basecamp.com/shapeup).

---

## Status

**Archived.** This project was built as a personal exploration of football analytics, predictive modeling, and hexagonal architecture in Python. It reached v1.15.0 with a functional end-to-end pipeline but is no longer under active development.

The codebase still serves as a reference for:
- Hexagonal architecture applied to a data science domain
- ML pipelines in production (ingest → train → calibrate → predict → value)
- Multi-provider integration with caching and fallback
- Iterative documentation via pitches + postmortems

Continuation ideas are documented under [`docs/pitches/`](docs/pitches/) (v1.16 monitored calibration, v1.17 hyperopt with Optuna + SHAP).

---

## Credits

Data:
- [StatsBomb Open Data](https://github.com/statsbomb/open-data) — events, lineups, 360 data
- [Sofascore](https://www.sofascore.com) — Brasileirão Série A
- [The Odds API](https://the-odds-api.com) — bookmaker odds
- [Betfair Exchange](https://www.betfair.com) — exchange odds via `betfairlightweight`

Methods:
- **Dixon-Coles** (1997) — low-score correction for Poisson models
- **xT** — Karun Singh (2018)
- **VAEP** — KU Leuven / Tom Decroos
- **RAPM** — Jeremias Engelmann / adapted from basketball
- **Pi-Rating** — Constantinou & Fenton (2013)
- **Shape Up** — Basecamp (pitch workflow)
