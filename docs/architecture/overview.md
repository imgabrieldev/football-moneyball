---
tags:
  - architecture
  - overview
  - hexagonal
---

# Arquitetura — Football Moneyball

## Visão Geral

CLI + API REST de analytics e betting value finder pro futebol (foco Brasileirão). Combina StatsBomb Open Data + Sofascore + The Odds API, persistência em PostgreSQL + pgvector, modelos probabilísticos (Dixon-Coles, Bivariate Poisson, CatBoost 1x2, Pi-Rating) e detecção de value bets via Kelly.

Arquitetura hexagonal (ports & adapters) — domínio puro sem dependências de infra.

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12+ |
| CLI | Typer + Rich |
| API REST | FastAPI + uvicorn |
| Frontend | React + Vite + Tailwind (diretório `frontend/`) |
| Banco | PostgreSQL 16 + pgvector |
| ORM | SQLAlchemy 2.x |
| ML | scikit-learn, CatBoost (lazy import), NumPy, pandas |
| Grafos | networkx |
| Viz | matplotlib + mplsoccer |
| Dados — eventos | statsbombpy (StatsBomb Open Data) |
| Dados — ligas BR | Sofascore API |
| Dados — odds | The Odds API + Betfair Exchange (betfairlightweight) |
| Infra | Minikube + Kustomize (sem Helm) |
| Observabilidade | `print`/Rich (sem OTel) |

## Camadas

```
┌─────────────────────────────────────────────────┐
│  Adapters IN                                    │
│  ├─ cli.py          (Typer, 22+ comandos)       │
│  └─ api.py          (FastAPI, endpoints REST)   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Use Cases (18 módulos em use_cases/)           │
│  Orquestram domain + ports                      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Domain (37 módulos em domain/)                 │
│  Lógica pura — numpy/pandas/sklearn/scipy       │
│  ZERO deps de infra                             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Ports (Protocol classes em ports/)             │
│  ├─ DataProvider      (eventos, lineups)        │
│  ├─ OddsProvider      (odds de casas)           │
│  ├─ MatchRepository   (persistência)            │
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

`config.py` implementa DI: `get_provider()`, `get_repository()`, `get_odds_provider()` — instancia adapters concretos e injeta nos use cases.

## Módulos

### `domain/` — Lógica pura

Agrupada por domínio:

| Grupo | Módulos | Responsabilidade |
|---|---|---|
| **Modelos base** | `models.py`, `constants.py` | Dataclasses (MatchInfo, PlayerMatchMetrics), thresholds, grid sizes |
| **Métricas StatsBomb** | `metrics.py`, `pressing.py`, `possession_value.py`, `network.py`, `embeddings.py`, `rapm.py` | ~45 métricas, PPDA, xT (Markov), grafo de passes, PCA+cluster, Ridge RAPM |
| **Predição 1x2** | `match_predictor.py`, `catboost_predictor.py`, `pi_rating.py`, `elo.py`, `ml_lambda.py`, `player_lambda.py` | Poisson/Dixon-Coles, CatBoost MultiClass, Pi-Rating (Constantinou), Elo, lambda adjustment |
| **Multi-market** | `markets.py`, `multi_monte_carlo.py`, `corners_predictor.py`, `cards_predictor.py`, `shots_predictor.py`, `player_props.py` | 1x2, correct score, Asian handicap, O/U, BTTS, corners, cards, shots, player props |
| **Features contextuais** | `feature_engineering.py`, `features.py`, `context_features.py`, `h2h_features.py`, `market_features.py`, `referee_features.py`, `referee.py` | EMA form, coach profile, rest days, H2H, odds features, árbitro |
| **Calibração** | `calibration.py` | Dixon-Coles τ, Platt, Isotonic, Temperature, Brier, ECE |
| **Betting** | `value_detector.py`, `bankroll.py` | Edge detection, Kelly sizing |
| **Lineups** | `lineup_prediction.py` | Previsão de escalação |
| **Track record** | `track_record.py` | Histórico imutável de predições resolvidas |

**Regra invariante:** domain não importa `statsbombpy`, `sqlalchemy`, `requests`, `matplotlib`, `fastapi`, `typer`. Só `numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx`. CatBoost é importado dentro das funções (lazy) pra manter top-level limpo.

### `ports/` — Interfaces

Protocol classes (PEP 544) — zero implementação:

- `DataProvider` — `get_events`, `get_lineups`, `list_competitions`, `list_matches`
- `OddsProvider` — `get_upcoming_odds`, `get_match_odds`, `get_historical_odds`
- `MatchRepository` — save/query matches, metrics, embeddings, stints, predictions, value bets
- `Visualizer` — `plot_*` (radar, heatmap, network, RAPM)

### `adapters/` — Implementações

| Arquivo | Port implementado | Observações |
|---|---|---|
| `statsbomb_provider.py` | DataProvider | Open data grátis via `statsbombpy` |
| `sofascore_provider.py` | DataProvider | Brasileirão (scraping leve) |
| `odds_provider.py` | OddsProvider | The Odds API (free tier) + cache PG |
| `postgres_repository.py` | MatchRepository | SQLAlchemy + pgvector, upsert helpers, similarity queries |
| `orm.py` | — | 20 classes Base SQLAlchemy (ver [#banco-de-dados](#banco-de-dados)) |
| `matplotlib_viz.py` | Visualizer | Dark theme, mplsoccer pitch |

### `use_cases/` — Orquestração

| Arquivo | CLI command | Responsabilidade |
|---|---|---|
| `analyze_match.py` | `analyze-match` | Extrai métricas de uma partida |
| `analyze_season.py` | `analyze-season` | Temporada + embeddings + RAPM |
| `compare_players.py` | `compare-players` | Comparação lado a lado |
| `find_similar.py` | `find-similar`, `recommend` | Busca vetorial pgvector |
| `generate_report.py` | `scout-report` | Scout report markdown |
| `ingest_matches.py` | `ingest` | Ingest batch multi-provider |
| `ingest_context.py` | `ingest-context` | Coach, injuries, standings |
| `ingest_lineups.py` | `ingest-lineups` | Escalações |
| `train_ml_models.py` | `train-models` | Treinos sklearn legados |
| `train_catboost.py` | `train-catboost` | CatBoost 1x2 end-to-end |
| `fit_calibration.py` | `fit-calibration` | Platt/Isotonic/Temperature auto-select |
| `predict_match.py` | `predict` | Previsão de uma partida |
| `predict_all.py` | `predict-all` | Previsão + salva em PG |
| `find_value_bets.py` | `value-bets` | Edge detection + Kelly |
| `snapshot_odds.py` | `snapshot-odds` | Snapshot de odds pra backtest |
| `backtest.py` | `backtest` | Walk-forward sobre histórico |
| `verify_predictions.py` | `verify` | Comparação predição × resultado |
| `resolve_predictions.py` | `resolve`, `track-record` | Marca predições como resolvidas, track record imutável |

### `cli.py` + `api.py` — Adapters IN

- [cli.py](football_moneyball/cli.py) — camada fina Typer, cada comando delega pra um use case. 1.6k linhas.
- [api.py](football_moneyball/api.py) — FastAPI read-only pro frontend: `/api/matches`, `/api/predictions`, `/api/value-bets`, `/api/players`, `/health`. Injeção do repo via `Depends`.

## Fluxo de Dados

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

## Banco de Dados

20 tabelas. Schema duplicado em dois lugares que **devem permanecer em sync**:

1. [football_moneyball/adapters/orm.py](football_moneyball/adapters/orm.py) — SQLAlchemy Base
2. [k8s/configmap.yaml](k8s/configmap.yaml) — init.sql executado na inicialização do container

### Tabelas

| Grupo | Tabelas |
|---|---|
| **Core analytics** | `matches`, `player_match_metrics`, `pass_networks`, `player_embeddings`, `stints`, `action_values`, `pressing_metrics` |
| **Predição** | `match_predictions`, `prediction_history`, `match_stats` |
| **Betting** | `match_odds`, `value_bets`, `value_bet_history`, `backtest_results` |
| **Contexto** | `referee_stats`, `team_coaches`, `player_injuries`, `league_standings`, `match_lineups` |

### pgvector

- `player_embeddings.embedding vector(16)` — arquetipos por posição
- Índice HNSW com `vector_cosine_ops`
- Operadores usados: `<=>` (cosine), `<->` (L2)
- Busca por similaridade delegada ao PG (não calcula no Python)

## Infra (Kubernetes)

Namespace `football-moneyball` via Kustomize puro:

```
k8s/
├── namespace.yaml              # football-moneyball
├── configmap.yaml              # init.sql (schema + pgvector extension)
├── configmap-app.yaml          # config do app
├── secret.yaml                 # POSTGRES_USER/PASSWORD/DB
├── pvc.yaml                    # 1Gi storage
├── deployment.yaml             # PostgreSQL (pgvector/pgvector:pg16)
├── service.yaml                # postgres ClusterIP:5432
├── app-deployment.yaml         # FastAPI (api.py) pod
├── app-service.yaml            # app ClusterIP:8000
├── frontend-deployment.yaml    # React + nginx
├── frontend-service.yaml       # frontend ClusterIP:80
├── cronjob-ingest.yaml         # ingest diário (StatsBomb + Sofascore)
├── cronjob-odds.yaml           # snapshot de odds (6h)
├── cronjob-predict.yaml        # predict-all antes das rodadas
└── kustomization.yaml          # aglutina tudo
```

CLI roda local apontando pro PG via `kubectl port-forward`. `DATABASE_URL` default: `postgresql://moneyball:moneyball@localhost:5432/moneyball`.

## Fluxo de Predição (end-to-end)

```
1. cronjob-ingest
   └─ ingest_matches → StatsBomb + Sofascore → PG
   └─ ingest_context → coach, injuries, standings → PG
   └─ ingest_lineups → PG

2. train-catboost (manual / periódico)
   └─ feature engineering (Pi-Rating + EMA + xG + odds + context)
   └─ temporal CV (expanding window)
   └─ CatBoost MultiClass → catboost_1x2.cbm

3. fit-calibration (manual / após train)
   └─ Platt / Isotonic / Temperature auto-select via CV
   └─ pickle com bundle de calibração

4. cronjob-predict
   └─ predict_all → CatBoost 1x2 + Poisson multi-market
   └─ Dixon-Coles correction em placares baixos
   └─ Calibração aplicada
   └─ PG (match_predictions)

5. cronjob-odds + find-value-bets
   └─ snapshot de odds (The Odds API / Betfair)
   └─ compara predicted_prob × fair_odds
   └─ edge filter + Kelly sizing → value_bets

6. frontend + api.py
   └─ React lê via /api/matches, /api/predictions, /api/value-bets
```

## Convenções

- **Código:** inglês
- **Docstrings:** português (brasileiro)
- **CLI output:** português (brasileiro)
- **Commits:** português ou inglês, descritivos
- **Métricas:** normalizadas por 90 minutos onde aplicável
- **StatsBomb:** coordenadas 120×80, gol em x=120
- **Domain puro:** zero import de statsbombpy/sqlalchemy/matplotlib/typer/fastapi
- **Schema sync:** ORM e init.sql devem mudar juntos
- **Tests:** unitários em domain = zero mocks, use cases mockam adapters via port
