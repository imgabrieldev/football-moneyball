<div align="center">

# Football Moneyball

### Motor de analytics, previsão probabilística e caça a value bets no futebol brasileiro

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

**CLI + API REST** que combina StatsBomb Open Data, Sofascore e odds de mercado (The Odds API / Betfair Exchange) para extrair ~45 métricas de jogadores, prever resultados de partidas do Brasileirão via **CatBoost + Pi-Rating + Dixon-Coles** e identificar apostas com edge positivo usando **Kelly sizing**.

[Arquitetura](#arquitetura) · [Stack](#stack) · [Quickstart](#quickstart) · [CLI](#cli) · [Status](#status)

</div>

---

## Sobre

`football-moneyball` é um projeto end-to-end de analytics de futebol construído com **arquitetura hexagonal** (ports & adapters). O domínio é 100% puro — zero dependências de infra — e testável sem mocks. Foi desenvolvido iterativamente em 15+ versões, cobrindo desde ingestão multi-provider até calibração bayesiana de probabilidades e detecção de value bets em todos os mercados da Betfair (1x2, Asian Handicap, Correct Score, BTTS, Over/Under, Corners, Cards).

Combina duas filosofias:

- **Analytics** — métricas StatsBomb, xT (Expected Threat), PPDA, grafo de passes, RAPM, embeddings de jogadores via PCA + pgvector
- **Prediction & Betting** — CatBoost 1x2, Dixon-Coles, Bivariate Poisson, Pi-Rating, calibração Platt/Isotonic/Temperature, Kelly fracionário

---

## Highlights

| Camada | Destaques |
|---|---|
| **Modelo preditivo** | CatBoost MultiClass end-to-end com 43 features (Pi-Rating diff, EMA form, xG rolling, coach profile, standings, market-devig) e temporal CV |
| **Multi-market** | Monte Carlo Dixon-Coles simulando 10k placares por partida → extrai 1x2, Correct Score, Asian Handicap, O/U, BTTS, Corners, Cards, Player Props |
| **Calibração** | Auto-seleção entre Platt, Isotonic e Temperature scaling via CV time-split — reduziu ECE de 0.060 para 0.028 |
| **Value detection** | Edge filter + Kelly fracionário sobre odds Betfair com dedup simétrico e draw floor de 26% |
| **Analytics** | xT (Markov), VAEP, PPDA, counter-pressing, grafo de passes com centralidade, RAPM Ridge, embeddings PCA → arquetipos por posição |
| **Busca vetorial** | Índice HNSW no pgvector (`vector(16)` cosine distance) para "encontre jogadores similares a X" |
| **Arquitetura** | 37 módulos de domain puro, 4 protocols em ports/, 6 adapters, 18 use cases — domain não importa `statsbombpy`, `sqlalchemy`, `matplotlib`, `fastapi` |
| **Observabilidade** | Track record imutável com resolução automática de predições, backtest walk-forward, métricas Brier + RPS + ECE |

---

## Arquitetura

Arquitetura hexagonal clássica. Dependências sempre apontam pra dentro — domínio não sabe que existe um banco, uma API externa ou um frontend.

```
┌──────────────────────────────────────────────────────────┐
│  Adapters IN                                             │
│  ├─ cli.py      Typer + Rich — 22 comandos               │
│  └─ api.py      FastAPI — endpoints read-only            │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Use Cases (18)                                          │
│  ingest, train, predict, value-bets, backtest, resolve,  │
│  fit-calibration, find-similar, scout-report, ...        │
└────────────────────────────┬─────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Domain (37 módulos) — Lógica pura                       │
│                                                          │
│  Predição          Calibração        Features           │
│  ├ match_predictor ├ calibration     ├ context_features │
│  ├ catboost_pred   ├ dixon_coles     ├ h2h_features     │
│  ├ pi_rating       └ temperature     └ market_features  │
│  ├ elo                                                   │
│  └ ml_lambda       Multi-market      Métricas           │
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
│  ├─ odds_provider        (The Odds API + cache PG)       │
│  ├─ postgres_repository  (SQLAlchemy + pgvector)         │
│  ├─ orm                  (20 tabelas)                    │
│  └─ matplotlib_viz       (mpl + mplsoccer, dark theme)   │
└──────────────────────────────────────────────────────────┘
```

Ver [`docs/architecture/overview.md`](docs/architecture/overview.md) pro mapa completo (fluxo de dados, schema PG, k8s).

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

**Dados & ML**
- NumPy / pandas / SciPy
- scikit-learn (PCA, KMeans, Ridge)
- CatBoost MultiClass
- networkx (grafos)
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

**Provedores de dados:** StatsBomb Open Data (`statsbombpy`), Sofascore (Brasileirão), The Odds API, Betfair Exchange (`betfairlightweight`).

---

## Estrutura

```
moneyball/
├── football_moneyball/
│   ├── domain/          37 módulos  Lógica pura, zero deps de infra
│   ├── ports/            4 protocols DataProvider, OddsProvider, Repository, Visualizer
│   ├── adapters/         6 adapters  StatsBomb, Sofascore, Odds, PG, ORM, matplotlib
│   ├── use_cases/       18 use cases Orquestração
│   ├── cli.py           22 comandos  Typer + Rich
│   ├── api.py                        FastAPI read-only
│   └── config.py                     DI: get_provider, get_repository, get_odds_provider
│
├── frontend/                         React + Vite dashboard
├── k8s/                              Kustomize (postgres + app + frontend + 3 cronjobs)
├── tests/                            pytest — domain tests = zero mocks
├── docs/
│   ├── architecture/                 Visão de sistema
│   ├── pitches/                      Propostas de features ativas
│   ├── postmortem/                   19 features shipped com retrospectiva
│   ├── research/                     16 research docs (métodos, benchmarks, datasets)
│   └── roadmap/central.md
│
└── pyproject.toml
```

---

## Banco de dados

**20 tabelas** no PostgreSQL 16, schema definido em paralelo no ORM SQLAlchemy ([`orm.py`](football_moneyball/adapters/orm.py)) e no init SQL ([`k8s/configmap.yaml`](k8s/configmap.yaml)):

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

**Predição**
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

**Contexto**
- `referee_stats`
- `team_coaches`
- `player_injuries`
- `league_standings`
- `match_lineups`

</td>
</tr>
</table>

**pgvector HNSW** em `player_embeddings.embedding` com `vector_cosine_ops` — busca de jogadores similares é delegada ao PG, não calculada em Python.

---

## Quickstart

```bash
# 1. Clone + instalar em modo editável
git clone <repo> moneyball && cd moneyball
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Subir PostgreSQL + app no Minikube
kubectl apply -k k8s/
kubectl port-forward -n football-moneyball svc/postgres 5432:5432 &

# 3. Ingerir dados da temporada
moneyball ingest --competition "Brasileirão Série A" --season 2026
moneyball ingest-context
moneyball ingest-lineups
moneyball snapshot-odds

# 4. Treinar o modelo preditivo + calibração
moneyball train-catboost
moneyball fit-calibration --method auto

# 5. Gerar previsões e detectar value bets
moneyball predict-all
moneyball value-bets --min-edge 0.05

# 6. Subir o dashboard React
cd frontend && npm install && npm run dev
# Em outro terminal:
uvicorn football_moneyball.api:app --reload
```

---

## CLI

`football-moneyball` expõe 22 comandos agrupados por domínio:

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

Exemplos:

```bash
# Busca por similaridade vetorial
moneyball find-similar "Pedro" --season 2025 --limit 10

# Scout report completo em markdown
moneyball scout-report "Endrick" --output endrick.md

# Previsão de uma partida
moneyball predict "Palmeiras" "Flamengo"

# Value bets da rodada com edge mínimo de 5%
moneyball value-bets --min-edge 0.05

# Backtest walk-forward da temporada
moneyball backtest --season 2025
```

---

## Métricas & Resultados

Resultados do modelo em backtest progressivo (`docs/postmortem/`):

| Versão | Contribuição | Métrica |
|---|---|---|
| v1.9.0 — Dixon-Coles + Platt | Correção em placares baixos | Brier **0.2437** |
| v1.11.0 — Isotonic/Temperature auto-select | Calibração não-paramétrica | ECE **0.060 → 0.028** |
| v1.12.0 — Bivariate Poisson + backfill 2022-2025 | 1610 matches treinados | n = 409 → 1610 |
| v1.13.0 — Market blend + draw floor | Blending odds Pinnacle × modelo | Brier melhora 9% |
| v1.14.0 — CatBoost 1x2 + Pi-Rating | Substituição do motor 1x2 | RPS **< 0.22** |
| v1.15.0 — Context features | xG form EMA, coach profile, standings | FEATURE_DIM 24 → 43 |

Ver [`docs/roadmap/central.md`](docs/roadmap/central.md) e cada postmortem em [`docs/postmortem/`](docs/postmortem/) pra métricas detalhadas e reliability diagrams.

---

## Desenvolvimento

```bash
# Pureza do domain (deve retornar vazio)
grep -r "from statsbombpy\|from sqlalchemy\|import matplotlib\|from fastapi\|from typer" football_moneyball/domain/

# Lint
python3 -m ruff check football_moneyball/

# Testes
pytest tests/ -v

# Deploy no Minikube
./deploy.sh 1.15.0
./connect.sh
```

### Workflow de contribuição

Cada feature nova segue:

1. **Pitch** em `docs/pitches/<feature>.md` — problema, solução, arquitetura, scope, testing, success criteria
2. **Research** em `docs/research/` quando necessário — papers, benchmarks, datasets
3. **Plan mode** — plano de implementação
4. **Implement** seguindo o plano, com testes unitários no domain
5. **Postmortem** em `docs/postmortem/` após shipped — métricas, decisões, próximos passos

Inspirado no [Shape Up](https://basecamp.com/shapeup) do Basecamp.

---

## Status

**Arquivado.** Este projeto foi desenvolvido como exploração pessoal de analytics de futebol, modelos preditivos e arquitetura hexagonal em Python. Chegou à v1.15.0 com pipeline end-to-end funcional, mas não está mais em desenvolvimento ativo.

O código continua servindo como referência de:
- Arquitetura hexagonal aplicada a um domínio de data science
- Pipeline ML em produção (ingest → train → calibrate → predict → value)
- Integração multi-provider com cache e fallback
- Documentação iterativa via pitches + postmortems

Ideias de continuidade estão documentadas em [`docs/pitches/`](docs/pitches/) (v1.16 calibração monitorada, v1.17 hyperopt com Optuna + SHAP).

---

## Créditos

Dados:
- [StatsBomb Open Data](https://github.com/statsbomb/open-data) — events, lineups, 360 data
- [Sofascore](https://www.sofascore.com) — Brasileirão Série A
- [The Odds API](https://the-odds-api.com) — bookmaker odds
- [Betfair Exchange](https://www.betfair.com) — exchange odds via `betfairlightweight`

Métodos:
- **Dixon-Coles** (1997) — correção de placares baixos em modelos Poisson
- **xT** — Karun Singh (2018)
- **VAEP** — KU Leuven / Tom Decroos
- **RAPM** — Jeremias Engelmann / adaptado do basquete
- **Pi-Rating** — Constantinou & Fenton (2013)
- **Shape Up** — Basecamp (workflow de pitches)
