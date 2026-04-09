---
tags:
  - pitch
  - k8s
  - cronjob
  - fastapi
  - automation
  - infra
---

# Pitch — Infra & K8s Automation (v0.6.0)

## Problem

The system works but is 100% manual and fragile:

1. **Odds in local JSON** — `data/odds_*.json` and `data/snapshots/` only exist on the local filesystem. If the PC restarts without a backup, we lose history. The `match_odds` table exists in PostgreSQL but isn't used.

2. **Manual ingestion** — to update Sofascore data, you need to run a script manually. Data gets stale between matchdays.

3. **Underused K8s** — Minikube cluster runs only PostgreSQL. No CronJob, no automation.

4. **CLI-only** — no REST API, impossible to integrate with frontend, bot or mobile.

5. **Manual port-forward** — need to run `kubectl port-forward` every time the PC restarts.

## Solution

### A. Persist odds in PostgreSQL

Move from local JSON to `match_odds` table (already in schema). odds_provider saves to the database via repository instead of file.

### B. Application container

Create Dockerfile to package moneyball as a container. Runs on K8s alongside PostgreSQL.

```
k8s/
├── postgres/ (already exists)
├── moneyball-app/
│   ├── Dockerfile
│   └── deployment.yaml
└── cronjobs/
    ├── ingest-sofascore.yaml    # every 6h
    ├── snapshot-odds.yaml        # 1x/day
    └── predict-matchday.yaml     # 2h before matches
```

### C. K8s CronJobs

3 CronJobs using the same moneyball container:

1. **ingest-sofascore** (`0 */6 * * *` = every 6h)
   - `moneyball ingest --provider sofascore`
   - Fetches new matches, persists player_match_metrics

2. **snapshot-odds** (`0 8 * * *` = every day 8am)
   - `moneyball snapshot-odds`
   - Fetches odds from The Odds API, persists in match_odds

3. **predict-matchday** (`0 16 * * 3,6` = Wed and Sat 4pm, ~2h before matches)
   - `moneyball predict-all`
   - Runs predictor for all matches of the day, persists in match_predictions

### D. FastAPI (basic endpoints)

Minimal API to serve data to future frontend (v0.7.0):

```
GET /api/predictions          — matchday predictions
GET /api/predictions/{id}     — single match prediction
GET /api/value-bets           — current value bets
GET /api/backtest             — backtesting results
GET /api/verify               — model vs reality
GET /health                   — healthcheck
```

Runs as a Deployment on K8s (port 8000).

### E. CLI: new automation commands

```bash
moneyball ingest --provider sofascore   # delta ingest
moneyball snapshot-odds                  # save odds to PG
moneyball predict-all                    # predict all matches of the day
```

## Architecture

### Affected modules

| Module | Action | Description |
|--------|------|-----------|
| `adapters/odds_provider.py` | MODIFY | Persist odds in PG via repo (not JSON) |
| `adapters/postgres_repository.py` | MODIFY | Queries for match_odds, save odds |
| `use_cases/ingest_matches.py` | NEW | Delta ingest Sofascore |
| `use_cases/snapshot_odds.py` | NEW | Snapshot odds → PG |
| `use_cases/predict_all.py` | NEW | Predict all matches of the day |
| `api.py` | NEW | FastAPI endpoints |
| `cli.py` | MODIFY | New commands |
| `Dockerfile` | NEW | Application container |
| `k8s/app-deployment.yaml` | NEW | moneyball deployment |
| `k8s/cronjob-ingest.yaml` | NEW | ingest CronJob |
| `k8s/cronjob-odds.yaml` | NEW | odds CronJob |
| `k8s/cronjob-predict.yaml` | NEW | predict CronJob |

### Schema

No changes — the `match_odds` table already exists, it just wasn't used.

### Infra (K8s)

```
Namespace: football-moneyball
├── Deployment: postgres (existing)
├── Deployment: moneyball-api (NEW — FastAPI port 8000)
├── Service: postgres (existing)
├── Service: moneyball-api (NEW)
├── CronJob: ingest-sofascore (NEW)
├── CronJob: snapshot-odds (NEW)
├── CronJob: predict-matchday (NEW)
├── ConfigMap: postgres-init (existing)
├── ConfigMap: moneyball-config (NEW — ODDS_API_KEY, etc.)
├── Secret: postgres-secret (existing)
└── PVC: postgres-pvc (existing)
```

## Scope

### In Scope

- [ ] Persist odds in PostgreSQL (use existing match_odds table)
- [ ] Dockerfile to package moneyball as container
- [ ] CronJob: ingest-sofascore (every 6h)
- [ ] CronJob: snapshot-odds (daily)
- [ ] CronJob: predict-matchday (pre-match)
- [ ] FastAPI with 6 basic endpoints (read-only)
- [ ] CLI: `ingest`, `snapshot-odds`, `predict-all`
- [ ] K8s manifests (deployment, service, cronjobs, configmap)
- [ ] Deployment: moneyball-api on K8s
- [ ] Remove dependency on `kubectl port-forward` for API

### Out of Scope

- Frontend/dashboard (v0.7.0)
- Telegram/Discord alerts (v0.7.0)
- API authentication
- CI/CD pipeline
- Monitoring/observability (Grafana, Prometheus)
- Multi-cluster (Minikube only)

## Research Needed

- [x] K8s CronJobs with Python — confirmed via official docs
- [x] FastAPI + SQLAlchemy async — abundant documentation
- [ ] Best way to share code between CLI and API (monorepo)
- [ ] Minimal Docker image for Python 3.14 + dependencies

## Testing Strategy

### Unit
- New use cases (ingest, snapshot, predict_all) with mocks

### Integration
- FastAPI endpoints with test client
- CronJobs: run manually and verify data in PG

### Manual
- `kubectl apply -k k8s/` → all resources created
- Verify CronJobs execute on schedule
- `curl http://localhost:8000/api/predictions` → valid JSON

## Success Criteria

- [ ] Odds persisted in PostgreSQL (no longer in local JSON)
- [ ] CronJobs run on schedule and update data
- [ ] `curl /api/predictions` returns valid predictions
- [ ] `curl /health` returns 200
- [ ] `moneyball ingest` updates data without manual intervention
- [ ] moneyball container runs on K8s without port-forward for the app
- [ ] Zero regression in existing CLI commands
