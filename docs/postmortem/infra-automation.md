---
tags:
  - pitch
  - k8s
  - cronjob
  - fastapi
  - automation
  - infra
---

# Pitch — Infra & Automação K8s (v0.6.0)

## Problema

O sistema funciona mas é 100% manual e frágil:

1. **Odds em JSON local** — `data/odds_*.json` e `data/snapshots/` existem só no filesystem local. Se o PC reiniciar sem backup, perdemos o histórico. A tabela `match_odds` existe no PostgreSQL mas não é usada.

2. **Ingestão manual** — pra atualizar dados do Sofascore, precisa rodar script manualmente. Dados ficam stale entre rodadas.

3. **K8s subutilizado** — cluster Minikube roda só PostgreSQL. Nenhum CronJob, nenhuma automação.

4. **CLI-only** — sem API REST, impossível integrar com frontend, bot ou mobile.

5. **Port-forward manual** — precisa rodar `kubectl port-forward` toda vez que reinicia o PC.

## Solução

### A. Persistir odds no PostgreSQL

Mover de JSON local pra tabela `match_odds` (já existe no schema). O odds_provider salva no banco via repository ao invés de arquivo.

### B. Container da aplicação

Criar Dockerfile pra empacotar o moneyball como container. Roda no K8s junto com o PostgreSQL.

```
k8s/
├── postgres/ (já existe)
├── moneyball-app/
│   ├── Dockerfile
│   └── deployment.yaml
└── cronjobs/
    ├── ingest-sofascore.yaml    # a cada 6h
    ├── snapshot-odds.yaml        # 1x/dia
    └── predict-matchday.yaml     # 2h antes dos jogos
```

### C. CronJobs K8s

3 CronJobs usando o mesmo container moneyball:

1. **ingest-sofascore** (`0 */6 * * *` = a cada 6h)
   - `moneyball ingest --provider sofascore`
   - Busca jogos novos, persiste player_match_metrics

2. **snapshot-odds** (`0 8 * * *` = todo dia 8h)
   - `moneyball snapshot-odds`
   - Busca odds da The Odds API, persiste em match_odds

3. **predict-matchday** (`0 16 * * 3,6` = qua e sab 16h, ~2h antes dos jogos)
   - `moneyball predict-all`
   - Roda predictor pra todos os jogos do dia, persiste em match_predictions

### D. FastAPI (endpoints básicos)

API mínima pra servir dados ao frontend futuro (v0.7.0):

```
GET /api/predictions          — previsões da rodada
GET /api/predictions/{id}     — previsão de um jogo
GET /api/value-bets           — value bets atuais
GET /api/backtest             — resultados do backtesting
GET /api/verify               — modelo vs realidade
GET /health                   — healthcheck
```

Roda como Deployment no K8s (porta 8000).

### E. CLI: novos comandos de automação

```bash
moneyball ingest --provider sofascore   # delta ingest
moneyball snapshot-odds                  # salvar odds no PG
moneyball predict-all                    # prever todos os jogos do dia
```

## Arquitetura

### Módulos afetados

| Módulo | Ação | Descrição |
|--------|------|-----------|
| `adapters/odds_provider.py` | MODIFICAR | Persistir odds no PG via repo (não JSON) |
| `adapters/postgres_repository.py` | MODIFICAR | Queries pra match_odds, save odds |
| `use_cases/ingest_matches.py` | NOVO | Delta ingest Sofascore |
| `use_cases/snapshot_odds.py` | NOVO | Snapshot odds → PG |
| `use_cases/predict_all.py` | NOVO | Predict todos jogos do dia |
| `api.py` | NOVO | FastAPI endpoints |
| `cli.py` | MODIFICAR | Novos comandos |
| `Dockerfile` | NOVO | Container da aplicação |
| `k8s/app-deployment.yaml` | NOVO | Deployment moneyball |
| `k8s/cronjob-ingest.yaml` | NOVO | CronJob ingest |
| `k8s/cronjob-odds.yaml` | NOVO | CronJob odds |
| `k8s/cronjob-predict.yaml` | NOVO | CronJob predict |

### Schema

Sem mudanças — tabela `match_odds` já existe, só não era usada.

### Infra (K8s)

```
Namespace: football-moneyball
├── Deployment: postgres (existente)
├── Deployment: moneyball-api (NOVO — FastAPI porta 8000)
├── Service: postgres (existente)
├── Service: moneyball-api (NOVO)
├── CronJob: ingest-sofascore (NOVO)
├── CronJob: snapshot-odds (NOVO)
├── CronJob: predict-matchday (NOVO)
├── ConfigMap: postgres-init (existente)
├── ConfigMap: moneyball-config (NOVO — ODDS_API_KEY, etc.)
├── Secret: postgres-secret (existente)
└── PVC: postgres-pvc (existente)
```

## Escopo

### Dentro do Escopo

- [ ] Persistir odds no PostgreSQL (usar tabela match_odds existente)
- [ ] Dockerfile pra empacotar moneyball como container
- [ ] CronJob: ingest-sofascore (a cada 6h)
- [ ] CronJob: snapshot-odds (diário)
- [ ] CronJob: predict-matchday (pré-jogo)
- [ ] FastAPI com 6 endpoints básicos (read-only)
- [ ] CLI: `ingest`, `snapshot-odds`, `predict-all`
- [ ] K8s manifests (deployment, service, cronjobs, configmap)
- [ ] Deployment: moneyball-api no K8s
- [ ] Remover dependência de `kubectl port-forward` pra API

### Fora do Escopo

- Frontend/dashboard (v0.7.0)
- Alertas Telegram/Discord (v0.7.0)
- Autenticação na API
- CI/CD pipeline
- Monitoring/observability (Grafana, Prometheus)
- Multi-cluster (só Minikube)

## Research Necessária

- [x] K8s CronJobs com Python — confirmado via docs oficiais
- [x] FastAPI + SQLAlchemy async — documentação abundante
- [ ] Melhor forma de compartilhar código entre CLI e API (monorepo)
- [ ] Imagem Docker mínima pra Python 3.14 + dependências

## Estratégia de Testes

### Unitários
- Use cases novos (ingest, snapshot, predict_all) com mocks

### Integração
- FastAPI endpoints com test client
- CronJobs: rodar manualmente e verificar dados no PG

### Manual
- `kubectl apply -k k8s/` → todos os recursos criados
- Verificar CronJobs executam no schedule
- `curl http://localhost:8000/api/predictions` → JSON válido

## Critérios de Sucesso

- [ ] Odds persistidas no PostgreSQL (não mais em JSON local)
- [ ] CronJobs rodam no schedule e atualizam dados
- [ ] `curl /api/predictions` retorna previsões válidas
- [ ] `curl /health` retorna 200
- [ ] `moneyball ingest` atualiza dados sem intervenção manual
- [ ] Container moneyball roda no K8s sem port-forward pra app
- [ ] Zero regressão nos comandos CLI existentes
