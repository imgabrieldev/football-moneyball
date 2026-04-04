---
tags:
  - index
  - pitch
---

# Pitches

Propostas de features em design ou implementação.

## Ativos

- [[advanced-predictor]] — v0.5.0: Pipeline 6 estágios (xG + pressing + xT + RAPM + regressão à média + Monte Carlo)

## Backlog

- **v0.6.0 — Infra & Automação K8s** — Persistir odds no PostgreSQL (sair de JSON local). CronJobs K8s: ingest Sofascore (6h), snapshot odds (diário), predict pré-jogo (2h antes). FastAPI para frontend. Alertas Telegram/Discord.
- **v0.7.0 — API REST + Frontend** — Endpoints: /predict, /value-bets, /backtest, /players, /verify. Dashboard web com resultados e value bets.

## Finalizados → `docs/postmortem/`

- [[state-of-the-art-engine]] — v0.2.0
- [[hexagonal-architecture]] — v0.3.0
- [[betting-value-finder]] — v0.4.0

## Template

Use `/pitch <nome-da-feature>` para criar um novo pitch.
