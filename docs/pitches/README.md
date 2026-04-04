---
tags:
  - index
  - pitch
---

# Pitches

Propostas de features em design ou implementação.

## Ativos

(nenhum pitch ativo)

## Backlog

- **v0.5.0 — Modelo Preditivo Avançado** — Brier score atual 0.76, precisa < 0.25. Integrar xT + pressing (PPDA) + RAPM no match_predictor. Regressão à média. Peso de escalação (RAPM dos titulares). Calibração Over/Under.
- **v0.6.0 — API REST + Automação** — FastAPI para integrar com frontend. Refresh automático Sofascore (cron/scheduler). Alertas de value bets (Telegram/Discord). Endpoints: /predict, /value-bets, /backtest, /players.

## Finalizados → `docs/postmortem/`

- [[state-of-the-art-engine]] — v0.2.0
- [[hexagonal-architecture]] — v0.3.0
- [[betting-value-finder]] — v0.4.0

## Template

Use `/pitch <nome-da-feature>` para criar um novo pitch.
