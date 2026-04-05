---
tags:
  - index
  - pitch
---

# Pitches

Propostas de features em design ou implementação.

## Ativos

- [[isotonic-calibration-v1.11]] — v1.11.0: Isotonic + Temperature + auto-select via CV Brier (shipped 2026-04-05)

## Backlog

- [[ux-polish]] — v0.8.0: Deduplicar value bets, cards com interpretação textual, filtro de bookmaker, datas, responsivo
- **v1.12.0 — Backfill 2022/2023** — ingestão histórica extra (~760 matches) pra destravar isotonic (precisa n≥1000)
- **v1.13.0 — Calibração monitorada** — cron de re-fit automático quando ECE > 0.015
- **v1.14.0 — Beta calibration / hybrid** — se isotonic sozinho não atingir meta Brier 0.20

## Finalizados → `docs/postmortem/`

Ver índice completo em [[../postmortem/README|postmortem/README]].

## Template

Use `/pitch <nome-da-feature>` para criar um novo pitch.
