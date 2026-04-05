---
tags:
  - roadmap
---

# Roadmap — Football Moneyball

## Shipped

### Foundation (v0.x)

- **v0.1.0** — Core Analytics: metrics (~45), pass networks, embeddings PCA, RAPM, pgvector similarity, scout reports, CLI + K8s
- **v0.2.0** — State-of-the-art engine: pipeline analytics StatsBomb
- **v0.3.0** — Hexagonal architecture: ports & adapters refactor
- **v0.4.0** — Betting value finder: Kelly + edge detection
- **v0.5.0** — Advanced predictor: Monte Carlo + Poisson
- **v0.6.0** — Infra automation: K8s cronjobs (ingest/predict/snapshot)
- **v0.7.0** — Frontend dashboard: React (moneyball-frontend)
- **v0.8.0** — UX polish (backlog)
- **v0.9.0** — Track record: resolve/verify + CLI track-record

### Core Predictor (v1.x)

- **v1.0.0** — Multi-market: todos mercados Betfair (CS, AH, corners, cards, HT)
- **v1.1.0-v1.4.0** — Comprehensive predictor: player-aware λ + multi-output Poisson + ML (Brier 0.241→0.216)
- **v1.5.0** — Feature-rich: Elo + 10 features SHAP-based
- **v1.6.0** — Context-aware: técnico, desfalques, standings, fadiga (16 features, FEATURE_DIM 40)
- **v1.7.0** — Post-match analysis + playing style stats
- **v1.8.0** — Playing style features (FEATURE_DIM 48)
- **v1.8.1** — 2024 data ingested + ensemble 60/40
- **v1.9.0** — Dixon-Coles ρ correction + Platt scaling 1x2
- **v1.10.0** — H2H + Referee features + market blending (65/35)
- **v1.11.0** — Isotonic/Temperature calibration + auto method selection (2026-04-05)

## Active

- [[pitches/isotonic-calibration-v1.11]] — shipped, aguardando mais dados pra isotonic dominar

## Backlog priorizado

### v1.12.0 — Backfill histórico 2022/2023
**Motivação**: n=409 ainda insuficiente pra isotonic (precisa n≥1000). +760 matches destravam.
- Ingestão Sofascore das temporadas 2022+2023 do Brasileirão
- Re-fit calibração com n~1170 — esperado isotonic vencer CV
- Re-train modelos ML goals/corners/cards com dataset maior

### v1.13.0 — Calibração monitorada
- Cron job diário: computar ECE em últimas N predições resolvidas
- Re-fit automático quando ECE > 0.015
- Alerta/log quando método vencedor mudar

### v1.14.0 — Beta calibration / Hybrid
- Beta calibration (3-param, tail skew)
- Hybrid: temperature scaling + isotonic nos resíduos
- Só implementar se v1.12/v1.13 não atingirem meta Brier 0.20

### v1.15.0 — UX polish (v0.8.0 backlog)
- Deduplicar value bets (1 linha = melhor odd por aposta)
- Cards com interpretação textual em PT-BR
- Filtro de bookmaker (só Betfair)
- Responsivo mobile básico

## Metas

| Métrica | Atual (v1.10) | Pitch original | Research-based |
|---|---|---|---|
| Brier 1x2 (full pipeline) | 0.244 | <0.19 | **0.20** |
| ECE | ~0.08 | — | <0.03 |
| Accuracy 1x2 | 42% | >54% | 50-54% |
| Feature importance top-10 | 0 context features | ≥2 | — |

**Descobertas empíricas (2026-04-05):**
- v1.6.0 context features estão no modelo mas **não aparecem no top-15 de importance** → signal já absorvido por Elo/xG
- v1.8.0 playing style features também ausentes do top-15
- Market blending (v1.10.0) reduz Brier ~30% em matches individuais overconfident
- Calibração atual (Platt) cai 81%→67% — ainda insuficiente, mas auto-select valida que Platt é ótimo pra n<1000

## Princípios

- **Calibração > Acurácia** pra ROI de betting (+34% vs -35% ROI, validado por Wilkens 2024)
- Meta realista: **Brier 0.20** (linha de bookmakers RPS 0.17-0.22)
- **Hexagonal arch mantida**: domain puro, adapters para DB/API/viz
- **Code inglês, output PT-BR, docs PT-BR**
