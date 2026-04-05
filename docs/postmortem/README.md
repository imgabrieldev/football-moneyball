---
tags:
  - index
  - postmortem
aliases:
  - postmortems
---

# Postmortems

Índice de features implementadas com retrospectivas.

## Foundation (v0.x)

- [[state-of-the-art-engine]] — v0.2.0: Pipeline analytics + ~45 métricas StatsBomb
- [[hexagonal-architecture]] — v0.3.0: Refactor pra ports & adapters
- [[betting-value-finder]] — v0.4.0: Detector de value bets (Kelly + edge)
- [[advanced-predictor]] — v0.5.0: Monte Carlo + Poisson
- [[infra-automation]] — v0.6.0: K8s cronjobs (ingest, predict, snapshot)
- [[frontend-dashboard]] — v0.7.0: React frontend (moneyball-frontend pod)
- [[track-record]] — v0.9.0: Resolve/verify + track-record CLI

## Core Predictor (v1.x)

- [[multi-market]] — v1.0.0: Todos os mercados Betfair (correct score, AH, corners, cards, HT)
- [[comprehensive-predictor]] — v1.1.0→v1.4.0: Framework 5-camadas (player-aware λ + multi-output Poisson + ML + player props) — Brier 0.2413→0.2158 em 92 jogos reais
- [[feature-rich-predictor]] — v1.5.0: Elo + 10 features novas pro XGBoost (SHAP-based)
- [[context-aware-predictor]] — v1.6.0: Features contextuais (técnico, desfalques, standings, fadiga) — 16 features, FEATURE_DIM 24→40
- [[calibration-dixon-coles-platt]] — v1.9.0: Dixon-Coles ρ correction + Platt scaling 1x2
- [[v1.10-h2h-referee-market]] — v1.10.0: H2H + Referee features + market blending (65/35)
