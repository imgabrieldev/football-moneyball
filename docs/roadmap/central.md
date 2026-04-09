---
tags:
  - roadmap
---

# Roadmap — Football Moneyball

## Shipped

### Foundation (v0.x)

- **v0.1.0** — Core Analytics: metrics (~45), pass networks, PCA embeddings, RAPM, pgvector similarity, scout reports, CLI + K8s
- **v0.2.0** — State-of-the-art engine: StatsBomb analytics pipeline
- **v0.3.0** — Hexagonal architecture: ports & adapters refactor
- **v0.4.0** — Betting value finder: Kelly + edge detection
- **v0.5.0** — Advanced predictor: Monte Carlo + Poisson
- **v0.6.0** — Infra automation: K8s cronjobs (ingest/predict/snapshot)
- **v0.7.0** — Frontend dashboard: React (moneyball-frontend)
- **v0.8.0** — UX polish (backlog)
- **v0.9.0** — Track record: resolve/verify + CLI track-record

### Core Predictor (v1.x)

- **v1.0.0** — Multi-market: all Betfair markets (CS, AH, corners, cards, HT)
- **v1.1.0-v1.4.0** — Comprehensive predictor: player-aware λ + multi-output Poisson + ML (Brier 0.241→0.216)
- **v1.5.0** — Feature-rich: Elo + 10 SHAP-based features
- **v1.6.0** — Context-aware: coach, absences, standings, fatigue (16 features, FEATURE_DIM 40)
- **v1.7.0** — Post-match analysis + playing style stats
- **v1.8.0** — Playing style features (FEATURE_DIM 48)
- **v1.8.1** — 2024 data ingested + 60/40 ensemble
- **v1.9.0** — Dixon-Coles ρ correction + 1x2 Platt scaling
- **v1.10.0** — H2H + referee features + market blending (65/35)
- **v1.11.0** — Isotonic/Temperature calibration + auto method selection (2026-04-05)
- **v1.12.0** — Bivariate Poisson + diagonal inflation + 2022-2025 backfill (1610 matches)
- **v1.13.0** — Market blend inversion + draw floor + odds features as input
- **v1.14.0** — CatBoost 1x2 + Pi-Rating (end-to-end predictor, replaced Poisson for 1x2)
- **v1.14.1** — Expanded features infra + 5% edge + Pi-Rating probs (2026-04-06)

## Active

No active pitches — session wrap-up.

## Prioritized Backlog

### v1.15.0 — UX polish (v0.8.0 backlog)
- Deduplicate value bets (1 row = best odd per bet)
- Cards with text interpretation in PT-BR
- Bookmaker filter (Betfair only)
- Basic mobile responsiveness

### v1.16.0 — Monitored calibration
- Daily cron job: compute ECE on the last N resolved predictions
- Automatic re-fit when ECE > threshold
- Alert/log when the winning method changes

### v1.17.0 — CatBoost hyperopt + feature selection
- Optuna/hyperopt for CatBoost params
- SHAP-based feature pruning
- Expanded temporal walk-forward validation

## Targets

| Metric | v1.10 | v1.14.1 | Target |
|---|---|---|---|
| Brier 1x2 (full pipeline) | 0.244 | TBD (CatBoost) | **0.20** |
| ECE | ~0.08 | TBD | <0.03 |
| 1x2 Accuracy | 42% | TBD | 50-54% |
| Dataset size | 409 | 1610 | — |
| 1x2 engine | Poisson MC | CatBoost MultiClass | — |
| Rating system | Elo | Pi-Rating | — |

**Architectural changes v1.12→v1.14:**
- Bivariate Poisson (v1.12) replaced independent Poisson for score sampling (corners, cards, CS, HT/FT)
- CatBoost (v1.14) replaced Poisson MC for 1x2 — learns P(H,D,A) end-to-end
- Pi-Rating (v1.14) replaced Elo — independent home/away ratings, faster convergence
- Market blend inversion (v1.13) — odds as model features, not just post-blend
- 4x larger dataset (409→1610) made ML viable

## Principles

- **Calibration > Accuracy** for betting ROI (+34% vs. -35% ROI, validated by Wilkens 2024)
- Realistic target: **Brier 0.20** (bookmaker line RPS 0.17-0.22)
- **Hexagonal arch preserved**: pure domain, adapters for DB/API/viz
- **English code, PT-BR output, PT-BR docs**
