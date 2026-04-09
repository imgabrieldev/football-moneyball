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
- **v1.12.0** — Bivariate Poisson + diagonal inflation + backfill 2022-2025 (1610 matches)
- **v1.13.0** — Market blend inversion + draw floor + odds features como input
- **v1.14.0** — CatBoost 1x2 + Pi-Rating (end-to-end predictor, substituiu Poisson para 1x2)
- **v1.14.1** — Expanded features infra + edge 5% + Pi-Rating probs (2026-04-06)

## Active

Nenhum pitch ativo — wrap-up de sessão.

## Backlog priorizado

### v1.15.0 — UX polish (v0.8.0 backlog)
- Deduplicar value bets (1 linha = melhor odd por aposta)
- Cards com interpretação textual em PT-BR
- Filtro de bookmaker (só Betfair)
- Responsivo mobile básico

### v1.16.0 — Calibração monitorada
- Cron job diário: computar ECE em últimas N predições resolvidas
- Re-fit automático quando ECE > threshold
- Alerta/log quando método vencedor mudar

### v1.17.0 — CatBoost hyperopt + feature selection
- Optuna/hyperopt para CatBoost params
- SHAP-based feature pruning
- Temporal walk-forward validation expandido

## Metas

| Métrica | v1.10 | v1.14.1 | Meta |
|---|---|---|---|
| Brier 1x2 (full pipeline) | 0.244 | TBD (CatBoost) | **0.20** |
| ECE | ~0.08 | TBD | <0.03 |
| Accuracy 1x2 | 42% | TBD | 50-54% |
| Dataset size | 409 | 1610 | — |
| Motor 1x2 | Poisson MC | CatBoost MultiClass | — |
| Rating system | Elo | Pi-Rating | — |

**Mudanças arquiteturais v1.12→v1.14:**
- Bivariate Poisson (v1.12) substituiu Poisson independente para score sampling (corners, cards, CS, HT/FT)
- CatBoost (v1.14) substituiu Poisson MC para 1x2 — aprende P(H,D,A) end-to-end
- Pi-Rating (v1.14) substituiu Elo — ratings home/away independentes, convergência mais rápida
- Market blend inversion (v1.13) — odds como features do modelo, não só pós-blend
- Dataset 4x maior (409→1610) viabilizou ML

## Princípios

- **Calibração > Acurácia** pra ROI de betting (+34% vs -35% ROI, validado por Wilkens 2024)
- Meta realista: **Brier 0.20** (linha de bookmakers RPS 0.17-0.22)
- **Hexagonal arch mantida**: domain puro, adapters para DB/API/viz
- **Code inglês, output PT-BR, docs PT-BR**
