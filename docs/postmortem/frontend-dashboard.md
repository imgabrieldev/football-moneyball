---
tags:
  - pitch
  - frontend
  - react
  - vite
  - dashboard
  - k8s
---

# Pitch — Frontend Dashboard (v0.7.0)

## Problem

The whole system is CLI-only. It has 15 commands, 7 API endpoints, data for 87 matches, Monte Carlo predictions and value bets — but everything is only accessible via terminal. There's no real-time visualization, no way to share analyses, no interface to follow the matchday.

The backend is complete (v0.1.0 to v0.6.0):
- FastAPI API with 7 working endpoints
- Dixon-Coles predictions with Brier 0.70
- Value bets with real odds from 39 bookmakers
- Backtesting with 87 Brasileirão 2026 matches
- K8s with PostgreSQL + automation CronJobs

The visual layer is missing.

## Solution

React + Vite SPA with 6 pages consuming the FastAPI. Dark-theme design (consistent with the matplotlib plots we already have). Deploy as an nginx container on Minikube.

### Pages

#### 1. Dashboard (Home) — `/`
Current matchday overview:
- Cards with next matches and predictions (P(H) / P(D) / P(A))
- Highlight value bets of the matchday (edge > 3%)
- Summary: matches analyzed, Brier score, backtest ROI

#### 2. Predictions — `/predictions`
All matchday predictions in a table:
- Home Team vs Away Team
- Expected xG (home/away)
- 1X2 probabilities with visual bar
- Over/Under 2.5, BTTS
- Most likely scoreline
- Expandable score matrix

#### 3. Value Bets — `/value-bets`
Value bet scanner:
- Table: match, market, bet, model%, odds, edge%, EV, Kelly stake
- Filters: minimum edge, market (1X2, O/U), bankroll
- Sort by edge or EV
- Colors: green (high edge), yellow (moderate)

#### 4. Players — `/players`
Players table with metrics:
- Filter by team
- Columns: matches, minutes, goals, xG, assists, xA, passes, tackles
- Sort by any column
- Visual xG vs actual goals bar (over/underperformance)

#### 5. Backtest — `/backtest`
Backtesting results:
- Metrics panel: ROI, hit rate, Brier, drawdown
- Bankroll-over-time chart (line chart)
- Recent bets table (won/lost)

#### 6. Verification — `/verify`
Model vs reality:
- Table: match, scoreline, prediction, result, hit?
- Metrics: 1X2 accuracy, O/U accuracy, Brier score
- Visual calibration (predicted prob vs actual frequency)

### Tech Stack

```
React 19 + Vite 6
├── React Router (client-side routing)
├── TanStack Query (data fetching + cache)
├── Tailwind CSS (styling — dark theme)
├── Recharts (charts — bankroll, calibration)
└── Lucide React (icons)
```

No additional backend — consumes the FastAPI directly (`/api/*`).

## Architecture

### Frontend structure

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.ts
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   └── client.ts          # fetch wrapper for API
│   ├── components/
│   │   ├── Layout.tsx          # sidebar + header
│   │   ├── MatchCard.tsx       # match card with prediction
│   │   ├── ProbabilityBar.tsx  # visual H/D/A bar
│   │   ├── ValueBetRow.tsx     # value bet row
│   │   ├── PlayerTable.tsx     # players table
│   │   └── BankrollChart.tsx   # evolution chart
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Predictions.tsx
│   │   ├── ValueBets.tsx
│   │   ├── Players.tsx
│   │   ├── Backtest.tsx
│   │   └── Verify.tsx
│   └── lib/
│       └── utils.ts            # formatting, colors, helpers
├── Dockerfile
└── nginx.conf                  # serves SPA + proxy /api → moneyball-api
```

### Affected backend modules

No Python module needs to change. The frontend consumes the existing API.

Possible improvement: add specific CORS origin in `api.py` instead of `*`.

### Schema

No changes in PostgreSQL.

### Infra (K8s)

```
k8s/
├── (existing)
├── frontend-deployment.yaml    # nginx + SPA
├── frontend-service.yaml       # port 3000
└── kustomization.yaml          # + frontend resources
```

Nginx config:
```nginx
server {
    listen 3000;
    root /usr/share/nginx/html;
    
    location /api/ {
        proxy_pass http://moneyball-api:8000;
    }
    
    location / {
        try_files $uri /index.html;
    }
}
```

Access: `kubectl port-forward -n football-moneyball svc/moneyball-frontend 3000:3000`

## Scope

### In Scope

- [ ] Setup React + Vite + Tailwind + React Router
- [ ] API client (`src/api/client.ts`)
- [ ] Layout with dark-theme sidebar
- [ ] Dashboard page (match cards + value bets)
- [ ] Predictions page (table with probabilities)
- [ ] Value Bets page (table with filters)
- [ ] Players page (table with metrics, team filter)
- [ ] Backtest page (metrics + bankroll chart)
- [ ] Verify page (table + accuracy)
- [ ] Dockerfile (nginx + SPA build)
- [ ] K8s manifests (deployment + service)
- [ ] Nginx proxy /api → moneyball-api

### Out of Scope

- Authentication / login
- Mobile mode (desktop-first, basic responsive)
- Real-time updates (WebSocket)
- E2E tests (Cypress, Playwright)
- CI/CD pipeline
- PWA / offline
- Internationalization (PT-BR only)

## Research Needed

- [ ] Confirm that Vite 6 + React 19 work with the Node version available on Arch
- [ ] Choose between Recharts vs Chart.js for charts

## Testing Strategy

### Frontend
- Manual: navigate each page and verify data
- Verify the API returns correct data (already tested in v0.6.0)

### Integration
- `docker build` of the frontend
- `kubectl apply` and verify nginx serves the SPA
- Verify the `/api` proxy works

## Success Criteria

- [ ] 6 navigable pages with real Brasileirão data
- [ ] Dashboard shows matchday predictions
- [ ] Value bets listed with edge, odds and stake
- [ ] Players table with team filter
- [ ] Backtest shows bankroll chart
- [ ] Runs on Minikube via `kubectl port-forward`
- [ ] Load time < 2s per page
- [ ] Consistent dark theme
